//! 轻量级缓存存储层：用 JSON 文件替代 SQLite，纯 Rust 无 C 编译依赖。
//!
//! 缓存图片指纹（dhash）与元信息，支持增量扫描。
//! 数据存于内存 HashMap，通过 [`Store::flush`] 显式写回磁盘。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// 线程安全的存储句柄。
pub type SharedStore = Arc<parking_lot::Mutex<Store>>;

/// 人脸/场景/闭眼缓存结构的当前 schema 版本。
///
/// 语义变化会断言这个版本：版本不符即视为未缓存，避免旧缓存被静默复用成 stale 值。
/// - v1：闭眼存 `bool is_any_closed`。
/// - v2：闭眼改为存 `min(open_l, open_r)` 连续概率（阶段二连续降权），旧 bool 缓存失效。
/// - v3：聚合由 `min` 改为 `max(open_l, open_r)`（阶段三：单眼 ROI 采到皮肤/眼镜致 0.00
///       不再压垮整张睁眼花，只有双眼都判闭才降权），旧 v2 缓存失效。
/// - v4：新增 `focus_score`（对焦分，增量重扫不再重算对焦），旧 v3 缓存失效。
pub const AI_FACE_CACHE_SCHEMA: u32 = 4;

/// 单张图片的人脸/场景/闭眼 AI 结果缓存（阶段一新增，阶段二升级闭眼语义）。
///
/// 与 `aesthetic_score`/`technical_score` 分离，独立判定"是否已计算"：
/// `None` 表示未计算（旧版缓存缺此字段），调用方需重算；
/// `Some(..)` 表示已计算可复用。字段间互相一致（同一次扫描产出）。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct AiFaceCache {
    /// 结构版本（迁移/失效用）。新版写入时置为 `AI_FACE_CACHE_SCHEMA`，
    /// 旧版/未来不匹配的记录读回时按版本号判为未缓存。
    #[serde(default)]
    pub schema_version: u32,
    pub has_face: bool,
    /// TOPIQ-NR-Face 人脸专评（1~10），无人脸为 None。
    pub face_score: Option<f32>,
    /// 场景（`crate::ai::scene::Scene` 的 repr(u8)，非人像时由场景分类器给出）。
    pub scene: u8,
    /// 开眼概率 `max(open_l, open_r) ∈ [0,1]`（1=至少一眼开，0=双眼全闭）。
    /// 无人脸时无意义，恒为 1.0。
    #[serde(default = "default_eye_open")]
    pub eye_open: f32,
    /// 对焦分（1.0~10.0）：人像/宠物为眼部对焦，其余为整图对焦。默认 1.0（不降权）。
    #[serde(default = "default_focus")]
    pub focus_score: f32,
}

/// 开眼概率默认值（字段缺失/无人脸时视为双眼全开，无闭眼降权）。
fn default_eye_open() -> f32 {
    1.0
}

/// 对焦分默认值（字段缺失/未启用时视为在焦，无对焦降权）。
fn default_focus() -> f32 {
    1.0
}

/// 单张图片的缓存记录。
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImageRecord {
    pub path: String,
    pub size: u64,
    pub modified: u64,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub dhash: u64,
    /// 平均哈希（用于双哈希校验，过滤 dhash 对平滑渐变图的误判）。
    /// `#[serde(default)]` 保证旧版缓存（无此字段）能正常读取。
    #[serde(default)]
    pub ahash: u64,
    /// CLIP 美学分缓存（0-10 分制），`#[serde(default)]` 兼容旧缓存。
    #[serde(default)]
    pub aesthetic_score: Option<f32>,
    /// NIMA 技术分缓存（0-10 分制）。
    #[serde(default)]
    pub technical_score: Option<f32>,
    /// 人脸/场景/闭眼结果缓存（阶段一：增量扫描复用）。
    /// `#[serde(default)]` 兼容旧缓存（读为 None → 按未缓存处理重算）。
    #[serde(default)]
    pub ai_face_cache: Option<AiFaceCache>,
}

/// 缓存文件内容。
#[derive(Debug, Default, Serialize, Deserialize)]
struct CacheFile {
    records: HashMap<String, ImageRecord>,
}

/// 存储层。
pub struct Store {
    path: PathBuf,
    inner: parking_lot::Mutex<CacheFile>,
}

impl Store {
    /// 打开（或创建）存储。
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let inner = if path.exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|c| serde_json::from_str(&c).ok())
                .unwrap_or_default()
        } else {
            CacheFile::default()
        };
        Ok(Self {
            path: path.to_path_buf(),
            inner: parking_lot::Mutex::new(inner),
        })
    }

    /// 创建线程安全句柄。
    pub fn shared(path: &Path) -> anyhow::Result<SharedStore> {
        Ok(Arc::new(parking_lot::Mutex::new(Self::open(path)?)))
    }

    /// 保存（插入或更新）一张图片的指纹信息（仅更新内存，需调用 [`Self::flush`] 落盘）。
    pub fn save_image(
        &self,
        file_hash: &str,
        path: &str,
        size: u64,
        modified: u64,
        width: u32,
        height: u32,
        format: &str,
        dhash: u64,
        ahash: u64,
    ) -> anyhow::Result<()> {
        let mut guard = self.inner.lock();
        // 保留旧的评分缓存（hash 未变则 AI 评分可复用）
        let prev = guard.records.get(file_hash).cloned();
        guard.records.insert(
            file_hash.to_string(),
            ImageRecord {
                path: path.to_string(),
                size,
                modified,
                width,
                height,
                format: format.to_string(),
                dhash,
                ahash,
                aesthetic_score: prev.as_ref().and_then(|r| r.aesthetic_score),
                technical_score: prev.as_ref().and_then(|r| r.technical_score),
                ai_face_cache: prev.as_ref().and_then(|r| r.ai_face_cache),
            },
        );
        Ok(())
    }

    /// 保存人脸/场景/闭眼/对焦 AI 结果缓存（仅更新内存）。
    /// `eye_open` 为开眼概率 `max(open_l, open_r) ∈ [0,1]`（至少一眼开）。
    pub fn save_ai_face(
        &self,
        file_hash: &str,
        has_face: bool,
        face_score: Option<f32>,
        scene: u8,
        eye_open: f32,
        focus_score: f32,
    ) -> anyhow::Result<()> {
        if let Some(r) = self.inner.lock().records.get_mut(file_hash) {
            r.ai_face_cache = Some(AiFaceCache {
                schema_version: AI_FACE_CACHE_SCHEMA,
                has_face,
                face_score,
                scene,
                eye_open,
                focus_score,
            });
        }
        Ok(())
    }

    /// 查询人脸/场景/闭眼 AI 结果缓存。
    /// 返回 `None` 表示未计算（旧版缓存或从未跑过人脸/场景/闭眼），
    /// 或缓存结构版本与当前 `AI_FACE_CACHE_SCHEMA` 不符（需按新语义重算）。
    pub fn get_cached_ai_face(&self, file_hash: &str) -> Option<AiFaceCache> {
        let cache = self.inner.lock().records.get(file_hash).and_then(|r| r.ai_face_cache)?;
        // 版本不符视为未缓存，避免旧缓存（如闭眼硬阈值语义）被静默复用
        if cache.schema_version != AI_FACE_CACHE_SCHEMA {
            return None;
        }
        Some(cache)
    }

    /// 保存 AI 双维度评分（美学 + 技术，仅更新内存）。
    pub fn save_ai_scores(
        &self,
        file_hash: &str,
        aesthetic: Option<f32>,
        technical: Option<f32>,
    ) -> anyhow::Result<()> {
        if let Some(r) = self.inner.lock().records.get_mut(file_hash) {
            if let Some(a) = aesthetic {
                r.aesthetic_score = Some(a);
            }
            if let Some(t) = technical {
                r.technical_score = Some(t);
            }
        }
        Ok(())
    }

    /// 查询 AI 双维度评分缓存（美学 + 技术）。
    /// 返回 `Some((aesthetic, technical))`，任一缺失则为 `None`（需重算）。
    pub fn get_cached_ai_scores(&self, file_hash: &str) -> Option<(f32, f32)> {
        let guard = self.inner.lock();
        let r = guard.records.get(file_hash)?;
        match (r.aesthetic_score, r.technical_score) {
            (Some(a), Some(t)) => Some((a, t)),
            _ => None,
        }
    }

    /// 清空缓存。
    pub fn clear_cache(&self) -> anyhow::Result<()> {
        self.inner.lock().records.clear();
        self.flush()
    }

    /// 查询某文件指纹的完整缓存记录（增量扫描用）。
    /// 仅当 dhash 与 ahash 都有效（非 0）时返回，否则视为需重新计算。
    pub fn get_cached_record(&self, file_hash: &str) -> Option<ImageRecord> {
        let guard = self.inner.lock();
        let r = guard.records.get(file_hash)?;
        if r.dhash == 0 || r.ahash == 0 {
            return None;
        }
        Some(r.clone())
    }

    /// 将内存中的缓存写回磁盘。
    pub fn flush(&self) -> anyhow::Result<()> {
        let guard = self.inner.lock();
        let content = serde_json::to_string(&*guard)?;
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 每次调用生成唯一临时文件路径（并行测试不互相干扰）。
    fn temp_store() -> (Store, std::path::PathBuf) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "pixsweep_store_test_{}_{}.json",
            std::process::id(),
            seq
        ));
        let _ = std::fs::remove_file(&path);
        let store = Store::open(&path).expect("打开临时 store 失败");
        (store, path)
    }

    fn seed(store: &Store, file_hash: &str) {
        store
            .save_image(file_hash, "E:/p/a.jpg", 1024, 0, 100, 100, "jpeg", 0xAAAA, 0xFF00)
            .expect("save_image 失败");
    }

    #[test]
    fn ai_face_cache_round_trip() {
        let (store, path) = temp_store();
        seed(&store, "h1");

        store
            .save_ai_face("h1", true, Some(6.5), 1, 0.9, 7.0)
            .expect("save_ai_face 失败");
        let cached = store.get_cached_ai_face("h1").expect("应有缓存");
        assert!(cached.has_face);
        assert_eq!(cached.face_score, Some(6.5));
        assert_eq!(cached.scene, 1);
        assert!(cached.eye_open > 0.5);

        // flush 后重新加载，验证持久化
        store.flush().expect("flush 失败");
        let reopened = Store::open(&path).expect("重新打开失败");
        let cached2 = reopened.get_cached_ai_face("h1").expect("应有缓存");
        assert_eq!(cached2.face_score, Some(6.5));
        assert_eq!(cached2.scene, 1);
        assert!((cached2.eye_open - 0.9).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ai_face_cache_missing_by_default() {
        // 旧缓存（无 ai_face_cache 字段）→ 读为 None，按"未缓存"处理
        let (store, path) = temp_store();
        seed(&store, "h1");
        store.flush().expect("flush 失败");

        // 模拟旧版记录：手动写入不含 ai_face_cache 的 JSON
        let json = r#"{"records":{"h1":{"path":"E:/p/a.jpg","size":1024,"modified":0,"width":100,"height":100,"format":"jpeg","dhash":43690,"ahash":65280,"nima_score":null,"aesthetic_score":null,"technical_score":null}}}"#;
        std::fs::write(&path, json).expect("写旧版缓存失败");
        let reopened = Store::open(&path).expect("重开失败");

        assert!(
            reopened.get_cached_ai_face("h1").is_none(),
            "旧记录缺失人脸字段应按未缓存处理"
        );
        // 且不影响既有字段读取
        assert_eq!(reopened.get_cached_record("h1").map(|r| r.dhash), Some(0xAAAA));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_image_preserves_ai_face_cache() {
        let (store, path) = temp_store();
        seed(&store, "h1");
        store
            .save_ai_face("h1", true, Some(7.0), 2, 0.2, 6.5)
            .expect("save_ai_face 失败");

        // 再次 save_image（模拟文件元数据刷新，hash 不变）应保留人脸缓存
        store
            .save_image("h1", "E:/p/a.jpg", 2048, 0, 200, 200, "jpeg", 0xBBBB, 0xFF00)
            .expect("save_image 失败");
        let cached = store.get_cached_ai_face("h1").expect("应保留");
        assert_eq!(cached.face_score, Some(7.0));
        assert_eq!(cached.scene, 2);
        assert!(cached.eye_open < 0.5, "闭眼概率应保留: {}", cached.eye_open);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ai_face_cache_schema_mismatch_is_ignored() {
        let (_store, path) = temp_store();
        // 写入一条 schema_version 与当前不符的人脸缓存（模拟未来版本/旧语义），
        // 读取时应判定为未缓存，避免 stale 值被静默复用。
        let json = r#"{"records":{"h1":{"path":"E:/p/a.jpg","size":1024,"modified":0,"width":100,"height":100,"format":"jpeg","dhash":43690,"ahash":65280,"nima_score":null,"aesthetic_score":null,"technical_score":null,"ai_face_cache":{"schema_version":1,"has_face":true,"face_score":6.0,"scene":1,"eye_closed":true}}}}"#;
        std::fs::write(&path, json).expect("写旧缓存失败");
        let reopened = Store::open(&path).expect("重开失败");

        assert!(
            reopened.get_cached_ai_face("h1").is_none(),
            "schema_version 不符应按未缓存处理"
        );

        let _ = std::fs::remove_file(&path);
    }
}

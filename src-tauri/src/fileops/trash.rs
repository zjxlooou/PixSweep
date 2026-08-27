//! 文件删除 + 应用内置"临时回收站"（隔离区 Quarantine）。
//!
//! 删除流程：
//! - `permanent = false`：把文件复制到 `{app_data_dir}/quarantine/files/{uuid}.{ext}`，
//!   在 `quarantine/index.json` 中追加元数据记录（原始路径、删除时间、大小）。
//!   原文件被删除（等于"先复制后删除"——确保数据完全转移后再清理原文件）。
//! - `permanent = true`：直接 `remove_file`。
//!
//! 恢复流程：根据 id 在 index.json 中查找原路径，把副本写回原位置后清理条目。
//!
//! 清空流程：删除整个 `quarantine/` 目录。

use crate::types::{DeleteFailure, DeleteResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use uuid::Uuid;

/// 隔离区中的一条记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineEntry {
    /// 唯一 ID（UUID）
    pub id: String,
    /// 删除前的原始路径
    pub original_path: String,
    /// 隔离区中的文件名（`files/{filename}`）
    pub quarantine_filename: String,
    /// 删除时间（Unix 秒）
    pub deleted_at: u64,
    /// 文件大小（字节）
    pub size: u64,
}

/// 删除时锁住 index.json 的全局互斥（防止并发损坏）。
static INDEX_LOCK: Mutex<()> = Mutex::new(());

fn quarantine_dir() -> PathBuf {
    // 隔离区放程序根数据目录（app_data_dir 已指向 exe 同级），与其余运行期文件同归位，
    // 不污染系统 Temp（用户要求所有临时文件在程序根目录）。
    crate::app_data_dir().join("quarantine")
}

fn quarantine_files_dir() -> PathBuf {
    quarantine_dir().join("files")
}

fn index_path() -> PathBuf {
    quarantine_dir().join("index.json")
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 读取 index.json 中所有记录。
pub fn load_index() -> Vec<QuarantineEntry> {
    let path = index_path();
    if !path.exists() {
        return Vec::new();
    }
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// 写回 index.json（持锁）。
fn save_index_locked(entries: &[QuarantineEntry]) -> anyhow::Result<()> {
    std::fs::create_dir_all(quarantine_dir())?;
    let json = serde_json::to_string_pretty(entries)?;
    std::fs::write(index_path(), json)?;
    Ok(())
}

/// 添加一条记录到 index.json（持锁）。
fn append_index_locked(entry: &QuarantineEntry) -> anyhow::Result<()> {
    let mut entries = load_index();
    entries.push(entry.clone());
    save_index_locked(&entries)
}

/// 删除单个文件。
///
/// `permanent = false` 时移动到隔离区；`permanent = true` 时直接删除。
pub fn delete_one(path: &Path, permanent: bool) -> anyhow::Result<()> {
    if !path.exists() {
        anyhow::bail!("文件不存在");
    }

    if permanent {
        if path.is_dir() {
            std::fs::remove_dir_all(path)?;
        } else {
            std::fs::remove_file(path)?;
        }
        Ok(())
    } else {
        move_to_quarantine(path)?;
        Ok(())
    }
}

/// 把文件移动到隔离区。
///
/// 流程：创建隔离区目录 → 生成 UUID 文件名 → 复制文件 → 写 index → 删除原文件。
fn move_to_quarantine(path: &Path) -> anyhow::Result<()> {
    if !path.is_file() {
        anyhow::bail!("仅支持文件（不支持目录）");
    }
    let meta = std::fs::metadata(path)?;
    let size = meta.len();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    let uuid = Uuid::new_v4().to_string();

    std::fs::create_dir_all(quarantine_files_dir())?;
    let q_filename = format!("{uuid}.{ext}");
    let q_path = quarantine_files_dir().join(&q_filename);

    // 1) 先复制文件
    std::fs::copy(path, &q_path).map_err(|e| anyhow::anyhow!("复制到隔离区失败: {e}"))?;
    // 验证复制成功（大小匹配）
    let copied_size = std::fs::metadata(&q_path).map(|m| m.len()).unwrap_or(0);
    if copied_size != size {
        // 回滚
        let _ = std::fs::remove_file(&q_path);
        anyhow::bail!(
            "隔离区文件大小不一致 (源 {} vs 目标 {})",
            size,
            copied_size
        );
    }

    // 2) 写 index
    let entry = QuarantineEntry {
        id: uuid.clone(),
        original_path: path.to_string_lossy().to_string(),
        quarantine_filename: q_filename.clone(),
        deleted_at: now_unix(),
        size,
    };
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = append_index_locked(&entry) {
        // 回滚：删除已复制的文件
        let _ = std::fs::remove_file(&q_path);
        return Err(e);
    }

    // 3) 删除原文件
    std::fs::remove_file(path).map_err(|e| {
        // 复制已成功、index 已记录，但原文件删除失败——下次启动可手动清理
        anyhow::anyhow!("隔离成功但原文件删除失败: {e}")
    })?;

    log::info!(
        "[隔离区] 已移动: {} -> {}",
        path.display(),
        q_path.display()
    );
    Ok(())
}

/// 列出隔离区中的所有条目（含进度回调）。
pub fn list_quarantine_with_progress<F: FnMut(usize)>(mut on_progress: F) -> Vec<QuarantineEntry> {
    let entries = list_quarantine();
    let total = entries.len();
    if total > 0 {
        on_progress(total);
    }
    let mut sorted = entries;
    sorted.sort_by(|a, b| b.deleted_at.cmp(&a.deleted_at));
    sorted
}

/// 列出隔离区中的所有条目。
pub fn list_quarantine() -> Vec<QuarantineEntry> {
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    load_index()
}

/// 根据 ID 恢复一个文件到原始路径。
pub fn restore_one(id: &str) -> anyhow::Result<String> {
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut entries = load_index();
    let pos = entries
        .iter()
        .position(|e| e.id == id)
        .ok_or_else(|| anyhow::anyhow!("未找到隔离区记录: {id}"))?;
    let entry = entries.remove(pos);

    let q_path = quarantine_files_dir().join(&entry.quarantine_filename);
    if !q_path.exists() {
        anyhow::bail!("隔离区文件已丢失: {}", q_path.display());
    }
    let target = PathBuf::from(&entry.original_path);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // 目标已存在则不覆盖
    if target.exists() {
        anyhow::bail!("目标已存在: {}", target.display());
    }
    std::fs::copy(&q_path, &target)?;
    // 校验大小
    let src_len = std::fs::metadata(&q_path).map(|m| m.len()).unwrap_or(0);
    let dst_len = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
    if src_len != dst_len {
        anyhow::bail!("恢复校验失败: {} vs {}", src_len, dst_len);
    }
    // 清理：删除隔离区文件 + 更新 index
    let _ = std::fs::remove_file(&q_path);
    save_index_locked(&entries)?;
    log::info!("[隔离区] 已恢复: {}", entry.original_path);
    Ok(entry.original_path)
}

/// 恢复所有文件到原始路径，返回成功数量。
pub fn restore_all_quarantine() -> anyhow::Result<Vec<String>> {
    let ids: Vec<String> = {
        let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        load_index().into_iter().map(|e| e.id).collect()
    };
    let mut paths = Vec::new();
    for id in ids {
        match restore_one(&id) {
            Ok(p) => paths.push(p),
            Err(e) => {
                log::warn!("[隔离区] 恢复失败 {id}: {e}");
            }
        }
    }
    log::info!("[隔离区] 全部恢复: 成功 {}", paths.len());
    Ok(paths)
}

/// 清空隔离区（删除所有文件 + 清空 index）。
pub fn clear_quarantine() -> anyhow::Result<u32> {
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let entries = load_index();
    let count = entries.len() as u32;
    for e in &entries {
        let p = quarantine_files_dir().join(&e.quarantine_filename);
        let _ = std::fs::remove_file(&p);
    }
    let _ = std::fs::remove_dir_all(quarantine_files_dir());
    save_index_locked(&[])?;
    log::info!("[隔离区] 清空: {count} 条");
    Ok(count)
}

/// 仅清空隔离区 index（文件已由调用方移入系统回收站后调用，保持记录一致）。
pub fn clear_quarantine_index() -> anyhow::Result<()> {
    let _guard = INDEX_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    save_index_locked(&[])
}

/// 返回隔离区目录路径（前端用于"打开目录"按钮）。
pub fn quarantine_dir_path() -> PathBuf {
    quarantine_dir()
}

/// 兼容旧版 trash::delete 调用——内部直接走 quarantine。
pub fn delete_files(paths: &[String], permanent: bool) -> DeleteResult {
    let mut deleted = Vec::new();
    let mut failed = Vec::new();
    for path in paths {
        match delete_one(Path::new(path), permanent) {
            Ok(()) => deleted.push(path.clone()),
            Err(e) => failed.push(DeleteFailure {
                path: path.clone(),
                reason: e.to_string(),
            }),
        }
    }
    DeleteResult { deleted, failed }
}

/// 写删除日志（兼容旧 API）。
pub fn append_delete_log(entries: &[String], permanent: bool) -> anyhow::Result<()> {
    let dir = crate::app_data_dir().join("logs");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("deletions.log");
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
    for entry in entries {
        writeln!(
            file,
            "[{}] mode={} path={}",
            now,
            if permanent { "permanent" } else { "quarantine" },
            entry
        )?;
    }
    Ok(())
}
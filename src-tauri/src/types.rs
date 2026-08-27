//! 前后端共享的数据类型（通过 serde 序列化传输）。

use serde::{Deserialize, Serialize};

/// 一张图片的元信息。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    /// 绝对路径
    pub path: String,
    /// 文件名
    pub name: String,
    /// 文件大小（字节）
    pub size: u64,
    /// 修改时间（unix 秒）
    pub modified: u64,
    /// 图片宽度（可能未知）
    pub width: u32,
    /// 图片高度（可能未知）
    pub height: u32,
    /// 文件格式（jpeg/png/webp/...）
    pub format: String,
    /// 内容指纹（blake3(path + size + mtime)）
    pub file_hash: String,
}

/// 扫描阶段枚举。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScanPhase {
    Scanning,
    Hashing,
    Clustering,
    Quality,
    /// 删除文件阶段（用于 delete-progress 事件）
    Deleting,
    Done,
    Error,
}

impl ScanPhase {
    pub fn label(&self) -> &'static str {
        match self {
            ScanPhase::Scanning => "扫描文件夹",
            ScanPhase::Hashing => "计算图片指纹",
            ScanPhase::Clustering => "聚类相似图片",
            ScanPhase::Quality => "AI 质量评分",
            ScanPhase::Deleting => "删除文件",
            ScanPhase::Done => "完成",
            ScanPhase::Error => "出错",
        }
    }
}

/// 扫描进度事件（后端推送给前端）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProgress {
    pub session_id: String,
    pub phase: ScanPhase,
    pub current: usize,
    pub total: usize,
    /// 当前处理的具体文件名（可选）
    pub current_file: Option<String>,
    /// 是否启用 AI 推理
    pub ai_enabled: bool,
    /// 当前阶段使用的硬件/技术（扫描/哈希/聚类=CPU；AI 评分=推理后端如 "CUDA (NVIDIA GPU)"）
    pub backend: String,
    /// 更细粒度的当前子阶段（如 "识别内容 / 识别眼部 / 对焦判断 / 美学评分"）
    pub detail: String,
}

/// 组内单张图片（含质量评分）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupImage {
    /// 图片元信息（保持嵌套，与前端 `img.info.path` 结构一致）
    pub info: ImageInfo,
    /// 综合评分（1.0 ~ 10.0），未启用 AI 时为 None
    pub score: Option<f32>,
    /// CLIP 美学评分（1.0 ~ 10.0），美学头缺失时为 None
    pub aesthetic_score: Option<f32>,
    /// NIMA 技术质量评分（1.0 ~ 10.0），技术模型缺失时为 None
    pub technical_score: Option<f32>,
    /// TOPIQ-NR-Face 人脸专评（1.0 ~ 10.0），无人脸或未启用时为 None
    #[serde(default)]
    pub face_score: Option<f32>,
    /// 是否检测到人脸（用于前端显示图标）
    #[serde(default)]
    pub has_face: bool,
    /// 场景分类（0=其他 1=人像 2=宠物 3=风景），前端展示场景标签
    #[serde(default)]
    pub scene: u8,
    /// 双眼都闭（OCEC 检测，`max(open_l,open_r) <= 0.5`，前端显示"闭眼"标签）
    #[serde(default)]
    pub is_eye_closed: bool,
    /// 对焦分（1.0 ~ 10.0）：人像/宠物为眼部对焦，其余为整图对焦；未启用时为 None
    #[serde(default)]
    pub focus_score: Option<f32>,
    /// 是否失焦（`focus_score` 低于阈值；前端显示"失焦"标签）
    #[serde(default)]
    pub is_out_of_focus: bool,
    /// 是否为推荐保留
    pub recommended: bool,
    /// 推荐/删除理由（AI + 启发式综合），如 "分辨率最高 (4032×3024)"
    pub reason: String,
}

/// 一组相似图片。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGroup {
    /// 唯一组号，格式 `{batch_id}-{6位序号}`，如 `20260818093107-000001`
    pub group_id: String,
    pub images: Vec<GroupImage>,
    /// 组内平均相似度（0~1）
    pub similarity: f32,
    /// 删除其余图片可释放的空间（字节）
    pub reclaimable_bytes: u64,
}

/// 扫描结果（推送给前端）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub session_id: String,
    /// 批次号 yyyyMMddHHmmSS（如 20260818093107），用于日志追溯
    pub batch_id: String,
    pub total_images: usize,
    pub groups: Vec<ImageGroup>,
    /// 可释放的总空间（字节）
    pub total_reclaimable_bytes: u64,
    pub ai_enabled: bool,
}

/// 删除操作结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResult {
    pub deleted: Vec<String>,
    pub failed: Vec<DeleteFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteFailure {
    pub path: String,
    pub reason: String,
}

/// 应用设置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// 相似度阈值（0~1），哈希距离低于该值视为相似
    pub similarity_threshold: f32,
    /// 是否启用 AI 推理（需要模型 + GPU）
    pub ai_enabled: bool,
    /// 删除方式：回收站 or 永久删除
    pub permanent_delete: bool,
    /// 是否启用增量扫描（跳过已缓存图片）
    pub incremental: bool,
    /// 是否启用 MCP server（供外部 AI Agent 操作应用）
    #[serde(default)]
    pub mcp_enabled: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            ai_enabled: true,
            permanent_delete: false,
            incremental: true,
            mcp_enabled: false,
        }
    }
}

/// 系统信息（GPU/模型检测结果）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// 是否检测到可用 GPU（DirectML / DirectX 12，覆盖 NVIDIA/AMD/Intel）
    pub gpu_available: bool,
    pub gpu_name: Option<String>,
    /// 主技术质量模型（TOPIQ-NR）文件是否存在
    pub technical_model_available: bool,
    /// 应用数据目录
    pub data_dir: String,
}

/// 扫描完成摘要（小 payload，通过事件推送；完整结果由前端 invoke 拉取）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanSummary {
    pub session_id: String,
    /// 批次号 yyyyMMddHHmmSS
    pub batch_id: String,
    pub total_images: usize,
    pub total_groups: usize,
    pub total_reclaimable_bytes: u64,
    pub ai_enabled: bool,
}

/// MCP server 状态（供设置面板展示）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpStatus {
    /// 是否正在运行
    pub running: bool,
    /// 监听端口
    pub port: u16,
    /// 端点 URL（如 `http://127.0.0.1:18765/mcp`）
    pub url: String,
}

/// 可清理的缓存类型（用户勾选，清理后移入系统回收站，非永久删）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheType {
    /// AI 代理图缓存（`app_data_dir()/proxy/`）
    Proxy,
    /// 缩略图缓存（`app_data_dir()/thumbnails/`）
    Thumbnails,
    /// AI 评分缓存（`pixsweep-cache.json`）
    AiCache,
    /// 日志（`pixsweep.log`、`logs/`）
    Logs,
    /// 临时回收站隔离区
    Quarantine,
}

/// 某类缓存的体积摘要（供前端"清理缓存"面板勾选）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheSummary {
    pub cache_type: CacheType,
    /// 文件数量
    pub count: usize,
    /// 总占用字节
    pub bytes: u64,
}

/// 缓存清理结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheCleanupResult {
    /// 成功移入系统回收站的数量
    pub moved: u32,
    /// 失败数量
    pub failed: u32,
}

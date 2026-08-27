//! 应用全局状态（由 Tauri 管理，注入到各个命令）。

use crate::db::store::SharedStore;
use crate::types::{AppSettings, ScanResult};

pub struct AppState {
    /// 数据库句柄
    pub db: SharedStore,
    /// 应用设置
    pub settings: parking_lot::Mutex<AppSettings>,
    /// 最近一次扫描结果（大对象，通过 invoke 按需拉取，避免事件大 payload 导致白屏）
    pub result: parking_lot::Mutex<Option<ScanResult>>,
    /// MCP server 运行时句柄（`Some` 表示服务正在运行）
    pub mcp: parking_lot::Mutex<Option<crate::mcp::McpRuntime>>,
    /// AI 推理引擎（启用 `ai` feature 时可用）
    #[cfg(feature = "ai")]
    pub ai: parking_lot::Mutex<Option<std::sync::Arc<crate::ai::engine::AiEngine>>>,
}

impl AppState {
    pub fn new(db: SharedStore) -> Self {
        Self {
            db,
            settings: parking_lot::Mutex::new(AppSettings::default()),
            result: parking_lot::Mutex::new(None),
            mcp: parking_lot::Mutex::new(None),
            #[cfg(feature = "ai")]
            ai: parking_lot::Mutex::new(None),
        }
    }
}

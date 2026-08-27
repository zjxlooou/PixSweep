//! # PixSweep —— 本地图片去重桌面应用（Tauri 2 + Rust 后端库）
//!
//! 本 crate 是 PixSweep 的 Rust 后端核心（`lib` 目标，被 `main.rs` 调用 `run()` 启动）。
//! 主要职责：
//!
//! - **图片扫描与去重**：递归扫描目录 → 感知哈希（pHash/dHash）初筛
//!   语义聚类 → Union-Find 分组，找出内容重复/高度相似的图片组。
//! - **双维度 AI 评分**（`ai` feature）：对组内图片做技术质量（TOPIQ-NR）+ 美学
//!   （TOPIQ-IAA）评分，综合分最高者标记为"推荐保留"。
//! - **文件操作**：删除（回收站）、缩略图缓存、回收站管理。
//! - **MCP server**：HTTP JSON-RPC 接口（127.0.0.1:18765），供外部 Agent 调用。
//!
//! ## 模块结构
//! - [`ai`]：AI 推理引擎（TOPIQ / NIMA），ONNX Runtime + DirectML。
//! - [`scanner`]：目录遍历 + 图片过滤。
//! - [`hashing`]：感知哈希（pHash）。
//! - [`cluster`]：相似度计算 + Union-Find 聚类。
//! - [`quality`]：综合评分推荐。
//! - [`db`]：JSON 文件缓存（图片元信息 + 评分缓存）。
//! - [`fileops`]：回收站删除（Windows 原生 API）。
//! - [`mcp`]：MCP server（HTTP JSON-RPC）。
//! - [`commands`]：Tauri IPC 命令（前端调用入口）。
//!
//! ## 关键约定
//! - 推理后端走 **DirectML**（Windows DirectX 12，全 GPU 通用，不依赖 CUDA），
//!   不兼容的模型自动回退 CPU EP。
//! - 模型文件放在可执行文件同级的 `models/` 目录（见 [`models_dir`]）。

pub mod cache;
pub mod cluster;
pub mod commands;
pub mod db;
pub mod fileops;
pub mod hashing;
pub mod image_io;
pub mod mcp;
pub mod quality;
pub mod scanner;
pub mod state;
pub mod types;

#[cfg(feature = "ai")]
pub mod ai;

use std::io::Write;
use std::path::PathBuf;
use tauri::Manager;

/// 返回应用数据目录（如 `%LOCALAPPDATA%/PixSweep`），不存在则创建。
/// 可执行文件所在目录（程序根目录）。
fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// 运行期数据目录：**一律放程序根目录**（exe 同级），
/// 不写入系统 `%LOCALAPPDATA%`，避免临时文件到处乱放。可执行目录需可写（便携运行）。
pub fn app_data_dir() -> PathBuf {
    let dir = exe_dir();
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 返回模型目录（存放 ONNX 模型文件）。
pub fn models_dir() -> PathBuf {
    // 优先取可执行文件同级的 models 目录，其次取应用数据目录。
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();
    let bundled = exe_dir.join("models");
    if bundled.join("topiq_nr.onnx").exists() {
        return bundled;
    }
    let data = app_data_dir().join("models");
    let _ = std::fs::create_dir_all(&data);
    data
}

// ---------------------------------------------------------------------------
// 文件日志：写入 %LOCALAPPDATA%/PixSweep/pixsweep.log
// GUI release 版无控制台窗口，必须落盘才能排查问题。
// ---------------------------------------------------------------------------

/// 双通道 logger：写入日志文件 + 同步输出到 stderr（开发控制台可见）。
struct FileLogger {
    file: std::sync::Mutex<std::fs::File>,
}

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        let line = format!(
            "{} [{}] {}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            record.level(),
            record.args()
        );
        if let Ok(mut f) = self.file.lock() {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
        // 同步到 stderr：有控制台（debug）时可见，GUI 无控制台时无害
        eprint!("{}", line);
    }

    fn flush(&self) {
        if let Ok(mut f) = self.file.lock() {
            let _ = f.flush();
        }
    }
}

/// 初始化日志：
/// - 若设置了 `RUST_LOG`（开发调试），用 env_logger 输出到控制台；
/// - 否则（用户正常使用）写入 `%LOCALAPPDATA%/PixSweep/pixsweep.log`，级别 Info。
pub fn init_logging() {
    if std::env::var("RUST_LOG").is_ok() {
        let _ = env_logger::Builder::from_default_env().try_init();
        log::info!("日志模式：控制台（RUST_LOG）");
        return;
    }

    let log_path = app_data_dir().join("pixsweep.log");
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        Ok(file) => {
            let logger = FileLogger {
                file: std::sync::Mutex::new(file),
            };
            if log::set_boxed_logger(Box::new(logger)).is_ok() {
                log::set_max_level(log::LevelFilter::Info);
            }
            log::info!("================ PixSweep 启动 ================");
            log::info!("日志文件: {}", log_path.display());
            log::info!("日志模式：文件（Info 级）");
        }
        Err(e) => {
            // 无法打开日志文件时退回 env_logger（尽力而为）
            let _ = env_logger::Builder::from_default_env().try_init();
            eprintln!("无法打开日志文件 {}: {}", log_path.display(), e);
        }
    }
}

/// Tauri 应用入口。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // 初始化日志
    init_logging();

    // 打开缓存（JSON 文件存储）
    let db_path = app_data_dir().join("pixsweep-cache.json");
    let store = db::store::Store::shared(&db_path).expect("无法打开缓存");

    log::info!("应用数据目录: {}", app_data_dir().display());
    log::info!("模型目录: {}", models_dir().display());
    log::info!("缓存文件: {}", db_path.display());

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .manage(state::AppState::new(store))
        .setup(|app| {
            // 启动 MCP server（条件：命令行 --mcp 参数 或 设置中 mcp_enabled）
            let arg_mcp = std::env::args().any(|a| a == "--mcp");
            let handle = app.handle().clone();
            let st = handle.state::<state::AppState>();
            let settings_mcp = st.settings.lock().mcp_enabled;
            let should_start_mcp = arg_mcp || settings_mcp;
            if should_start_mcp {
                let runtime = mcp::start(handle.clone(), mcp::DEFAULT_MCP_PORT);
                *handle.state::<state::AppState>().mcp.lock() = Some(runtime);
                log::info!("[MCP] 启动时已自动启动 MCP server");
            }

            // 强制窗口居中并可见（修复"窗口位置记忆到屏幕外"问题）
            // - center:true 仅首次启动生效；多显示器拔掉后，记忆的位置会变成屏幕外
            // - 每次启动都重新定位到主屏幕中心 + 取消最小化 + 聚焦
            if let Some(win) = handle.get_webview_window("main") {
                if let Ok(Some(monitor)) = win.primary_monitor() {
                    let screen_size = monitor.size();
                    let win_size = win.outer_size().unwrap_or(tauri::PhysicalSize::new(1280, 800));
                    let x = ((screen_size.width as i32 - win_size.width as i32) / 2).max(0);
                    let y = ((screen_size.height as i32 - win_size.height as i32) / 2).max(0);
                    let _ = win.set_position(tauri::PhysicalPosition::new(x, y));
                    log::info!("[窗口] 居中到主屏幕 {}x{}, 位置 ({},{})", screen_size.width, screen_size.height, x, y);
                } else {
                    log::warn!("[窗口] 无法获取主屏幕信息");
                }
                let _ = win.unminimize();
                let _ = win.show();
                let _ = win.set_focus();
            } else {
                log::warn!("[窗口] 未找到 main webview 窗口");
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_system_info,
            commands::get_settings,
            commands::set_settings,
            commands::start_scan,
            commands::get_thumbnail,
            commands::get_full_image,
            commands::delete_files,
            commands::get_scan_result,
            commands::list_trash_images,
            commands::restore_trash_item,
            commands::restore_all_trash_images,
            commands::clear_trash_bin,
            commands::open_trash_bin_in_explorer,
            commands::get_mcp_status,
            commands::set_mcp_enabled,
            commands::get_cache_summary,
            commands::cleanup_cache,
            commands::count_images,
        ])
        .run(tauri::generate_context!())
        .expect("PixSweep 启动失败");
}

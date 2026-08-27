//! MCP (Model Context Protocol) 服务端。
//!
//! 通过 **HTTP (JSON-RPC 2.0)** 对外提供 PixSweep 的完整能力，让外部 AI Agent
//! 能够远程操作本应用（扫描、删除、回收站管理、设置、导出等），
//! 从而在**真实环境**中对应用进行端到端测试。
//!
//! ## 传输方式
//! - 监听 `127.0.0.1:18765`（仅本机，不对外网开放）
//! - `POST /mcp`：请求体为 JSON-RPC 2.0（`initialize` / `tools/list` / `tools/call`）
//! - 响应为 JSON（非流式），`Content-Type: application/json`
//!
//! ## 启动方式
//! - App 启动时带 `--mcp` 参数：强制启动 MCP server
//! - App 设置中 `mcp_enabled = true`：启动时自动启动
//! - 运行中可通过命令 `set_mcp_enabled(true/false)` 热启停
//!
//! ## 安全
//! 只监听 loopback 地址，仅本机 Agent 可访问；所有操作与 GUI 共用同一份状态，
//! Agent 的每次操作都会实时反映在界面上。

use crate::state::AppState;
use crate::types::{AppSettings, ScanResult, SystemInfo};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Manager};

/// MCP server 默认监听端口。
pub const DEFAULT_MCP_PORT: u16 = 18765;

/// MCP server 运行时句柄（用于热停止）。
pub struct McpRuntime {
    stop: Arc<AtomicBool>,
    port: u16,
}

impl McpRuntime {
    pub fn port(&self) -> u16 {
        self.port
    }

    /// 请求停止 MCP server（异步生效）。
    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// 启动 MCP server（阻塞运行直到被停止），返回运行时句柄。
pub fn start(app: AppHandle, port: u16) -> McpRuntime {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();

    std::thread::spawn(move || {
        serve(app, port, stop2);
    });

    log::info!("[MCP] 服务已启动: http://127.0.0.1:{}/mcp", port);
    McpRuntime { stop, port }
}

/// MCP server 主循环。
fn serve(app: AppHandle, port: u16, stop: Arc<AtomicBool>) {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            log::error!("[MCP] 端口绑定失败 {}: {}", port, e);
            return;
        }
    };
    // 非阻塞 accept，配合停止标志实现优雅退出
    let _ = listener.set_nonblocking(true);

    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let app2 = app.clone();
                std::thread::spawn(move || {
                    let _ = handle_connection(app2, stream);
                });
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                log::error!("[MCP] accept 失败: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
    log::info!("[MCP] 服务已停止");
}

/// 处理单个 HTTP 连接：读取请求 → 解析 → 执行 JSON-RPC → 响应。
fn handle_connection(app: AppHandle, mut stream: TcpStream) -> std::io::Result<()> {
    // 读取请求头（直到空行）
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 64 * 1024 {
            return Ok(()); // 请求头过大，直接丢弃
        }
    };

    let headers_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let body = buf[header_end..].to_vec();

    // 解析 Content-Length
    let content_length: usize = headers_str
        .lines()
        .find_map(|l| {
            let lower = l.to_ascii_lowercase();
            lower.strip_prefix("content-length:").and_then(|v| v.trim().parse().ok())
        })
        .unwrap_or(0);

    // 读取剩余 body
    let mut body = body;
    while body.len() < content_length {
        let n = stream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }

    // 解析 JSON-RPC 请求
    let request_str = String::from_utf8_lossy(&body).to_string();
    let response = match serde_json::from_str::<RpcRequest>(&request_str) {
        Ok(req) => handle_request(&req, &app),
        Err(e) => rpc_error(Value::Null, -32700, &format!("Parse error: {e}")),
    };

    let response_body = serde_json::to_string(&response).unwrap_or_default();
    let response_headers = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        response_body.len()
    );

    stream.write_all(response_headers.as_bytes())?;
    stream.write_all(response_body.as_bytes())?;
    stream.flush()?;
    Ok(())
}

/// MCP 工具定义。
#[derive(Debug, Clone, Serialize)]
struct ToolDef {
    name: &'static str,
    description: &'static str,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

/// 所有可用的 MCP 工具（覆盖应用完整能力）。
fn all_tools() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "get_system_info",
            description: "获取系统信息：GPU 型号、模型可用性、数据目录路径",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "get_settings",
            description: "获取当前应用设置（相似度阈值、AI 开关、增量扫描、删除方式、MCP 开关等）",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "set_settings",
            description: "更新应用设置",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "description": "完整的 AppSettings 对象（可省略不想改的字段）",
                        "properties": {
                            "similarity_threshold": { "type": "number", "description": "相似度阈值 0~1" },
                            "ai_enabled": { "type": "boolean", "description": "是否启用 AI 评分" },
                            "permanent_delete": { "type": "boolean", "description": "true=永久删除, false=移至临时回收站" },
                            "incremental": { "type": "boolean", "description": "是否启用增量扫描" }
                        }
                    }
                },
                "required": ["settings"]
            }),
        },
        ToolDef {
            name: "start_scan",
            description: "同步扫描指定文件夹，找出相似图片并分组。返回完整扫描结果（含分组、评分、推荐）。支持增量模式。",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "folders": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "要扫描的文件夹绝对路径列表"
                    },
                    "similarity_threshold": { "type": "number", "description": "可选：覆盖设置中的相似度阈值" },
                    "incremental": { "type": "boolean", "description": "可选：覆盖设置中的增量扫描开关" }
                },
                "required": ["folders"]
            }),
        },
        ToolDef {
            name: "get_scan_result",
            description: "获取最近一次扫描的完整结果（不触发新扫描）",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "delete_files",
            description: "删除指定路径的文件。permanent=false 时移至临时回收站（可恢复），permanent=true 时永久删除",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "paths": { "type": "array", "items": { "type": "string" }, "description": "要删除的文件绝对路径列表" },
                    "permanent": { "type": "boolean", "description": "true=永久删除, false=移至临时回收站" }
                },
                "required": ["paths"]
            }),
        },
        ToolDef {
            name: "list_trash",
            description: "列出临时回收站中的所有文件",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "restore_trash_item",
            description: "从临时回收站恢复单个文件到原路径",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "回收站条目 ID" }
                },
                "required": ["id"]
            }),
        },
        ToolDef {
            name: "restore_all_trash",
            description: "恢复临时回收站中的所有文件",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "clear_trash",
            description: "清空临时回收站（永久删除所有隔离文件）",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "clear_cache",
            description: "清空扫描缓存（哈希 + AI 评分），下次扫描将全量重算",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "export_report",
            description: "将最近一次扫描结果导出为 CSV 文件",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "output_path": { "type": "string", "description": "CSV 文件保存路径" }
                },
                "required": ["output_path"]
            }),
        },
        ToolDef {
            name: "get_cache_summary",
            description: "查询各缓存类型（代理图/缩略图/AI评分缓存/日志/临时回收站）的文件数与占用字节",
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDef {
            name: "cleanup_cache",
            description: "清理选中的缓存类型：把对应文件移入系统回收站（可恢复，非永久删）",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "types": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["proxy", "thumbnails", "ai_cache", "logs", "quarantine"] },
                        "description": "要清理的缓存类型列表"
                    }
                },
                "required": ["types"]
            }),
        },
    ]
}

/// JSON-RPC 2.0 请求。
#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Value,
    method: String,
    #[serde(default)]
    params: Value,
}

/// 构建 JSON-RPC 成功响应。
fn rpc_ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// 构建 JSON-RPC 错误响应。
fn rpc_error(id: Value, code: i32, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// 工具调用结果（MCP 协议要求 content 数组）。
fn tool_result(text: String, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": text }],
        "isError": is_error
    })
}

/// 处理单个 JSON-RPC 请求。
fn handle_request(req: &RpcRequest, app: &AppHandle) -> Value {
    match req.method.as_str() {
        "initialize" => rpc_ok(
            req.id.clone(),
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "pixsweep",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        ),
        "initialized" => rpc_ok(req.id.clone(), json!({})),
        "ping" => rpc_ok(req.id.clone(), json!({})),
        "tools/list" => rpc_ok(req.id.clone(), json!({ "tools": all_tools() })),
        "tools/call" => {
            let name = req.params.get("name").and_then(|v| v.as_str());
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            match name {
                Some(tool_name) => {
                    let (text, is_error) = execute_tool(tool_name, &args, app);
                    rpc_ok(req.id.clone(), tool_result(text, is_error))
                }
                None => rpc_error(req.id.clone(), -32602, "Missing 'name' in tools/call"),
            }
        }
        _ => rpc_error(req.id.clone(), -32601, &format!("Method not found: {}", req.method)),
    }
}

/// 执行工具调用，返回 (输出文本, 是否错误)。
fn execute_tool(name: &str, args: &Value, app: &AppHandle) -> (String, bool) {
    let state = app.state::<AppState>();

    match name {
        "get_system_info" => {
            let info = SystemInfo {
                gpu_available: crate::commands::gpu_available(),
                gpu_name: crate::commands::gpu_name(),
                clip_model_available: crate::models_dir().join("clip-vit-b32-visual.onnx").exists(),
                #[cfg(feature = "ai")]
                technical_model_available: crate::models_dir()
                    .join(crate::ai::engine::TOPIQ_NR_MODEL)
                    .exists(),
                #[cfg(not(feature = "ai"))]
                technical_model_available: false,
                data_dir: crate::app_data_dir().to_string_lossy().to_string(),
            };
            (serde_json::to_string_pretty(&info).unwrap_or_default(), false)
        }

        "get_settings" => {
            let settings = state.settings.lock().clone();
            (serde_json::to_string_pretty(&settings).unwrap_or_default(), false)
        }

        "set_settings" => {
            let settings_val = match args.get("settings") {
                Some(s) => s,
                None => return ("Missing 'settings' argument".into(), true),
            };
            let settings: AppSettings = match serde_json::from_value(settings_val.clone()) {
                Ok(s) => s,
                Err(e) => return (format!("Invalid settings: {e}"), true),
            };
            *state.settings.lock() = settings;
            ("Settings updated".into(), false)
        }

        "start_scan" => {
            let folders: Vec<String> = match args
                .get("folders")
                .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok())
            {
                Some(f) if !f.is_empty() => f,
                _ => return ("Missing or empty 'folders' argument".into(), true),
            };
            let threshold = args
                .get("similarity_threshold")
                .and_then(|v| v.as_f64())
                .map(|f| f as f32);
            let incremental = args.get("incremental").and_then(|v| v.as_bool());

            let settings = state.settings.lock().clone();
            let threshold = threshold.unwrap_or(settings.similarity_threshold);
            let use_incremental = incremental.unwrap_or(settings.incremental);
            let db = state.db.clone();

            let folder_paths: Vec<std::path::PathBuf> =
                folders.iter().map(std::path::PathBuf::from).collect();

            #[cfg(feature = "ai")]
            let ai_engine = state.ai.lock().clone();
            #[cfg(not(feature = "ai"))]
            let ai_engine: Option<Arc<()>> = None;

            let session_id = uuid::Uuid::new_v4().to_string();
            let app2 = app.clone();

            // 同步执行扫描（MCP 场景下 Agent 需要等待结果）
            let db_guard = db.lock();
            crate::commands::run_scan(
                &db_guard,
                &folder_paths,
                threshold,
                use_incremental,
                ai_engine,
                &app2,
                &session_id,
            );
            drop(db_guard);

            // run_scan 已将结果存入 state.result
            match state.result.lock().clone() {
                Some(r) => {
                    let summary = format!(
                        "扫描完成：{} 张图片，{} 组，可释放 {}",
                        r.total_images,
                        r.groups.len(),
                        format_bytes(r.total_reclaimable_bytes)
                    );
                    (
                        format!("{summary}\n\n{}", serde_json::to_string_pretty(&r).unwrap_or_default()),
                        false,
                    )
                }
                None => ("Scan failed".into(), true),
            }
        }

        "get_scan_result" => match state.result.lock().clone() {
            Some(r) => (serde_json::to_string_pretty(&r).unwrap_or_default(), false),
            None => ("No scan result available".into(), false),
        },

        "delete_files" => {
            let paths: Vec<String> = match args.get("paths").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                Some(p) => p,
                None => return ("Missing 'paths' argument".into(), true),
            };
            let permanent = args.get("permanent").and_then(|v| v.as_bool()).unwrap_or(false);

            let result = crate::fileops::trash::delete_files(&paths, permanent);
            let summary = format!(
                "删除完成：成功 {} 个，失败 {} 个",
                result.deleted.len(),
                result.failed.len()
            );
            let detail = if result.failed.is_empty() {
                String::new()
            } else {
                let fails: Vec<String> = result
                    .failed
                    .iter()
                    .map(|f| format!("  - {} ({})", f.path, f.reason))
                    .collect();
                format!("\n失败列表:\n{}", fails.join("\n"))
            };
            (format!("{summary}{detail}"), !result.failed.is_empty())
        }

        "list_trash" => {
            let items = crate::fileops::trash::list_quarantine();
            let summary = format!("临时回收站：{} 个文件", items.len());
            let detail = serde_json::to_string_pretty(&items).unwrap_or_default();
            (format!("{summary}\n\n{detail}"), false)
        }

        "restore_trash_item" => {
            let id = match args.get("id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None => return ("Missing 'id' argument".into(), true),
            };
            match crate::fileops::trash::restore_one(&id) {
                Ok(path) => (format!("已恢复到原路径: {path}"), false),
                Err(e) => (format!("恢复失败: {e}"), true),
            }
        }

        "restore_all_trash" => match crate::fileops::trash::restore_all_quarantine() {
            Ok(paths) => (format!("已恢复 {} 个文件", paths.len()), false),
            Err(e) => (format!("恢复失败: {e}"), true),
        },

        "clear_trash" => match crate::fileops::trash::clear_quarantine() {
            Ok(count) => (format!("已清空临时回收站，删除 {count} 个文件"), false),
            Err(e) => (format!("清空失败: {e}"), true),
        },

        "clear_cache" => {
            let db = state.db.lock();
            match db.clear_cache() {
                Ok(()) => ("缓存已清空".into(), false),
                Err(e) => (format!("清空缓存失败: {e}"), true),
            }
        }

        "export_report" => {
            let output_path = match args.get("output_path").and_then(|v| v.as_str()) {
                Some(p) => p.to_string(),
                None => return ("Missing 'output_path' argument".into(), true),
            };
            match state.result.lock().clone() {
                Some(result) => match build_and_save_csv(&result, &output_path) {
                    Ok(()) => (format!("报告已导出到: {output_path}"), false),
                    Err(e) => (format!("导出失败: {e}"), true),
                },
                None => ("No scan result to export".into(), true),
            }
        }

        "get_cache_summary" => {
            let summary = crate::commands::get_cache_summary();
            (serde_json::to_string_pretty(&summary).unwrap_or_default(), false)
        }

        "cleanup_cache" => {
            let types: Vec<crate::types::CacheType> =
                match args.get("types").and_then(|v| serde_json::from_value(v.clone()).ok()) {
                    Some(t) => t,
                    None => return ("Missing or invalid 'types' argument".into(), true),
                };
            let r = crate::commands::cleanup_cache_sync(types);
            (
                format!("已移入系统回收站 {} 个文件，失败 {} 个", r.moved, r.failed),
                r.failed > 0,
            )
        }

        _ => (format!("Unknown tool: {name}"), true),
    }
}

/// 格式化字节为人类可读字符串。
pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// 构建并保存 CSV 报告。
fn build_and_save_csv(result: &ScanResult, path: &str) -> anyhow::Result<()> {
    let mut lines = vec!["分组,推荐保留,评分,文件路径,大小(字节)".to_string()];
    for g in &result.groups {
        for img in &g.images {
            let score = img.score.map(|s| s.to_string()).unwrap_or_default();
            lines.push(format!(
                "{},{},{},\"{}\",{}",
                g.group_id,
                if img.recommended { "是" } else { "否" },
                score,
                img.info.path.replace('"', "\"\""),
                img.info.size
            ));
        }
    }
    let content = format!("\u{FEFF}{}", lines.join("\n"));
    std::fs::write(path, content)?;
    Ok(())
}

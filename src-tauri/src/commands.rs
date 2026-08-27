//! Tauri IPC 命令层：前端调用的入口，以及核心扫描流程编排。

use crate::types::{
    AppSettings, CacheCleanupResult, CacheSummary, CacheType, DeleteResult, ImageInfo,
    McpStatus, ScanPhase, ScanProgress, ScanResult, SystemInfo,
};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

use crate::state::AppState;

/// 返回系统信息（GPU / 模型可用性）。
#[tauri::command]
pub fn get_system_info(state: State<'_, AppState>) -> SystemInfo {
    let models = crate::models_dir();
    let (gpu_available, gpu_name) = {
        #[cfg(feature = "ai")]
        {
            // 触发引擎初始化，确保读取真实后端（与扫描进度条一致），
            // 避免启动时 state.ai 尚未加载而读到 fallback DirectML 造成两处不一致。
            let ai_enabled = state.settings.lock().ai_enabled;
            let _ = get_ai_engine(state.inner(), ai_enabled);
            let eng = state.ai.lock();
            match eng.as_ref() {
                Some(e) => (e.backend().is_gpu(), Some(e.backend().label().to_string())),
                None => (gpu_available(), gpu_name()),
            }
        }
        #[cfg(not(feature = "ai"))]
        {
            (false, None)
        }
    };
    let info = SystemInfo {
        gpu_available,
        gpu_name,
        clip_model_available: models.join("clip-vit-b32-visual.onnx").exists(),
        technical_model_available: models.join(crate::ai::engine::TOPIQ_NR_MODEL).exists(),
        data_dir: crate::app_data_dir().to_string_lossy().to_string(),
    };
    log::info!(
        "[系统信息] GPU 可用: {}, GPU 名称: {:?}, CLIP 模型: {}, TOPIQ-NR 模型: {}",
        info.gpu_available,
        info.gpu_name,
        info.clip_model_available,
        info.technical_model_available
    );
    log::info!(
        "[系统信息] 后备模型 —— CLIP-IQA: {}, NIMA: {}",
        models.join(crate::ai::engine::CLIP_IQA_MODEL).exists(),
        models.join(crate::ai::engine::NIMA_TECH_MODEL).exists(),
    );
    info
}

/// 获取当前设置。
#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> AppSettings {
    state.settings.lock().clone()
}

/// 更新设置。
#[tauri::command]
pub fn set_settings(state: State<'_, AppState>, settings: AppSettings) -> Result<(), String> {
    *state.settings.lock() = settings;
    Ok(())
}

/// 启动一次扫描（异步执行，进度通过事件推送）。
#[tauri::command]
pub async fn start_scan(
    state: State<'_, AppState>,
    app: AppHandle,
    folders: Vec<String>,
    similarity_threshold: Option<f32>,
    incremental: Option<bool>,
) -> Result<String, String> {
    let session_id = Uuid::new_v4().to_string();

    let settings = state.settings.lock().clone();
    let threshold = similarity_threshold.unwrap_or(settings.similarity_threshold);
    let use_incremental = incremental.unwrap_or(settings.incremental);
    let db = state.db.clone();

    let folder_paths: Vec<PathBuf> = folders.iter().map(PathBuf::from).collect();
    if folder_paths.is_empty() {
        return Err("请至少选择一个文件夹".to_string());
    }

    // 获取 AI 引擎（feature gated）
    #[cfg(feature = "ai")]
    let ai_engine = get_ai_engine(&state, settings.ai_enabled);
    #[cfg(not(feature = "ai"))]
    let ai_engine: Option<Arc<()>> = None;

    // 后台执行扫描（spawn_blocking 避免阻塞 async runtime）
    let scan_session_id = session_id.clone();
    tokio::spawn(async move {
        let app2 = app.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let db_guard = db.lock();
            run_scan(
                &db_guard,
                &folder_paths,
                threshold,
                use_incremental,
                ai_engine,
                &app2,
                &scan_session_id,
            )
        })
        .await;
    });

    Ok(session_id)
}

/// 获取单张图片的缩略图（base64 data URL）。
#[tauri::command]
pub fn get_thumbnail(path: String, file_hash: String) -> Result<String, String> {
    let bytes = crate::cache::thumbnail::generate_thumbnail(&path, &file_hash)
        .map_err(|e| e.to_string())?;
    Ok(crate::cache::thumbnail::to_data_url(&bytes))
}

/// 获取单张图片的原图（限制最大边 1600px 为 JPEG data URL，用于预览对比）。
///
/// 大图通过 invoke 返回，避免事件大 payload 导致白屏（沿用 get_thumbnail 的策略）。
#[tauri::command]
pub fn get_full_image(path: String) -> Result<String, String> {
    use image::GenericImageView;

    const MAX_SIDE: u32 = 3072;
    const JPEG_QUALITY: u8 = 92;

    let img = crate::image_io::load_image_oriented(&path).map_err(|e| {
        log::warn!("[预览] 打开图片失败 {}: {}", path, e);
        format!("无法打开图片: {}", e)
    })?;

    // 若超过最大边则等比缩放（保留原图质量，仅降采样）。
    // 统一转 RGB8：JPEG 编码器不支持 alpha 通道（PNG/WebP 等格式可能有 alpha），
    // 对预览用图丢弃 alpha 是无损的（alpha=不透明时等价 RGB）。
    let (w, h) = img.dimensions();
    let scaled: image::DynamicImage = if w.max(h) > MAX_SIDE {
        let scale = MAX_SIDE as f32 / w.max(h) as f32;
        image::DynamicImage::ImageRgb8(image::imageops::resize(
            &img.to_rgb8(),
            ((w as f32 * scale).round() as u32).max(1),
            ((h as f32 * scale).round() as u32).max(1),
            image::imageops::FilterType::Triangle,
        ))
    } else {
        image::DynamicImage::ImageRgb8(img.to_rgb8())
    };

    let mut buf: Vec<u8> = Vec::new();
    {
        use image::ImageEncoder;
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, JPEG_QUALITY);
        encoder
            .write_image(
                scaled.as_bytes(),
                scaled.width(),
                scaled.height(),
                scaled.color().into(),
            )
            .map_err(|e| format!("编码失败: {}", e))?;
    }

    log::info!(
        "[预览] 原图加载成功 {} ({}x{} -> {}x{}, {:.0}KB)",
        path,
        w,
        h,
        scaled.width(),
        scaled.height(),
        buf.len() as f64 / 1024.0
    );
    Ok(crate::cache::thumbnail::to_data_url(&buf))
}

/// 快速统计文件夹内图片总数（只遍历 + 扩展名过滤，不读图、不哈希、不 AI）。
#[tauri::command]
pub fn count_images(folders: Vec<String>) -> usize {
    let fpaths: Vec<PathBuf> = folders.into_iter().map(PathBuf::from).collect();
    crate::scanner::walker::scan_folders(&fpaths).len()
}

/// 删除一批文件（默认回收站，可永久删除）。
///
/// 异步执行：立即返回，通过 `delete-progress` 事件推送进度（done/total），
/// 完成后通过 `delete-done` 事件推送结果。避免大量文件删除时 UI 无反馈。
#[tauri::command]
pub fn delete_files(
    app: AppHandle,
    paths: Vec<String>,
    permanent: bool,
) -> Result<(), String> {
    log::info!(
        "[删除] 请求删除 {} 个文件, 永久删除: {}",
        paths.len(),
        permanent
    );
    if paths.is_empty() {
        return Ok(());
    }

    // 后台线程执行，避免阻塞主线程
    std::thread::spawn(move || {
        let total = paths.len();
        let mut deleted = Vec::with_capacity(total);
        let mut failed = Vec::new();

        for (i, path) in paths.iter().enumerate() {
            match crate::fileops::trash::delete_one(std::path::Path::new(path), permanent) {
                Ok(()) => {
                    log::info!("deleted: {} (permanent={})", path, permanent);
                    deleted.push(path.clone());
                }
                Err(e) => {
                    log::warn!("failed to delete {}: {}", path, e);
                    failed.push(crate::types::DeleteFailure {
                        path: path.clone(),
                        reason: e.to_string(),
                    });
                }
            }
            // 进度事件：每 10 个或最后一个推送（文件少时也能看到进度）
            if i % 10 == 0 || i + 1 == total {
                let _ = app.emit(
                    "delete-progress",
                    crate::types::ScanProgress {
                        session_id: String::new(),
                        phase: ScanPhase::Deleting,
                        current: i + 1,
                        total,
                        current_file: Some(path.clone()),
                        ai_enabled: false,
                        backend: "CPU".to_string(),
                        detail: "删除文件".to_string(),
                    },
                );
            }
        }

        let result = DeleteResult { deleted, failed };
        log::info!(
            "[删除] 结果: 成功 {} 个, 失败 {} 个",
            result.deleted.len(),
            result.failed.len()
        );
        for (i, p) in result.deleted.iter().enumerate() {
            log::info!("[删除] 已删除[{}]: {}", i, p);
        }
        for f in &result.failed {
            log::warn!("[删除] 失败: {} ({})", f.path, f.reason);
        }
        // 记录日志
        if !result.deleted.is_empty() {
            let _ = crate::fileops::trash::append_delete_log(&result.deleted, permanent);
        }
        // 完成事件
        let _ = app.emit("delete-done", &result);
    });

    Ok(())
}

/// 获取最近一次扫描的完整结果（通过 invoke 按需拉取，避免事件大 payload 导致白屏）。
#[tauri::command]
pub fn get_scan_result(state: State<'_, AppState>) -> Option<ScanResult> {
    state.result.lock().clone()
}

// ============================ 临时回收站（隔离区）===========================

/// 列出临时回收站中的全部图片（按删除时间倒序）。
///
/// **异步执行**：立即返回，通过 `trashbin-progress` 事件推送进度，
/// `trashbin-done` 事件推送结果。避免扫描阻塞 UI 线程。
#[tauri::command]
pub fn list_trash_images(app: AppHandle) -> Result<(), String> {
    std::thread::spawn(move || {
        // 进度回调：每 64 个文件推一次进度
        let app_progress = app.clone();
        let items = crate::fileops::trash::list_quarantine_with_progress(move |count| {
            let _ = app_progress.emit("trashbin-progress", count);
        });
        log::info!("[临时回收站] 列出完成: {} 张图片", items.len());
        // 用 owned payload（避免借用生命周期问题，Tauri 2 要求 'static + Serialize）
        let _ = app.emit("trashbin-done", items);
    });
    Ok(())
}

/// 恢复临时回收站中的单个条目。
#[tauri::command]
pub fn restore_trash_item(id: String) -> Result<String, String> {
    crate::fileops::trash::restore_one(&id).map_err(|e| e.to_string())
}

/// 恢复临时回收站中的全部图片，返回成功恢复的路径列表。
/// async + spawn_blocking：批量复制回原路径可能较慢，避免阻塞主线程导致界面假死。
#[tauri::command]
pub async fn restore_all_trash_images() -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(|| crate::fileops::trash::restore_all_quarantine())
        .await
        .map_err(|e| format!("恢复任务失败: {e}"))?
        .map_err(|e| e.to_string())
}

/// 清空临时回收站，返回删除的文件数量。
/// async + spawn_blocking：批量删除可能较慢，避免阻塞主线程。
#[tauri::command]
pub async fn clear_trash_bin() -> Result<u32, String> {
    tauri::async_runtime::spawn_blocking(|| crate::fileops::trash::clear_quarantine())
        .await
        .map_err(|e| format!("清空任务失败: {e}"))?
        .map_err(|e| e.to_string())
}

/// 在系统文件管理器中打开临时回收站目录（Windows: explorer.exe）。
#[tauri::command]
pub fn open_trash_bin_in_explorer() -> Result<(), String> {
    let path = crate::fileops::trash::quarantine_dir_path();
    std::fs::create_dir_all(&path).map_err(|e| format!("创建目录失败: {e}"))?;
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer.exe")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("启动资源管理器失败: {e}"))?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("xdg-open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("打开目录失败: {e}"))?;
    }
    log::info!("[临时回收站] 已在文件管理器打开: {}", path.display());
    Ok(())
}

/// 获取 MCP server 状态。
#[tauri::command]
pub fn get_mcp_status(state: State<'_, AppState>) -> McpStatus {
    let mcp = state.mcp.lock();
    match mcp.as_ref() {
        Some(r) => McpStatus {
            running: true,
            port: r.port(),
            url: format!("http://127.0.0.1:{}/mcp", r.port()),
        },
        None => McpStatus {
            running: false,
            port: crate::mcp::DEFAULT_MCP_PORT,
            url: format!("http://127.0.0.1:{}/mcp", crate::mcp::DEFAULT_MCP_PORT),
        },
    }
}

/// 启停 MCP server（热切换），同时更新设置持久化。
#[tauri::command]
pub fn set_mcp_enabled(state: State<'_, AppState>, app: AppHandle, enabled: bool) -> Result<McpStatus, String> {
    {
        let mut mcp_guard = state.mcp.lock();
        let currently_running = mcp_guard.is_some();
        if enabled && !currently_running {
            let runtime = crate::mcp::start(app, crate::mcp::DEFAULT_MCP_PORT);
            *mcp_guard = Some(runtime);
            log::info!("[MCP] 已启动 (set_mcp_enabled=true)");
        } else if !enabled && currently_running {
            if let Some(r) = mcp_guard.take() {
                r.stop();
            }
            log::info!("[MCP] 已停止 (set_mcp_enabled=false)");
        }
    }
    // 持久化设置
    state.settings.lock().mcp_enabled = enabled;
    Ok(get_mcp_status(state))
}

// ---------------------------------------------------------------------------
// 内部实现
// ---------------------------------------------------------------------------

#[cfg(feature = "ai")]
fn get_ai_engine(state: &AppState, enabled: bool) -> Option<Arc<crate::ai::engine::AiEngine>> {
    let mut guard = state.ai.lock();
    if enabled {
        if guard.is_none() {
            match crate::ai::engine::AiEngine::new(&crate::models_dir()) {
                Ok(e) => *guard = Some(Arc::new(e)),
                Err(e) => log::warn!("AI 引擎初始化失败: {}", e),
            }
        }
        guard.clone()
    } else {
        None
    }
}

pub fn gpu_available() -> bool {
    #[cfg(feature = "ai")]
    {
        crate::ai::engine::directml_available()
    }
    #[cfg(not(feature = "ai"))]
    {
        false
    }
}

pub fn gpu_name() -> Option<String> {
    #[cfg(feature = "ai")]
    {
        if crate::ai::engine::directml_available() {
            Some("GPU (DirectML / DirectX 12)".to_string())
        } else {
            None
        }
    }
    #[cfg(not(feature = "ai"))]
    {
        None
    }
}

/// 核心扫描流程（CPU 密集，在 spawn_blocking 中运行）。
/// 公开供 MCP server 复用。
#[allow(clippy::too_many_arguments)]
pub fn run_scan(
    db: &crate::db::store::Store,
    folders: &[PathBuf],
    threshold: f32,
    incremental: bool,
    #[cfg(feature = "ai")] ai_engine: Option<Arc<crate::ai::engine::AiEngine>>,
    #[cfg(not(feature = "ai"))] ai_engine: Option<Arc<()>>,
    app: &AppHandle,
    session_id: &str,
) {
    // 实际使用的推理后端（供进度 UI 展示 CUDA/DirectML/CPU）
    let backend_label = {
        #[cfg(feature = "ai")]
        {
            ai_engine
                .as_ref()
                .map(|e| e.backend().label().to_string())
                .unwrap_or_else(|| "CPU".to_string())
        }
        #[cfg(not(feature = "ai"))]
        {
            "CPU".to_string()
        }
    };

    let emit = |phase: ScanPhase, current: usize, total: usize, current_file: Option<String>, detail: &str| {
        // 按阶段/子阶段展示实际硬件：扫描/哈希/聚类是纯 CPU；
        // AI 评分展示推理后端（"对焦判断"子阶段是纯 CPU 拉普拉斯方差，不跑模型）
        let backend = match phase {
            ScanPhase::Quality if detail == "对焦判断" => "CPU".to_string(),
            ScanPhase::Quality => backend_label.clone(),
            _ => "CPU".to_string(),
        };
        let _ = app.emit(
            "scan-progress",
            ScanProgress {
                session_id: session_id.to_string(),
                phase,
                current,
                total,
                current_file,
                ai_enabled: ai_enabled(),
                backend,
                detail: detail.to_string(),
            },
        );
    };

    // 1. 扫描文件夹
    log::info!("[扫描] 开始 session={}", session_id);
    log::info!("[扫描] 文件夹: {}", folders.iter().map(|f| f.display().to_string()).collect::<Vec<_>>().join(", "));
    log::info!("[扫描] 相似度阈值: {}", threshold);
    emit(ScanPhase::Scanning, 0, 0, None, "");
    let mut infos = crate::scanner::walker::scan_folders(folders);
    let total = infos.len();
    emit(ScanPhase::Scanning, total, total, None, "");
    log::info!("[扫描] 发现 {} 张图片", total);

    if total == 0 {
        let batch_id_empty = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
        let _ = app.emit(
            "scan-complete",
            ScanResult {
                session_id: session_id.to_string(),
                batch_id: batch_id_empty,
                total_images: 0,
                groups: Vec::new(),
                total_reclaimable_bytes: 0,
                ai_enabled: ai_enabled(),
            },
        );
        log::info!("[扫描] 无图片，提前结束");
        return;
    }

    // 2. 计算感知哈希（增量：命中缓存则复用；非增量强制全量重算）
    log::info!(
        "[扫描] 模式: {}",
        if incremental { "增量（复用未变化文件缓存）" } else { "全量（忽略缓存）" }
    );
    emit(ScanPhase::Hashing, 0, total, None, "");
    let hash_start = std::time::Instant::now();
    let (hashes, ahashs) = compute_hashes(db, &mut infos, incremental, &emit);
    log::info!(
        "[扫描] 哈希计算完成: {} 张, 耗时 {:.1}s",
        hashes.len(),
        hash_start.elapsed().as_secs_f64()
    );
    let _ = db.flush(); // 落盘缓存

    // 3. 聚类（双哈希：dhash + ahash，过滤渐变图误判）
    emit(ScanPhase::Clustering, 0, total, None, "");
    let groups = crate::cluster::similarity::cluster_by_hash(&hashes, &ahashs, threshold);
    let group_count = groups.iter().filter(|g| g.len() > 1).count();
    log::info!("[扫描] 聚类完成: {} 组 (重复组 {} 个)", groups.len(), group_count);

    // 4. AI 质量评分（可选）：返回 (综合分, 美学分, 技术分, 人脸专评分, 有人脸标记, 场景, 闭眼)
    let (scores, aesthetic_scores, technical_scores, face_scores, has_faces, scenes, eye_closed, focus_scores) = if ai_enabled() {
        emit(ScanPhase::Quality, 0, groups.len(), None, "");
        #[cfg(feature = "ai")]
        {
            if let Some(engine) = &ai_engine {
                log::info!("[扫描] AI 双维度评分开始, GPU 加速: {}", engine.gpu_enabled());
                let ai_start = std::time::Instant::now();
                let (s, a, t, f, hf, sc, ec, foc) = score_groups_with_ai(db, engine, &infos, &groups, incremental, &emit);
                log::info!(
                    "[扫描] AI 评分完成, 耗时 {:.1}s (美学+技术+人脸+场景+闭眼+对焦, 增量={})",
                    ai_start.elapsed().as_secs_f64(),
                    incremental
                );
                let _ = db.flush(); // 落盘评分缓存
                emit(ScanPhase::Quality, groups.len(), groups.len(), None, "");
                (s, a, t, f, hf, sc, ec, foc)
            } else {
                log::warn!("[扫描] AI 引擎未初始化，跳过 AI 评分");
                empty_scores(infos.len())
            }
        }
        #[cfg(not(feature = "ai"))]
        {
            let _ = &ai_engine;
            empty_scores(infos.len())
        }
    } else {
        log::info!("[扫描] AI 未启用，跳过质量评分");
        empty_scores(infos.len())
    };

    // 5. 组装结果：生成批次号（yyyyMMddHHmmSS）用于全局唯一组号
    let batch_id = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
    log::info!("[扫描] 批次号: {}", batch_id);

    let mut image_groups = crate::quality::recommender::build_groups(
        &infos,
        &groups,
        &scores,
        &aesthetic_scores,
        &technical_scores,
        &face_scores,
        &has_faces,
        &scenes,
        &eye_closed,
        &focus_scores,
        &batch_id,
    );

    // 补充组内平均相似度（用 HashMap 预构建映射，避免 O(M×N) 查找）
    let index_map: std::collections::HashMap<&str, usize> = infos
        .iter()
        .enumerate()
        .map(|(idx, i)| (i.path.as_str(), idx))
        .collect();
    for g in image_groups.iter_mut() {
        let indices: Vec<usize> = g
            .images
            .iter()
            .filter_map(|gi| index_map.get(gi.info.path.as_str()).copied())
            .collect();
        g.similarity = crate::cluster::similarity::average_hash_similarity(&hashes, &indices);
    }

    let total_reclaimable: u64 = image_groups.iter().map(|g| g.reclaimable_bytes).sum();

    // 日志：打印每组完整 ID（批次号-序号）与成员，方便用户对照界面反馈误判组
    // （debug 级：大目录下逐图打印会拖慢"进度 100% → 完成"，故仅在诊断时输出）
    for g in &image_groups {
        log::debug!(
            "[分组 {}] 相似度 {:.1}%, {} 张, 可释放 {} KB",
            g.group_id,
            g.similarity * 100.0,
            g.images.len(),
            g.reclaimable_bytes / 1024
        );
        for gi in &g.images {
            log::debug!(
                "    [{}] {} ({}x{}) {}",
                if gi.recommended { "保留" } else { "删除" },
                gi.info.path,
                gi.info.width,
                gi.info.height,
                gi.reason
            );
        }
    }

    let result = ScanResult {
        session_id: session_id.to_string(),
        batch_id: batch_id.clone(),
        total_images: total,
        groups: image_groups,
        total_reclaimable_bytes: total_reclaimable,
        ai_enabled: ai_enabled(),
    };

    // 把完整结果存到后端 state，事件只推送小摘要，前端通过 invoke 按需拉取，
    // 避免大 payload 通过 evaluate_script 传输导致白屏。
    {
        use tauri::Manager;
        if let Some(state) = app.try_state::<AppState>() {
            *state.result.lock() = Some(result.clone());
        }
    }

    emit(ScanPhase::Done, total, total, None, "");
    log::info!(
        "[扫描] 完成: {} 张图片, {} 组, 可释放 {} MB",
        result.total_images,
        result.groups.len(),
        result.total_reclaimable_bytes as f64 / 1024.0 / 1024.0
    );
    let _ = app.emit(
        "scan-complete",
        crate::types::ScanSummary {
            session_id: result.session_id,
            batch_id: result.batch_id.clone(),
            total_images: result.total_images,
            total_groups: result.groups.len(),
            total_reclaimable_bytes: result.total_reclaimable_bytes,
            ai_enabled: result.ai_enabled,
        },
    );
}

fn ai_enabled() -> bool {
    cfg!(feature = "ai")
}

/// 构造"AI 未启用 / 引擎缺失"时的空评分占位，与 [`score_groups_with_ai`] 返回结构一致。
fn empty_scores(
    len: usize,
) -> (
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<bool>,
    Vec<crate::ai::scene::Scene>,
    Vec<bool>,
    Vec<f32>,
) {
    (
        vec![None; len],
        vec![None; len],
        vec![None; len],
        vec![None; len],
        vec![false; len],
        vec![crate::ai::scene::Scene::Other; len],
        vec![false; len],
        vec![1.0; len],
    )
}

/// 计算每张图片的感知哈希（并行 + 增量缓存）。
///
/// - `incremental = true`：文件指纹未变化且缓存有双哈希 → 直接复用缓存（含 width/height），
///   完全跳过文件读取与解码，显著加速二次扫描。
/// - `incremental = false`：忽略缓存，全量重新计算（哈希 + 解码尺寸）。
///
/// 返回与 `infos` 等长的 `Vec<u64>`，其中解码失败的图片哈希记为 0。
fn compute_hashes(
    db: &crate::db::store::Store,
    infos: &mut [ImageInfo],
    incremental: bool,
    emit: &(impl Fn(ScanPhase, usize, usize, Option<String>, &str) + Sync),
) -> (Vec<u64>, Vec<u64>) {
    let total = infos.len();

    // 1. 查询缓存（dhash + ahash + 完整记录）
    let cached_records: Vec<Option<crate::db::store::ImageRecord>> = if incremental {
        infos
            .iter()
            .map(|i| db.get_cached_record(&i.file_hash))
            .collect()
    } else {
        vec![None; infos.len()]
    };

    // 2. 并行计算未命中缓存的图片。用原子计数器追踪进度，避免 rayon 并行时
    //    emit 乱序导致进度条跳动（每 128 张或最后一张推送一次）。
    use std::sync::atomic::{AtomicUsize, Ordering};
    let counter = AtomicUsize::new(0);
    let last_emitted = parking_lot::Mutex::new(0usize);

    let results: Vec<(u64, u64, u32, u32)> = infos
        .par_iter()
        .zip(cached_records.par_iter())
        .map(|(info, cached)| {
            let r = if let Some(rec) = cached {
                // 增量命中：文件未变化 + 缓存有双哈希，直接复用（含尺寸）
                (rec.dhash, rec.ahash, rec.width, rec.height)
            } else {
                hash_from_path(&info.path).unwrap_or((0, 0, 0, 0))
            };

            let done = counter.fetch_add(1, Ordering::SeqCst) + 1;
            // 只在进度单调递增时才推送（每 128 张或最后一张）
            if done % 128 == 0 || done == total {
                let mut last = last_emitted.lock();
                if done > *last {
                    *last = done;
                    drop(last);
                    emit(ScanPhase::Hashing, done, total, None, "");
                }
            }
            r
        })
        .collect();

    // 确保最终进度为 100%
    emit(ScanPhase::Hashing, total, total, None, "");

    // 3. 写回缓存 + 更新尺寸
    for (info, (hash, ahash, width, height)) in infos.iter_mut().zip(results.iter()) {
        info.width = *width;
        info.height = *height;
        if *hash != 0 {
            let _ = db.save_image(
                &info.file_hash,
                &info.path,
                info.size,
                info.modified,
                info.width,
                info.height,
                &info.format,
                *hash,
                *ahash,
            );
        }
    }

    (
        results.iter().map(|(h, _, _, _)| *h).collect(),
        results.iter().map(|(_, a, _, _)| *a).collect(),
    )
}

/// 从路径解码图片并计算 dhash、ahash 与尺寸。
fn hash_from_path(path: &str) -> Option<(u64, u64, u32, u32)> {
    use image::GenericImageView;
    let img = crate::image_io::load_image_oriented(path).ok()?;
    let (w, h) = img.dimensions();
    Some((
        crate::hashing::phash::dhash(&img),
        crate::hashing::phash::ahash(&img),
        w,
        h,
    ))
}

/// 使用 AI 引擎对组内图片做双维度综合评分（CLIP 美学 + NIMA 技术 + 启发式）。
/// 返回 (综合分, 美学分, 技术分) 三个数组，分别对应 infos 的索引。
#[cfg(feature = "ai")]
fn score_groups_with_ai(
    db: &crate::db::store::Store,
    engine: &crate::ai::engine::AiEngine,
    infos: &[ImageInfo],
    groups: &[Vec<usize>],
    incremental: bool,
    emit: &impl Fn(ScanPhase, usize, usize, Option<String>, &str),
) -> (
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<Option<f32>>,
    Vec<bool>,
    Vec<crate::ai::scene::Scene>,
    Vec<bool>, // is_eye_closed
    Vec<f32>,  // focus_vals
) {
    let mut scores: Vec<Option<f32>> = vec![None; infos.len()];
    let mut aesthetic: Vec<Option<f32>> = vec![None; infos.len()];
    let mut technical: Vec<Option<f32>> = vec![None; infos.len()];
    let mut face_scores: Vec<Option<f32>> = vec![None; infos.len()];
    let mut has_faces: Vec<bool> = vec![false; infos.len()];
    let mut scenes: Vec<crate::ai::scene::Scene> = vec![crate::ai::scene::Scene::Other; infos.len()];
    let mut eye_closed: Vec<bool> = vec![false; infos.len()];
    // 阶段三：每张图 max(open_l, open_r) ∈ [0,1]（至少一眼开），供综合分连续降权；默认 1.0（不罚）。
    let mut eye_open_vals: Vec<f32> = vec![1.0; infos.len()];
    // 对焦分（人像→眼部对焦，非人像→整图对焦；默认 1.0 不降权）。下方按需填充。
    let mut focus_vals: Vec<f32> = vec![1.0; infos.len()];

    // 收集所有组内图片索引（去重）
    let mut to_score: Vec<usize> = Vec::new();
    for g in groups {
        to_score.extend(g.iter().copied());
    }
    to_score.sort_unstable();
    to_score.dedup();

    if to_score.is_empty() {
        return (scores, aesthetic, technical, face_scores, has_faces, scenes, eye_closed, focus_vals);
    }

    // 增量模式：先查评分缓存，命中者跳过推理（唯一可能变化的是启发式权重，
    // 它依赖 width/height/size——这些已从缓存记录补齐，与推理结果无关）。
    let mut cached_aes: Vec<Option<f32>> = vec![None; infos.len()];
    let mut cached_tech: Vec<Option<f32>> = vec![None; infos.len()];
    let mut need_infer: Vec<usize> = Vec::new();
    if incremental {
        for &i in &to_score {
            if let Some((a, t)) = db.get_cached_ai_scores(&infos[i].file_hash) {
                cached_aes[i] = Some(a);
                cached_tech[i] = Some(t);
            } else {
                need_infer.push(i);
            }
        }
        log::info!(
            "[AI 评分] 增量: 缓存命中 {} 张, 需推理 {} 张",
            to_score.len() - need_infer.len(),
            need_infer.len()
        );
    } else {
        need_infer = to_score.clone();
    }

    // 阶段一：人脸/场景/闭眼缓存独立判定（与美学/技术分缓存解耦，避免首跑漏算）。
    // 命中者直接复用 has_face/face_score/scene/eye_closed；未命中者进 need_face 重算。
    let mut need_face: Vec<usize> = Vec::new();
    if incremental {
        for &i in &to_score {
            if let Some(fc) = db.get_cached_ai_face(&infos[i].file_hash) {
                has_faces[i] = fc.has_face;
                face_scores[i] = fc.face_score;
                scenes[i] = crate::ai::scene::Scene::from_u8(fc.scene);
                eye_open_vals[i] = fc.eye_open;
                eye_closed[i] = crate::ai::eye::is_closed(fc.eye_open);
                focus_vals[i] = fc.focus_score;
            } else {
                need_face.push(i);
            }
        }
        log::info!(
            "[AI 评分] 人脸/场景/闭眼: 缓存命中 {} 张, 需重算 {} 张",
            to_score.len() - need_face.len(),
            need_face.len()
        );
    } else {
        need_face = to_score.clone();
    }

    // 分批推理：双缓冲流水线（producer 用 rayon 并行预处理下一批 tensor，consumer 用
    // 现有单 session 逐张推理 → 解码与 GPU 推理重叠）。批内顺序、模型调用顺序、
    // session 串行语义不变 → 分数确定性不变。
    const BATCH: usize = 16;
    let need_paths: Vec<String> = need_infer.iter().map(|&i| infos[i].path.clone()).collect();
    let mut progress = |done: usize, total: usize| emit(ScanPhase::Quality, done, total, None, "");
    let (batch_aes, batch_tech, timing) =
        crate::ai::engine::score_batch_scores(engine, &need_paths, BATCH, &mut progress);
    log::info!(
        "[AI 评分] 双缓冲流水线: 预处理(与推理重叠) {:.2}s, 推理 {:.2}s, wall {:.2}s",
        timing.prep_sec(),
        timing.infer_sec,
        timing.wall_sec
    );

    for (j, &idx) in need_infer.iter().enumerate() {
        let a = batch_aes.get(j).copied().flatten();
        let t = batch_tech.get(j).copied().flatten();
        cached_aes[idx] = a;
        cached_tech[idx] = t;
        // 写回评分缓存（增量模式下下次命中）
        let _ = db.save_ai_scores(&infos[idx].file_hash, a, t);
    }

    // 场景分类（MobileNetV3）：只对需要重算的图片分类，人像由 has_faces 覆盖。逐块推进度。
    if engine.scene_scoring_available() && !need_face.is_empty() {
        const CHUNK: usize = 64;
        emit(ScanPhase::Quality, 0, need_face.len(), None, "识别内容");
        let mut done = 0usize;
        let scene_start = std::time::Instant::now();
        let mut pet = 0;
        let mut landscape = 0;
        for chunk in need_face.chunks(CHUNK) {
            let cpaths: Vec<String> = chunk.iter().map(|&i| infos[i].path.clone()).collect();
            let classified = engine.scene_scores(&cpaths);
            for (k, &idx) in chunk.iter().enumerate() {
                let sc = classified.get(k).copied().unwrap_or(crate::ai::scene::Scene::Other);
                // 人像覆盖：检测到人脸 → 强制人像（ImageNet 无 person 类，MobileNetV3 判不出人像）
                scenes[idx] = if has_faces[idx] {
                    crate::ai::scene::Scene::Portrait
                } else {
                    match sc {
                        crate::ai::scene::Scene::Pet => {
                            pet += 1;
                            sc
                        }
                        crate::ai::scene::Scene::Landscape => {
                            landscape += 1;
                            sc
                        }
                        _ => sc,
                    }
                };
            }
            done += chunk.len();
            emit(ScanPhase::Quality, done, need_face.len(), None, "识别内容");
        }
        log::info!(
            "[AI 评分] 场景分类完成, 耗时 {:.1}s (宠物 {} 张, 风景 {} 张)",
            scene_start.elapsed().as_secs_f64(),
            pet,
            landscape
        );
    }

    // 人脸专评：仅对需要重算的图片（增量模式未命中人脸缓存者）做检测 + TOPIQ-NR-Face。
    // 逐块检测并推送进度，避免子阶段内进度条不动。
    if engine.face_scoring_available() && !need_face.is_empty() {
        const CHUNK: usize = 64;
        let face_start = std::time::Instant::now();
        emit(ScanPhase::Quality, 0, need_face.len(), None, "识别面部");
        let mut done = 0usize;
        let mut has_face_total = 0usize;
        for chunk in need_face.chunks(CHUNK) {
            let cpaths: Vec<String> = chunk.iter().map(|&i| infos[i].path.clone()).collect();
            let (f_scores, f_has) = engine.face_scores(&cpaths);
            for (k, &idx) in chunk.iter().enumerate() {
                face_scores[idx] = f_scores.get(k).copied().flatten();
                has_faces[idx] = f_has.get(k).copied().unwrap_or(false);
                if has_faces[idx] {
                    scenes[idx] = crate::ai::scene::Scene::Portrait;
                }
            }
            has_face_total += f_has.iter().filter(|&&v| v).count();
            done += chunk.len();
            emit(ScanPhase::Quality, done, need_face.len(), None, "识别面部");
        }
        log::info!(
            "[AI 评分] 人脸专评完成, 耗时 {:.1}s (检测 {} 张, 有人脸 {} 张)",
            face_start.elapsed().as_secs_f64(),
            need_face.len(),
            has_face_total
        );
    }

    // 闭眼检测（OCEC）：仅对有脸的图做（人像场景才有意义）。返回 max(open_l, open_r)。
    // 只在检测到的人脸图上逐块推进度。
    if engine.eye_status_available() {
        const CHUNK: usize = 64;
        let face_idxs: Vec<usize> = need_face.iter().copied().filter(|&i| has_faces[i]).collect();
        let face_cnt = face_idxs.len();
        if face_cnt > 0 {
            emit(ScanPhase::Quality, 0, face_cnt, None, "识别眼部");
            let mut done = 0usize;
            let eye_start = std::time::Instant::now();
            let mut closed_count = 0usize;
            for chunk in face_idxs.chunks(CHUNK) {
                let cpaths: Vec<String> = chunk.iter().map(|&i| infos[i].path.clone()).collect();
                let chas = vec![true; chunk.len()];
                let opens = engine.eye_open_probs(&cpaths, &chas);
                for (k, &idx) in chunk.iter().enumerate() {
                    let open = opens.get(k).copied().unwrap_or(1.0);
                    eye_open_vals[idx] = open;
                    let closed = crate::ai::eye::is_closed(open);
                    eye_closed[idx] = closed;
                    if closed {
                        closed_count += 1;
                    }
                }
                done += chunk.len();
                emit(ScanPhase::Quality, done, face_cnt, None, "识别眼部");
            }
            log::info!(
                "[AI 评分] 闭眼检测完成, 耗时 {:.1}s (识别眼部 {} 张, 闭眼 {} 张)",
                eye_start.elapsed().as_secs_f64(),
                face_cnt,
                closed_count
            );
        }
    }

    // 对焦分：人像→眼部对焦，非人像→整图对焦。仅对未命中缓存的 need_face 计算，
    // 命中者已从 AiFaceCache 恢复 focus_score。逐块推进度。
    if !need_face.is_empty() {
        const CHUNK: usize = 64;
        emit(ScanPhase::Quality, 0, need_face.len(), None, "对焦判断");
        let mut done = 0usize;
        for chunk in need_face.chunks(CHUNK) {
            let cpaths: Vec<String> = chunk.iter().map(|&i| infos[i].path.clone()).collect();
            let chas: Vec<bool> = chunk.iter().map(|&i| has_faces[i]).collect();
            let f_scores = engine.focus_scores(&cpaths, &chas);
            for (k, &idx) in chunk.iter().enumerate() {
                focus_vals[idx] = f_scores.get(k).copied().unwrap_or(1.0);
            }
            done += chunk.len();
            emit(ScanPhase::Quality, done, need_face.len(), None, "对焦判断");
        }
    }

    // 综合评分：用缓存/推理结果 + 启发式（权重依赖尺寸，与哈希缓存一致）
    let cached_aes_arr = cached_aes;
    let cached_tech_arr = cached_tech;
    let widths: Vec<u32> = infos.iter().map(|i| i.width).collect();
    let heights: Vec<u32> = infos.iter().map(|i| i.height).collect();
    let sizes: Vec<u64> = infos.iter().map(|i| i.size).collect();
    let comp = engine.composite_scores(
        Some(&cached_aes_arr.iter().map(|o| o.unwrap_or(0.0)).collect::<Vec<_>>()),
        Some(&focus_vals),
        Some(&face_scores.iter().map(|o| o.unwrap_or(0.0)).collect::<Vec<_>>()),
        &has_faces,
        &scenes,
        &eye_open_vals,
        &widths,
        &heights,
        &sizes,
    );
    match comp {
        Ok(s) if !s.is_empty() => {
            // 【修复】s 是全长数组（宽度 = infos.len()），须用图片索引 idx 取值，
            // 而非 to_score 内的位置 i。此前用 s.get(i) 在 to_score 非连续
            // （含未入组的唯一图）时会错位，导致推荐分写错。
            for &idx in &to_score {
                scores[idx] = s.get(idx).copied().or_else(|| cached_aes_arr[idx].map(|_| 5.0));
            }
            for &idx in &to_score {
                aesthetic[idx] = cached_aes_arr[idx];
                technical[idx] = cached_tech_arr[idx];
            }
        }
        Ok(_) => {
            log::warn!("综合评分返回空，本批使用启发式推荐");
        }
        Err(e) => log::warn!("综合评分失败: {}", e),
    }

    // 阶段一：写回人脸/场景/闭眼缓存（hash 未变，下次增量扫描直接命中）。
    // 非增量模式也会写，使紧随其后的增量重扫受益。
    // 仅当三类模型都真正产出结果时才写——若任一模型缺失，has_face/scene/eye_closed
    // 是默认占位值，写回会把"假结果"持久化成有效缓存，日后模型补齐也会被静默复用。
    if engine.face_scoring_available()
        && engine.scene_scoring_available()
        && engine.eye_status_available()
    {
        for &idx in &need_face {
            let _ = db.save_ai_face(
                &infos[idx].file_hash,
                has_faces[idx],
                face_scores[idx],
                scenes[idx] as u8,
                eye_open_vals[idx],
                focus_vals[idx],
            );
        }
    }

    (scores, aesthetic, technical, face_scores, has_faces, scenes, eye_closed, focus_vals)
}

// ---- 缓存清理（移入系统回收站，可恢复） ----

fn dir_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map(|e| e.flatten().count()).unwrap_or(0)
}

fn dir_bytes(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|e| e.flatten().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum())
        .unwrap_or(0)
}

fn dir_files(dir: &std::path::Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .map(|e| e.flatten().map(|f| f.path()).collect())
        .unwrap_or_default()
}

/// 单文件缓存（存在才返回）。
fn cache_file(path: &std::path::Path) -> Option<PathBuf> {
    if path.is_file() {
        Some(path.to_path_buf())
    } else {
        None
    }
}

/// 日志文件列表（顶层 pixsweep.log + logs/ 子目录）。
fn log_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = cache_file(&crate::app_data_dir().join("pixsweep.log")) {
        out.push(p);
    }
    out.extend(dir_files(&crate::app_data_dir().join("logs")));
    out
}

/// 隔离区文件列表（files/ 下的文件）。
fn quarantine_files() -> Vec<PathBuf> {
    crate::fileops::trash::list_quarantine()
        .into_iter()
        .map(|e| {
            crate::fileops::trash::quarantine_dir_path()
                .join("files")
                .join(&e.quarantine_filename)
        })
        .collect()
}

/// 查询各缓存类型的体积摘要（供前端"清理缓存"面板勾选）。
#[tauri::command]
pub fn get_cache_summary() -> Vec<CacheSummary> {
    let ai_cache = cache_file(&crate::app_data_dir().join("pixsweep-cache.json"));
    let quarantines = crate::fileops::trash::list_quarantine();
    vec![
        CacheSummary {
            cache_type: CacheType::Proxy,
            count: crate::cache::proxy::proxy_cache_files().len(),
            bytes: crate::cache::proxy::proxy_cache_bytes(),
        },
        CacheSummary {
            cache_type: CacheType::Thumbnails,
            count: dir_count(&crate::app_data_dir().join("thumbnails")),
            bytes: dir_bytes(&crate::app_data_dir().join("thumbnails")),
        },
        CacheSummary {
            cache_type: CacheType::AiCache,
            count: usize::from(ai_cache.is_some()),
            bytes: ai_cache
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .unwrap_or(0),
        },
        CacheSummary {
            cache_type: CacheType::Logs,
            count: log_files().len(),
            bytes: log_files()
                .iter()
                .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                .sum(),
        },
        CacheSummary {
            cache_type: CacheType::Quarantine,
            count: quarantines.len(),
            bytes: quarantines.iter().map(|e| e.size).sum(),
        },
    ]
}

/// 清理选中的缓存类型：把对应文件移入**系统回收站**（可恢复，非永久删）。
/// 同步清理核心（供异步命令与 MCP 复用）。逐个把文件移入系统回收站。
pub fn cleanup_cache_sync(types: Vec<CacheType>) -> CacheCleanupResult {
    let mut files: Vec<PathBuf> = Vec::new();
    for t in &types {
        match t {
            CacheType::Proxy => files.extend(crate::cache::proxy::proxy_cache_files()),
            CacheType::Thumbnails => files.extend(dir_files(&crate::app_data_dir().join("thumbnails"))),
            CacheType::AiCache => {
                if let Some(p) = cache_file(&crate::app_data_dir().join("pixsweep-cache.json")) {
                    files.push(p);
                }
            }
            CacheType::Logs => files.extend(log_files()),
            CacheType::Quarantine => files.extend(quarantine_files()),
        }
    }
    let mut moved = 0u32;
    let mut failed = 0u32;
    for f in &files {
        match trash::delete(f) {
            Ok(()) => moved += 1,
            Err(e) => {
                log::warn!("[清理] 移入系统回收站失败 {}: {}", f.display(), e);
                failed += 1;
            }
        }
    }
    // 隔离区文件移入系统回收站后，清空 index 保持记录一致
    if types.contains(&CacheType::Quarantine) {
        let _ = crate::fileops::trash::clear_quarantine_index();
    }
    CacheCleanupResult { moved, failed }
}

/// 清理选中的缓存类型：把对应文件移入**系统回收站**（可恢复，非永久删）。
/// async + spawn_blocking：逐文件移入回收站可能较慢，避免阻塞主线程导致界面假死。
#[tauri::command]
pub async fn cleanup_cache(types: Vec<CacheType>) -> CacheCleanupResult {
    tauri::async_runtime::spawn_blocking(move || cleanup_cache_sync(types))
        .await
        .unwrap_or(CacheCleanupResult { moved: 0, failed: 0 })
}

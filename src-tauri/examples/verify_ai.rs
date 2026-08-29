//! 验证脚本：加载 AI 引擎，对一批真实图片打分，检查
//! 1) TOPIQ-NR/IAA 在三级回退上成功加载（backend + has_topiq_*）
//! 2) 同一张图两次打分一致（确定性，修复"同图不同分"）
//! 3) 打印技术/美学/综合分，便于人工核对分布是否合理
//!
//! 用法：cargo run --example verify_ai -- <图片目录> [最多N张]

use pixsweep_lib::ai::engine::AiEngine;
use pixsweep_lib::image_io;
use pixsweep_lib::models_dir;
use image::GenericImageView;
use std::path::Path;

struct Img {
    path: String,
    name: String,
    width: u32,
    height: u32,
    size: u64,
}

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).cloned() else {
        eprintln!("用法: cargo run --example verify_ai -- <图片目录> [最多N张]");
        std::process::exit(2);
    };
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);

    // 验证脚本优先使用源码树中的 models 目录（cargo run --example 时
    // current_exe 在 target/debug/examples 下，models_dir() 会回退到 AppData）。
    let model_dir = {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        if manifest.join("topiq_nr.onnx").exists() {
            manifest
        } else {
            models_dir()
        }
    };
    println!("[verify] 模型目录: {}", model_dir.display());
    let engine = match AiEngine::new(&model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[verify] 引擎初始化失败: {}", err);
            std::process::exit(1);
        }
    };
    println!(
        "[verify] 后端: {} | TOPIQ-NR 可用: {} | TOPIQ-IAA 可用: {}",
        engine.backend().label(),
        engine.has_topiq_nr(),
        engine.has_topiq_iia(),
    );

    // 收集可解码的图片（同时校验 EXIF Orientation 解码不报错）
    let mut imgs: Vec<Img> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            // 与生产扫描器同一判定（含相机 RAW）
            if !pixsweep_lib::scanner::walker::is_supported_image(&p) {
                continue;
            }
            let path = p.to_string_lossy().to_string();
            match image_io::load_image_oriented(&path) {
                Ok(img) => {
                    // 与生产扫描同口径：RAW 用传感器原生分辨率（EXIF 转正），
                    // 解码尺寸仅是机内嵌预览，会低估分辨率启发式
                    let (w, h) =
                        pixsweep_lib::image_io::raw_source_dimensions(std::path::Path::new(&path))
                            .unwrap_or_else(|| img.dimensions());
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    let name = Path::new(&path)
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    println!("[verify]   {}  {}x{}  (EXIF 方向解码 OK)", name, w, h);
                    imgs.push(Img {
                        path,
                        name,
                        width: w,
                        height: h,
                        size,
                    });
                }
                Err(err) => println!("[verify]   （解码失败）{}: {}", p.display(), err),
            }
            if imgs.len() >= limit {
                break;
            }
        }
    }
    if imgs.is_empty() {
        eprintln!("[verify] 目录中未找到可解码图片: {}", dir);
        std::process::exit(2);
    }
    println!("[verify] 选取 {} 张图片进行评分", imgs.len());

    let paths: Vec<String> = imgs.iter().map(|i| i.path.clone()).collect();
    let widths: Vec<u32> = imgs.iter().map(|i| i.width).collect();
    let heights: Vec<u32> = imgs.iter().map(|i| i.height).collect();
    let sizes: Vec<u64> = imgs.iter().map(|i| i.size).collect();

    // ---- 双缓冲批量评分（性能基准 seam）：producer rayon 并行预处理下一批 tensor，
    //      consumer 用单 session 逐张推理，解码与 GPU 推理重叠。返回与 paths 等长分值。----
    let mut noop = |_done: usize, _total: usize| {};
    let (aes_opt, tech_opt, timing) =
        pixsweep_lib::ai::engine::score_batch_scores(&engine, &paths, 16, &mut noop);
    let tech: Vec<f32> = tech_opt.iter().map(|o| o.unwrap_or(0.0)).collect();
    let aes: Vec<f32> = aes_opt.iter().map(|o| o.unwrap_or(0.0)).collect();

    // 综合分
    let n = tech.len();
    let face_none: Vec<Option<f32>> = vec![None; n];
    let has_face: Vec<bool> = vec![false; n];
    let scenes = vec![pixsweep_lib::ai::scene::Scene::Other; n];
    let eye_open: Vec<f32> = vec![1.0; n];
    let t4 = std::time::Instant::now();
    let focus = engine.focus_scores(&paths, &has_face);
    let focus_sec = t4.elapsed().as_secs_f64();
    let t5 = std::time::Instant::now();
    let comp = engine
        .composite_scores(
            if aes.is_empty() { None } else { Some(&aes) },
            Some(&focus),
            Some(&face_none.iter().map(|o| o.unwrap_or(0.0)).collect::<Vec<_>>()),
            &has_face,
            &scenes,
            &eye_open,
            &widths,
            &heights,
            &sizes,
        )
        .expect("综合评分失败");
    let comp_sec = t5.elapsed().as_secs_f64();

    println!("\n[verify] 评分结果（技术=TOPIQ-NR/CLIP-IQA/NIMA 1~10, 美学=TOPIQ-IAA/CLIP 1~10, 综合=加权）");
    println!(
        "{:<24} {:<8} {:<8} {:<8}",
        "文件名", "技术", "美学", "综合"
    );
    for (i, im) in imgs.iter().enumerate() {
        println!(
            "{:<24} {:<8.2} {:<8.2} {:<8.2}",
            im.name.chars().take(22).collect::<String>(),
            tech.get(i).copied().unwrap_or(0.0),
            aes.get(i).copied().unwrap_or(0.0),
            comp.get(i).copied().unwrap_or(0.0)
        );
    }

    // 确定性检查：单张图走完整双缓冲流水线两次，首图技术分（与美学分）应逐位一致
    let one: Vec<String> = vec![paths[0].clone()];
    let (a1, t1, _) = pixsweep_lib::ai::engine::score_batch_scores(&engine, &one, 1, &mut noop);
    let (a2, t2, _) = pixsweep_lib::ai::engine::score_batch_scores(&engine, &one, 1, &mut noop);
    let tech0_before = t1.first().copied().flatten().unwrap_or(0.0);
    let tech0_after = t2.first().copied().flatten().unwrap_or(0.0);
    let aes0_before = a1.first().copied().flatten().unwrap_or(0.0);
    let aes0_after = a2.first().copied().flatten().unwrap_or(0.0);
    println!(
        "\n[verify] 确定性检查：首图技术分 第一次={:.6} 第二次={:.6} 一致={} | 美学分 第一次={:.6} 第二次={:.6} 一致={}",
        tech0_before,
        tech0_after,
        (tech0_before - tech0_after).abs() < 1e-5,
        aes0_before,
        aes0_after,
        (aes0_before - aes0_after).abs() < 1e-5
    );

    // 双缓冲流水线计时（性能基准 seam）：同一目录 + 同一 N 下对比优化前后。
    // wall 含重叠，通常 < prep + infer —— 解码隐藏在 GPU 推理期间即优化生效。
    println!(
        "\n[verify] 计时(双缓冲流水线): 预处理(与推理重叠) TOPIQ={:.2}s NIMA={:.2}s 推理={:.2}s wall={:.2}s 每张≈{:.0}ms 对焦+综合={:.2}s",
        timing.prep_topiq_sec,
        timing.prep_nima_sec,
        timing.infer_sec,
        timing.wall_sec,
        timing.wall_sec * 1000.0 / n.max(1) as f64,
        focus_sec + comp_sec
    );
}

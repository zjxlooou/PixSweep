//! 真图验证：InsightFace 人脸检测 + TOPIQ-NR-Face 专评。
//!
//! 用法: cargo run --example verify_face -- <照片目录> [最大张数]

use pixsweep_lib::ai::engine::AiEngine;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).map(|s| s.as_str()) else {
        eprintln!("用法: cargo run --example verify_face -- <照片目录> [最大张数]");
        std::process::exit(2);
    };
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    // 模型目录：相对 src-tauri 运行
    let model_dir = Path::new("models");
    println!("[face] 模型目录: {}", model_dir.display());

    // 初始化 AI 引擎（会加载 InsightFace + TOPIQ-NR-Face）
    let engine = match AiEngine::new(model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[face] AI 引擎初始化失败: {err}");
            return;
        }
    };

    println!(
        "[face] 人脸专评可用: {} | GPU: {}",
        engine.face_scoring_available(),
        engine.gpu_enabled()
    );
    if !engine.face_scoring_available() {
        eprintln!("[face] InsightFace 或 TOPIQ-NR-Face 模型缺失，退出");
        return;
    }

    // 收集图片
    let mut paths: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s.to_ascii_lowercase())
                .unwrap_or_default();
            if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
                paths.push(p.to_string_lossy().to_string());
            }
            if paths.len() >= limit {
                break;
            }
        }
    }
    println!("[face] 测试 {} 张照片", paths.len());

    // 人脸检测 + 专评
    let start = std::time::Instant::now();
    let (scores, has_face) = engine.face_scores(&paths);
    let elapsed = start.elapsed().as_secs_f64();

    let mut face_count = 0;
    for (i, (s, h)) in scores.iter().zip(has_face.iter()).enumerate() {
        let name = Path::new(&paths[i])
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if *h {
            face_count += 1;
            println!(
                "{:<28} 有人脸 人脸专评 {:.2}/10",
                name.chars().take(26).collect::<String>(),
                s.unwrap_or(0.0)
            );
        } else {
            println!(
                "{:<28} 无人脸（跳过专评）",
                name.chars().take(26).collect::<String>()
            );
        }
    }
    println!(
        "\n[face] 完成: {} 张, 检测到人脸 {} 张, 耗时 {:.1}s ({:.0}ms/张)",
        paths.len(),
        face_count,
        elapsed,
        elapsed * 1000.0 / paths.len() as f64
    );
}

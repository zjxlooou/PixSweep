//! 真图验证：OCEC 闭眼检测（InsightFace 人脸检测 → 眼点 ROI → 闭眼判定）。
//!
//! 用法: cargo run --example verify_eye -- <照片目录> [最大张数]

use pixsweep_lib::ai::engine::AiEngine;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).map(|s| s.as_str()) else {
        eprintln!("用法: cargo run --example verify_eye -- <照片目录> [最大张数]");
        std::process::exit(2);
    };
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let model_dir = Path::new("models");
    let engine = match AiEngine::new(model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[eye] AI 引擎初始化失败: {err}");
            return;
        }
    };

    println!(
        "[eye] 闭眼检测可用: {} | GPU: {}",
        engine.eye_status_available(),
        engine.gpu_enabled()
    );
    if !engine.eye_status_available() {
        eprintln!("[eye] OCEC 或 InsightFace 模型缺失，退出");
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
    println!("[eye] 测试 {} 张照片", paths.len());

    // 先做人脸检测（确定哪些图有人脸），再对有人脸的图做闭眼检测
    let (_, has_faces) = engine.face_scores(&paths);
    println!("[eye] 检测到人脸 {} 张", has_faces.iter().filter(|&&v| v).count());

    let start = std::time::Instant::now();
    let eye_flags = engine.eye_status(&paths, &has_faces);
    let elapsed = start.elapsed().as_secs_f64();

    let mut closed_count = 0;
    for (i, (flag, has)) in eye_flags.iter().zip(has_faces.iter()).enumerate() {
        let name = Path::new(&paths[i])
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if *has {
            if *flag {
                closed_count += 1;
                println!("{:<28} 有人脸 闭眼 ⚠", name.chars().take(26).collect::<String>());
            } else {
                println!("{:<28} 有人脸 睁眼 ✓", name.chars().take(26).collect::<String>());
            }
        } else {
            println!("{:<28} 无人脸", name.chars().take(26).collect::<String>());
        }
    }
    println!(
        "\n[eye] 完成: {} 张, 有人脸 {} 张, 闭眼 {} 张, 耗时 {:.1}s",
        paths.len(),
        has_faces.iter().filter(|&&v| v).count(),
        closed_count,
        elapsed
    );
}
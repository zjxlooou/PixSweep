//! 真图验证：MobileNetV3 场景分类（人像/宠物/风景/其他）。
//!
//! 用法: cargo run --example verify_scene -- <照片目录> [最大张数]
//! 注意：此验证只测场景分类本身（不含人脸覆盖）。实际业务中人像由 InsightFace 覆盖。

use pixsweep_lib::ai::engine::AiEngine;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).map(|s| s.as_str()) else {
        eprintln!("用法: cargo run --example verify_scene -- <照片目录> [最大张数]");
        std::process::exit(2);
    };
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let model_dir = Path::new("models");
    let engine = match AiEngine::new(model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[scene] AI 引擎初始化失败: {err}");
            return;
        }
    };

    println!(
        "[scene] 场景分类可用: {} | GPU: {}",
        engine.scene_scoring_available(),
        engine.gpu_enabled()
    );
    // 运行时验证 SCENE_MAP（排查 include! 是否编译了最新映射）
    println!(
        "[scene] SCENE_MAP[975]={} (应为 3=风景) | SCENE_MAP[151]={} (应为 2=宠物)",
        pixsweep_lib::ai::scene::SCENE_MAP[975],
        pixsweep_lib::ai::scene::SCENE_MAP[151]
    );
    if !engine.scene_scoring_available() {
        eprintln!("[scene] MobileNetV3 模型缺失，退出");
        return;
    }

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
    println!("[scene] 测试 {} 张照片", paths.len());

    let start = std::time::Instant::now();
    let scenes = engine.scene_scores(&paths);
    let elapsed = start.elapsed().as_secs_f64();

    let mut counts = std::collections::HashMap::new();
    for (i, sc) in scenes.iter().enumerate() {
        let name = Path::new(&paths[i])
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        *counts.entry(sc.label()).or_insert(0) += 1;
        // 打印 argmax 索引（辅助排查）
        let argmax = engine.scene_argmax(&paths[i]);
        println!(
            "{:<28} {} (argmax={})",
            name.chars().take(26).collect::<String>(),
            sc.label(),
            argmax
        );
    }
    println!("\n[scene] 分类统计: {:?}", counts);
    println!(
        "[scene] 完成: {} 张, 耗时 {:.1}s ({:.0}ms/张)",
        paths.len(),
        elapsed,
        elapsed * 1000.0 / paths.len() as f64
    );
}

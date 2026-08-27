//! 全维度评分诊断：对指定图片逐张打印
//!   技术分 / 美学分 / 人脸专评分 / 是否有人脸 / 场景 / 闭眼 / EXIF / 综合分。
//!
//! 用法：cargo run --example verify_full -- <图片路径> [更多路径...]
//!
//! 等价于 commands.rs 里 `score_groups_with_ai` 的推理部分（不含分组/缓存）。

use pixsweep_lib::ai::engine::AiEngine;
use pixsweep_lib::ai::preprocess;
use pixsweep_lib::ai::scene::Scene;
use pixsweep_lib::image_io::exif_orientation;
use pixsweep_lib::models_dir;
use std::path::Path;

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let paths: Vec<String> = args[1..].to_vec();
    if paths.is_empty() {
        eprintln!("usage: verify_full <image> [image...]");
        std::process::exit(2);
    }

    // 优先使用源码树 models（cargo run --example 时 current_exe 在 target/debug/examples）
    let model_dir = {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        if manifest.join("clip-vit-b32-visual.onnx").exists() {
            manifest
        } else {
            models_dir()
        }
    };
    println!("[verify_full] 模型目录: {}", model_dir.display());
    let engine = match AiEngine::new(&model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[verify_full] 引擎初始化失败: {}", err);
            std::process::exit(1);
        }
    };

    // 人脸 + 场景 + 闭眼
    let (face_scores, has_faces) = engine.face_scores(&paths);
    let mut scenes = engine.scene_scores(&paths);
    for s in scenes.iter_mut() {
        if *s == Scene::Portrait {
            // scene_scores 已可能给 Portrait（MobileNet 人像类）
        }
    }
    let eye_open = engine.eye_open_probs(&paths, &has_faces);
    let focus = engine.focus_scores(&paths, &has_faces);
    // 有脸 → 场景覆盖为人像（与 commands.rs 一致）
    for (i, has) in has_faces.iter().enumerate() {
        if *has {
            scenes[i] = Scene::Portrait;
        }
    }

    // 技术 / 美学
    let clip_batch = preprocess::images_to_batch(&paths).expect("CLIP 预处理失败");
    let topiq_batch = preprocess::images_to_batch_topiq(&paths).expect("TOPIQ 预处理失败");
    let tech = if engine.has_topiq_nr() {
        engine.topiq_nr_scores(&topiq_batch).expect("TOPIQ-NR 推理失败")
    } else if engine.has_clipiqa() {
        engine.clipiqa_scores(&clip_batch).expect("CLIP-IQA 推理失败")
    } else {
        let nima_batch = preprocess::images_to_batch_nima(&paths).expect("NIMA 预处理失败");
        engine.nima_technical_scores(&nima_batch).expect("NIMA 推理失败")
    };
    let aes = if engine.has_topiq_iia() {
        engine.topiq_iia_scores(&topiq_batch).expect("TOPIQ-IAA 推理失败")
    } else {
        let embeds = engine.extract_embeddings(&clip_batch).expect("CLIP 推理失败");
        engine.aesthetic_scores(&embeds).unwrap_or_default()
    };

    let widths: Vec<u32> = paths.iter().map(|p| image::image_dimensions(p).unwrap_or((0, 0)).0).collect();
    let heights: Vec<u32> = paths.iter().map(|p| image::image_dimensions(p).unwrap_or((0, 0)).1).collect();
    let sizes: Vec<u64> = paths.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
    let face_vals: Vec<f32> = face_scores.iter().map(|o| o.unwrap_or(0.0)).collect();

    let comp = engine
        .composite_scores(Some(&aes), Some(&focus), Some(&face_vals), &has_faces, &scenes, &eye_open, &widths, &heights, &sizes)
        .expect("综合评分失败");

    println!("\n{:<22} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6}", "文件名", "技术", "美学", "人脸", "hasFace", "场景", "闭眼", "EXIF", "综合");
    for (i, p) in paths.iter().enumerate() {
        let name = Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let exif = exif_orientation(p).map(|o| o.to_exif()).unwrap_or(0);
        let scene_tag = match scenes[i] {
            Scene::Portrait => "人像",
            Scene::Landscape => "风景",
            Scene::Pet => "宠物",
            Scene::Other => "其他",
        };
        let eye_info = if has_faces[i] {
            engine.eye_probs(p).map(|(l, r)| format!("L={:.2} R={:.2}", l, r)).unwrap_or_else(|| "-".into())
        } else {
            "-".into()
        };
        if has_faces[i] {
            if let Some(lms) = engine.largest_face_landmarks(p) {
                println!("  [lmks] {} bbox={:?} 眼1=({:.0},{:.0}) 眼2=({:.0},{:.0}) 鼻=({:.0},{:.0}) 嘴角1=({:.0},{:.0}) 嘴角2=({:.0},{:.0})",
                    name.chars().take(14).collect::<String>(), lms.0, lms.1[0].0, lms.1[0].1, lms.1[1].0, lms.1[1].1, lms.1[2].0, lms.1[2].1, lms.1[3].0, lms.1[3].1, lms.1[4].0, lms.1[4].1);
            }
        }
        println!(
            "{:<22} {:>6.2} {:>6.2} {:>6.2} {:>7} {:>6} {:>10} {:>6} {:>6.2}",
            name.chars().take(20).collect::<String>(),
            tech[i],
            aes[i],
            face_scores[i].map(|v| v).unwrap_or(f32::NAN),
            has_faces[i],
            scene_tag,
            eye_info,
            exif,
            comp[i],
        );
    }
}
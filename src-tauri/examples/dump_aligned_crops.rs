//! 一次性研究工具：把指定目录里的图片经生产同款链路
//! （InsightFace 检测 → 最大脸 → align_face 512×512）导出对齐后人脸 crop，
//! 供 Python 侧公平评测人脸质量/美学候选模型。用完即删。
//!
//! 用法：cargo run --release --example dump_aligned_crops -- <图片目录> <输出目录>

use image::GenericImageView;
use pixsweep_lib::ai::insightface::InsightFaceEngine;
use pixsweep_lib::image_io;
use std::path::Path;

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).cloned() else {
        eprintln!("用法: dump_aligned_crops -- <图片目录> <输出目录>");
        std::process::exit(2);
    };
    let out_dir = args.get(2).cloned().unwrap_or_else(|| dir.clone() + "_crops");
    std::fs::create_dir_all(&out_dir).expect("创建输出目录");

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let face_engine = InsightFaceEngine::new();
    face_engine
        .load(&manifest.join("insightface"), false)
        .expect("InsightFace 加载失败");

    let mut n_face = 0usize;
    let mut n_total = 0usize;
    let entries = std::fs::read_dir(&dir).expect("读目录");
    for e in entries.flatten() {
        let p = e.path();
        let Some(ext) = p.extension().and_then(|s| s.to_str()) else { continue };
        if !matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png") { continue; }
        n_total += 1;
        let Ok(img) = image_io::load_image_oriented(p.to_string_lossy().as_ref()) else { continue };
        let (w, h) = img.dimensions();
        let rgb = img.to_rgb8().into_raw();
        let faces = face_engine.detect(&rgb, h, w).unwrap_or_default();
        let Some(face) = faces.iter().max_by(|a, b| {
            ((a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]))
                .partial_cmp(&((b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])))
                .unwrap()
        }) else {
            println!("无脸: {}", p.file_name().unwrap().to_string_lossy());
            continue;
        };
        let crop = face_engine.align_face(&rgb, h, w, face, 512);
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        image::save_buffer(
            Path::new(&out_dir).join(format!("{stem}_face.png")),
            &crop,
            512,
            512,
            image::ColorType::Rgb8,
        )
        .expect("保存 crop");
        n_face += 1;
        println!("已导出: {}（脸 {:.2}）", stem, face.score);
    }
    println!("\n共 {n_total} 图，其中 {n_face} 张导出对齐人脸");
}

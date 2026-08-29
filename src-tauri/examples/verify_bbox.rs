//! 诊断：把旋转图 + 每张检测到的"人脸"区域单独裁剪放大，供人工核对
//! det 认为的脸到底是不是脸。同时输出整图带粗红框。
//! 用法：cargo run --example verify_bbox -- <图片路径...>

use pixsweep_lib::ai::eye::debug_sample_eyes_rgb;
use pixsweep_lib::ai::insightface::InsightFaceEngine;
use pixsweep_lib::ai::eye::EYE_H;
use pixsweep_lib::ai::eye::EYE_W;
use pixsweep_lib::image_io::load_image_oriented;
use std::path::Path;

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let paths: Vec<String> = args[1..].to_vec();
    if paths.is_empty() {
        eprintln!("usage: verify_bbox <image>...");
        std::process::exit(2);
    }
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("insightface");
    let mut eng = InsightFaceEngine::new();
    if let Err(err) = eng.load(&model_dir, false) {
        eprintln!("[bbox] InsightFace 加载失败: {err}");
        std::process::exit(1);
    }

    for p in paths {
        let Ok(img) = load_image_oriented(&p) else {
            eprintln!("[bbox] 解码失败 {p}");
            continue;
        };
        let (w, h) = (img.width(), img.height());
        let mut rgb = img.to_rgb8();
        // 干净副本（未画红框/绿点），供采样眼 ROI 用，避免把标记点采进去
        let rgb_pure = rgb.clone();
        let faces = match eng.detect(rgb.as_raw(), h, w) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[bbox] 检测失败: {e}");
                continue;
            }
        };
        let name = Path::new(&p).file_stem().unwrap().to_string_lossy();
        println!("[bbox] {} 旋转图 {}x{} 检测到 {} 张脸", name, w, h, faces.len());

        for (fi, f) in faces.iter().enumerate() {
            let b = f.bbox;
            let cx = ((b[0] + b[2]) / 2.0) as i32;
            let cy = ((b[1] + b[3]) / 2.0) as i32;
            let bw = (b[2] - b[0]) as i32;
            let bh = (b[3] - b[1]) as i32;
            println!("   #{} score={:.3} bbox=[{:.0},{:.0},{:.0},{:.0}] 中心=({}, {}) 宽高=({}x{})",
                fi, f.score, b[0], b[1], b[2], b[3], cx, cy, bw, bh);

            // 在整图上画粗红框（5px）
            for t in -2i32..=2 {
                for x in (b[0].max(0.0) as i32)..=(b[2] as i32) {
                    for y in [b[1] as i32 + t, b[3] as i32 + t] {
                        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                            rgb.put_pixel(x as u32, y as u32, image::Rgb([255, 0, 0]));
                        }
                    }
                }
                for y in (b[1].max(0.0) as i32)..=(b[3] as i32) {
                    for x in [b[0] as i32 + t, b[2] as i32 + t] {
                        if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                            rgb.put_pixel(x as u32, y as u32, image::Rgb([255, 0, 0]));
                        }
                    }
                }
            }
            // 关键点：粗绿点（半径 12），并标序号
            let lms_with_id = [
                ("LE", f.landmarks.left_eye),
                ("RE", f.landmarks.right_eye),
                ("N", f.landmarks.nose),
                ("LM", f.landmarks.left_mouth),
                ("RM", f.landmarks.right_mouth),
            ];
            for (_, lm) in lms_with_id.iter() {
                for dy in -12i32..=12 { for dx in -12i32..=12 {
                    if dx*dx + dy*dy > 144 { continue; }
                    let (x, y) = (lm.0 as i32 + dx, lm.1 as i32 + dy);
                    if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                        rgb.put_pixel(x as u32, y as u32, image::Rgb([0, 255, 0]));
                    }
                }}
            }

            // 裁剪这个"人脸"区域，放大到 600px 宽，单独存一张
            let x0 = b[0].max(0.0) as u32;
            let y0 = b[1].max(0.0) as u32;
            let crop_w = (b[2] - b[0]) as u32;
            let crop_h = (b[3] - b[1]) as u32;
            if crop_w > 4 && crop_h > 4 {
                let sub = image::imageops::crop_imm(&rgb, x0.min(w-1), y0.min(h-1), crop_w.min(w-x0), crop_h.min(h-y0)).to_image();
                let big_w = 600u32;
                let big_h = (big_w * sub.height() / sub.width().max(1)).max(1);
                let big = image::imageops::resize(&sub, big_w, big_h, image::imageops::FilterType::Triangle);
                let out = std::env::temp_dir().join(format!("det_{}_f{}.jpg", name, fi));
                let mut fout = std::fs::File::create(&out).unwrap();
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut fout, 88);
                let _ = image::DynamicImage::ImageRgb8(big).write_with_encoder(enc);
                println!("   det 认为的脸区域 -> {}", out.display());
            }

            // 导出 align_face 输出 (256x256 正立人脸) + 按 112 模板切出的左右眼 ROI 拼图
            let aligned = eng.align_face(rgb.as_raw(), h, w, &f, 256);
            if aligned.len() == 256 * 256 * 3 {
                let crop_img = image::RgbImage::from_raw(256, 256, aligned.clone()).unwrap();
                let crop_path = std::env::temp_dir().join(format!("crop_{}_f{}.jpg", name, fi));
                let mut cf = std::fs::File::create(&crop_path).unwrap();
                let ce = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut cf, 90);
                let _ = image::DynamicImage::ImageRgb8(crop_img.clone()).write_with_encoder(ce);
                println!("   align_face crop -> {}", crop_path.display());

                // 导出 eye.rs::sample_eye_rgb_internal 实际采到的 ROI（不是按 crop 切）
                if let Some((lroi, rroi)) = debug_sample_eyes_rgb(rgb_pure.as_raw(), h, w, &f) {
                    let mut big2 = image::RgbImage::new(280, 180);
                    let lw = EYE_W as u32; let lh = EYE_H as u32;
                    for y in 0..lh { for x in 0..lw {
                        let p = image::Rgb([lroi[((y*lw+x)*3) as usize], lroi[((y*lw+x)*3+1) as usize], lroi[((y*lw+x)*3+2) as usize]]);
                        big2.put_pixel(20+x, 30+y, p);
                    }}
                    for y in 0..lh { for x in 0..lw {
                        let p = image::Rgb([rroi[((y*lw+x)*3) as usize], rroi[((y*lw+x)*3+1) as usize], rroi[((y*lw+x)*3+2) as usize]]);
                        big2.put_pixel(140+x, 30+y, p);
                    }}
                    let big2 = image::imageops::resize(&big2, 560, 360, image::imageops::FilterType::Triangle);
                    let eye2_path = std::env::temp_dir().join(format!("eye_real_{}_f{}.jpg", name, fi));
                    let mut pf = std::fs::File::create(&eye2_path).unwrap();
                    let pe = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut pf, 95);
                    let _ = image::DynamicImage::ImageRgb8(big2).write_with_encoder(pe);
                    println!("   eye.rs::sample_eye 实际 ROI (左=左眼, 右=右眼) -> {}", eye2_path.display());
                }
            }
        }

        // 整图缩小存 JPEG（带粗框）
        let scale_f = 900.0 / w as f32;
        let tw = 900u32;
        let th = ((h as f32 * scale_f) as u32).max(1);
        let small = image::imageops::resize(&rgb, tw, th, image::imageops::FilterType::Triangle);
        let out = std::env::temp_dir().join(format!("full_{}.jpg", name));
        let mut fout = std::fs::File::create(&out).unwrap();
        let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut fout, 85);
        let _ = image::DynamicImage::ImageRgb8(small).write_with_encoder(enc);
        println!("   整图(粗框) -> {}", out.display());

        // 红框区域原分辨率单独存图（绿点在此尺度清晰可见）
        if let Some(f) = faces.first() {
            let b = f.bbox;
            let x0 = b[0].max(0.0) as u32;
            let y0 = b[1].max(0.0) as u32;
            let cw = ((b[2] - b[0]) as u32).min(w - x0);
            let ch = ((b[3] - b[1]) as u32).min(h - y0);
            if cw > 4 && ch > 4 {
                let sub = image::imageops::crop_imm(&rgb, x0, y0, cw, ch).to_image();
                let out = std::env::temp_dir().join(format!("bbox_face_{}.jpg", name));
                let mut fout = std::fs::File::create(&out).unwrap();
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut fout, 92);
                let _ = image::DynamicImage::ImageRgb8(sub).write_with_encoder(enc);
                println!("   红框原图 -> {}", out.display());
            }
        }
    }
}
//! 验证 EXIF Orientation 处理是否生效。
//!
//! 用法：
//!   - 传单个图片路径：对比原始像素尺寸与 `load_image_oriented` 旋转后尺寸，
//!     并打印文件头/旧 from_exif_chunk 结果/新 decoder.orientation() 结果。
//!   - 传目录 [N]：扫描 N 张 JPG，统计非 1 orientation 与旋转生效情况。
//!
//! cargo run --example verify_orient -- <file-or-dir> [N]

use image::GenericImageView;
use pixsweep_lib::image_io;
use std::io::{Read, Seek};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(target) = args.get(1).cloned() else {
        eprintln!("用法: cargo run --example verify_orient -- <file-or-dir> [N]");
        std::process::exit(2);
    };
    let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    if Path::new(&target).is_file() {
        probe_one(&target);
        return;
    }
    scan_dir(&target, limit);
}

/// 单文件定向诊断
fn probe_one(s: &str) {
    use image::metadata::Orientation;

    let raw = image::image_dimensions(s).unwrap_or((0, 0));
    println!("file: {}", s);

    // 文件头前若干字节（JPEG 以 FF D8 开头，含 SOI + APP markers）
    if let Ok(mut f) = std::fs::File::open(s) {
        let mut head = vec![0u8; 16];
        let n = f.read(&mut head).unwrap_or(0);
        println!("  head[0..{}] = {:02X?}", n, &head[..n.min(16)]);
        // 旧实现：把整个文件头丢给 from_exif_chunk（它在偏移 0 找 TIFF magic → 必然失败）
        let mut big = vec![0u8; 65536];
        let _ = f.seek(std::io::SeekFrom::Start(0));
        let m = f.read(&mut big).unwrap_or(0);
        big.truncate(m);
        println!("  from_exif_chunk(文件头) = {:?}", Orientation::from_exif_chunk(&big));
    }

    println!("  raw (image::image_dimensions) = {}x{}", raw.0, raw.1);
    match image_io::load_image_oriented(s) {
        Ok(img) => {
            let (w, h) = img.dimensions();
            println!("  oriented (load_image_oriented) = {}x{}", w, h);
            // 旋转后落地一张 PNG 便于人工比对方向
            let out = std::env::temp_dir().join("pixsweep_orient_check.png");
            let _ = img.save(&out);
            println!("  saved preview -> {}", out.display());
        }
        Err(e) => println!("  load_image_oriented FAIL: {}", e),
    }
}

/// 目录批量统计
fn scan_dir(dir: &str, limit: usize) {
    use image::metadata::Orientation;

    let mut n = 0usize;
    let mut tag_present = 0usize;
    let mut rotated_ok = 0usize;
    let mut rotated_fail = 0usize;
    let mut errors = 0usize;

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let p = entry.path();
        if !p.is_file() {
            continue;
        }
        let Some(ext) = p.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg") {
            continue;
        }
        n += 1;

        let s = p.to_string_lossy().to_string();
        let raw = image::image_dimensions(p).unwrap_or((0, 0));

        let orient = match pixsweep_lib::image_io::exif_orientation(&s) {
            Ok(o) => o,
            Err(_) => Orientation::NoTransforms,
        };

        if orient != Orientation::NoTransforms {
            tag_present += 1;
        }
        let ori_num = orient.to_exif();
        if ori_num == 1 {
            continue; // 不需要旋转
        }
        match image_io::load_image_oriented(&s) {
            Ok(img) => {
                let (ow, oh) = img.dimensions();
                let swapped = raw.0 != ow || raw.1 != oh;
                if swapped {
                    rotated_ok += 1;
                } else {
                    rotated_fail += 1;
                }
                println!(
                    "[{}] orient={} raw={}x{} oriented={}x{}  {}",
                    if swapped { "OK  " } else { "FAIL" },
                    ori_num,
                    raw.0,
                    raw.1,
                    ow,
                    oh,
                    p.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()
                );
            }
            Err(e) => {
                errors += 1;
                println!("[ERR] {} {}", s, e);
            }
        }
        if n >= limit {
            break;
        }
    }
    println!(
        "\n扫描 {} 张：非默认 orientation {} 张；旋转生效 {} 张；未旋转(BUG) {} 张；解码失败 {} 张",
        n, tag_present, rotated_ok, rotated_fail, errors
    );
}
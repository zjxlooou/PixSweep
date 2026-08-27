//! 真图验证：对给定图片调用 `largest_face_landmarks` 输出人脸框 + 5 关键点
//! （EXIF 转向后的原图坐标），并做基本 sanity 校验。
//!
//! 用法：cargo run --example verify_landmarks -- <图片路径...>

use pixsweep_lib::ai::engine::AiEngine;
use pixsweep_lib::image_io::load_image_oriented;

const NAME: [&str; 5] = ["左眼", "右眼", "鼻尖", "左嘴角", "右嘴角"];

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let paths: Vec<String> = args[1..].to_vec();
    if paths.is_empty() {
        eprintln!("usage: verify_landmarks <image>...");
        std::process::exit(2);
    }
    let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let engine = match AiEngine::new(&model_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[landmarks] AiEngine 加载失败: {err}");
            std::process::exit(1);
        }
    };

    let mut ok = 0usize;
    for p in &paths {
        // 原图尺寸（EXIF 转向后）—— 校验竖屏图是否被正确转正（h>w）
        let (iw, ih) = load_image_oriented(p)
            .map(|i| (i.width(), i.height()))
            .unwrap_or((0, 0));
        match engine.largest_face_landmarks(p) {
            Some((bbox, pts)) => {
                let [bx1, by1, bx2, by2] = bbox;
                println!("[FOUND] {}  ({iw}x{ih})", p);
                println!(
                    "    bbox=[{bx1:.0},{by1:.0},{bx2:.0},{by2:.0}]  宽高 {:.0}x{:.0}",
                    bx2 - bx1,
                    by2 - by1
                );
                for (i, (x, y)) in pts.iter().enumerate() {
                    println!(
                        "    {:>3} ({:.0}, {:.0})",
                        NAME.get(i).copied().unwrap_or("?"),
                        x,
                        y
                    );
                }
                let in_bounds = pts.iter().all(|(x, y)| {
                    *x >= 0.0 && *y >= 0.0 && *x <= iw as f32 && *y <= ih as f32
                });
                let eyes_dx = pts.get(1).map(|r| r.0).unwrap_or(0.0) - pts.get(0).map(|l| l.0).unwrap_or(0.0);
                println!("    关键点界内: {in_bounds} | 眼距 dx={eyes_dx:.0}px");
                if in_bounds {
                    ok += 1;
                }
            }
            None => println!("[无脸] {}  未检测到人脸", p),
        }
    }
    if ok > 0 {
        println!("=== 完成: {ok}/{} 张有脸且关键点界内 ===", paths.len());
    } else {
        println!("=== 完成: 未检出有脸图 ===");
    }
}

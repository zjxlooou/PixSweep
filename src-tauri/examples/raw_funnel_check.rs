//! 一次性研究工具：验证生产解码漏斗 load_image_oriented 的 RAW 分支。
//! 用法：cargo run --release --example raw_funnel_check -- <RAW文件...>
use pixsweep_lib::image_io::{is_raw_image, load_image_oriented};

fn main() {
    for p in std::env::args().skip(1) {
        assert!(is_raw_image(&p), "扩展名未识别为 RAW: {p}");
        match load_image_oriented(&p) {
            Ok(img) => println!(
                "✓ {p} -> {}x{} {:?}",
                img.width(),
                img.height(),
                img.color()
            ),
            Err(e) => println!("✗ {p} -> {e:#}"),
        }
    }
}

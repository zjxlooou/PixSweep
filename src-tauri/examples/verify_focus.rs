//! 对焦指标校准：打印每张图的原始拉普拉斯方差 + 对焦分（1~10）。
//! 用法：cargo run --example verify_focus -- <图片...>
//!
//! 用于校准 `focus.rs` 的 V_MIN/V_MAX/FOCUS_OUT_THRESHOLD：对比清晰(实焦)与
//! 失焦(虚焦)图的方差分布，取能区分二者的区间。

use pixsweep_lib::ai::focus;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let paths = &args[1..];
    if paths.is_empty() {
        eprintln!("usage: verify_focus <image>...");
        std::process::exit(2);
    }
    println!("{:<26} {:>12} {:>9}", "图片", "方差", "对焦分");
    for p in paths {
        let img = match pixsweep_lib::cache::proxy::ai_proxy(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("加载失败 {}: {}", p, e);
                continue;
            }
        };
        let v = focus::focus_variance_of(&img);
        let s = focus::focus_score(&img);
        let name = Path::new(p)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        println!("{:<26} {:>12.2} {:>9.2}", name.chars().take(24).collect::<String>(), v, s);
    }
}

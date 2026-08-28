//! 一次性研究工具：验证生产扫描入口 scan_folder 对 RAW 的收录与信息提取。
//! 用法：cargo run --release --example scan_check -- <目录>
use pixsweep_lib::scanner::walker::scan_folder;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "D:/ai-code/pixsweep/.scratch/model-research/raw-samples".into());
    let infos = scan_folder(std::path::Path::new(&dir));
    println!("scan_folder 收录 {} 张:", infos.len());
    for i in &infos {
        println!(
            "  {}  {}x{}  {}  {:.1}MB",
            i.path.rsplit(['/', '\\']).next().unwrap_or("?"),
            i.width,
            i.height,
            i.format,
            i.size as f64 / 1048576.0
        );
    }
}

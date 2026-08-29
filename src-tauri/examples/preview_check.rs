//! 验证 RAW 全显影预览路径（get_full_image 的 RAW 分支核心）：
//! 传感器原生分辨率 + EXIF 转正 + 耗时。
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        anyhow::bail!("用法: preview_check <RAW路径>...");
    }
    for path in &args {
        let t0 = Instant::now();
        let img = pixsweep_lib::image_io::load_raw_developed(std::path::Path::new(path))?;
        println!(
            "{path}\n  全显影: {}x{} ({:.1}MP)  {:.2}s",
            img.width(),
            img.height(),
            img.width() as f32 * img.height() as f32 / 1_048_576.0,
            t0.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

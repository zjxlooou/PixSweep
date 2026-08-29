//! 验证 RAW 源口径分辨率探针：解码尺寸（机内嵌预览）vs 传感器原生尺寸
//! （`raw_source_dimensions`，dummy 探针不解码像素）。
//!
//! 运行：cargo run --example raw_dims_check -- <RAW 路径>...

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        anyhow::bail!("用法: raw_dims_check <RAW路径>...");
    }

    for path in &args {
        let p = std::path::Path::new(path);
        let t0 = Instant::now();
        let decoded = pixsweep_lib::image_io::load_image_oriented(p)?;
        let t_decode = t0.elapsed();

        let t0 = Instant::now();
        let src = pixsweep_lib::image_io::raw_source_dimensions(p);
        let t_probe = t0.elapsed();

        let mp = |(w, u): (u32, u32)| w as f32 * u as f32 / 1_048_576.0;
        match src {
            Some((w, h)) => println!(
                "{path}\n  解码(预览): {d}x{e} ({dm:.1}MP, {td:.2}s) | 源口径: {w}x{h} ({sm:.1}MP, 探针 {tp:.3}s)",
                d = decoded.width(),
                e = decoded.height(),
                dm = mp((decoded.width(), decoded.height())),
                td = t_decode.as_secs_f32(),
                sm = mp((w, h)),
                tp = t_probe.as_secs_f32(),
            ),
            None => println!("{path}\n  源口径探针失败（应回退解码尺寸）"),
        }
    }
    Ok(())
}

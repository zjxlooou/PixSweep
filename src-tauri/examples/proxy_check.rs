//! 验证统一前置代理规格：触发条件（RAW / >2MB / >2K）→ 代理 <2K 且 <2MB →
//! 缓存写入临时文件夹 → 二次访问命中缓存。
//!
//! 运行：cargo run --example proxy_check -- <图片路径>...

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        anyhow::bail!("用法: proxy_check <图片路径>...");
    }

    for path in &args {
        println!("== {path}");
        let size = std::fs::metadata(path)?.len();

        // 基线：原图全分辨率解码
        let t0 = Instant::now();
        let full = pixsweep_lib::image_io::load_image_oriented(std::path::Path::new(path))?;
        let t_full = t0.elapsed();
        let mp = full.width() as f32 * full.height() as f32 / 1_048_576.0;
        let src_raw =
            pixsweep_lib::image_io::is_raw_image(std::path::Path::new(path));
        println!(
            "  源图        : {}x{} ({mp:.1} MP)  {:.2}MB  raw={src_raw}  {:.2}s",
            full.width(),
            full.height(),
            size as f64 / 1_048_576.0,
            t_full.as_secs_f32()
        );

        // 第一次 ai_proxy：触发判定 → 压缩 → 写缓存
        let t0 = Instant::now();
        let p1 = pixsweep_lib::cache::proxy::ai_proxy(path)?;
        let t_gen = t0.elapsed();
        // 第二次：命中缓存
        let t0 = Instant::now();
        let p2 = pixsweep_lib::cache::proxy::ai_proxy(path)?;
        let t_hit = t0.elapsed();

        let cache = {
            let key = blake3::hash(path.as_bytes());
            pixsweep_lib::app_data_dir()
                .join("quarantine")
                .join("proxy")
                .join(format!("v3-{}.jpg", key.to_hex()))
        };
        let cache_size = std::fs::metadata(&cache).map(|m| m.len()).unwrap_or(0);
        let edge = p1.width().max(p1.height());
        println!(
            "  代理        : {}x{}  {:.2}MB  生成 {:.2}s / 命中 {:.3}s",
            p1.width(),
            p1.height(),
            cache_size as f64 / 1_048_576.0,
            t_gen.as_secs_f32(),
            t_hit.as_secs_f32()
        );
        println!(
            "  断言        : 边长<2K {} | 体积<2MB {} | 缓存存在 {}",
            edge < 2048,
            cache_size < 2 * 1024 * 1024,
            cache.exists()
        );
        let _ = p2;
    }
    Ok(())
}

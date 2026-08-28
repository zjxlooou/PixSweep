//! 一次性研究工具：多品牌 RAW 解码探针。
//! 对 raw-samples/ 下每个文件报告：解码器、嵌入预览（full/preview/thumbnail）
//! 可用性与尺寸、全显影（demosaic→sRGB）耗时与尺寸。用完即删。
//!
//! 用法：cargo run --release --example probe_raw_decode -- <样张目录>

use rayon::prelude::*;
use std::time::Instant;

use rawler::decoders::RawDecodeParams;
use rawler::imgop::develop::RawDevelop;
use rawler::rawsource::RawSource;
use rawler::get_decoder;

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "D:/ai-code/pixsweep/.scratch/model-research/raw-samples".into());
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("读样张目录")
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()),
                Some(ref s) if ["rw2","nef","arw","cr2","cr3","raf","orf","dng","raw","rwl"].contains(&s.as_str())
            )
        })
        .collect();
    files.sort();

    println!("共 {} 个 RAW 样张\n", files.len());

    let results: Vec<String> = files
        .par_iter()
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let mut line = format!("{name:40}");
            let rawfile = match RawSource::new(path) {
                Ok(f) => f,
                Err(e) => return format!("{line}  RawSource 失败: {e}"),
            };
            let decoder = match get_decoder(&rawfile) {
                Ok(d) => d,
                Err(e) => return format!("{line}  无解码器: {e}"),
            };
            let params = RawDecodeParams::default();

            // 1) 机内嵌预览（快路径）
            let full = decoder.full_image(&rawfile, &params);
            let prev = decoder.preview_image(&rawfile, &params);
            let thumb = decoder.thumbnail_image(&rawfile, &params);
            let preview_desc = match (&full, &prev, &thumb) {
                (Ok(Some(f)), _, _) => format!("full {}x{}", f.width(), f.height()),
                (_, Ok(Some(p)), _) => format!("preview {}x{}", p.width(), p.height()),
                (_, _, Ok(Some(t))) => format!("thumb {}x{}", t.width(), t.height()),
                _ => "无嵌入预览".to_string(),
            };

            // 2) 全显影（demosaic → sRGB，慢路径）
            let t0 = Instant::now();
            let dev_desc = match decoder.raw_image(&rawfile, &params, false) {
                Ok(raw) => {
                    let dims = format!("{}x{}", raw.width, raw.height);
                    match RawDevelop::default().develop_intermediate(&raw) {
                        Ok(inter) => match inter.to_dynamic_image() {
                            Some(img) => format!(
                                "显影OK {dims}→{}x{} {:.1}s",
                                img.width(),
                                img.height(),
                                t0.elapsed().as_secs_f32()
                            ),
                            None => format!("显影转换失败 {dims}"),
                        },
                        Err(e) => format!("显影失败: {e}"),
                    }
                }
                Err(e) => format!("raw_image 失败: {e}"),
            };

            line.push_str(&format!("  {preview_desc:<26} | {dev_desc}"));
            line
        })
        .collect();

    for r in &results {
        println!("{r}");
    }
}

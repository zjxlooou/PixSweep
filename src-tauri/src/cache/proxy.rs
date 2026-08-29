//! 统一前置代理图（2026-08-28 规格重做）。
//!
//! 满足**任一条件**的源图必须先生成代理图，后续全部 AI 处理（评分/人脸/闭眼/
//! 对焦/场景）都只读代理：
//! - 分辨率 > 2K（最长边 > 2048）
//! - 磁盘占用 > 2MB
//! - 相机原片格式（RAW，见 [`crate::image_io::is_raw_image`]）
//!
//! 代理图是一张**方向正确（EXIF 转正）、最长边 < 2K（1920）、体积 < 2MB** 的
//! JPEG。生成后按路径键缓存在**临时文件夹** `app_data_dir()/quarantine/proxy/`
//! （工具栏"临时文件夹"按钮显示的占用即整个隔离区目录，含代理）。
//!
//! 不满足条件的源图（小且轻的普通 JPG）直接解码使用，不落缓存。
//!
//! 精度依据：对焦整图归一到 1024、眼 ROI 归一到 24×40 再算拉普拉斯方差，
//! SCRFD 固定 letterbox 640×640，闭眼网格用的是几何比例——三者都对输入分辨率
//! 不敏感，统一代理不影响判定（2026-08-28 用 ARW/NEF/RW2 与同名 JPG 成对集实测）。
//!
//! 代理图是纯缓存，删掉会按原图重建；清空临时文件夹不影响源图。

use std::path::PathBuf;
use std::sync::Once;

/// 触发阈值：最长边超过此值视为"分辨率 > 2K"。
pub const SRC_MAX_EDGE: u32 = 2048;
/// 触发阈值：源文件超过此字节数（2MiB）视为"磁盘占用 > 2MB"。
pub const SRC_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// 代理输出上限：最长边 < 2K（取 1920）。
pub const PROXY_MAX_EDGE: u32 = 1920;
/// 代理输出上限：体积 < 2MB（取 2MiB）。
pub const PROXY_MAX_BYTES: u64 = 2 * 1024 * 1024;
/// 代理图缓存版本。触发/编码逻辑变化时递增，避免命中旧逻辑生成的缓存。
const PROXY_VERSION: &str = "v3";
/// JPEG 质量阶梯：从高到低取第一个满足 <2MB 的档位。
const QUALITY_LADDER: [u8; 5] = [95, 88, 80, 70, 60];
/// 全部档位仍超限时再降到此边长重试一轮。
const FALLBACK_EDGE: u32 = 1280;

/// 代理图缓存目录：临时文件夹（隔离区）下的 `proxy/` 子目录。
fn proxy_dir() -> PathBuf {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // 旧版代理缓存在程序根 proxy/，一次性迁出（纯缓存，直接删，按需重建）
        let legacy = crate::app_data_dir().join("proxy");
        if legacy.is_dir() {
            let _ = std::fs::remove_dir_all(&legacy);
        }
    });
    let dir = crate::app_data_dir().join("quarantine").join("proxy");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 按路径算出代理图缓存路径。路径键用 blake3(path)，避免文件系统非法字符。
fn proxy_cache_path(path: &str) -> PathBuf {
    let key = blake3::hash(path.as_bytes());
    proxy_dir().join(format!("{}-{}.jpg", PROXY_VERSION, key.to_hex()))
}

/// 该源图是否需要生成代理图（满足任一触发条件）。
pub fn needs_proxy(path: &str, max_edge: u32) -> bool {
    if crate::image_io::is_raw_image(std::path::Path::new(path)) {
        return true;
    }
    if std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) > SRC_MAX_BYTES {
        return true;
    }
    max_edge > SRC_MAX_EDGE
}

/// 读取/生成代理图（<2K 且 <2MB、方向正确），返回 `RgbImage`。
///
/// 首次访问：`load_image_oriented`（EXIF 转正）→ 命中触发条件则压缩到
/// 最长边 ≤ [`PROXY_MAX_EDGE`]、JPEG 编码 < [`PROXY_MAX_BYTES`] 并写缓存。
/// 之后访问：直接解码缓存的小 JPG（快得多）。未触发条件的图直接解码返回。
pub fn ai_proxy(path: &str) -> anyhow::Result<image::RgbImage> {
    let cache = proxy_cache_path(path);
    if cache.exists() {
        if let Ok(img) = image::open(&cache) {
            return Ok(img.to_rgb8());
        }
    }

    let oriented = crate::image_io::load_image_oriented(std::path::Path::new(path))
        .map_err(|e| anyhow::anyhow!("无法解码图片 {}: {}", path, e))?;
    let rgb = oriented.to_rgb8();
    if !needs_proxy(path, rgb.width().max(rgb.height())) {
        return Ok(rgb);
    }

    let (proxy, jpeg) = encode_proxy(&rgb);
    let _ = std::fs::write(&cache, &jpeg);
    Ok(proxy)
}

/// 把源图压成满足 <2K 且 <2MB 的代理：先缩边到 [`PROXY_MAX_EDGE`]，
/// 按质量阶梯编码；全部超限再降边到 [`FALLBACK_EDGE`] 重试；保底取最小档。
/// 返回 (代理像素, 代理 JPEG 字节)。
fn encode_proxy(src: &image::RgbImage) -> (image::RgbImage, Vec<u8>) {
    let downscaled = downscale(src, PROXY_MAX_EDGE);
    if let Some(jpeg) = encode_under_limit(&downscaled) {
        return (downscaled, jpeg);
    }
    let further = downscale(&downscaled, FALLBACK_EDGE);
    if let Some(jpeg) = encode_under_limit(&further) {
        return (further, jpeg);
    }
    // 保底：最小质量档（理论上极难到达）
    let jpeg = encode_jpeg(&further, *QUALITY_LADDER.last().unwrap());
    (further, jpeg)
}

/// 最长边超过 `max_edge` 则等比缩小（Triangle），否则原样返回。
fn downscale(img: &image::RgbImage, max_edge: u32) -> image::RgbImage {
    let longest = img.width().max(img.height());
    if longest <= max_edge {
        return img.clone();
    }
    let scale = max_edge as f64 / longest as f64;
    let nw = ((img.width() as f64 * scale).round() as u32).max(1);
    let nh = ((img.height() as f64 * scale).round() as u32).max(1);
    image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle)
}

/// 按质量阶梯返回第一个 < [`PROXY_MAX_BYTES`] 的 JPEG；全超限返回 None。
fn encode_under_limit(img: &image::RgbImage) -> Option<Vec<u8>> {
    for &q in &QUALITY_LADDER {
        let jpeg = encode_jpeg(img, q);
        if (jpeg.len() as u64) < PROXY_MAX_BYTES {
            return Some(jpeg);
        }
    }
    None
}

fn encode_jpeg(img: &image::RgbImage, quality: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality);
    let _ = img.write_with_encoder(enc);
    out
}

/// 代理图缓存占用字节数（供"清理缓存"面板显示）。
pub fn proxy_cache_bytes() -> u64 {
    cache_dir_bytes(&proxy_dir())
}

/// 清理代理图缓存（放入系统回收站，由调用方 `fileops::trash` 处理）。
/// 返回清理的文件路径列表，供逐个移入系统回收站。
pub fn proxy_cache_files() -> Vec<std::path::PathBuf> {
    cache_dir_files(&proxy_dir())
}

// ---- 缓存目录小工具 ----

/// 目录内文件总字节数。
fn cache_dir_bytes(dir: &PathBuf) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// 目录内全部文件路径。
fn cache_dir_files(dir: &PathBuf) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries.flatten().map(|e| e.path()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_output_limits() {
        assert_eq!(PROXY_MAX_EDGE, 1920); // < 2K
        assert_eq!(PROXY_MAX_BYTES, 2 * 1024 * 1024); // < 2MB
        assert!(PROXY_MAX_EDGE <= SRC_MAX_EDGE);
    }

    #[test]
    fn downscale_only_shrinks() {
        let big = image::RgbImage::from_raw(4000, 3000, vec![128u8; 4000 * 3000 * 3]).unwrap();
        let d = downscale(&big, PROXY_MAX_EDGE);
        assert_eq!((d.width(), d.height()), (1920, 1440));
        let small = image::RgbImage::new(800, 600);
        let s = downscale(&small, PROXY_MAX_EDGE);
        assert_eq!((s.width(), s.height()), (800, 600));
    }

    #[test]
    fn encode_proxy_meets_both_limits() {
        // 4000×3000 噪声图是压缩最坏的用例：验证输出仍满足 <2K 且 <2MB
        let mut img = image::RgbImage::new(4000, 3000);
        let mut x: u32 = 12345;
        for px in img.pixels_mut() {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            *px = image::Rgb([ (x & 0xff) as u8, (x >> 8 & 0xff) as u8, (x >> 16 & 0xff) as u8 ]);
        }
        let (proxy, jpeg) = encode_proxy(&img);
        assert!(proxy.width().max(proxy.height()) <= PROXY_MAX_EDGE);
        assert!((jpeg.len() as u64) < PROXY_MAX_BYTES, "jpeg = {} bytes", jpeg.len());
    }

    #[test]
    fn proxy_cache_path_is_stable_and_sanitized() {
        let a = proxy_cache_path("C:/x/abc def.jpg");
        let b = proxy_cache_path("E:/other/photo.jpg");
        // 不同路径 → 不同键；同一路径 → 相同键；文件名不含非法字符
        assert_ne!(a, b);
        assert_eq!(proxy_cache_path("C:/x/abc def.jpg"), a);
        assert!(!a.to_string_lossy().contains(" "));
        assert!(a.extension().is_some());
    }
}

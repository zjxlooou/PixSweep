//! AI 推理代理图缓存。
//!
//! 生成一张**低分辨率且方向正确**（EXIF 转正）的代理图，供整图模型共用，
//! 避免对 6000×4000 大图逐模型反复全分辨率解码。代理图按**路径键**缓存到
//! 程序根数据目录 `app_data_dir()/proxy/`，可像缩略图一样 LRU 驱逐。
//!
//! ## 约定（两级）
//! - **走代理**：整图模型（场景分类 / CLIP / TOPIQ-NR & IAA / NIMA / 整图对焦）。
//!   它们都会 resize 到 224/384，对输入分辨率不敏感，代理 2048 结果稳定
//!   （实测技术/美学分几乎不变）。
//! - **走原图全分辨率**：人脸检测 / 闭眼 / 眼下对焦。这些依赖眼部细节与关键点
//!   精度，小脸（如 8.8% 画面占比）降到 2048 实测仍使 OCEC 由 0.93 掉到 0.09
//!   （破坏 A1 闭眼/眼对焦），故人脸/眼路径不可用代理。
//!
//! 代理图是纯性能/规范性缓存，删掉会按原图重建；清理进系统回收站不影响源图。

use std::path::PathBuf;

/// 代理图目标最长边（像素）。
///
/// 权衡：越低解码越快，但小脸/眼部细节越糊（实测 1024 会让 8.8% 画面占比的
/// 脸眼部从 OCEC 0.93 掉到 0.01，破坏 A1 的闭眼/眼部对焦）。取 2048 在
/// "省解码"与"保头部细节"间取得平衡：24MP → 约 2.8MP（约 8.6× 少像素）。
pub const PROXY_MAX: u32 = 2048;
/// 代理图缓存版本。图像解码/方向/缩放逻辑变化时递增，避免命中旧逻辑生成的缓存。
const PROXY_VERSION: &str = "v2";
/// JPEG 编码质量。对焦/失焦阈值按此码率校准。
const PROXY_JPEG_QUALITY: u8 = 95;

/// 代理图缓存目录（程序根数据目录下，与缩略图同归位）。
fn proxy_dir() -> PathBuf {
    let dir = crate::app_data_dir().join("proxy");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 按路径算出代理图缓存路径。路径键用 blake3(path)，避免文件系统非法字符。
fn proxy_cache_path(path: &str) -> PathBuf {
    let key = blake3::hash(path.as_bytes());
    proxy_dir().join(format!("{}-{}.jpg", PROXY_VERSION, key.to_hex()))
}

/// 读取/生成代理图（低分辨率、方向正确），返回 `RgbImage`。
///
/// 首次访问：`load_image_oriented`（EXIF 转正）→ 若最长边 > [`PROXY_MAX`] 则降采样 →
/// 编码 JPEG 写缓存。之后访问：直接解码缓存的低分辨代理图（快得多）。
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
    let proxy = if rgb.width().max(rgb.height()) > PROXY_MAX {
        let scale = PROXY_MAX as f64 / rgb.width().max(rgb.height()) as f64;
        let nw = ((rgb.width() as f64 * scale).round() as u32).max(1);
        let nh = ((rgb.height() as f64 * scale).round() as u32).max(1);
        image::imageops::resize(&rgb, nw, nh, image::imageops::FilterType::Triangle)
    } else {
        rgb
    };

    let _ = write_jpeg(&cache, &proxy);
    Ok(proxy)
}

/// 将代理图编码为 JPEG 并写入缓存。
fn write_jpeg(path: &PathBuf, img: &image::RgbImage) -> anyhow::Result<()> {
    let mut out = Vec::new();
    let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, PROXY_JPEG_QUALITY);
    img.write_with_encoder(enc)?;
    Ok(std::fs::write(path, &out)?)
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
    fn proxy_max_is_2048() {
        assert_eq!(PROXY_MAX, 2048);
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

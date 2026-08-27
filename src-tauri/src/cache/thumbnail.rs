//! 缩略图生成与磁盘缓存。
//!
//! 生成 256px 的 JPEG 缩略图，以文件指纹为缓存键存储到磁盘，
//! 前端通过 `get_thumbnail` 命令按需获取（返回 base64 data URL）。

use std::path::{Path, PathBuf};

/// 缩略图目标边长（像素）。
pub const THUMBNAIL_SIZE: u32 = 256;

/// 缩略图缓存版本。图像解码/旋转逻辑变化（如 EXIF 修复）时递增，
/// 避免命中用旧逻辑生成的缓存（旧竖图未旋转）。
const THUMB_CACHE_VERSION: &str = "v2";

/// 缩略图缓存目录。
fn thumbnail_dir() -> PathBuf {
    let dir = crate::app_data_dir().join("thumbnails");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// 获取缩略图缓存路径。
fn cache_path(file_hash: &str) -> PathBuf {
    thumbnail_dir().join(format!("{}-{}.jpg", THUMB_CACHE_VERSION, file_hash))
}

/// 生成缩略图：读取原图 → 缩放 → 编码 JPEG → 缓存 → 返回字节。
pub fn generate_thumbnail(path: &str, file_hash: &str) -> anyhow::Result<Vec<u8>> {
    // 命中缓存则直接返回
    let cached = cache_path(file_hash);
    if cached.exists() {
        if let Ok(bytes) = std::fs::read(&cached) {
            return Ok(bytes);
        }
    }

    let img = crate::image_io::load_image_oriented(Path::new(path))
        .map_err(|e| anyhow::anyhow!("无法解码图片 {}: {}", path, e))?;

    let thumbnail = img.thumbnail(THUMBNAIL_SIZE, THUMBNAIL_SIZE);

    // 编码为 JPEG
    let mut out = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 80);
    thumbnail.write_with_encoder(encoder)?;

    // 写入缓存
    let _ = std::fs::write(&cached, &out);

    Ok(out)
}

/// 将缩略图字节编码为 base64 data URL，便于前端直接渲染。
pub fn to_data_url(bytes: &[u8]) -> String {
    format!("data:image/jpeg;base64,{}", base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes))
}

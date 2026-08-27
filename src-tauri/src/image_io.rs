//! 统一的图片加载与 EXIF Orientation 处理。
//!
//! 之前 `image::open()` 不读取 EXIF Orientation，导致手机/相机拍摄的"竖图"被按 EXIF 的
//! 横向像素方向展示——和图片查看软件（系统相册/Photos）显示效果不一致。
//!
//! 本模块用 `ImageReader` 识别格式 → `into_decoder()` 拿到底层 decoder，调
//! decoder 的 `orientation()`（JPEG/PNG/WebP 各自解析 Exif/APP1/eXIf 段）→
//! `decode()` 后 `apply_orientation()` 旋转到位。
//!
//! > 坑：不要用 `image::metadata::Orientation::from_exif_chunk(文件头)` ——该函数
//! > 期望 chunk 从 TIFF header（`49 49 2A 00`）开始，而 JPEG 文件头以 `FF D8`(SOI)
//! > 与各 APP marker 开头，直接传整个文件头必然解析失败，EXIF 旋转永远不生效。
//!
//! 用法：
//! ```ignore
//! let img = load_image_oriented("path/to/photo.jpg")?;
//! // img 现在是按用户预期方向显示的 RGB8 DynamicImage
//! ```

use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader};

/// 读取图片并自动应用 EXIF Orientation，返回按用户预期方向显示的 DynamicImage。
///
/// 流程：识别格式 → into_decoder() 取 orientation → decode → apply。
/// 任何格式转换错误都向上传播。
pub fn load_image_oriented<P: AsRef<Path>>(path: P) -> anyhow::Result<DynamicImage> {
    let path_ref = path.as_ref();

    let reader = ImageReader::open(path_ref)
        .map_err(|e| anyhow::anyhow!("打开图片失败 {}: {}", path_ref.display(), e))?
        .with_guessed_format()
        .map_err(|e| anyhow::anyhow!("识别图片格式失败 {}: {}", path_ref.display(), e))?;

    // decoder 内部按各格式解析 EXIF(JPEG APP1 / PNG eXIf / WebP EXIF box)并暴露 orientation
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| anyhow::anyhow!("获取解码器失败 {}: {}", path_ref.display(), e))?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);

    let mut img = DynamicImage::from_decoder(decoder)
        .map_err(|e| anyhow::anyhow!("解码失败 {}: {}", path_ref.display(), e))?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// 仅读取图片的 EXIF Orientation（不解码像素）。
///
/// 各格式 decoder 内部正确解析（JPEG APP1 / PNG eXIf / WebP EXIF box），
/// 无方向标签或格式不支持时返回 `NoTransforms`。供诊断/测试复用。
pub fn exif_orientation<P: AsRef<Path>>(path: P) -> anyhow::Result<image::metadata::Orientation> {
    let path_ref = path.as_ref();
    let reader = ImageReader::open(path_ref)
        .map_err(|e| anyhow::anyhow!("打开图片失败 {}: {}", path_ref.display(), e))?
        .with_guessed_format()
        .map_err(|e| anyhow::anyhow!("识别图片格式失败 {}: {}", path_ref.display(), e))?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| anyhow::anyhow!("获取解码器失败 {}: {}", path_ref.display(), e))?;
    Ok(decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms))
}
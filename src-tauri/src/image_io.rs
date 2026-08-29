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

/// 支持的相机 RAW 扩展名（小写）。覆盖主流品牌原生态格式 + DNG 通用容器。
pub const RAW_EXTENSIONS: &[&str] = &[
    "rw2",             // Panasonic
    "nef", "nrw",      // Nikon
    "arw", "srw",      // Sony
    "cr2", "cr3", "crw", // Canon
    "raf",             // Fujifilm
    "orf",             // Olympus
    "pef", "ptx",      // Pentax
    "dng",             // Adobe/DNG 通用容器（Leica/大疆等）
    "raw", "rwl",      // Leica / Panasonic 旧款
    "x3f",             // Sigma Foveon
    "3fr",             // Hasselblad
    "erf",             // Epson
    "mrw",             // Minolta
    "iiq",             // Phase One
    "gpr", "kdc", "dcr", // GoPro / Kodak
];

/// 判断路径是否为相机 RAW 文件（按扩展名）。
pub fn is_raw_image<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref()
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| RAW_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// 读取图片并自动应用 EXIF Orientation，返回按用户预期方向显示的 DynamicImage。
///
/// 流程：识别格式 → into_decoder() 取 orientation → decode → apply。
/// 任何格式转换错误都向上传播。
///
/// **RAW 分支**（2026-08-27，rawler 0.7）：扩展名命中 [`is_raw_image`] 时改走
/// rawler 解码——优先取机内嵌预览（`full_image` > `preview_image` > `thumbnail_image`，
/// 相机端已完成去马赛克/白平衡/降噪，毫秒级），全无嵌入预览时回退全显影
/// （demosaic → sRGB，秒级）。两种路径都按 RAW 内 EXIF orientation 旋转。
pub fn load_image_oriented<P: AsRef<Path>>(path: P) -> anyhow::Result<DynamicImage> {
    let path_ref = path.as_ref();
    if is_raw_image(path_ref) {
        return load_raw_oriented(path_ref);
    }

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
/// RAW 的传感器原生分辨率（源口径），不解码像素。
///
/// 用 rawler `raw_image(dummy=true)` 探针模式：只解析容器与尺寸、不分配不解码
/// 像素，毫秒级。尺寸优先取 `crop_area`（相机标称有效像素）→ `active_area` →
/// 全幅读出区。返回**EXIF 方向转正后**的宽高（与显示/解码口径一致）。
/// 解析失败返回 None，调用方应回退到解码尺寸。
///
/// 用途：RAW 的机内嵌预览往往远小于传感器（如 Sony 1080p 预览 vs 24MP 传感器），
/// 分辨率启发式等**尺寸比较必须用源口径**，否则 RAW 在与同画面 JPG 对比时被
/// 预览尺寸低估（2026-08-28 用户约定）。
pub fn raw_source_dimensions(path: &Path) -> Option<(u32, u32)> {
    use rawler::decoders::RawDecodeParams;
    use rawler::get_decoder;
    use rawler::rawsource::RawSource;

    let rawfile = RawSource::new(path).ok()?;
    let decoder = get_decoder(&rawfile).ok()?;
    let params = RawDecodeParams::default();
    let exif_u16 = decoder
        .raw_metadata(&rawfile, &params)
        .ok()
        .and_then(|md| md.exif.orientation);
    let raw = decoder.raw_image(&rawfile, &params, true).ok()?;

    let (mut w, mut h) = raw
        .crop_area
        .or(raw.active_area)
        .map(|rect| (rect.d.w, rect.d.h))
        .unwrap_or((raw.width, raw.height));
    if w == 0 || h == 0 {
        return None;
    }
    // 竖拍（EXIF 旋转 90/270 族）宽高互换，保持与转正后的解码结果同口径
    let rotated = exif_u16
        .and_then(|v| u8::try_from(v).ok())
        .and_then(image::metadata::Orientation::from_exif)
        .map(|o| {
            matches!(
                o,
                image::metadata::Orientation::Rotate90
                    | image::metadata::Orientation::Rotate270
                    | image::metadata::Orientation::Rotate90FlipH
                    | image::metadata::Orientation::Rotate270FlipH
            )
        })
        .unwrap_or(false);
    if rotated {
        std::mem::swap(&mut w, &mut h);
    }
    Some((u32::try_from(w).ok()?, u32::try_from(h).ok()?))
}

/// RAW 解码：机内嵌预览优先（毫秒级），全无嵌入预览时回退全显影（秒级）。
///
/// 两条路径都不自带旋转，统一按 RAW 内 EXIF orientation 手动应用
/// （预览走 `raw_metadata.exif.orientation`，显影走 `raw.orientation`）。
fn load_raw_oriented(path: &Path) -> anyhow::Result<DynamicImage> {
    use rawler::decoders::RawDecodeParams;
    use rawler::imgop::develop::RawDevelop;
    use rawler::rawsource::RawSource;
    use rawler::get_decoder;

    let rawfile = RawSource::new(path)
        .map_err(|e| anyhow::anyhow!("RAW 打开失败 {}: {e}", path.display()))?;
    let decoder = get_decoder(&rawfile)
        .map_err(|e| anyhow::anyhow!("RAW 无可用解码器 {}: {e}", path.display()))?;
    let params = RawDecodeParams::default();

    // EXIF 方向（元数据提取失败按无旋转处理）
    let exif_u16 = decoder
        .raw_metadata(&rawfile, &params)
        .ok()
        .and_then(|md| md.exif.orientation);
    let apply = |img: DynamicImage| -> DynamicImage {
        let mut img = img;
        if let Some(o) = exif_u16.and_then(|v| u8::try_from(v).ok()).and_then(image::metadata::Orientation::from_exif) {
            img.apply_orientation(o);
        }
        img
    };

    // 快路径：机内嵌预览（full > preview > thumbnail，取到即用）
    for attempt in [
        decoder.full_image(&rawfile, &params),
        decoder.preview_image(&rawfile, &params),
        decoder.thumbnail_image(&rawfile, &params),
    ] {
        if let Ok(Some(img)) = attempt {
            return Ok(apply(img));
        }
    }

    // 慢路径：全显影（demosaic → 白平衡 → 色彩转换 → sRGB）
    log::info!("[RAW] 无机内嵌预览，走全显影: {}", path.display());
    let raw = decoder.raw_image(&rawfile, &params, false)?;
    let orientation = exif_u16.or_else(|| Some(raw.orientation.to_u16()));
    let img = RawDevelop::default()
        .develop_intermediate(&raw)?
        .to_dynamic_image()
        .ok_or_else(|| anyhow::anyhow!("RAW 显影输出为空: {}", path.display()))?;
    let mut img = img;
    if let Some(o) = orientation.and_then(|v| u8::try_from(v).ok()).and_then(image::metadata::Orientation::from_exif) {
        img.apply_orientation(o);
    }
    Ok(img)
}


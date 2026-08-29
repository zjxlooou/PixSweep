//! TOPIQ-NR-Face 人脸技术质量评分（专用版）。
//!
//! 与 `topiq.rs`（TOPIQ-NR 通用技术质量）共用 CFANet 架构，但权重不同：
//! - **TOPIQ-NR**：KonIQ-10k 训练，通用画质（模糊/噪声/曝光）
//! - **TOPIQ-NR-Face**：CGFIQA-40k 训练，**人脸区域画质**，对虚焦敏感
//!
//! ## 使用流程
//! 1. **前置**：调用 InsightFace buffalo_l 做 5 关键点对齐（仿射 warp），得到 512×512 人脸 crop
//! 2. **本模块**：`face_quality_scores(session, &crops)` → 0~1 MOS 分数列表
//! 3. **业务侧**：`1 + clip(score, 0, 1) * 9` 映射到 1~10
//!
//! ## 推理后端
//! 复用 `engine::build_session` 的三级回退链（CUDA → DirectML → CPU），无需重复。

use anyhow::Context;
use ndarray::Array4;
use ort::inputs;
use ort::session::Session;

/// TOPIQ-NR-Face 模型文件名（ResNet50，CGFIQA-40k，输出 0~1，输入 512×512）。
pub const MODEL_NAME: &str = "topiq_nr_face.onnx";
/// TOPIQ-NR-Face 输入张量名称。
pub const INPUT_NAME: &str = "face_crop";
/// TOPIQ-NR-Face 输出张量名称。
pub const OUTPUT_NAME: &str = "quality";
/// TOPIQ-NR-Face 输入尺寸（宽高），导出时固定。
pub const INPUT_SIZE: u32 = 512;

/// 将任意尺寸的 RGB 人脸 crop 缩放到 512×512 并做 ImageNet 归一化。
///
/// 输入 `crop_rgb`：HWC 格式 RGB `u8` 数据，长度必须为 `side*side*3`。
/// 返回 `[1, 3, 512, 512]` `f32` 张量（已 HWC→CHW + ImageNet mean/std 归一化）。
fn preprocess_face_crop(crop_rgb: &[u8], side: u32) -> Array4<f32> {
    debug_assert_eq!(crop_rgb.len(), (side * side * 3) as usize);
    let mut img = image::RgbImage::new(side, side);
    img.copy_from_slice(crop_rgb);
    // resize 到 512×512（Triangle 滤波器）
    let resized = image::imageops::resize(
        &img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let mut arr = ndarray::Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = resized.get_pixel(x, y);
            for c in 0..3 {
                arr[[0, c, y as usize, x as usize]] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    arr
}

/// 对多张人脸 crop 推理。
///
/// `face_crops`：长度 N 的 `Vec`，每项是 `(crop_rgb, side)`：
/// - `crop_rgb`：已对齐到正方形的 RGB u8 数据（side × side × 3）
/// - `side`：正方形边长（任意尺寸，内部 resize 到 512）
///
/// 返回：长度 N 的 `Vec<f32>`，每项 ∈ [0, 1]（CGFIQA MOS）。
///
/// 模型有两种形态：
/// - **动态 batch 导出**（2026-08-29，pyiqa 语义重导出）：8 张一次前向；
/// - 旧 fix batch=1：批量输入报错，自动回退逐张。
pub fn face_quality_scores(
    session: &mut Session,
    face_crops: &[(Vec<u8>, u32)],
) -> anyhow::Result<Vec<f32>> {
    if face_crops.is_empty() {
        return Ok(Vec::new());
    }
    const BATCH: usize = 8;
    let mut scores = Vec::with_capacity(face_crops.len());
    for chunk in face_crops.chunks(BATCH) {
        match face_quality_scores_batch(session, chunk) {
            Ok(vals) => scores.extend(vals),
            Err(_) => {
                // 旧 fix batch=1 模型：批量输入失败，回退逐张
                for (rgb, side) in chunk {
                    scores.push(face_quality_score_single(session, rgb, *side)?);
                }
            }
        }
    }
    Ok(scores)
}

/// 批量前向：[N,3,512,512] 一次推理。
fn face_quality_scores_batch(
    session: &mut Session,
    face_crops: &[(Vec<u8>, u32)],
) -> anyhow::Result<Vec<f32>> {
    let n = face_crops.len();
    let mut batch = Array4::<f32>::zeros((n, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for (i, (rgb, side)) in face_crops.iter().enumerate() {
        batch.slice_mut(ndarray::s![i, .., .., ..])
            .assign(&preprocess_face_crop(rgb, *side).slice(ndarray::s![0, .., .., ..]));
    }
    let tensor = ort::value::Tensor::from_array(batch)
        .context("构造 TOPIQ-NR-Face 批量输入张量失败")?;
    let outputs = session
        .run(inputs![INPUT_NAME => tensor])
        .context("TOPIQ-NR-Face 批量推理失败")?;
    let arr = outputs[OUTPUT_NAME]
        .try_extract_array::<f32>()
        .context("解析 TOPIQ-NR-Face 输出失败")?;
    let flat = arr.as_slice().context("TOPIQ-NR-Face 输出非连续")?;
    anyhow::ensure!(flat.len() >= n, "TOPIQ-NR-Face 输出数量不足");
    Ok(flat[..n].to_vec())
}

/// 单张前向（旧 fix batch=1 模型的回退路径）。
fn face_quality_score_single(session: &mut Session, rgb: &[u8], side: u32) -> anyhow::Result<f32> {
    let single = preprocess_face_crop(rgb, side);
    let tensor =
        ort::value::Tensor::from_array(single).context("构造 TOPIQ-NR-Face 输入张量失败")?;
    let outputs = session
        .run(inputs![INPUT_NAME => tensor])
        .context("TOPIQ-NR-Face 推理失败")?;
    let arr = outputs[OUTPUT_NAME]
        .try_extract_array::<f32>()
        .context("解析 TOPIQ-NR-Face 输出失败")?;
    arr.as_slice()
        .context("TOPIQ-NR-Face 输出非连续")?
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("TOPIQ-NR-Face 输出为空"))
}

/// 将 [0, 1] MOS 映射到 [1, 10]（与 TOPIQ-NR 统一）。
pub fn map_to_ten_scale(score: f32) -> f32 {
    1.0 + score.clamp(0.0, 1.0) * 9.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_maps_zero_to_one() {
        assert!((map_to_ten_scale(0.0) - 1.0).abs() < 1e-5);
        assert!((map_to_ten_scale(1.0) - 10.0).abs() < 1e-5);
        assert!((map_to_ten_scale(0.5) - 5.5).abs() < 1e-5);
    }

    #[test]
    fn face_clamps_out_of_range() {
        assert!((map_to_ten_scale(-1.0) - 1.0).abs() < 1e-5);
        assert!((map_to_ten_scale(2.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn constants_match_exported_onnx() {
        // 防止导出器/输入名改动后忘了同步这里
        assert_eq!(MODEL_NAME, "topiq_nr_face.onnx");
        assert_eq!(INPUT_NAME, "face_crop");
        assert_eq!(OUTPUT_NAME, "quality");
        assert_eq!(INPUT_SIZE, 512);
    }
}
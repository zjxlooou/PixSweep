//! 图像预处理：将图片解码并转换为模型输入 tensor。

use ndarray::{Array, Ix4};
use rayon::prelude::*;

/// NIMA 模型的标准输入尺寸。
pub const INPUT_SIZE: u32 = 224;


/// ImageNet 归一化均值（RGB）——NIMA / MobileNet / TOPIQ 使用。
pub const MEAN_MOBILENET: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet 归一化标准差（RGB）——NIMA / MobileNet / TOPIQ 使用。
pub const STD_MOBILENET: [f32; 3] = [0.229, 0.224, 0.225];

/// TOPIQ 模型输入尺寸（384×384，TOPIQ 官方训练尺寸）。
pub const TOPIQ_INPUT_SIZE: u32 = 384;

/// 单张图解码 → 居中裁剪 → resize → 归一化，产出 `[1, C, H, W]`（`chan_first`）或
/// `[1, H, W, C]`（NHWC）tensor。
///
/// 每张图独立计算，输出内容只依赖输入图本身 → 并行/串行执行结果逐位一致。
fn image_to_tensor_layout(
    path: &str,
    size: u32,
    chan_first: bool,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> anyhow::Result<Array<f32, Ix4>> {
    let rgb = crate::cache::proxy::ai_proxy(path)?;

    // 居中裁剪为正方形，避免拉伸变形
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let x = (w - side) / 2;
    let y = (h - side) / 2;
    let cropped = image::imageops::crop_imm(&rgb, x, y, side, side).to_image();

    // resize 到目标尺寸
    let resized = image::imageops::resize(
        &cropped,
        size,
        size,
        image::imageops::FilterType::Triangle,
    );

    let s = size as usize;
    let mut t = if chan_first {
        Array::<f32, Ix4>::zeros((1, 3, s, s))
    } else {
        Array::<f32, Ix4>::zeros((1, s, s, 3))
    };
    for (x, y, px) in resized.enumerate_pixels() {
        let (r, g, b) = (px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0);
        let (yu, xv) = (y as usize, x as usize);
        if chan_first {
            t[[0, 0, yu, xv]] = (r - mean[0]) / std[0];
            t[[0, 1, yu, xv]] = (g - mean[1]) / std[1];
            t[[0, 2, yu, xv]] = (b - mean[2]) / std[2];
        } else {
            t[[0, yu, xv, 0]] = (r - mean[0]) / std[0];
            t[[0, yu, xv, 1]] = (g - mean[1]) / std[1];
            t[[0, yu, xv, 2]] = (b - mean[2]) / std[2];
        }
    }

    Ok(t)
}


/// 批量预处理：每张图独立解码/裁剪/resize/归一化，rayon 并行跑（解码是 CPU 瓶颈，
/// 并行后 GPU 等数据的时间显著缩短），再按原顺序组装 batch tensor。
///
/// 单张图的像素计算与串行完全一致，rayon collect 保序 → 并行只改调度顺序，不改分数确定性。
fn images_to_batch_layout(
    paths: &[String],
    size: u32,
    chan_first: bool,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> anyhow::Result<Array<f32, Ix4>> {
    let n = paths.len();
    let s = size as usize;
    let mut batch = if chan_first {
        Array::<f32, Ix4>::zeros((n, 3, s, s))
    } else {
        Array::<f32, Ix4>::zeros((n, s, s, 3))
    };

    let tensors: anyhow::Result<Vec<_>> = paths
        .par_iter()
        .map(|p| image_to_tensor_layout(p, size, chan_first, mean, std))
        .collect();

    for (i, t) in tensors?.into_iter().enumerate() {
        batch
            .slice_mut(ndarray::s![i, .., .., ..])
            .assign(&t.slice(ndarray::s![0, .., .., ..]));
    }

    Ok(batch)
}


/// 将多张图片预处理为 NIMA（MobileNet）所需的 `[N, 224, 224, 3]` NHWC 批量 tensor。
///
/// NIMA 输入是 NHWC 布局，使用 MobileNet 的归一化参数。
pub fn images_to_batch_nima(paths: &[String]) -> anyhow::Result<Array<f32, Ix4>> {
    images_to_batch_layout(paths, INPUT_SIZE, false, &MEAN_MOBILENET, &STD_MOBILENET)
}

/// 将多张图片预处理为 TOPIQ 所需的 `[N, 3, 384, 384]` CHW 批量 tensor。
///
/// TOPIQ 使用 CHW 布局 + ImageNet 归一化（与 NIMA 相同的 mean/std，但尺寸 384×384）。
pub fn images_to_batch_topiq(paths: &[String]) -> anyhow::Result<Array<f32, Ix4>> {
    images_to_batch_layout(paths, TOPIQ_INPUT_SIZE, true, &MEAN_MOBILENET, &STD_MOBILENET)
}

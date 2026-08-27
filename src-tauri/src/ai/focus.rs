//! 对焦/清晰度指标（模型无关）：灰度图 Laplacian 方差。
//!
//! 高通能量越高 → 越在焦/锐利；越低 → 越失焦/模糊。用于：
//! - **整图对焦**（风景/其他/宠物回退）：`focus_score`。
//! - **眼部对焦**（人像/宠物）：`eye_focus_score`（对眼部 ROI 计算）。
//!
//! ⚠️ 分辨率影响方差的绝对值（越大越糊），故计算前把图归一到 [`FOCUS_REF`] 最长边，
//! 保证跨图可比。人脸/眼路径用**原图**（见 proxy.rs 两级约定），整图对焦用代理图。

/// 对焦参考边长。计算前图归一到该最长边，消除分辨率差异，保证跨图可比。
pub const FOCUS_REF: u32 = 1024;
/// 拉普拉斯方差 → 对焦分（1~10）线性映射：方差 ≤ `V_MIN` → 1 分；≥ `V_MAX` → 10 分。
///
/// 校准自 `Desktop/test1`（实焦 148 / 虚焦 137）与真实人像（宋宇芳 202 / 邹存 180）：
/// 真实照片方差约 130~210，映射到 5~8 分（明确在焦）；低纹理/极小图方差 < 40 落 1 分。
/// ⚠️ 该指标对整图内容敏感（噪声/高纹理拉满、纯色/平滑压低），跨场景区分度有限：
/// 作为"是否明显失焦"的保守信号使用，实焦/虚焦这类细差由综合分其他轴衡量。
const V_MIN: f32 = 40.0;
const V_MAX: f32 = 260.0;
/// "失焦"阈值：对焦分低于此值 → 标失焦（只标明显低清晰度，避免误伤轻微虚焦）。
pub const FOCUS_OUT_THRESHOLD: f32 = 4.0;

/// 整图对焦分（1.0~10.0）。输入归一到 [`FOCUS_REF`] 后算灰度拉普拉斯方差再映射。
pub fn focus_score(img: &image::RgbImage) -> f32 {
    variance_to_score(focus_variance_of(img))
}

/// 对焦分 → 是否失焦。
pub fn is_out_of_focus(score: f32) -> bool {
    score < FOCUS_OUT_THRESHOLD
}

/// 眼睛 ROI 对焦分（左、右眼各 24×40×3 RGB 字节），取两只眼中更清晰者。
/// 用于人像/宠物"眼部对焦"，避免任一单眼 ROI 轻微偏位拉低整脸。
pub fn eye_focus_score(lroi: &[u8], rroi: &[u8]) -> f32 {
    let l = variance_to_score(laplacian_variance_of(EYE_W as usize, EYE_H as usize, &gray24x40(lroi)));
    let r = variance_to_score(laplacian_variance_of(EYE_W as usize, EYE_H as usize, &gray24x40(rroi)));
    l.max(r)
}

/// 原始拉普拉斯方差（对焦映射前的值），供校准/诊断。
pub fn focus_variance_of(img: &image::RgbImage) -> f32 {
    let (w, h, g) = gray_normalized(img);
    laplacian_variance_of(w, h, &g)
}

// ---- 内部实现 ----

const EYE_W: u32 = 40;
const EYE_H: u32 = 24;

/// 归一化到 `FOCUS_REF` 最长边后转灰度，返回 (宽, 高, 灰度数组)。
fn gray_normalized(img: &image::RgbImage) -> (usize, usize, Vec<f32>) {
    let (w, h) = img.dimensions();
    let (nw, nh, src) = if w.max(h) > FOCUS_REF {
        let scale = FOCUS_REF as f64 / w.max(h) as f64;
        let nw = ((w as f64 * scale).round() as u32).max(1);
        let nh = ((h as f64 * scale).round() as u32).max(1);
        let resized = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Triangle);
        (nw as usize, nh as usize, resized)
    } else {
        (w as usize, h as usize, img.clone())
    };
    let mut g = Vec::with_capacity(nw * nh);
    for p in src.pixels() {
        g.push(lum(p[0], p[1], p[2]));
    }
    (nw, nh, g)
}

/// 24×40×3 RGB 字节 → 灰度。
fn gray24x40(rgb: &[u8]) -> Vec<f32> {
    rgb.chunks_exact(3).map(|c| lum(c[0], c[1], c[2])).collect()
}

#[inline]
fn lum(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// 灰度拉普拉斯方差（3×3 核，仅内部像素）。
fn laplacian_variance_of(w: usize, h: usize, g: &[f32]) -> f32 {
    if w < 3 || h < 3 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut sq_sum = 0.0f32;
    let mut count = 0usize;
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let i = y * w + x;
            let lap = 4.0 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
            sum += lap;
            sq_sum += lap * lap;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / count as f32;
    sq_sum / count as f32 - mean * mean
}

/// 方差 → 对焦分（1~10，单调线性）。
fn variance_to_score(v: f32) -> f32 {
    let t = ((v - V_MIN) / (V_MAX - V_MIN)).clamp(0.0, 1.0);
    1.0 + 9.0 * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_maps_monotonically() {
        // 方差越大 → 分越高
        assert!(variance_to_score(10.0) < variance_to_score(50.0));
        assert!(variance_to_score(50.0) < variance_to_score(500.0));
        // 端点
        assert!((variance_to_score(0.0) - 1.0).abs() < 1e-6);
        assert!((variance_to_score(1e6) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_focus_threshold() {
        assert!(is_out_of_focus(3.5));
        assert!(!is_out_of_focus(4.5));
    }

    #[test]
    fn uniform_image_has_zero_variance() {
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128]));
        assert!((focus_variance_of(&img)).abs() < 1e-3);
    }

    #[test]
    fn image_with_high_frequency_has_higher_variance() {
        // 棋盘格（高频）方差应远大于纯色
        let mut checker = image::RgbImage::new(64, 64);
        for (x, y, p) in checker.enumerate_pixels_mut() {
            let v = if (x / 2 + y / 2) % 2 == 0 { 200 } else { 60 };
            *p = image::Rgb([v, v, v]);
        }
        assert!(focus_variance_of(&checker) > focus_variance_of(&image::RgbImage::from_pixel(64, 64, image::Rgb([90, 90, 90]))));
    }
}

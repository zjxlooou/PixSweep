//! NIMA（Neural Image Assessment）模型相关常量与后处理。

/// 模型输入张量名称（tf2onnx 转换自 Keras MobileNet，输入为 NHWC）。
pub const INPUT_NAME: &str = "input_1";
/// 模型输出张量名称（10-bin 美学分布）。
pub const OUTPUT_NAME: &str = "nima_dense";
/// NIMA 输入为 NHWC 布局（Keras 原始布局）。
pub const IS_NHWC: bool = true;

/// 从 NIMA 输出的 10-bin 概率分布计算美学评分（1.0~10.0）。
///
/// 评分 = Σ(bin_i × (i+1)) / Σ(bin_i)，与 TOPIQ-IAA 同一公式
/// （[`crate::ai::mos_from_bins`]）。
pub fn nima_score_from_distribution(dist: &[f32]) -> f32 {
    crate::ai::mos_from_bins(dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_ranges_from_one_to_ten() {
        // 全部分配到最高分（第10档）
        let mut d = [0.0f32; 10];
        d[9] = 1.0;
        assert!((nima_score_from_distribution(&d) - 10.0).abs() < 1e-5);

        // 全部分配到最低分（第1档）
        let mut d2 = [0.0f32; 10];
        d2[0] = 1.0;
        assert!((nima_score_from_distribution(&d2) - 1.0).abs() < 1e-5);

        // 均匀分布 → 5.5
        let d3 = [0.1f32; 10];
        assert!((nima_score_from_distribution(&d3) - 5.5).abs() < 1e-5);
    }
}

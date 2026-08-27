//! TOPIQ（Top-down Image Quality）模型相关常量与后处理。
//!
//! TOPIQ 论文：Chen et al., IEEE TIP 2024。ResNet50 backbone + 多尺度 Transformer 头。
//! 两个变体：
//! - **TOPIQ-NR**（技术质量，KonIQ-10k 训练，输出 0~1 标量 MOS）
//! - **TOPIQ-IAA**（美学，AVA 训练，输出 10-bin softmax 概率分布）

/// TOPIQ 模型输入张量名称（两个变体一致）。
pub const INPUT_NAME: &str = "input";
/// TOPIQ-NR 输出张量名称（0~1 标量质量分）。
pub const NR_OUTPUT_NAME: &str = "quality_score";
/// TOPIQ-IAA 输出张量名称（10-bin softmax 分布）。
pub const IAA_OUTPUT_NAME: &str = "output";
/// TOPIQ 输入尺寸（正方形边长，TOPIQ 官方训练尺寸 384×384）。
pub const INPUT_SIZE: u32 = 384;

/// 从 TOPIQ-NR 的 0~1 标量映射到 1.0~10.0。
///
/// TOPIQ-NR 输出已经过 sigmoid，值域 [0,1]，线性映射到 [1,10]。
pub fn topiq_nr_to_score(v: f32) -> f32 {
    1.0 + v.clamp(0.0, 1.0) * 9.0
}

/// 从 TOPIQ-IAA 的 10-bin softmax 概率分布计算美学评分（1.0~10.0）。
///
/// 与 NIMA 相同：加权平均 Σ(p_i × (i+1))。IAA 输出已是 softmax 概率（和≈1），
/// 但仍做归一化保护。
pub fn topiq_iaa_to_score(dist: &[f32]) -> f32 {
    let sum: f32 = dist.iter().sum();
    if sum <= 0.0 {
        return 5.0;
    }
    let weighted: f32 = dist
        .iter()
        .enumerate()
        .map(|(i, &p)| p * (i as f32 + 1.0))
        .sum();
    weighted / sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nr_maps_zero_to_one() {
        assert!((topiq_nr_to_score(0.0) - 1.0).abs() < 1e-5);
        assert!((topiq_nr_to_score(1.0) - 10.0).abs() < 1e-5);
        assert!((topiq_nr_to_score(0.5) - 5.5).abs() < 1e-5);
    }

    #[test]
    fn nr_clamps_out_of_range() {
        assert!((topiq_nr_to_score(-1.0) - 1.0).abs() < 1e-5);
        assert!((topiq_nr_to_score(2.0) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn iaa_scores_from_one_to_ten() {
        let mut high = [0.0f32; 10];
        high[9] = 1.0;
        assert!((topiq_iaa_to_score(&high) - 10.0).abs() < 1e-5);

        let mut low = [0.0f32; 10];
        low[0] = 1.0;
        assert!((topiq_iaa_to_score(&low) - 1.0).abs() < 1e-5);

        let uniform = [0.1f32; 10];
        assert!((topiq_iaa_to_score(&uniform) - 5.5).abs() < 1e-5);
    }
}

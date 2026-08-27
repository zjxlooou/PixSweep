//! CLIP ViT-B/32 模型相关常量与辅助。

/// 模型输入张量名称。
pub const INPUT_NAME: &str = "pixel_values";
/// 模型输出张量名称（图像 embedding）。
pub const OUTPUT_NAME: &str = "image_embeds";
/// CLIP ViT-B/32 的 embedding 维度。
pub const CLIP_EMBEDDING_DIM: usize = 512;
/// 模型输入尺寸（正方形边长）。
pub const INPUT_SIZE: u32 = 224;

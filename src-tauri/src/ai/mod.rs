//! AI 推理模块（需启用 `ai` feature）。
//!
//! 通过 ONNX Runtime 的三级 GPU 回退链（CUDA → DirectML → CPU）：
//! - CLIP ViT-B/32：提取 512 维图像 embedding，用于语义级相似度检测（去重核心）
//! - TOPIQ-NR：技术质量评分（主用，ResNet50，KonIQ-10k，通用画质）
//! - TOPIQ-NR-Face：人脸技术质量评分（专用，ResNet50，CGFIQA-40k，需前置人脸检测+对齐）
//! - TOPIQ-IAA：美学评分（主用，ResNet50，AVA）
//! - CLIP-IQA+：零样本技术质量评分（后备），NIMA MobileNet-v2 作二级后备
//! - LAION Aesthetics V1 线性头：美学评分后备（依赖 CLIP embedding）
//! - MobileNetV3-Large：场景分类（无脸图的风景/宠物识别，人像由人脸检测覆盖）
//! - OCEC (PINTO0309)：闭眼检测，人像场景闭眼照片自动降权
//!
//! 模型文件（ONNX 格式）需放置在 [`crate::models_dir`] 目录下。

pub mod clip;
pub mod engine;
pub mod eye;
pub mod focus;
pub mod insightface;
pub mod nima;
pub mod preprocess;
pub mod scene;
pub mod topiq;
pub mod topiq_face;

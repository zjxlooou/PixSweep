//! 场景分类器（MobileNetV3-Large，ImageNet 1000 类 → 人像/宠物/风景/其他）。
//!
//! ## 设计要点
//! - **人像场景不靠本分类器**：ImageNet 1000 类没有直接 "person" 类（groom/scuba diver
//!   等仅 2 类，召回极低）。人像判定完全由 InsightFace 人脸检测覆盖（有脸 → 人像）。
//! - 本分类器用于**无脸图**的风景/宠物识别：
//!   - 宠物：犬种 0-based 151-268（118 类）+ 猫 281-284 + 兔/仓鼠/豚鼠 330-333
//!   - 风景：ImageNet 尾部自然景观 970-980（alp/cliff/coral reef/geyser/lakeside/promontory/
//!     sandbar/seashore/valley/volcano，除 971 bubble）
//!   - 其余：其他（文档/食物/建筑/工具等）
//!
//! ## 模型
//! `models/scene/mobilenet_v3_large.onnx`（74KB 图）+ `mobilenet_v3_large.data`（21MB 权重）
//! - 输入 `image_tensor` [1,3,224,224] float32，**值域 [0,1]**（仅 /255，不做 ImageNet 归一化）
//! - 输出 `class_logits` [1,1000]（logits，取 argmax）
//!
//! ## 来源
//! Qualcomm AI Hub `mobilenet_v3_large-onnx-float`（timm 导出，ImageNet 1000，Top-1 ~75%）

use anyhow::Context;
use ndarray::Array4;
use ort::inputs;
use ort::session::Session;

/// 模型文件名。
pub const MODEL_NAME: &str = "mobilenet_v3_large.onnx";
/// 输入张量名。
pub const INPUT_NAME: &str = "image_tensor";
/// 输出张量名。
pub const OUTPUT_NAME: &str = "class_logits";
/// 输入尺寸。
pub const INPUT_SIZE: u32 = 224;

/// 场景分类结果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Scene {
    /// 其他（文档/食物/建筑/工具等，未分类）
    Other = 0,
    /// 人像（由人脸检测覆盖，MobileNetV3 不直接产生产物）
    Portrait = 1,
    /// 宠物（猫/狗/兔/仓鼠/豚鼠）
    Pet = 2,
    /// 风景（自然景观：山/海/湖/火山等）
    Landscape = 3,
}

impl Scene {
    pub fn label(&self) -> &'static str {
        match self {
            Scene::Other => "其他",
            Scene::Portrait => "人像",
            Scene::Pet => "宠物",
            Scene::Landscape => "风景",
        }
    }

    /// 从 repr(u8) 反序列化（与 `Scene as u8` 对应），未知值安全回退为 `Other`。
    pub fn from_u8(v: u8) -> Scene {
        match v {
            1 => Scene::Portrait,
            2 => Scene::Pet,
            3 => Scene::Landscape,
            _ => Scene::Other,
        }
    }
}

// ImageNet 1000 类 → 场景映射（0=其他 1=人像 2=宠物 3=风景）。
// 由 scripts/gen_scene_map.py 生成，索引对应 MobileNetV3 输出顺序。
// 映射表单独文件 scene_map.rs，避免主文件过长。
include!("scene_map.rs");

/// 把代理图转成 MobileNetV3 输入张量 [1,3,224,224]，值域 [0,1]（仅 /255，无 ImageNet 归一化）。
fn scene_input_tensor(img: &image::RgbImage) -> Array4<f32> {
    let resized = image::imageops::resize(
        img,
        INPUT_SIZE,
        INPUT_SIZE,
        image::imageops::FilterType::Triangle,
    );
    let mut batch = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let p = resized.get_pixel(x, y);
            for c in 0..3 {
                batch[[0, c, y as usize, x as usize]] = p[c] as f32 / 255.0;
            }
        }
    }
    batch
}

/// f32 切片的 argmax（并列取首个）。
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

/// 调试辅助：单张图推理并返回 argmax 类索引（供 verify_scene 排查预处理差异）。
pub fn argmax_of(session: &mut Session, path: &str) -> anyhow::Result<usize> {
    let img = crate::cache::proxy::ai_proxy(path)?;
    let tensor = ort::value::Tensor::from_array(scene_input_tensor(&img)).context("输入张量失败")?;
    let outputs = session
        .run(inputs![INPUT_NAME => tensor])
        .context("推理失败")?;
    let logits = outputs[OUTPUT_NAME]
        .try_extract_array::<f32>()
        .context("输出解析失败")?;
    Ok(argmax(logits.slice(ndarray::s![0, ..]).as_slice().unwrap_or(&[])))
}

/// 将 MobileNetV3 输出 logits（[1, 1000]）映射为场景。
fn logits_to_scene(logits: &[f32]) -> Scene {
    let code = SCENE_MAP.get(argmax(logits)).copied().unwrap_or(0);
    match code {
        2 => Scene::Pet,
        3 => Scene::Landscape,
        _ => Scene::Other,
    }
}

/// 对一批图片做场景分类（MobileNetV3）。
///
/// 返回每张图的场景。**人像场景由调用方通过人脸检测覆盖**
/// （有脸 → Portrait），本函数只产 Other/Pet/Landscape。
///
/// 注意：MobileNetV3 ONNX **固定 batch=1**（Qualcomm 导出 shape [1,3,224,224]），
/// 不支持批量推理，故逐张 run（224×224 单张 CUDA 约 10ms，CPU 约 50ms，可接受）。
pub fn classify(session: &mut Session, paths: &[String]) -> anyhow::Result<Vec<Scene>> {
    let n = paths.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut scenes = Vec::with_capacity(n);
    for path in paths.iter() {
        let sc = classify_one(session, path).unwrap_or_else(|e| {
            log::warn!("[场景] 单张分类失败 {}: {}", path, e);
            Scene::Other
        });
        scenes.push(sc);
    }
    Ok(scenes)
}

/// 单张图分类（batch=1）。
fn classify_one(session: &mut Session, path: &str) -> anyhow::Result<Scene> {
    let img = crate::cache::proxy::ai_proxy(path)?;
    let tensor =
        ort::value::Tensor::from_array(scene_input_tensor(&img)).context("场景分类输入张量失败")?;
    let outputs = session
        .run(inputs![INPUT_NAME => tensor])
        .context("场景分类推理失败")?;
    let logits = outputs[OUTPUT_NAME]
        .try_extract_array::<f32>()
        .context("场景分类输出解析失败")?;

    let row: Vec<f32> = logits.slice(ndarray::s![0, ..]).iter().copied().collect();
    Ok(logits_to_scene(&row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_map_has_expected_classes() {
        // 宠物：犬种 151-268（118 类）+ 猫 281-284 + 兔/仓鼠 330-333
        assert_eq!(SCENE_MAP[151], 2); // Chihuahua
        assert_eq!(SCENE_MAP[268], 2); // Mexican hairless
        assert_eq!(SCENE_MAP[281], 2); // tabby
        assert_eq!(SCENE_MAP[284], 2); // Siamese cat
        assert_eq!(SCENE_MAP[330], 2); // wood rabbit
        assert_eq!(SCENE_MAP[333], 2); // Angora
        // 风景：970-980（除 971 bubble）
        assert_eq!(SCENE_MAP[970], 3); // alp
        assert_eq!(SCENE_MAP[980], 3); // volcano
        assert_eq!(SCENE_MAP[971], 0); // bubble 排除
        // 其他常见类
        assert_eq!(SCENE_MAP[0], 0); // tench（鱼类，归其他）
        assert_eq!(SCENE_MAP[500], 0); // 非风景
    }

    #[test]
    fn logits_argmax_maps_correctly() {
        let mut logits = vec![0.0f32; 1000];
        logits[0] = 10.0; // 其他类
        assert_eq!(logits_to_scene(&logits), Scene::Other);

        let mut logits = vec![0.0f32; 1000];
        logits[200] = 10.0; // 犬种
        assert_eq!(logits_to_scene(&logits), Scene::Pet);

        let mut logits = vec![0.0f32; 1000];
        logits[975] = 10.0; // lakeside
        assert_eq!(logits_to_scene(&logits), Scene::Landscape);
    }
}
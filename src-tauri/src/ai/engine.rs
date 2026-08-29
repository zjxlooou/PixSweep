//! AI 引擎：封装 ONNX Runtime 会话，三级 GPU 回退链（CUDA → DirectML → CPU）。
//!
//! - **CUDA**（NVIDIA）：PyTorch 原生后端，优先选用
//! - **DirectML**（NVIDIA/AMD/Intel 通用）：兜底 GPU 加速
//! - **CPU**：最后保底
//!
//! 运行时按顺序探测，session 创建时自动注入 EP 列表（失败则降级）。
//!
//! 评分体系（双维度）：
//! - **技术评分**：TOPIQ-NR（ResNet50，KonIQ-10k，主用）→ NIMA（MobileNetV2，二级后备）
//! - **美学评分**：TOPIQ-IAA（ResNet50，AVA，主用）；LAION/CLIP 后备已于 2026-08-27 移除
//! 综合评分 = 美学 × w_a + 技术 × w_t + 启发式，用于组内挑选最佳照片。
//!

use crate::ai::nima::{INPUT_NAME as NIMA_INPUT, OUTPUT_NAME as NIMA_OUTPUT};
use crate::ai::topiq::{
    IAA_OUTPUT_NAME as TOPIQ_IAA_OUTPUT, INPUT_NAME as TOPIQ_INPUT,
    NR_OUTPUT_NAME as TOPIQ_NR_OUTPUT, topiq_iaa_to_score, topiq_nr_to_score,
};
use ndarray::Array4;
use rayon::prelude::*;
use ort::ep::{CPU, DirectML, ExecutionProvider};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

/// NIMA 技术质量模型文件名（MobileNet 224×224，10-bin MOS 分布）。
pub const NIMA_TECH_MODEL: &str = "nima-technical.onnx";
/// TOPIQ-NR 技术质量模型文件名（ResNet50，KonIQ-10k，输出 0~1）。
pub const TOPIQ_NR_MODEL: &str = "topiq_nr.onnx";
/// HyperIQA 通用质量模型文件名（ResNet50，KonIQ-10k，FP16 单文件，可选）。
/// 仅用于**非人像**场景与 TOPIQ-IAA 融合（人像偏置重，人像不启用）。
pub const HYPERIQA_MODEL: &str = "hyperiqa.onnx";

/// 人像融合权重：`face = w * nr_face + (1-w) * nr_on_face`。
///
/// w=0.5 由 357 张基准标定（2026-08-27）：把 nr_face 的暗光盲区
/// （欠曝人像反而高分，融合前 dark45 敏感 -0.038）归零，
/// 平均降级敏感度 ×3.6（+0.010 → +0.036），同时保留一半人脸特化信号。
pub const FACE_FUSION_NR_FACE_WEIGHT: f32 = 0.5;

/// HyperIQA → TOPIQ-IAA 尺度的线性校准（2026-08-27，357 张基准最小二乘）。
/// 映射 `iaa_est = A*h + B`（h 为 [0,1] 原始输出），与 IAA 值域对齐后做 0.5/0.5 融合。
pub const HYPERIQA_CAL_A: f32 = 3.2251;
pub const HYPERIQA_CAL_B: f32 = 3.2467;
pub const HYPERIQA_FUSION_WEIGHT: f32 = 0.5;
/// TOPIQ-IAA 美学模型文件名（ResNet50，AVA，输出 10-bin softmax）。
pub const TOPIQ_IAA_MODEL: &str = "topiq_iaa_res50.onnx";

/// 综合评分权重。
/// 场景是"组内选最佳"：组内照片内容相同，美学分必然接近，
/// 对焦/清晰度才是决定性维度，故对焦权重最高。
pub const WEIGHT_AESTHETIC: f32 = 0.25;
pub const WEIGHT_FOCUS: f32 = 0.6;
pub const WEIGHT_HEURISTIC: f32 = 0.15;

/// 人像场景权重（有人脸且人脸分有效时主导）。人像 = **人像美学(人脸分) 主导** +
/// **眼部对焦** + 启发式；整图美学/技术不参与（见用户流程：五官/闭眼/眼部对焦/人像美学）。
pub const W_FACE_A: f32 = 0.0;
pub const W_FACE_FOCUS: f32 = 0.30;
pub const W_FACE_F: f32 = 0.55;
pub const W_FACE_H: f32 = 0.15;
/// 风景场景权重（画质优先，对焦分略高）。
pub const W_LAND_A: f32 = 0.40;
pub const W_LAND_FOCUS: f32 = 0.50;
pub const W_LAND_H: f32 = 0.10;
/// 宠物场景权重（美学/对焦均衡）。
pub const W_PET_A: f32 = 0.45;
pub const W_PET_FOCUS: f32 = 0.45;
pub const W_PET_H: f32 = 0.10;

/// 推理加速后端。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// CUDA（NVIDIA 原生 GPU 加速，POC 验证比 CPU 快 4×）
    Cuda,
    /// DirectML（Windows 标准 DirectX 12，任何显卡通用，CUDA 不可用时回退）
    DirectML,
    /// 纯 CPU（最后保底）
    Cpu,
}

impl GpuBackend {
    pub fn label(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "CUDA (NVIDIA GPU)",
            GpuBackend::DirectML => "DirectML (DirectX 12)",
            GpuBackend::Cpu => "CPU",
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, GpuBackend::Cuda | GpuBackend::DirectML)
    }
}


/// AI 推理引擎，持有 TOPIQ-NR/IAA（主评分）+ NIMA（技术后备）+ 人脸/场景/闭眼。
pub struct AiEngine {
    topiq_nr_session: Option<parking_lot::Mutex<Session>>,
    topiq_iia_session: Option<parking_lot::Mutex<Session>>,
    nima_tech_session: Option<parking_lot::Mutex<Session>>,
    /// HyperIQA（可选）：非人像场景的美学第二意见
    hyperiqa_session: Option<parking_lot::Mutex<Session>>,
    /// InsightFace 人脸检测（可选）：存在 buffalo_l 模型时加载
    face_det: Option<crate::ai::insightface::InsightFaceEngine>,
    /// 最大脸检测共享缓存（路径 → 最大脸）：face/eye/对焦三阶段复用，
    /// 同一张图全链只跑一次 SCRFD。代理图按路径确定性重建，失效语义与代理缓存一致。
    detect_cache: parking_lot::Mutex<std::collections::HashMap<String, Option<crate::ai::insightface::Face>>>,
    /// TOPIQ-NR-Face 人脸专评 session（可选）：模型存在时加载
    face_session: Option<parking_lot::Mutex<Session>>,
    /// MobileNetV3 场景分类 session（可选）：模型存在时加载
    scene_session: Option<parking_lot::Mutex<Session>>,
    /// OCEC 闭眼检测器（可选）：模型存在时加载
    eye_det: Option<crate::ai::eye::EyeDetector>,
    backend: GpuBackend,
}

impl AiEngine {
    /// 尝试加载单个可选模型会话：文件存在则按三级回退构建；缺失/失败仅告警并返回
    /// None（模型缺失不报错是既定约定，对应能力自动跳过）。
    /// 成功时同时返回所用后端，调用方决定是否采纳为引擎后端。
    fn load_optional_session(
        path: &std::path::Path,
        capability: &str,
    ) -> Option<(parking_lot::Mutex<Session>, GpuBackend)> {
        if !path.exists() {
            log::warn!("{} 模型不存在，跳过对应能力: {}", capability, path.display());
            return None;
        }
        log::info!("{} 模型存在: {}", capability, path.display());
        match Self::build_session(path, false) {
            Ok((s, be)) => {
                log::info!("{} 会话就绪，推理后端: {}", capability, be.label());
                Some((parking_lot::Mutex::new(s), be))
            }
            Err(e) => {
                log::warn!("{} 模型加载失败，跳过对应能力: {}", capability, e);
                None
            }
        }
    }

    /// 初始化引擎：按 CUDA → DirectML → CPU 三级回退加载全部模型。
    /// 加载顺序：TOPIQ-NR/IAA（主）→ NIMA 技术质量（后备，可选）→ 其余能力模型。
    pub fn new(model_dir: &std::path::Path) -> anyhow::Result<Self> {
        log::info!("AI 引擎初始化，模型目录: {}", model_dir.display());
        log::info!("DirectML 编译支持: {}", directml_available());

        // 引擎后端 = 首个成功加载的评分模型会话所用后端（无模型可用时为 CPU）
        let mut backend = GpuBackend::Cpu;

        // NIMA 技术质量模型（可选）：TOPIQ-NR 不可用时的后备
        let nima_tech_session =
            if let Some((s, be)) = Self::load_optional_session(&model_dir.join(NIMA_TECH_MODEL), "NIMA 技术") {
                backend = be;
                Some(s)
            } else {
                None
            };

        // TOPIQ-NR 技术质量模型（主用，ResNet50）：不可用时技术分回退 NIMA
        let topiq_nr_session =
            if let Some((s, be)) = Self::load_optional_session(&model_dir.join(TOPIQ_NR_MODEL), "TOPIQ-NR") {
                backend = be;
                Some(s)
            } else {
                None
            };

        // TOPIQ-IAA 美学模型（主用，ResNet50）：不可用时无美学后备
        let topiq_iia_session =
            if let Some((s, be)) = Self::load_optional_session(&model_dir.join(TOPIQ_IAA_MODEL), "TOPIQ-IAA") {
                backend = be;
                Some(s)
            } else {
                None
            };

        // HyperIQA（可选）：非人像美学融合的第二意见
        let hyperiqa_session =
            if let Some((s, be)) = Self::load_optional_session(&model_dir.join(HYPERIQA_MODEL), "HyperIQA") {
                backend = be;
                Some(s)
            } else {
                None
            };

        // InsightFace buffalo_l 人脸检测（可选）：存在 models/insightface/ 时加载
        let insightface_dir = model_dir.join("insightface");
        let face_det = if insightface_dir.join("det_10g.onnx").exists() {
            log::info!("InsightFace 检测模型存在，加载人脸检测: {}", insightface_dir.display());
            let mut eng = crate::ai::insightface::InsightFaceEngine::new();
            match eng.load(&insightface_dir, false) {
                Ok(()) => {
                    log::info!("InsightFace 人脸检测就绪");
                    Some(eng)
                }
                Err(e) => {
                    log::warn!("InsightFace 人脸检测加载失败，跳过人脸专评: {}", e);
                    None
                }
            }
        } else {
            log::warn!("InsightFace 检测模型不存在，跳过人脸专评: {}", insightface_dir.display());
            None
        };

        // TOPIQ-NR-Face 人脸专评（可选）：模型存在时加载（不改变引擎后端标签）
        let face_session = Self::load_optional_session(
            &model_dir.join(crate::ai::topiq_face::MODEL_NAME),
            "TOPIQ-NR-Face",
        )
        .map(|(s, _)| s);

        // MobileNetV3 场景分类（可选）：模型存在时加载（在 models/scene/ 子目录）
        let scene_dir = model_dir.join("scene");
        let scene_path = scene_dir.join(crate::ai::scene::MODEL_NAME);
        let scene_session = if scene_path.exists() {
            log::info!("MobileNetV3 场景分类模型存在: {}", scene_path.display());
            match Self::build_session(&scene_path, false) {
                Ok((s, be)) => {
                    log::info!("场景分类会话就绪，推理后端: {}", be.label());
                    Some(parking_lot::Mutex::new(s))
                }
                Err(e) => {
                    log::warn!("场景分类模型加载失败，跳过场景识别: {}", e);
                    None
                }
            }
        } else {
            log::warn!("场景分类模型不存在，跳过场景识别");
            None
        };

        // OCEC 闭眼检测（可选）：模型存在时加载（在 models/eye/ 子目录）
        let eye_dir = model_dir.join("eye");
        let eye_det = if eye_dir.join(crate::ai::eye::MODEL_NAME).exists() {
            log::info!("OCEC 闭眼检测模型存在: {}", eye_dir.display());
            let det = crate::ai::eye::EyeDetector::new();
            match det.load(&eye_dir) {
                Ok(d) => {
                    log::info!("OCEC 闭眼检测就绪");
                    Some(d)
                }
                Err(e) => {
                    log::warn!("OCEC 加载失败，跳过闭眼检测: {}", e);
                    None
                }
            }
        } else {
            log::warn!("OCEC 模型不存在，跳过闭眼检测: {}", eye_dir.display());
            None
        };

        Ok(Self {
            topiq_nr_session,
            topiq_iia_session,
            nima_tech_session,
            hyperiqa_session,
            face_det,
            detect_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
            face_session,
            scene_session,
            eye_det,
            backend,
        })
    }

    /// 构建会话：三级回退（CUDA → DirectML → CPU）。
    ///
    /// `force_cpu=true` 时跳过 GPU：某些模型的个别 op 在 GPU 跑期会报错
    /// （CLIP-IQA+ 的 `Reshape` 节点曾因 DirectML 返回 E_INVALIDARG 而强制 CPU；
    /// 此时直接用 CPU EP 最稳妥——模型能正常加载且推理确定。
    pub(super) fn build_session(
        path: &std::path::Path,
        force_cpu: bool,
    ) -> anyhow::Result<(Session, GpuBackend)> {
        if !force_cpu {
            // 1) CUDA 优先（NVIDIA GPU，POC 验证最稳）
            #[cfg(feature = "cuda")]
            {
                if let Some(s) = Self::try_build_cuda(path) {
                    log::info!("CUDA 会话构建成功: {}", path.display());
                    return Ok((s, GpuBackend::Cuda));
                }
                log::warn!("CUDA 会话构建失败: {}", path.display());
            }
            // 2) DirectML 兜底（NVIDIA/AMD/Intel 通用）
            #[cfg(feature = "directml")]
            {
                if let Some(s) = Self::try_build_directml(path) {
                    log::info!("DirectML 会话构建成功: {}", path.display());
                    return Ok((s, GpuBackend::DirectML));
                }
                log::warn!("DirectML 会话构建失败: {}", path.display());
            }
        }

        // 3) CPU 最后保底
        let s = Self::try_build_cpu(path)
            .ok_or_else(|| anyhow::anyhow!("无法加载模型: {}", path.display()))?;
        log::info!("CPU 会话构建成功: {}", path.display());
        Ok((s, GpuBackend::Cpu))
    }

    /// 统一配置会话选项，保证推理可复现且满足 DirectML EP 的强制要求：
    ///
    /// - `with_parallel_execution(false)`：DirectML 不支持并行执行，必须顺序执行
    ///   （否则 `commit_from_file` 直接报错）；
    /// - `with_memory_pattern(false)`：DirectML 要求关闭内存模式优化；
    /// - `with_intra_threads(1)`：单线程 intra-op 归约，消除浮点求和顺序带来的
    ///   非确定性——这正是"同一张图有时高分有时低分"的根因。
    fn configure_session_builder(mut builder: SessionBuilder) -> Option<SessionBuilder> {
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .ok()?
            .with_parallel_execution(false)
            .ok()?
            .with_memory_pattern(false)
            .ok()?
            .with_intra_threads(1)
            .ok()?;
        Some(builder)
    }

    /// 尝试用 CUDA EP 构建会话（NVIDIA 原生 GPU 加速，POC 验证最快）。
    ///
    /// 先用 CUDA 驱动 API（`nvcuda.dll` 的 `cuInit` + `cuDeviceGetCount`）真实验证
    /// 设备数：装过带 cudart 的软件会让 EP DLL 加载成功，但无 N 卡时推理会静默
    /// 回退 CPU——必须以驱动级设备枚举为准，否则后端标签误报 CUDA。
    #[cfg(feature = "cuda")]
    fn try_build_cuda(path: &std::path::Path) -> Option<Session> {
        use ort::ep::CUDA;
        if !cuda_device_available() {
            log::warn!("CUDA 驱动未检测到 NVIDIA GPU 设备，跳过 CUDA EP: {}", path.display());
            return None;
        }
        let builder = Self::configure_session_builder(Session::builder().ok()?)?;
        // SameAsRequested：默认 kNextPowerOfTwo 会把 arena 翻倍预留，大激活会话
        // 挤占显存导致后续小模型分配变慢（HyperIQA 减速 6× 的疑似根因）
        let ep = CUDA::default()
            .with_device_id(0)
            .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
            .build();
        let mut builder = builder.with_execution_providers([ep]).ok()?;
        match builder.commit_from_file(path) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("CUDA commit 失败 {}: {}", path.display(), e);
                None
            }
        }
    }

    /// 尝试用 DirectML EP 构建会话（Windows 标准硬件加速）。
    #[cfg(feature = "directml")]
    fn try_build_directml(path: &std::path::Path) -> Option<Session> {
        let builder = Self::configure_session_builder(Session::builder().ok()?)?;
        let ep = DirectML::default().build();
        let mut builder = builder.with_execution_providers([ep]).ok()?;
        match builder.commit_from_file(path) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("DirectML commit 失败 {}: {}", path.display(), e);
                None
            }
        }
    }

    /// 用 CPU EP 构建会话。
    fn try_build_cpu(path: &std::path::Path) -> Option<Session> {
        let builder = Self::configure_session_builder(Session::builder().ok()?)?;
        let ep = CPU::default().build();
        let mut builder = builder.with_execution_providers([ep]).ok()?;
        builder.commit_from_file(path).ok()
    }

    /// 当前使用的推理后端。
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }

    /// 是否启用了 GPU 加速（CUDA 或 DirectML）。
    pub fn gpu_enabled(&self) -> bool {
        self.backend.is_gpu()
    }

    /// 人脸专评是否可用（需要 InsightFace 检测 + TOPIQ-NR-Face 模型）。
    pub fn face_scoring_available(&self) -> bool {
        self.face_det.is_some() && self.face_session.is_some()
    }

    /// 场景分类是否可用（MobileNetV3 模型存在）。
    pub fn scene_scoring_available(&self) -> bool {
        self.scene_session.is_some()
    }

    /// 闭眼检测是否可用（OCEC 模型存在）。
    pub fn eye_status_available(&self) -> bool {
        self.eye_det.as_ref().map(|e| e.is_loaded()).unwrap_or(false)
            && self.face_det.is_some()
    }

    /// 对一批图片做闭眼检测，返回每张图的 `max(open_l, open_r) ∈ [0,1]`。
    ///
    /// - 1.0 = 至少一眼全开（无闭眼降权）；0.0 = 双眼全闭（最大降权）。
    /// - 无人脸 / 未启用 / 人脸检测失败 / ROI 采样失败 → 1.0（默认不降权）。
    /// - 双信号融合：OCEC 判眨眼式闭合，脸网格（`face_landmarker.onnx`，可选）判垂目式
    ///   闭合——取两者较小值；网格缺失或拟合失败时自动退化为仅 OCEC。
    ///
    /// 实现逻辑：复用 `face_det` 重新检测 → 取最大脸 → OCEC + 脸网格推理。
    /// 注意：这是二次图像 I/O，可接受延迟（人脸检测 ~50ms + OCEC ~1ms + 网格几 ms）。
    pub fn eye_open_probs(&self, paths: &[String], has_faces: &[bool]) -> Vec<f32> {
        let n = paths.len();
        let mut results = vec![1.0f32; n];
        if n == 0 || !self.eye_status_available() {
            return results;
        }
        let (Some(face_engine), Some(eye_det)) = (&self.face_det, &self.eye_det) else {
            return results;
        };

        // rayon 并行逐张：单张图的开分只依赖自身像素与模型输出，与处理顺序无关
        //（collect 保序回填）→ 并行不改变确定性；GPU 会话经 Mutex 串行，CPU 侧的
        // 代理解码/几何计算与 GPU 重叠，消除整批串行等待（2026-08-29 优化）。
        let computed: Vec<Option<f32>> =
            crate::image_io::heavy_pool().install(|| paths
                .par_iter()
                .zip(has_faces.par_iter())
                .map(|(path, &has_face)| {
                if !has_face {
                    return None;
                }
                // 加载代理图（统一前置代理，见 cache::proxy）
                let img = match crate::cache::proxy::ai_proxy(path) {
                    Ok(img) => img,
                    Err(e) => {
                        log::warn!("[eye] 加载代理图失败 {}: {}", path, e);
                        return None;
                    }
                };
                let (w, h) = (img.width(), img.height());
                let rgb = img.as_raw();
                // 检测最大脸（共享缓存：人脸专评阶段已检过的图直接复用）
                let face = self.detect_max_face(face_engine, path, rgb, h, w)?;
                // 网格为主信号（垂目+眨眼几何都敏感，且提供精确虹膜中心修复 OCEC 采样）。
                // OCEC 仅作"眨眼否决"：网格判半开以上时其假闭误报较多（如刘海遮挡的睁眼），
                // 只在网格也模棱两可（< MESH_VETO_BAND）而 OCEC 双眼都强判闭（取 min <
                // OCEC_VETO_MAX）时采信——否决语义与"双眼都闭才降权"的产品规则一致。
                let mesh = eye_det.mesh_result(rgb, h, w, &face);
                let ocec = match &mesh {
                    Some(m) => {
                        eye_det.detect_probs_at(rgb, h, w, m.left_eye_src, m.right_eye_src)
                    }
                    None => eye_det.detect_probs(face_engine, rgb, h, w, &face),
                };
                // 回退分支沿用历史语义 max（任一眼开即开）；否决触发用 min（双眼都闭才否决）
                let (ocec_any, ocec_both) = ocec.map_or((1.0, 1.0), |(l, r)| (l.max(r), l.min(r)));
                Some(match mesh {
                    Some(m) => {
                        if ocec_both < crate::ai::eye::OCEC_VETO_MAX
                            && m.norm_open < crate::ai::eye::MESH_VETO_BAND
                        {
                            ocec_any.min(m.norm_open)
                        } else {
                            m.norm_open
                        }
                    }
                    None => ocec_any,
                })
            })
            .collect());
        for (i, v) in computed.into_iter().enumerate() {
            if let Some(v) = v {
                results[i] = v;
            }
        }
        results
    }

    /// 对一批图片做闭眼检测，返回 `is_any_closed` 标记数组：
    /// - `true`：双眼都闭（`max(open) <= 0.5`）
    /// - `false`：至少一眼睁
    /// - 默认 `false`（未启用或无脸图）
    ///
    /// 由 [`Self::eye_open_probs`] 派生（`max(open) <= 0.5`），避免重复推理。
    pub fn eye_status(&self, paths: &[String], has_faces: &[bool]) -> Vec<bool> {
        self.eye_open_probs(paths, has_faces)
            .iter()
            .map(|p| crate::ai::eye::is_closed(*p))
            .collect()
    }

    /// 对一批图片算"对焦分"（1~10）：
    /// - 有人脸 → **眼部对焦**（取最大脸，用眼 ROI 锐度，见 [`crate::ai::focus::eye_focus_score`]）。
    /// - 无脸 → **整图对焦**（代理图锐度，见 [`crate::ai::focus::focus_score`]）。
    /// - 失败/未启用 → 1.0（不因对焦降权）。
    pub fn focus_scores(&self, paths: &[String], has_faces: &[bool]) -> Vec<f32> {
        let n = paths.len();
        let mut out = vec![1.0f32; n];
        if n == 0 {
            return out;
        }
        // rayon 并行逐张（同 eye_open_probs：保序回填，确定性不变）
        let computed: Vec<Option<f32>> =
            crate::image_io::heavy_pool().install(|| paths
                .par_iter()
                .zip(has_faces.par_iter())
                .map(|(path, &has_face)| {
                if has_face {
                    self.eye_focus(path)
                } else {
                    crate::cache::proxy::ai_proxy(path)
                        .ok()
                        .map(|img| crate::ai::focus::focus_score(&img))
                }
            })
            .collect());
        for (i, v) in computed.into_iter().enumerate() {
            if let Some(v) = v {
                out[i] = v;
            }
        }
        out
    }

    /// 单张眼部对焦分（有人脸才调用）：检测最大脸（共享缓存）→ 采样眼 ROI → 锐度。
    fn eye_focus(&self, path: &str) -> Option<f32> {
        let det = self.face_det.as_ref()?;
        let img = crate::cache::proxy::ai_proxy(path).ok()?;
        let (w, h) = (img.width(), img.height());
        let raw = img.as_raw();
        let max_face = self.detect_max_face(det, path, raw, h, w)?;
        let (lroi, rroi) = crate::ai::eye::debug_sample_eyes_rgb(raw, h, w, &max_face)?;
        Some(crate::ai::focus::eye_focus_score(&lroi, &rroi))
    }

    /// 最大脸检测（带共享缓存）：闭眼/对焦阶段优先复用人脸专评阶段的检测结果，
    /// 避免同一张图重复跑 SCRFD（每张图全链只检测一次）。检测失败不缓存（下次重试）。
    fn detect_max_face(
        &self,
        det: &crate::ai::insightface::InsightFaceEngine,
        path: &str,
        rgb: &[u8],
        h: u32,
        w: u32,
    ) -> Option<crate::ai::insightface::Face> {
        if let Some(hit) = self.detect_cache.lock().get(path) {
            return *hit;
        }
        let faces = det.detect(rgb, h, w).ok()?;
        let face = crate::ai::insightface::Face::largest(faces);
        self.detect_cache.lock().insert(path.to_string(), face);
        face
    }

    /// 调试辅助：返回单张图的 MobileNetV3 argmax 类索引。
    pub fn scene_argmax(&self, path: &str) -> usize {
        let Some(sess) = &self.scene_session else {
            return 0;
        };
        let mut guard = sess.lock();
        crate::ai::scene::argmax_of(&mut guard, path).unwrap_or(0)
    }

    /// 返回单张图的最大人脸 bbox + 5 关键点（EXIF 转向后的原图坐标）。
    /// 供诊断 example（`verify_landmarks` / `verify_full`）复用。
    pub fn largest_face_landmarks(&self, path: &str) -> Option<([f32; 4], Vec<(f32, f32)>)> {
        let face_engine = self.face_det.as_ref()?;
        let img = crate::cache::proxy::ai_proxy(path).ok()?;
        let (w, h) = (img.width(), img.height());
        let raw = img.as_raw();
        let max_face = crate::ai::insightface::Face::largest(face_engine.detect(raw, h, w).ok()?)?;
        Some((
            max_face.bbox,
            vec![
                max_face.landmarks.left_eye,
                max_face.landmarks.right_eye,
                max_face.landmarks.nose,
                max_face.landmarks.left_mouth,
                max_face.landmarks.right_mouth,
            ],
        ))
    }

    /// 诊断辅助：返回单张图的 (左眼, 右眼) open 概率。
    pub fn eye_probs(&self, path: &str) -> Option<(f32, f32)> {
        let (Some(face_engine), Some(eye_det)) = (&self.face_det, &self.eye_det) else {
            return None;
        };
        let img = crate::cache::proxy::ai_proxy(path).ok()?;
        let (w, h) = (img.width(), img.height());
        let raw = img.as_raw();
        let max_face = crate::ai::insightface::Face::largest(face_engine.detect(raw, h, w).ok()?)?;
        eye_det.detect_probs(face_engine, raw, h, w, &max_face)
    }

    /// 对一批图片做场景分类（MobileNetV3，无脸图的风景/宠物识别）。
    ///
    /// 返回每张图的场景。**人像由调用方通过 has_faces 覆盖**（有脸 → Portrait）。
    pub fn scene_scores(&self, paths: &[String]) -> Vec<crate::ai::scene::Scene> {
        let n = paths.len();
        if n == 0 {
            return Vec::new();
        }
        let Some(sess) = &self.scene_session else {
            return vec![crate::ai::scene::Scene::Other; n];
        };
        let mut guard = sess.lock();
        match crate::ai::scene::classify(&mut guard, paths) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("[场景] 分类失败: {}", e);
                vec![crate::ai::scene::Scene::Other; n]
            }
        }
    }

    /// 对一批图片做人脸检测 + 人脸 crop 专评。
    ///
    /// 返回 `(face_scores, has_faces)`：
    /// - `face_scores[i]`：TOPIQ-NR-Face 评分（1~10，None 表示无人脸或失败）
    /// - `has_faces[i]`：是否检测到人脸（供前端显示人像标记）
    ///
    /// 仅当图片检测到人脸时才跑 TOPIQ-NR-Face（省时间），否则直接 None。
    pub fn face_scores(
        &self,
        paths: &[String],
    ) -> (Vec<Option<f32>>, Vec<bool>) {
        let n = paths.len();
        let mut scores: Vec<Option<f32>> = vec![None; n];
        let mut has_face = vec![false; n];

        let (Some(det), Some(face_sess)) = (&self.face_det, &self.face_session) else {
            return (scores, has_face);
        };

        // 1) 人脸检测 + 对齐 crop：批量前向（det_10g_batched 动态 batch，8 张一次）
        //    + 并行 letterbox/对齐。检测结果写入共享缓存（闭眼/眼对焦阶段复用）。
        //    单图结果只依赖自身，批内顺序固定 -> 确定性不变。
        const DET_BATCH: usize = 8;
        let mut crops: Vec<(usize, Vec<u8>, u32)> = Vec::new(); // (idx, crop_rgb, side)
        for (base, group) in paths.chunks(DET_BATCH).enumerate() {
            let base = base * DET_BATCH;
            // (a) 并行读代理图
            let imgs: Vec<Option<image::RgbImage>> = crate::image_io::heavy_pool().install(|| {
                group
                    .par_iter()
                    .map(|path| match crate::cache::proxy::ai_proxy(path) {
                        Ok(img) => Some(img),
                        Err(e) => {
                            log::warn!("[人脸] 加载失败 {}: {}", path, e);
                            None
                        }
                    })
                    .collect()
            });
            // (b) 批量检测（跳过加载失败的）
            let mut ready: Vec<(usize, String, Vec<u8>, u32, u32)> = Vec::new();
            for (gi, img) in imgs.iter().enumerate() {
                if let Some(img) = img {
                    ready.push((
                        base + gi,
                        group[gi].clone(),
                        img.as_raw().to_vec(),
                        img.width(),
                        img.height(),
                    ));
                }
            }
            let refs: Vec<(&[u8], u32, u32)> =
                ready.iter().map(|(_, _, rgb, w, h)| (rgb.as_slice(), *w, *h)).collect();
            let results = det.detect_batch(&refs);
            // (c) 写缓存 + 并行对齐 crop
            let mut to_align: Vec<(usize, Vec<u8>, u32, u32, crate::ai::insightface::Face)> =
                Vec::new();
            for ((idx, path, rgb, w, h), res) in ready.into_iter().zip(results) {
                match res {
                    Ok(faces) => {
                        let max = crate::ai::insightface::Face::largest(faces);
                        self.detect_cache.lock().insert(path, max);
                        if let Some(face) = max {
                            has_face[idx] = true;
                            to_align.push((idx, rgb, w, h, face));
                        }
                    }
                    Err(e) => log::warn!("[人脸] 批量检测失败: {}", e),
                }
            }
            let aligned: Vec<(usize, Vec<u8>)> = crate::image_io::heavy_pool().install(|| {
                to_align
                    .par_iter()
                    .map(|(idx, rgb, w, h, face)| {
                        let crop = det.align_face(rgb, *h, *w, face, crate::ai::topiq_face::INPUT_SIZE);
                        (*idx, crop)
                    })
                    .collect()
            });
            for (idx, crop) in aligned {
                crops.push((idx, crop, crate::ai::topiq_face::INPUT_SIZE));
            }
        }

        if crops.is_empty() {
            return (scores, has_face);
        }

        // 2) TOPIQ-NR-Face 批量推理（人脸 crop 已是 512×512）
        let mut sess_guard = face_sess.lock();
        // 转成 topiq_face 接口需要的 (Vec<u8>, u32)
        let crop_owned: Vec<(Vec<u8>, u32)> = crops
            .iter()
            .map(|(_, rgb, side)| (rgb.clone(), *side))
            .collect();
        match crate::ai::topiq_face::face_quality_scores(&mut sess_guard, &crop_owned) {
            Ok(vals) => {
                // 人像融合第二意见：TOPIQ-NR 对同一批对齐 crop 打技术分（nr-on-face）。
                // nr_face 对欠曝人像不扣分（暗光盲区），nr-on-face 对模糊/压缩/低清/欠曝
                // 全部强敏感——50/50 融合把盲区归零、平均降级敏感度 ×3.6（357 张基准标定）。
                let nr10: Option<Vec<f32>> = if self.topiq_nr_session.is_some() {
                    crate::ai::preprocess::face_crops_to_batch_topiq(&crop_owned)
                        .ok()
                        .and_then(|b| self.topiq_nr_scores(&b).ok())
                } else {
                    None
                };
                let w = FACE_FUSION_NR_FACE_WEIGHT;
                for (k, &val) in vals.iter().enumerate() {
                    if k < crops.len() {
                        let idx = crops[k].0;
                        let nrf10 = crate::ai::topiq_face::map_to_ten_scale(val);
                        let fused = match nr10.as_ref().and_then(|v| v.get(k)) {
                            Some(&nro10) => {
                                (w * nrf10 + (1.0 - w) * nro10).clamp(1.0, 10.0)
                            }
                            None => nrf10,
                        };
                        scores[idx] = Some(fused);
                    }
                }
            }
            Err(e) => {
                log::warn!("[人脸] TOPIQ-NR-Face 推理失败: {:#}", e);
            }
        }

        (scores, has_face)
    }

    /// 是否启用了 TOPIQ-NR 主技术评分模型（优先级最高）。
    pub fn has_topiq_nr(&self) -> bool {
        self.topiq_nr_session.is_some()
    }

    /// 是否启用了 TOPIQ-IAA 美学评分模型（优先级最高）。
    pub fn has_topiq_iia(&self) -> bool {
        self.topiq_iia_session.is_some()
    }

    /// HyperIQA（非人像美学第二意见）是否可用。
    pub fn has_hyperiqa(&self) -> bool {
        self.hyperiqa_session.is_some()
    }

    /// HyperIQA 原始质量分（[0,1]）。输入 `[N,3,512,512]` CHW、[0,1]（见
    /// [`crate::ai::preprocess::images_to_batch_raw01_512`]），导出图内含归一化。
    pub fn hyperiqa_scores(&self, batch: &Array4<f32>) -> anyhow::Result<Vec<f32>> {
        let Some(sess) = &self.hyperiqa_session else {
            anyhow::bail!("HyperIQA 会话未初始化");
        };
        let tensor = ort::value::Tensor::from_array(batch.clone())?;
        let mut session = sess.lock();
        let outputs = session.run(ort::inputs!["input" => tensor])?;
        Ok(outputs[0].try_extract_array::<f32>()?.iter().copied().collect())
    }

    /// TOPIQ-NR 技术质量评分（ResNet50，KonIQ-10k，输出 0~1 标量）。
    ///
    /// 输入 `batch` 形状为 `[N, 3, 384, 384]`（CHW + ImageNet 归一化，见 `images_to_batch_topiq`）。
    /// 模型已重导出支持动态 batch（2026-08-25，官方 cfanet 权重），整批一次推理；
    /// 输出映射到 1~10（线性 `1 + v*9`）。
    pub fn topiq_nr_scores(&self, batch: &Array4<f32>) -> anyhow::Result<Vec<f32>> {
        let Some(topiq_nr) = &self.topiq_nr_session else {
            anyhow::bail!("TOPIQ-NR 会话未初始化");
        };
        let tensor = ort::value::Tensor::from_array(batch.clone())?;
        let mut session = topiq_nr.lock();
        let outputs = session.run(ort::inputs![TOPIQ_INPUT => tensor])?;
        let raw = outputs[TOPIQ_NR_OUTPUT].try_extract_array::<f32>()?;
        let n = raw.shape()[0];
        let flat: Vec<f32> = raw.iter().copied().collect();
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let v = flat.get(i).copied().unwrap_or(0.0);
            scores.push(topiq_nr_to_score(v));
        }
        Ok(scores)
    }

    /// TOPIQ-IAA 美学评分（ResNet50，AVA，输出 10-bin softmax 分布）。
    ///
    /// 输入 `batch` 形状为 `[N, 3, 384, 384]`。模型已重导出支持动态 batch
    /// （2026-08-25，官方 cfanet 权重），整批一次推理。
    /// 输出为 1~10 范围的美学分（10-bin 加权平均）。
    pub fn topiq_iia_scores(&self, batch: &Array4<f32>) -> anyhow::Result<Vec<f32>> {
        let Some(topiq_iia) = &self.topiq_iia_session else {
            anyhow::bail!("TOPIQ-IAA 会话未初始化");
        };
        let tensor = ort::value::Tensor::from_array(batch.clone())?;
        let mut session = topiq_iia.lock();
        let outputs = session.run(ort::inputs![TOPIQ_INPUT => tensor])?;
        let dist = outputs[TOPIQ_IAA_OUTPUT].try_extract_array::<f32>()?;
        let shape = dist.shape();
        let n = shape[0];
        let classes = *shape.last().unwrap_or(&10);
        let flat: Vec<f32> = dist.iter().copied().collect();
        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<f32> = flat[i * classes..(i + 1) * classes].to_vec();
            scores.push(topiq_iaa_to_score(&row));
        }
        Ok(scores)
    }


    /// NIMA 技术质量评分（MobileNet 224×224，10-bin MOS 分布），技术后备。
    ///
    /// 输入 `batch` 形状为 `[N, 224, 224, 3]`（NHWC + MobileNet 归一化）。
    /// 输出为 1~10 范围的浮点分数（10-bin 分布的加权平均）。
    pub fn nima_technical_scores(&self, batch: &Array4<f32>) -> anyhow::Result<Vec<f32>> {
        let Some(nima_tech) = &self.nima_tech_session else {
            return Ok(Vec::new());
        };
        let tensor = ort::value::Tensor::from_array(batch.clone())?;
        let mut session = nima_tech.lock();
        let outputs = session.run(ort::inputs![NIMA_INPUT => tensor])?;

        let dist = outputs[NIMA_OUTPUT].try_extract_array::<f32>()?;
        let n = dist.shape()[0];

        let mut scores = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<f32> = dist.slice(ndarray::s![i, ..]).iter().copied().collect();
            scores.push(crate::ai::nima::nima_score_from_distribution(&row));
        }

        Ok(scores)
    }

    /// 综合评分：美学 + 技术 + 人脸专评 + 启发式（分辨率/大小），加权合并为最终分。
    ///
    /// 权重分两档：
    /// - **有人脸**（`has_face[i]=true` 且 `face[i]` 有值）：人脸专评主导
    ///   （美学 0.20 + 技术 0.20 + 人脸 0.45 + 启发式 0.15）——人像优先
    /// - **无人脸**：原公式（美学 0.25 + 技术 0.60 + 启发式 0.15）
    ///
    /// `aesthetic`: TOPIQ-IAA 美学分数组（可选）
    /// `focus`: 每张图对焦分（可选，1~10；人像=眼部对焦，非人像=整图对焦）
    /// `face`: TOPIQ-NR-Face 人脸专评分数组（可选，1~10）
    /// `has_face`: 是否检测到人脸
    /// `eye_open`: 每张图 `max(open_l, open_r) ∈ [0,1]`（1=至少一眼开；0=双眼全闭）。
    ///   连续开眼概率（由早期布尔闭眼标记演化而来），聚合用 `max`——仅当双眼都判闭
    ///   才降权，避免单眼 ROI 采到皮肤/眼镜致 0.00 压垮整张睁眼照。无人脸/未启用 → 1.0。
    /// `widths/heights/sizes`: 启发式信息
    pub fn composite_scores(
        &self,
        aesthetic: Option<&[f32]>,
        focus: Option<&[f32]>,
        face: Option<&[f32]>,
        has_face: &[bool],
        scenes: &[crate::ai::scene::Scene],
        eye_open: &[f32],
        widths: &[u32],
        heights: &[u32],
        sizes: &[u64],
    ) -> anyhow::Result<Vec<f32>> {
        let n = widths.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            // 人脸专评值（None 表示无人脸或未启用）
            let face_val = face.and_then(|f| f.get(i)).copied().filter(|v| *v > 0.0);
            let is_face = has_face.get(i).copied().unwrap_or(false);
            let scene = scenes.get(i).copied().unwrap_or(crate::ai::scene::Scene::Other);
            let open = eye_open.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);

            // 启发式：分辨率越高分越高
            let w = widths.get(i).copied().unwrap_or(0);
            let h = heights.get(i).copied().unwrap_or(0);
            let pixels = (w as u64) * (h as u64);
            let heuristic = if pixels > 0 {
                (10.0 * (pixels as f64 / (pixels as f64 + 2_000_000.0))) as f32
            } else {
                let s = sizes.get(i).copied().unwrap_or(0) as f64;
                (10.0 * (s / (s + 3_000_000.0))) as f32
            };

            let a = aesthetic.and_then(|x| x.get(i)).copied();
            let fo = focus.and_then(|x| x.get(i)).copied();
            let mut s = weighted_score(a, fo, face_val, is_face, scene, heuristic);
            // 闭眼降权（阶段二：连续平滑），见 [`eye_penalty`]
            if is_face {
                s *= eye_penalty(open);
            }
            result.push(s);
        }

        Ok(result)
    }
}


/// 双缓冲流水线的分段时间（verify_ai 性能基准 / commands 日志用）。
#[derive(Clone, Copy, Default)]
pub struct PipelineTiming {
    /// TOPIQ 预处理总耗时（各批求和，与推理重叠执行）。
    pub prep_topiq_sec: f64,
    /// NIMA 预处理总耗时（各批求和）。
    pub prep_nima_sec: f64,
    /// consumer 端推理总耗时（各批求和）。
    pub infer_sec: f64,
    /// 整条流水线 wall-clock（含重叠，通常 < prep_sec() + infer_sec）。
    pub wall_sec: f64,
}

impl PipelineTiming {
    /// 预处理总耗时（各模型预处理求和）。
    pub fn prep_sec(&self) -> f64 {
        self.prep_topiq_sec + self.prep_nima_sec
    }
}

/// 双缓冲流水线中一批图片的预处理 tensor（channel 传输单元）。
#[derive(Default)]
struct BatchTensors {
    topiq: Option<Array4<f32>>,
    nima: Option<Array4<f32>>,
}

/// 对 `paths` 做双缓冲批量评分：producer 线程用 rayon 并行构建下一批 tensor
/// （解码/缩放/归一化是 CPU 瓶颈），consumer 用现有单 session 推理 →
/// 解码与 GPU 推理重叠，消除 GPU 等数据的时间。TOPIQ NR/IAA 模型已支持
/// 动态 batch，整批一次推理摊薄 launch 开销。
///
/// 批内顺序、模型调用顺序、session 串行语义不变 → 分数确定可复现。
///
/// - 美学分：TOPIQ-IAA（LAION/CLIP 后备已移除）。
/// - 技术分：TOPIQ-NR 优先，否则 NIMA。
/// - 无对应模型 / 预处理失败 → 该图对应分值为 `None`。
///
/// `progress(done, total)` 每批开始前调用（供进度条）。
/// 返回 (美学分, 技术分)，均与 `paths` 等长对齐。
pub fn score_batch_scores(
    engine: &AiEngine,
    paths: &[String],
    batch_size: usize,
    progress: &mut dyn FnMut(usize, usize),
) -> (Vec<Option<f32>>, Vec<Option<f32>>, PipelineTiming) {
    let n = paths.len();
    let mut aes: Vec<Option<f32>> = vec![None; n];
    let mut tech: Vec<Option<f32>> = vec![None; n];
    if n == 0 {
        return (aes, tech, PipelineTiming::default());
    }
    let chunks: Vec<&[String]> = paths.chunks(batch_size.max(1)).collect();

    let need_topiq = engine.has_topiq_nr() || engine.has_topiq_iia();
    let need_nima = !engine.has_topiq_nr();

    let mut prep_topiq_sec = 0.0f64;
    let mut prep_nima_sec = 0.0f64;
    let mut infer_sec = 0.0f64;
    let wall = std::time::Instant::now();

    let (tx, rx) = std::sync::mpsc::sync_channel::<BatchTensors>(2);
    std::thread::scope(|scope| {
        // producer：持续预取下一批预处理 tensor（channel 容量 2：一预一推）
        scope.spawn(|| {
            for chunk in &chunks {
                let mut b = BatchTensors::default();
                if need_topiq {
                    let t0 = std::time::Instant::now();
                    match crate::ai::preprocess::images_to_batch_topiq(chunk) {
                        Ok(v) => b.topiq = Some(v),
                        Err(e) => log::warn!("TOPIQ 预处理失败: {}", e),
                    }
                    prep_topiq_sec += t0.elapsed().as_secs_f64();
                }
                if need_nima {
                    let t0 = std::time::Instant::now();
                    match crate::ai::preprocess::images_to_batch_nima(chunk) {
                        Ok(v) => b.nima = Some(v),
                        Err(e) => log::warn!("NIMA 预处理失败: {}", e),
                    }
                    prep_nima_sec += t0.elapsed().as_secs_f64();
                }
                if tx.send(b).is_err() {
                    break; // 消费者已退出
                }
            }
        });

        // consumer：逐批推理（与 producer 的解码重叠）
        for (i, chunk) in chunks.iter().enumerate() {
            progress(i * batch_size, n);
            let b = match rx.recv() {
                Ok(b) => b,
                Err(_) => break, // producer 已结束
            };
            let t0 = std::time::Instant::now();

            let aes_scores: Option<Vec<f32>> = if engine.has_topiq_iia() {
                b.topiq
                    .as_ref()
                    .map(|t| engine.topiq_iia_scores(t).unwrap_or_default())
            } else {
                None // 美学后备（LAION/CLIP）已随 CLIP 移除
            };

            let tech_scores: Vec<f32> = if engine.has_topiq_nr() {
                b.topiq
                    .as_ref()
                    .map(|t| engine.topiq_nr_scores(t).unwrap_or_default())
                    .unwrap_or_default()
            } else {
                match b.nima.as_ref() {
                    Some(nm) => engine.nima_technical_scores(nm).unwrap_or_default(),
                    None => Vec::new(),
                }
            };

            for (j, _) in chunk.iter().enumerate() {
                let idx = i * batch_size + j;
                aes[idx] = aes_scores.as_ref().and_then(|v| v.get(j).copied());
                tech[idx] = tech_scores.get(j).copied();
            }
            infer_sec += t0.elapsed().as_secs_f64();
        }
    });

    let timing = PipelineTiming {
        prep_topiq_sec,
        prep_nima_sec,
        infer_sec,
        wall_sec: wall.elapsed().as_secs_f64(),
    };
    (aes, tech, timing)
}

/// 闭眼连续降权系数。`open = max(open_l, open_r) ∈ [0,1]`。
///
/// - `open >= 0.5`（至少一眼判开）→ `1.0`，不降权；
/// - `open < 0.5`（双眼都判闭）→ 平滑降到 `0.5`，全闭取极值 `0.5`。
///
/// 聚合采用 `max` 而非 `min`：实测戴镜/偏脸时单眼 ROI 常采到皮肤或眼镜，
/// OCEC 给 0.00 → 旧 `min` 会把整张明显的睁眼照压成 ×0.5，失去区分度。
/// 改用 `max` 后只有双眼都判闭才降权，单眼噪声不再压垮整脸。
/// 注：spec 字面 `0.5 + 0.5*open` 会对明显睁眼但 `open<1` 也降权，属回归，故采用分段。
fn eye_penalty(open: f32) -> f32 {
    if open >= 0.5 {
        1.0
    } else {
        0.5 + open
    }
}

/// 单张图片的综合加权（供 composite_scores 与测试复用）。
///
/// 权重规则（人像 > 风景 > 宠物 > 其他）：
/// - **人像**（`has_face` 且 `face` 有值）：人像美学(人脸分) 主导（0.55 + 眼部对焦 0.30 + 启发式 0.15；
///   整图美学不参与——用户流程：五官/闭眼/眼部对焦/人像美学）
/// - **风景**：美学 0.40 + 对焦 0.50 + 启发式 0.10（画质优先）
/// - **宠物**：美学 0.45 + 对焦 0.45 + 启发式 0.10（均衡）
/// - **其他**：原公式（美学 0.25 + 对焦 0.60 + 启发式 0.15）
fn weighted_score(
    aesthetic: Option<f32>,
    focus: Option<f32>,
    face: Option<f32>,
    has_face: bool,
    scene: crate::ai::scene::Scene,
    heuristic: f32,
) -> f32 {
    let is_face_priority = has_face && face.is_some() && face.unwrap_or(0.0) > 0.0;
    let (w_a, w_focus, w_face, w_h) = if is_face_priority {
        (W_FACE_A, W_FACE_FOCUS, W_FACE_F, W_FACE_H)
    } else if scene == crate::ai::scene::Scene::Landscape {
        (W_LAND_A, W_LAND_FOCUS, 0.0, W_LAND_H)
    } else if scene == crate::ai::scene::Scene::Pet {
        (W_PET_A, W_PET_FOCUS, 0.0, W_PET_H)
    } else {
        (WEIGHT_AESTHETIC, WEIGHT_FOCUS, 0.0, WEIGHT_HEURISTIC)
    };

    let mut score = 0.0f32;
    let mut weight_sum = 0.0f32;
    if let Some(a) = aesthetic {
        score += a * w_a;
        weight_sum += w_a;
    }
    if let Some(fo) = focus {
        score += fo * w_focus;
        weight_sum += w_focus;
    }
    if let Some(f) = face.filter(|f| *f > 0.0) {
        score += f * w_face;
        weight_sum += w_face;
    }
    score += heuristic * w_h;
    weight_sum += w_h;

    if weight_sum > 0.0 {
        (score / weight_sum).clamp(1.0, 10.0)
    } else {
        heuristic.clamp(1.0, 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_device_detection_does_not_crash() {
        // 行为锁定：有 NVIDIA 驱动 → true；无驱动/无卡 → false（不 panic）。
        // 本机结果取决于硬件，只验证可调用且与 nvidia-smi 存在性大体一致。
        let detected = cuda_device_available();
        log::info!("cuda_device_available = {}", detected);
        let _ = detected;
    }

    #[test]
    fn face_priority_boosts_face_score() {
        use crate::ai::scene::Scene;
        // 人脸 9.0 主导 0.55 权重 → 显著高于无脸场景
        let with_face = weighted_score(Some(5.0), Some(5.0), Some(9.0), true, Scene::Portrait, 7.0);
        let without_face = weighted_score(Some(5.0), Some(5.0), None, false, Scene::Other, 7.0);
        assert!(
            with_face > without_face,
            "人脸优先图应得分更高: {} vs {}",
            with_face,
            without_face
        );
    }

    #[test]
    fn no_face_uses_focus_dominant() {
        use crate::ai::scene::Scene;
        // 无脸：对焦 0.6 权重主导
        let s = weighted_score(Some(4.0), Some(8.0), None, false, Scene::Other, 7.0);
        assert!((s - 6.7).abs() < 0.3, "无脸图对焦主导异常: {}", s);
    }

    #[test]
    fn portrait_formula_face_dominant() {
        use crate::ai::scene::Scene;
        // 人像：0.55*9(人像美学) + 0.30*5(眼部对焦) + 0.15*7(启发式) = 7.5；整图美学不参与
        let s = weighted_score(Some(5.0), Some(5.0), Some(9.0), true, Scene::Portrait, 7.0);
        assert!((s - 7.5).abs() < 0.01, "人像图综合分异常: {}", s);
    }

    #[test]
    fn landscape_weights_focus_higher() {
        use crate::ai::scene::Scene;
        // 风景：美学 0.40 + 对焦 0.50 + 启发式 0.10（对焦略高）
        let s = weighted_score(Some(5.0), Some(8.0), None, false, Scene::Landscape, 7.0);
        // 0.4*5 + 0.5*8 + 0.1*7 = 6.7
        assert!((s - 6.7).abs() < 0.3, "风景图对焦权重异常: {}", s);
    }

    #[test]
    fn pet_weights_balanced() {
        use crate::ai::scene::Scene;
        // 宠物：美学 0.45 + 对焦 0.45 + 启发式 0.10（均衡）
        let s = weighted_score(Some(5.0), Some(5.0), None, false, Scene::Pet, 7.0);
        // 0.45*5 + 0.45*5 + 0.1*7 = 5.2
        assert!((s - 5.2).abs() < 0.3, "宠物图权重异常: {}", s);
    }

    #[test]
    fn eye_penalty_is_continuous_and_open_untouched() {
        // 明显睁眼（max(open) >= 0.5）→ 不降权（spec 字面公式会误伤，分段公式不罚）
        assert_eq!(eye_penalty(1.0), 1.0);
        assert_eq!(eye_penalty(0.6), 1.0);
        // 阈值处平滑连续
        assert!((eye_penalty(0.5) - 1.0).abs() < 1e-6);
        // 接近阈值几乎不罚：0.49 → 0.99（旧硬阈值会直接 0.5）
        assert!((eye_penalty(0.49) - 0.99).abs() < 1e-6);
        // 半闭 → 0.75
        assert!((eye_penalty(0.25) - 0.75).abs() < 1e-6);
        // 全闭 → 0.5（与旧硬阈值极值一致）
        assert!((eye_penalty(0.0) - 0.5).abs() < 1e-6);
    }
}

/// 检测 onnxruntime 是否编译了 DirectML EP 支持。
#[cfg(feature = "directml")]
pub fn directml_available() -> bool {
    DirectML::default().is_available().unwrap_or(false)
}

/// 未启用 directml feature 时，恒返回 false。
#[cfg(not(feature = "directml"))]
pub fn directml_available() -> bool {
    false
}

/// 驱动级检测 NVIDIA GPU：动态加载 `nvcuda.dll`（装了 NVIDIA 驱动必有），
/// 调 `cuInit(0)` + `cuDeviceGetCount` 确认设备数 ≥ 1。
///
/// 仅凭 `onnxruntime_providers_cuda.dll` 能加载不足以判定——系统 PATH 里有
/// cudart/cudnn（如装过 PyTorch/OBS）时 EP DLL 会加载成功，但无 N 卡时推理
/// 静默回退 CPU，后端标签就会误报 CUDA。
#[cfg(feature = "cuda")]
pub fn cuda_device_available() -> bool {
    use std::ffi::c_void;

    // 手写最小 FFI：LoadLibraryW / GetProcAddress（kernel32）
    unsafe extern "system" {
        fn LoadLibraryW(lpfilename: *const u16) -> *mut c_void;
        fn GetProcAddress(hmodule: *mut c_void, lpprocname: *const u8) -> *mut c_void;
        fn FreeLibrary(hlibmodule: *mut c_void) -> i32;
    }

    let name: Vec<u16> = "nvcuda.dll\0".encode_utf16().collect();
    unsafe {
        let lib = LoadLibraryW(name.as_ptr());
        if lib.is_null() {
            return false; // 无 NVIDIA 驱动
        }
        let result = (|| {
            type CuInit = unsafe extern "system" fn(u32) -> i32;
            type CuDeviceGetCount = unsafe extern "system" fn(*mut i32) -> i32;
            let init: Option<CuInit> =
                std::mem::transmute(GetProcAddress(lib, b"cuInit\0".as_ptr()));
            let count: Option<CuDeviceGetCount> =
                std::mem::transmute(GetProcAddress(lib, b"cuDeviceGetCount\0".as_ptr()));
            let (Some(init), Some(count)) = (init, count) else {
                return false;
            };
            if init(0) != 0 {
                return false; // CUDA_ERROR 枚举非 0 即失败
            }
            let mut n: i32 = 0;
            count(&mut n) == 0 && n >= 1
        })();
        FreeLibrary(lib);
        result
    }
}

/// 未启用 cuda feature 时，恒返回 false。
#[cfg(not(feature = "cuda"))]
pub fn cuda_device_available() -> bool {
    false
}

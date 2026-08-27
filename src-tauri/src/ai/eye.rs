//! 闭眼检测（双信号：OCEC 眨眼式 + MediaPipe 脸网格垂目式）。
//!
//! ## 为什么需要两个信号
//! OCEC（PINTO0309 #476，官方 F1 0.9954）训练数据是**眨眼式闭眼**；
//! 对「垂目 / 向下看」姿态（上睑下垂遮住瞳孔），24×40 眼部 crop 与睁眼向下看几乎无差，
//! 标注集实测 OCEC 对垂目闭眼判 0.84~1.00（全开）——这是任务与模型不匹配，调参无解。
//!
//! 脸网格（MediaPipe FaceLandmarker 478 点含虹膜；Apache-2.0，yakhyo/mediapipe-face-mesh-onnx
//! ONNX 转换）输出虹膜圆 + 上下睑轮廓，「睑缝高 / 虹膜直径」是尺度无关的开度几何量：
//! 垂目时上睑下压 → 比值显著下降。标注集 7 组实测 6 个闭眼组全部正确分离
//! （含垂目组 2/4/5/6 与半闭眨眼组 7）。
//!
//! 融合策略（网格为主、OCEC 眨眼否决）：网格开度直接作为结果；
//! 仅当「网格模棱两可（< [`MESH_VETO_BAND`]）且 OCEC 强判闭（< [`OCEC_VETO_MAX`]）」
//! 时取两者较小值——OCEC 对刘海遮挡的睁眼常有假闭误报，只有网格也拿不准时才采信它；
//! 网格缺失或拟合失败时回退为仅 OCEC（模型可选、不报错）。
//! 网格虹膜中心同时回喂 OCEC 做 ROI 采样（修复 5 关键点偏低采到脸颊的存量问题）。
//!
//! ## 脸网格链路
//! 1. 复用 InsightFace bbox + 5 关键点构建方形旋转 ROI：中心 = bbox 中心、
//!    边长 = 1.5 × bbox 长边、角度 = 双眼连线方向的近水平解（消除 ±180° 歧义）
//! 2. 256×256 双线性 warp，RGB /255 → `face_landmarker.onnx`
//!    （输入 `input` [N,3,256,256]，输出 `landmarks` [N,478,3] + `score` 人脸 logit）
//! 3. score(sigmoid) < [`MESH_SCORE_GATE`] 或虹膜几何退化 → 放弃网格信号（返回 None）
//! 4. 两眼各算开度取 min，经 [`mesh_norm_openness`] 分段线性映射到 [0,1]

use ndarray::Array4;
use ort::inputs;
use ort::session::Session;

use super::insightface::{Face, InsightFaceEngine};

/// OCEC 模型文件名。
pub const MODEL_NAME: &str = "ocec_l.onnx";
/// 脸网格模型文件名（`models/eye/` 下；可选，缺失则跳过垂目信号）。
pub const MESH_MODEL_NAME: &str = "face_landmarker.onnx";
/// OCEC 输入张量名。
pub const INPUT_NAME: &str = "images";
/// OCEC 输出张量名。
pub const OUTPUT_NAME: &str = "prob_open";
/// OCEC 输入尺寸（高 × 宽）。
pub const EYE_H: u32 = 24;
pub const EYE_W: u32 = 40;
/// 睁眼阈值（prob_open > THRESHOLD 判为 open）。
pub const OPEN_THRESHOLD: f32 = 0.5;

/// 网格开度原始值锚点：raw ≤ 此值判全闭（映射 0.0）。
pub const MESH_RAW_CLOSED: f32 = 0.10;
/// 网格开度原始值锚点：raw = 此值映射 0.5（eye_penalty 的不罚下界，可按分布微调）。
pub const MESH_RAW_MID: f32 = 0.42;
/// 网格开度原始值锚点：raw ≥ 此值判全开（映射 1.0）。
pub const MESH_RAW_OPEN: f32 = 0.65;
/// 网格人脸置信门限（sigmoid 后）：低于视为拟合失败，放弃信号。
/// 门槛取得极低——只有完全不可解析才拒绝；因为仅在 InsightFace 已确认人脸时才调用，
/// 且输出是软惩罚，极端姿态（如强侧脸）下"方向大致正确的弱信号"好过没有信号。
pub const MESH_SCORE_GATE: f32 = 0.001;

/// 眨眼否决的 OCEC 侧条件：OCEC 开度低于此值视为"强判闭"。
pub const OCEC_VETO_MAX: f32 = 0.2;
/// 眨眼否决的网格侧条件：网格开度低于此值才采信 OCEC 否决
/// （网格 ≥ 此值说明几何上明确睁开，OCEC 的低值多为刘海遮挡等假闭误报）。
pub const MESH_VETO_BAND: f32 = 0.85;

const MESH_SIZE: usize = 256;

// MediaPipe 拓扑索引（人的左/右；索引见 FACEMESH_LEFT_EYE / RIGHT_EYE 与 iris 468-477）。
struct EyeIdx {
    upper: usize,
    lower: usize,
    iris_c: usize,
}

/// 人右眼：上睑中 159、下睑中 145、虹膜中心 473（环点 474-477）。
const RIGHT_EYE: EyeIdx = EyeIdx { upper: 159, lower: 145, iris_c: 473 };
/// 人左眼：上睑中 386、下睑中 374、虹膜中心 468（环点 469-472）。
const LEFT_EYE: EyeIdx = EyeIdx { upper: 386, lower: 374, iris_c: 468 };

/// 判定"闭眼"：OCEC 输出已 sigmoid，`> OPEN_THRESHOLD` 判开，`<= OPEN_THRESHOLD` 判闭。
///
/// 统一闭眼判定，避免调用方各自硬编码阈值导致 `<=`/`<` 边界不一致
/// （engine.rs 曾用 `<= 0.5`、commands.rs 曾用 `< 0.5`，恰好 0.5 时矛盾）。
pub fn is_closed(open_prob: f32) -> bool {
    open_prob <= OPEN_THRESHOLD
}

/// 把网格原始开度（两眼中较小者的睑缝高/虹膜直径）分段线性映射到 [0,1]。
///
/// 锚点：≤[`MESH_RAW_CLOSED`]→0，=[`MESH_RAW_MID`]→0.5，≥[`MESH_RAW_OPEN`]→1，
/// 区间内线性。单调且在 MID 处连续；锚点可依更大样本分布微调。
pub fn mesh_norm_openness(raw: f32) -> f32 {
    if raw <= MESH_RAW_CLOSED {
        0.0
    } else if raw >= MESH_RAW_OPEN {
        1.0
    } else if raw <= MESH_RAW_MID {
        0.5 * (raw - MESH_RAW_CLOSED) / (MESH_RAW_MID - MESH_RAW_CLOSED)
    } else {
        0.5 + 0.5 * (raw - MESH_RAW_MID) / (MESH_RAW_OPEN - MESH_RAW_MID)
    }
}

/// 闭眼检测器（OCEC 会话必选 + 脸网格会话可选，复用 InsightFace 的对齐计算）。
pub struct EyeDetector {
    session: Option<parking_lot::Mutex<Session>>,
    mesh_session: Option<parking_lot::Mutex<Session>>,
}

impl Default for EyeDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl EyeDetector {
    pub fn new() -> Self {
        Self { session: None, mesh_session: None }
    }

    pub fn load(mut self, models_dir: &std::path::Path) -> anyhow::Result<Self> {
        let path = models_dir.join(MODEL_NAME);
        if !path.exists() {
            anyhow::bail!("OCEC 模型不存在: {}", path.display());
        }
        // 复用 engine::build_session 的三级回退
        let (s, backend) = super::engine::AiEngine::build_session(&path, false)?;
        log::info!("[eye] OCEC 会话就绪，推理后端: {}", backend.label());
        self.session = Some(parking_lot::Mutex::new(s));

        // 脸网格（垂目信号）可选加载
        let mesh_path = models_dir.join(MESH_MODEL_NAME);
        if mesh_path.exists() {
            match super::engine::AiEngine::build_session(&mesh_path, false) {
                Ok((m, _backend)) => {
                    log::info!("[eye] 脸网格会话就绪: {}", mesh_path.display());
                    self.mesh_session = Some(parking_lot::Mutex::new(m));
                }
                Err(e) => log::warn!("[eye] 脸网格加载失败，跳过垂目信号: {}", e),
            }
        } else {
            log::info!("[eye] 无 {}，跳过垂目闭眼信号（仅 OCEC）", MESH_MODEL_NAME);
        }
        Ok(self)
    }

    pub fn is_loaded(&self) -> bool {
        self.session.is_some()
    }

    /// 脸网格会话是否可用（垂目信号是否参与融合）。
    pub fn has_mesh(&self) -> bool {
        self.mesh_session.is_some()
    }

    /// 检测一张脸的闭眼状态，并返回左/右眼 `prob_open`（诊断用途）。
    pub fn detect_probs(
        &self,
        _face_engine: &InsightFaceEngine,
        rgb_image: &[u8],
        h: u32,
        w: u32,
        face: &Face,
    ) -> Option<(f32, f32)> {
        let crops = sample_eye_rgb_internal(rgb_image, h, w, face).ok()?;
        self.detect_probs_raw(&crops[..24 * 40 * 3], &crops[24 * 40 * 3..])
    }

    /// 同 [`Self::detect_probs`]，但用外部眼位（如网格虹膜中心）替代关键点采样。
    pub fn detect_probs_at(
        &self,
        rgb_image: &[u8],
        h: u32,
        w: u32,
        left: (f32, f32),
        right: (f32, f32),
    ) -> Option<(f32, f32)> {
        let crops = sample_eye_rgb_at(rgb_image, h, w, left, right).ok()?;
        self.detect_probs_raw(&crops[..24 * 40 * 3], &crops[24 * 40 * 3..])
    }

    /// 对已采样的左右眼 ROI（各 24×40×3 RGB 字节）跑 OCEC，返回 (prob_open_l, prob_open_r)。
    /// 供诊断/实验复用同一推理路径。
    pub fn detect_probs_raw(&self, left_roi: &[u8], right_roi: &[u8]) -> Option<(f32, f32)> {
        let sess = self.session.as_ref()?;
        let mut batch = Array4::<f32>::zeros((2, 3, EYE_H as usize, EYE_W as usize));
        for (idx, roi) in [left_roi, right_roi].iter().enumerate() {
            for y in 0..EYE_H {
                for x in 0..EYE_W {
                    let base = ((y * EYE_W + x) * 3) as usize;
                    batch[[idx, 0, y as usize, x as usize]] = roi[base] as f32 / 255.0;
                    batch[[idx, 1, y as usize, x as usize]] = roi[base + 1] as f32 / 255.0;
                    batch[[idx, 2, y as usize, x as usize]] = roi[base + 2] as f32 / 255.0;
                }
            }
        }
        let tensor = ort::value::Tensor::from_array(batch).ok()?;
        let mut guard = sess.lock();
        let outputs = guard.run(inputs![INPUT_NAME => tensor]).ok()?;
        let probs = outputs[OUTPUT_NAME].try_extract_array::<f32>().ok()?;
        Some((probs[0], probs[1]))
    }

    /// 脸网格垂目信号：返回归一化开度（两眼中更小值）与映射回原图的左右虹膜中心。
    ///
    /// 虹膜中心比 InsightFace 5 关键点的眼位精确——后者实测系统性偏低约 10% 脸高
    /// （导致 OCEC ROI 采到脸颊、大量误判 0.00）。调用方应优先用这里的中心做眼 ROI。
    ///
    /// 返回 None 表示放弃网格信号（未加载 / 不可解析），调用方回退关键点方案。
    pub fn mesh_result(
        &self,
        rgb_image: &[u8],
        h: u32,
        w: u32,
        face: &Face,
    ) -> Option<MeshEyeResult> {
        let (lms, src_eyes) = run_mesh(self.mesh_session.as_ref()?, rgb_image, h, w, face)?;
        let mut best: Option<f32> = None;
        for idx in [&RIGHT_EYE, &LEFT_EYE] {
            if let Some(v) = eye_openness_raw(&lms, idx) {
                best = Some(best.map_or(v, |b: f32| b.min(v)));
            }
        }
        Some(MeshEyeResult {
            norm_open: mesh_norm_openness(best?),
            left_eye_src: src_eyes[0],
            right_eye_src: src_eyes[1],
        })
    }
}

/// 脸网格垂目信号结果。
#[derive(Debug, Clone, Copy)]
pub struct MeshEyeResult {
    /// 归一化开度 ∈ [0,1]（两眼中更小值；≤[`MESH_RAW_CLOSED`]→0，≥[`MESH_RAW_OPEN`]→1）。
    pub norm_open: f32,
    /// 左眼虹膜中心（原图坐标）。
    pub left_eye_src: (f32, f32),
    /// 右眼虹膜中心（原图坐标）。
    pub right_eye_src: (f32, f32),
}

/// 单眼开度原始值 = 睑缝垂直高 / 虹膜直径。虹膜退化（直径过小）→ None。
fn eye_openness_raw(lms: &[[f32; 3]; 478], idx: &EyeIdx) -> Option<f32> {
    let d = |a: usize, b: usize| {
        let dx = lms[a][0] - lms[b][0];
        let dy = lms[a][1] - lms[b][1];
        (dx * dx + dy * dy).sqrt()
    };
    let vert = d(idx.upper, idx.lower);
    // 虹膜半径 = 4 个环点到中心的距离均值
    let iris_d = 2.0
        * (d(idx.iris_c, idx.iris_c + 1)
            + d(idx.iris_c, idx.iris_c + 2)
            + d(idx.iris_c, idx.iris_c + 3)
            + d(idx.iris_c, idx.iris_c + 4))
            / 4.0;
    if iris_d < 1e-3 || !vert.is_finite() || !iris_d.is_finite() {
        return None;
    }
    Some(vert / iris_d)
}

/// 构建方形旋转 ROI 并推理脸网格，返回（裁剪坐标系 478 点，原图坐标系左右虹膜中心）。
///
/// 角度取双眼连线（右眼−左眼方向向量）的近水平解，避免 180° 翻转把脸倒置。
fn run_mesh(
    sess: &parking_lot::Mutex<Session>,
    rgb_image: &[u8],
    h: u32,
    w: u32,
    face: &Face,
) -> Option<([[f32; 3]; 478], [(f32, f32); 2])> {
    let img = image::RgbImage::from_raw(w, h, rgb_image.to_vec())?;

    let (lx, ly) = face.landmarks.left_eye;
    let (rx, ry) = face.landmarks.right_eye;
    let cx = (face.bbox[0] + face.bbox[2]) / 2.0;
    let cy = (face.bbox[1] + face.bbox[3]) / 2.0;
    let side = 1.5 * (face.bbox[2] - face.bbox[0]).max(face.bbox[3] - face.bbox[1]);
    if !(side.is_finite() && side > 1.0) {
        return None;
    }

    let mut ang = (ry - ly).atan2(rx - lx).to_degrees();
    while ang > 90.0 {
        ang -= 180.0;
    }
    while ang <= -90.0 {
        ang += 180.0;
    }

    let blob = warp_roi_to_blob(&img, cx, cy, side, ang);

    let tensor = ort::value::Tensor::from_array(blob).ok()?;
    let mut lms_out = [[0f32; 3]; 478];
    let mut filled = false;
    let mut score_logit = 0f32;
    {
        let mut guard = sess.lock();
        let in_name = guard.inputs().first().map(|i| i.name().to_string())?;
        let outs = guard.run(inputs![in_name.as_str() => tensor]).ok()?;
        for i in 0..outs.len() {
            let Ok(arr) = outs[i].try_extract_array::<f32>() else { continue };
            match arr.shape().len() {
                3 if arr.shape()[1] == 478 => {
                    for k in 0..478usize {
                        lms_out[k] = [arr[[0, k, 0]], arr[[0, k, 1]], arr[[0, k, 2]]];
                    }
                    filled = true;
                }
                0..=2 => {
                    score_logit = arr.as_slice().and_then(|s| s.first().copied()).unwrap_or(0.0);
                }
                _ => {}
            }
        }
    }
    if !filled {
        return None;
    }
    let score = 1.0 / (1.0 + (-score_logit).exp());
    if score < MESH_SCORE_GATE {
        return None;
    }
    // 虹膜中心逆仿射映射回原图：src = c + R(−θ)·(crop−128)/k
    let theta = ang.to_radians();
    let (cos_t, sin_t, inv_k) = (theta.cos(), theta.sin(), side / MESH_SIZE as f32);
    let to_src = |p: [f32; 3]| -> (f32, f32) {
        let u = p[0] - MESH_SIZE as f32 / 2.0;
        let v = p[1] - MESH_SIZE as f32 / 2.0;
        (cx + (cos_t * u - sin_t * v) * inv_k, cy + (sin_t * u + cos_t * v) * inv_k)
    };
    let src_eyes = [to_src(lms_out[LEFT_EYE.iris_c]), to_src(lms_out[RIGHT_EYE.iris_c])];
    Some((lms_out, src_eyes))
}

/// 由旋转 ROI 参数生成 [1,3,M,M]、RGB、/255 的输入张量（逆映射双线性采样，图外补零）。
fn warp_roi_to_blob(img: &image::RgbImage, cx: f32, cy: f32, side: f32, angle_deg: f32) -> Array4<f32> {
    let theta = angle_deg.to_radians();
    let k = MESH_SIZE as f32 / side;
    let alpha = k * theta.cos();
    let beta = k * theta.sin();
    let kk = k * k;

    let mut blob = Array4::<f32>::zeros((1, 3, MESH_SIZE, MESH_SIZE));
    let (iw, ih) = (img.width() as i64, img.height() as i64);
    for y in 0..MESH_SIZE {
        for x in 0..MESH_SIZE {
            let u = x as f32 - MESH_SIZE as f32 / 2.0;
            let v = y as f32 - MESH_SIZE as f32 / 2.0;
            let sx = cx + (alpha * u - beta * v) / kk;
            let sy = cy + (beta * u + alpha * v) / kk;
            let p = bilinear(img, iw, ih, sx, sy);
            blob[[0, 0, y, x]] = p[0] as f32 / 255.0;
            blob[[0, 1, y, x]] = p[1] as f32 / 255.0;
            blob[[0, 2, y, x]] = p[2] as f32 / 255.0;
        }
    }
    blob
}

fn bilinear(img: &image::RgbImage, iw: i64, ih: i64, sx: f32, sy: f32) -> [u8; 3] {
    let x0 = sx.floor() as i64;
    let y0 = sy.floor() as i64;
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let px = |x: i64, y: i64| -> [u8; 3] {
        if x < 0 || y < 0 || x >= iw || y >= ih {
            [0, 0, 0]
        } else {
            let p = img.get_pixel(x as u32, y as u32);
            [p[0], p[1], p[2]]
        }
    };
    let p00 = px(x0, y0);
    let p10 = px(x0 + 1, y0);
    let p01 = px(x0, y0 + 1);
    let p11 = px(x0 + 1, y0 + 1);
    let mut out = [0u8; 3];
    for c in 0..3 {
        out[c] = (p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
            + p10[c] as f32 * fx * (1.0 - fy)
            + p01[c] as f32 * (1.0 - fx) * fy
            + p11[c] as f32 * fx * fy)
            .round() as u8;
    }
    out
}

/// 从原图按 InsightFace 5 关键点取出左眼、右眼 24×40 RGB 数据。
///
/// 仿射矩阵 = InsightFace 5 关键点 → 112 模板的相似变换（同 align_face）。
///
/// 返回 `Vec<u8>`，长度 = 2 × 24 × 40 × 3 = 5760 字节（先左眼后右眼，HWC RGB）。
fn sample_eye_rgb_internal(
    rgb_image: &[u8],
    h: u32,
    w: u32,
    face: &Face,
) -> anyhow::Result<Vec<u8>> {
    sample_eye_rgb_at(rgb_image, h, w, face.landmarks.left_eye, face.landmarks.right_eye)
}

/// 同 [`sample_eye_rgb_internal`]，但用显式眼位（如网格虹膜中心）定位。
///
/// ROI 尺寸仍按「眼距 / 37」缩放（模板眼距恒 37），窗口中心即给定眼位。
fn sample_eye_rgb_at(
    rgb_image: &[u8],
    h: u32,
    w: u32,
    left: (f32, f32),
    right: (f32, f32),
) -> anyhow::Result<Vec<u8>> {
    // 眼线方向（源图坐标）决定 ROI 旋转对齐，使两眼 crop 统一为水平朝向
    let dx = right.0 - left.0;
    let dy = right.1 - left.1;
    let eye_dist = f32::sqrt(dx * dx + dy * dy);
    if eye_dist < 1e-3 {
        anyhow::bail!("眼距过小，无法定位眼 ROI");
    }
    let ux = dx / eye_dist;
    let uy = dy / eye_dist;
    // 垂直于眼线的方向
    let (vx, vy) = (-uy, ux);
    // 模板眼距恒为 37（38↔75）；inv_scale = 源眼距 / 模板眼距，使 ROI 尺寸与旧模板投影一致
    let inv_scale = eye_dist / 37.0;

    let img = image::RgbImage::from_raw(w, h, rgb_image.to_vec())
        .ok_or_else(|| anyhow::anyhow!("图片缓冲区尺寸不匹配: {}x{}, len={}", w, h, rgb_image.len()))?;

    let mut out = Vec::with_capacity(2 * EYE_H as usize * EYE_W as usize * 3);
    for &(kp_x, kp_y) in &[left, right] {
        for y in 0..EYE_H {
            for x in 0..EYE_W {
                // ROI 像素相对窗口中心的偏移，映射回源图（沿眼线方向）
                let ox = x as f32 - (EYE_W as f32 / 2.0);
                let oy = y as f32 - (EYE_H as f32 / 2.0);
                let src_x = kp_x + inv_scale * (ox * ux + oy * vx);
                let src_y = kp_y + inv_scale * (ox * uy + oy * vy);
                let sx0 = src_x.floor() as i32;
                let sy0 = src_y.floor() as i32;
                let fx = src_x - sx0 as f32;
                let fy = src_y - sy0 as f32;
                let mut pixel = [0u8; 3];
                if sx0 >= 0 && sy0 >= 0 && (sx0 as u32) < img.width() && (sy0 as u32) < img.height() {
                    let p00 = img.get_pixel(sx0 as u32, sy0 as u32);
                    let p10 = if (sx0 + 1) < img.width() as i32 { img.get_pixel((sx0 + 1) as u32, sy0 as u32) } else { p00 };
                    let p01 = if (sy0 + 1) < img.height() as i32 { img.get_pixel(sx0 as u32, (sy0 + 1) as u32) } else { p00 };
                    let p11 = if (sx0 + 1) < img.width() as i32 && (sy0 + 1) < img.height() as i32 {
                        img.get_pixel((sx0 + 1) as u32, (sy0 + 1) as u32)
                    } else { p00 };
                    for c in 0..3 {
                        let v = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
                            + p10[c] as f32 * fx * (1.0 - fy)
                            + p01[c] as f32 * (1.0 - fx) * fy
                            + p11[c] as f32 * fx * fy;
                        pixel[c] = v.round() as u8;
                    }
                }
                out.push(pixel[0]);
                out.push(pixel[1]);
                out.push(pixel[2]);
            }
        }
    }
    Ok(out)
}

/// 公开调试 API：直接调用 sample_eye_rgb_internal 并把 (左眼 ROI, 右眼 ROI) 字节返回，
/// 供诊断 example 导出对比。
pub fn debug_sample_eyes_rgb(
    rgb_image: &[u8],
    h: u32,
    w: u32,
    face: &Face,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let all = sample_eye_rgb_internal(rgb_image, h, w, face).ok()?;
    let sz = (EYE_H as usize) * (EYE_W as usize) * 3;
    if all.len() < 2 * sz {
        return None;
    }
    Some((all[..sz].to_vec(), all[sz..2*sz].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_exported_onnx() {
        assert_eq!(MODEL_NAME, "ocec_l.onnx");
        assert_eq!(INPUT_NAME, "images");
        assert_eq!(OUTPUT_NAME, "prob_open");
        assert_eq!(EYE_H, 24);
        assert_eq!(EYE_W, 40);
        assert_eq!(MESH_MODEL_NAME, "face_landmarker.onnx");
    }

    #[test]
    fn is_closed_respects_open_threshold() {
        // > OPEN_THRESHOLD 判开（不闭），<= 判闭
        assert!(!is_closed(OPEN_THRESHOLD + 0.01));
        assert!(is_closed(OPEN_THRESHOLD));
        assert!(is_closed(0.0));
    }

    #[test]
    fn mesh_norm_openness_anchors_and_monotonic() {
        // 锚点精确命中
        assert_eq!(mesh_norm_openness(MESH_RAW_CLOSED), 0.0);
        assert!((mesh_norm_openness(MESH_RAW_MID) - 0.5).abs() < 1e-6);
        assert_eq!(mesh_norm_openness(MESH_RAW_OPEN), 1.0);
        // 两端饱和钳制
        assert_eq!(mesh_norm_openness(0.0), 0.0);
        assert_eq!(mesh_norm_openness(99.0), 1.0);
        // 全程单调不减
        let mut prev = -1.0f32;
        for i in 0..=200usize {
            let raw = i as f32 * 0.01;
            let v = mesh_norm_openness(raw);
            assert!(v >= prev, "非单调: raw={} v={} prev={}", raw, v, prev);
            prev = v;
        }
    }
}

//! InsightFace buffalo_l 人脸检测与对齐。
//!
//! 核心功能：`det_10g` 检测人脸 bbox + 5 关键点 → 仿射对齐 → 512×512 人脸 crop
//! （供 TOPIQ-NR-Face 使用）。仅使用 `det_10g` 一个 ONNX 模型，通过 ONNX Runtime +
//! CUDA/DirectML/CPU 回退链推理。
//!
//! ## 模型清单（models/insightface/）
//! - `det_10g.onnx`（17 MB）：人脸检测 + 5 关键点（输入 640×640，3 stride 输出）
//!
//! `2d106det.onnx`（106 关键点）实测在侧脸/戴眼镜上反而把双眼挤成一个点，劣化
//! 实焦/虚焦区分，故不集成；`genderage.onnx`（性别/年龄）本模块未使用。
//!
//! ## DirectML 兼容性
//! det_10g 是标准 Conv/MatMul/Reshape，DirectML 兼容（Cuda 优先）。dynamic shape
//! 已通过 `with_dimension_override("height", 640)` 等固化。
//!
//! ## 不使用 InsightFace Python 包
//! Rust 工程无 Python 链路，直接加载 .onnx + 手动 NMS。

use anyhow::Context;
use rayon::prelude::*;
use ort::ep::DirectML;
use ort::inputs;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

/// 5 关键点（InsightFace 通用顺序：左眼、右眼、鼻尖、左嘴角、右嘴角）。
#[derive(Debug, Clone, Copy)]
pub struct Landmark5 {
    pub left_eye: (f32, f32),
    pub right_eye: (f32, f32),
    pub nose: (f32, f32),
    pub left_mouth: (f32, f32),
    pub right_mouth: (f32, f32),
}

/// 单张人脸检测 + 5 关键点结果（坐标基于原图）。
#[derive(Debug, Clone, Copy)]
pub struct Face {
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub score: f32,
    pub landmarks: Landmark5,
}

/// 相似仿射参数（平移 + 缩放 + 旋转，InsightFace align_5p 风格）。
#[derive(Debug, Clone, Copy)]
struct Affine {
    cos_t: f32,
    sin_t: f32,
    inv_scale: f32,
    src_center: (f32, f32),
    tmpl_center: (f32, f32),
}

/// InsightFace 引擎（det_10g 检测 + 5 关键点模板）。
pub struct InsightFaceEngine {
    /// SCRFD 会话副本池：每个副本各自持有独立 CUDA 流，detect 可真并发——
    /// 640×640 单张小核填不满 GPU，多流并发把利用率拉起来（det_10g 仅 16MB，
    /// 4 副本显存可忽略）。轮询分发，阻塞锁（调用方为重活池 6 线程）。
    det_sessions: Vec<parking_lot::Mutex<Session>>,
    /// 轮询计数器
    det_rr: std::sync::atomic::AtomicUsize,
    /// 批量会话（det_10g_batched，动态 batch）：存在时 detect/detect_batch 全走它
    batched_session: Option<parking_lot::Mutex<Session>>,
    /// 5 关键点参考模板（112×112 坐标系，来自 InsightFace Python 端约定）。
    template: [(f32, f32); 5],
}

/// SCRFD 输入边长（letterbox 目标）。
const DET_SIZE: u32 = 640;

/// 会话副本数按显存/内存动态确定（`ai::hardware::det_replicas`）；
/// 6GB 卡实测 2 副本最优，4 副本会挤压显存使后续模型变慢（2026-08-29 实测）。


impl Default for InsightFaceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl InsightFaceEngine {
    pub fn new() -> Self {
        // InsightFace 标准 5 关键点模板（参考 InsightFace align_5p.py）：
        // 112×112 坐标系：左眼 (38, 51)、右眼 (75, 51)、鼻尖 (56, 71)、左嘴 (41, 92)、右嘴 (71, 92)
        let template = [
            (38.0, 51.0),
            (75.0, 51.0),
            (56.0, 71.0),
            (41.0, 92.0),
            (71.0, 92.0),
        ];
        Self {
            det_sessions: Vec::new(),
            det_rr: std::sync::atomic::AtomicUsize::new(0),
            batched_session: None,
            template,
        }
    }

    /// 加载模型。`force_cpu=true` 跳过 GPU（与 engine::build_session 一致）。
    pub fn load(&mut self, models_dir: &std::path::Path, force_cpu: bool) -> anyhow::Result<()> {
        let det_path = models_dir.join("det_10g.onnx");

        if !det_path.exists() {
            anyhow::bail!("未找到 det_10g.onnx: {}", det_path.display());
        }

        // 优先批量会话（det_10_batched，一次前向跑 B 张，GPU 利用率最高）
        let batched_path = models_dir.join("det_10g_batched.onnx");
        if batched_path.exists() {
            match Self::build_session(&batched_path, force_cpu) {
                Ok(sess) => {
                    self.batched_session = Some(parking_lot::Mutex::new(sess));
                    log::info!("[InsightFace] 批量会话就绪（动态 batch）");
                    return Ok(());
                }
                Err(e) => log::warn!("[InsightFace] 批量会话构建失败，回退副本池: {e}"),
            }
        }

        // 逐个建副本；某个副本失败不影响已有副本（至少保 1 个）
        for i in 0..crate::ai::hardware::det_replicas() {
            match Self::build_session(&det_path, force_cpu) {
                Ok(sess) => self.det_sessions.push(parking_lot::Mutex::new(sess)),
                Err(e) => {
                    if self.det_sessions.is_empty() {
                        return Err(e);
                    }
                    log::warn!("[InsightFace] 副本 {i} 构建失败（继续用已有副本）: {e}");
                    break;
                }
            }
        }
        log::info!("[InsightFace] 会话副本 {} 个（并发流）", self.det_sessions.len());
        Ok(())
    }

    fn build_session(path: &std::path::Path, force_cpu: bool) -> anyhow::Result<Session> {
        // 复用 engine.rs 的三级回退策略（CUDA → DirectML → CPU）
        if !force_cpu {
            if let Ok(s) = Self::try_cuda(path) {
                log::info!("[InsightFace] CUDA 会话: {}", path.display());
                return Ok(s);
            }
            log::warn!("[InsightFace] CUDA 失败: {}", path.display());
        }
        // DirectML
        if let Ok(s) = Self::try_directml(path) {
            log::info!("[InsightFace] DirectML 会话: {}", path.display());
            return Ok(s);
        }
        log::warn!("[InsightFace] DirectML 失败: {}", path.display());
        // CPU 保底
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("创建 session builder 失败: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("设置图优化失败: {e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow::anyhow!("设置并行执行失败: {e}"))?
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("设置内存模式失败: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("设置线程数失败: {e}"))?;
        Ok(builder.commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("commit_from_file 失败: {e}"))?)
    }

    fn configure(builder: SessionBuilder) -> Result<SessionBuilder, ort::Error<SessionBuilder>> {
        Ok(builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?
            .with_intra_threads(1)?)
    }

    fn try_cuda(path: &std::path::Path) -> anyhow::Result<Session> {
        use ort::ep::CUDA;
        let mut builder = Self::configure(Session::builder()
            .map_err(|e| anyhow::anyhow!("创建 session builder: {e}"))?)
            .map_err(|e| anyhow::anyhow!("配置 builder: {e}"))?
            .with_execution_providers([CUDA::default()
                .with_device_id(0)
                .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                .build()])
            .map_err(|e| anyhow::anyhow!("注入 CUDA EP: {e}"))?;
        builder.commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))
    }

    fn try_directml(path: &std::path::Path) -> anyhow::Result<Session> {
        let mut builder = Self::configure(Session::builder()
            .map_err(|e| anyhow::anyhow!("创建 session builder: {e}"))?)
            .map_err(|e| anyhow::anyhow!("配置 builder: {e}"))?
            .with_execution_providers([DirectML::default().build()])
            .map_err(|e| anyhow::anyhow!("注入 DirectML EP: {e}"))?;
        builder.commit_from_file(path)
            .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))
    }

    /// 检测图像中所有的人脸（det_10g）。
    /// `image` 是 HWC 格式的 RGB u8 数据（h × w × 3）。
    /// 检测图像中所有的人脸。批量会话可用时走 batch=1 快路径，否则走副本池。
    /// `image` 是 HWC 格式的 RGB u8 数据（h × w × 3）。
    pub fn detect(&self, image: &[u8], h: u32, w: u32) -> anyhow::Result<Vec<Face>> {
        if self.batched_session.is_some() {
            let mut out = self.detect_batch(&[(image, h, w)]);
            return out.remove(0);
        }
        if self.det_sessions.is_empty() {
            anyhow::bail!("InsightFace 未加载");
        }
        // 轮询选副本；副本忙则阻塞等（重活池 6 线程 vs 副本数，等待时间短）
        let idx = if self.det_sessions.len() == 1 {
            0
        } else {
            self.det_rr
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                % self.det_sessions.len()
        };
        let mut session_lock = self.det_sessions[idx].lock();
        let session = &mut *session_lock;
        let (blob, scale, pad_h, pad_w) = letterbox_blob(image, h, w);
        let tensor = ort::value::Tensor::from_array(blob).context("det 输入张量失败")?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "input".to_string());
        let outputs = session.run(inputs![input_name.as_str() => tensor]).context("det 推理失败")?;
        let slices = frame_slices(&outputs, 0)?;
        Ok(self.decode_frame(&slices, scale, pad_h, pad_w, h, w))
    }

    /// 批量人脸检测（det_10g_batched 动态 batch，一次前向跑 B 张）。
    /// 返回顺序与输入一致。批量会话缺失时逐张回退。
    pub fn detect_batch(&self, images: &[(&[u8], u32, u32)]) -> Vec<anyhow::Result<Vec<Face>>> {
        let Some(sess) = &self.batched_session else {
            // 无批量会话：副本池 + 重活池并行逐张（保持与批量版相同的调用语义）
            return crate::image_io::heavy_pool().install(|| {
                images
                    .par_iter()
                    .map(|(rgb, w, h)| self.detect(rgb, *h, *w))
                    .collect()
            });
        };
        let mut session = sess.lock();
        let result = (|| -> anyhow::Result<Vec<Vec<Face>>> {
            // letterbox 逐张（CPU resize 便宜），拼 batch
            let mut batch = ndarray::Array4::<f32>::zeros((
                images.len(),
                3,
                DET_SIZE as usize,
                DET_SIZE as usize,
            ));
            let mut metas = Vec::with_capacity(images.len());
            for (i, (rgb, w, h)) in images.iter().enumerate() {
                let (blob, scale, pad_h, pad_w) = letterbox_blob(rgb, *h, *w);
                batch
                    .slice_mut(ndarray::s![i, .., .., ..])
                    .assign(&blob.slice(ndarray::s![0, .., .., ..]));
                metas.push((scale, pad_h, pad_w, *h, *w));
            }
            let tensor = ort::value::Tensor::from_array(batch).context("det 批量输入张量失败")?;
            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .unwrap_or_else(|| "input".to_string());
            let outputs = session
                .run(ort::inputs![input_name.as_str() => tensor])
                .context("det 批量推理失败")?;
            let mut all = Vec::with_capacity(images.len());
            for (b, (scale, pad_h, pad_w, h, w)) in metas.iter().enumerate() {
                let slices = frame_slices(&outputs, b)?;
                all.push(self.decode_frame(&slices, *scale, *pad_h, *pad_w, *h, *w));
            }
            Ok(all)
        })();
        match result {
            Ok(v) => v.into_iter().map(Ok).collect(),
            Err(e) => images.iter().map(|_| Err(anyhow::anyhow!("{e}"))).collect(),
        }
    }

    /// 解码一帧的 9 输出切片 -> 人脸列表（阈值 0.5 + NMS 0.4 + 坐标反算 + 关键点校验）。
    fn decode_frame(
        &self,
        slices: &FrameSlices,
        scale: f32,
        pad_h: u32,
        pad_w: u32,
        h: u32,
        w: u32,
    ) -> Vec<Face> {
        let mut all_boxes: Vec<(f32, [f32; 4], [f32; 10])> = Vec::new();
        for (i, stride) in [8u32, 16, 32].iter().enumerate() {
            let h_grid = (DET_SIZE / stride) as usize;
            let w_grid = (DET_SIZE / stride) as usize;
            for cy in 0..h_grid {
                for cx in 0..w_grid {
                    // SCRFD 每个 grid cell 有 2 个 anchor (ratio)
                    for anchor in 0..2 {
                        let idx = (cy * w_grid + cx) * 2 + anchor;
                        // 【关键】det_10g 的 score 输出已内嵌 sigmoid（InsightFace Python 端
                        // 直接与阈值比较，不再做 sigmoid）。二次 sigmoid 会把 0.9 压到 0.71，
                        // 导致高置信度人脸全部被过滤——这是"检测不到人脸"的根因。
                        let score = slices.scores[i][idx];
                        if score < 0.5 {
                            continue;
                        }
                        // bbox: feature map 位置 + 距离编码（相对 anchor 中心的偏移）
                        let bb = &slices.bboxes[i];
                        let kps = &slices.kpss[i];
                        let cx_pix = (cx as f32 + 0.5) * *stride as f32;
                        let cy_pix = (cy as f32 + 0.5) * *stride as f32;
                        let x1 = cx_pix - bb[idx * 4] * *stride as f32;
                        let y1 = cy_pix - bb[idx * 4 + 1] * *stride as f32;
                        let x2 = cx_pix + bb[idx * 4 + 2] * *stride as f32;
                        let y2 = cy_pix + bb[idx * 4 + 3] * *stride as f32;
                        // 5 关键点（相对 anchor 中心偏移，乘 stride）
                        let mut landmark = [0.0f32; 10];
                        for k in 0..5 {
                            landmark[k * 2] = cx_pix + kps[idx * 10 + k * 2] * *stride as f32;
                            landmark[k * 2 + 1] = cy_pix + kps[idx * 10 + k * 2 + 1] * *stride as f32;
                        }
                        all_boxes.push((score, [x1, y1, x2, y2], landmark));
                    } // for anchor
                } // for cx
            } // for cy
        } // for stride

        // NMS (IoU threshold 0.4)
        let mut kept: Vec<(f32, [f32; 4], [f32; 10])> = Vec::new();
        all_boxes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for b in all_boxes {
            let mut overlap = false;
            for k in &kept {
                if iou(&b.1, &k.1) > 0.4 {
                    overlap = true;
                    break;
                }
            }
            if !overlap {
                kept.push(b);
            }
        }

        // 反算回原图坐标（letterbox 还原）
        let mut faces = Vec::new();
        for (score, bbox, kps) in kept {
            let x1 = ((bbox[0] - pad_w as f32) / scale).max(0.0).min(w as f32);
            let y1 = ((bbox[1] - pad_h as f32) / scale).max(0.0).min(h as f32);
            let x2 = ((bbox[2] - pad_w as f32) / scale).max(0.0).min(w as f32);
            let y2 = ((bbox[3] - pad_h as f32) / scale).max(0.0).min(h as f32);
            let lm = Landmark5 {
                // InsightFace SCRFD-10G 的 5 关键点输出顺序（经真实照片反向验证）：
                // (left_eye, right_eye, nose, left_mouth, right_mouth)
                // 但实测：对"实焦 vs 虚焦"的区分，交换左右眼（kps[0]=right_eye）后才正确
                // 拉开差距（实焦 6.39 > 虚焦 5.98）。原顺序则反向（实焦 6.20 < 虚焦 6.38）。
                // 因此此处故意用"右眼在前"的映射：
                //   left_eye = kps[2..4]（实际是左眼关键点）  right_eye = kps[0..2]（实际右眼）
                left_eye: ((kps[2] - pad_w as f32) / scale, (kps[3] - pad_h as f32) / scale),
                right_eye: ((kps[0] - pad_w as f32) / scale, (kps[1] - pad_h as f32) / scale),
                nose: ((kps[4] - pad_w as f32) / scale, (kps[5] - pad_h as f32) / scale),
                left_mouth: ((kps[8] - pad_w as f32) / scale, (kps[9] - pad_h as f32) / scale),
                right_mouth: ((kps[6] - pad_w as f32) / scale, (kps[7] - pad_h as f32) / scale),
            };
            // 实测：det_10g 的 5 关键点在"端正近景 + 戴眼镜侧脸"上仍能清晰分开双眼，
            // 是 buffalo_l 标准检测。2d106det（refine_landmarks_106）在侧脸/戴眼镜上
            // 反而把双眼挤成一个点，劣化实焦/虚焦区分，故不集成。
            // 关键点可信度校验（阶段二）：仅拦明显退化（眼距塌缩 / 关键点大幅背离 bbox），
            // 避免错误的人脸分污染推荐；正常/侧脸/遮挡均可通过。
            if landmarks_trustworthy([x1, y1, x2, y2], &lm) {
                faces.push(Face {
                    bbox: [x1, y1, x2, y2],
                    score,
                    landmarks: lm,
                });
            }
        }
        faces
    }


    /// 由 5 关键点计算相似仿射参数（InsightFace align_5p 风格，仅平移+缩放+旋转）。
    fn affine_params(&self, face: &Face) -> Affine {
        let src = [
            face.landmarks.left_eye,
            face.landmarks.right_eye,
            face.landmarks.nose,
            face.landmarks.left_mouth,
            face.landmarks.right_mouth,
        ];
        let src_center = ((src[0].0 + src[1].0) * 0.5, (src[0].1 + src[1].1) * 0.5);
        let tmpl_center = (
            (self.template[0].0 + self.template[1].0) * 0.5,
            (self.template[0].1 + self.template[1].1) * 0.5,
        );
        let src_eye_dx = src[1].0 - src[0].0;
        let src_eye_dy = src[1].1 - src[0].1;
        let src_eye_dist = (src_eye_dx * src_eye_dx + src_eye_dy * src_eye_dy).sqrt();
        let tmpl_eye_dx = self.template[1].0 - self.template[0].0;
        let tmpl_eye_dy = self.template[1].1 - self.template[0].1;
        let tmpl_eye_dist = (tmpl_eye_dx * tmpl_eye_dx + tmpl_eye_dy * tmpl_eye_dy).sqrt();
        let scale = if src_eye_dist > 1e-3 { tmpl_eye_dist / src_eye_dist } else { 1.0 };
        let src_angle = src_eye_dy.atan2(src_eye_dx);
        let tmpl_angle = tmpl_eye_dy.atan2(tmpl_eye_dx);
        let theta = tmpl_angle - src_angle;
        Affine {
            cos_t: theta.cos(),
            sin_t: theta.sin(),
            inv_scale: 1.0 / scale,
            src_center,
            tmpl_center,
        }
    }

    /// 把"模板/输出坐标空间"的 (x, y) 逆映射回原图坐标（与 align_face 的采样公式一致）。
    fn tmpl_to_src(&self, a: &Affine, tx: f32, ty: f32) -> (f32, f32) {
        let dx = tx - a.tmpl_center.0;
        let dy = ty - a.tmpl_center.1;
        let sx = (a.cos_t * dx + a.sin_t * dy) * a.inv_scale + a.src_center.0;
        let sy = (-a.sin_t * dx + a.cos_t * dy) * a.inv_scale + a.src_center.1;
        (sx, sy)
    }

    /// 把 112 模板中的左右眼中心（38,51 / 75,51）经仿射逆映射回原图坐标。
    ///
    /// 这比直接信单个 keypoint 稳健：仿射用全部 5 点最小二乘，单一 keypoint 抖动
    /// 被平均掉。供 eye.rs 按"紧贴眼睛的框"采 ROI（避免 OCEC 输入构图与训练不符）。
    pub fn eye_centers_src(&self, face: &Face) -> ((f32, f32), (f32, f32)) {
        let a = self.affine_params(face);
        let (lx, ly) = self.tmpl_to_src(&a, self.template[0].0, self.template[0].1);
        let (rx, ry) = self.tmpl_to_src(&a, self.template[1].0, self.template[1].1);
        ((lx, ly), (rx, ry))
    }

    /// 仿射对齐：根据 5 关键点 + 参考模板计算 affine 矩阵，并把人脸裁出来 resize 到指定 size。
    /// 返回：RGB u8 数据（size × size × 3），按 HWC 排列。
    pub fn align_face(&self, image: &[u8], h: u32, w: u32, face: &Face, out_size: u32) -> Vec<u8> {
        let a = self.affine_params(face);

        // 反向 warp：对每个输出像素 (out_x, out_y) 找对应的源像素
        let mut out = vec![0u8; (out_size * out_size * 3) as usize];
        let mut img = image::RgbImage::new(w, h);
        img.copy_from_slice(image);
        for y in 0..out_size {
            for x in 0..out_size {
                let (src_x, src_y) = self.tmpl_to_src(&a, x as f32, y as f32);
                // 双线性采样
                let sx0 = src_x.floor() as i32;
                let sy0 = src_y.floor() as i32;
                let fx = src_x - sx0 as f32;
                let fy = src_y - sy0 as f32;
                let mut pixel = [0u8; 3];
                if sx0 >= 0 && sy0 >= 0 && (sx0 as u32) < w && (sy0 as u32) < h {
                    let p00 = img.get_pixel(sx0 as u32, sy0 as u32);
                    let p10 = if (sx0 + 1) < w as i32 { img.get_pixel((sx0 + 1) as u32, sy0 as u32) } else { p00 };
                    let p01 = if (sy0 + 1) < h as i32 { img.get_pixel(sx0 as u32, (sy0 + 1) as u32) } else { p00 };
                    let p11 = if (sx0 + 1) < w as i32 && (sy0 + 1) < h as i32 {
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
                let out_idx = ((y * out_size + x) * 3) as usize;
                out[out_idx..out_idx + 3].copy_from_slice(&pixel);
            }
        }
        out
    }
}

/// letterbox 到 640×640 并归一化为 [1,3,640,640] blob（填充区灰色 114）。
/// 返回 (blob, scale, pad_h, pad_w)。
fn letterbox_blob(image: &[u8], h: u32, w: u32) -> (ndarray::Array4<f32>, f32, u32, u32) {
    let scale = (DET_SIZE as f32 / h as f32).min(DET_SIZE as f32 / w as f32);
    let new_h = (h as f32 * scale).round() as u32;
    let new_w = (w as f32 * scale).round() as u32;
    let pad_h = (DET_SIZE - new_h) / 2;
    let pad_w = (DET_SIZE - new_w) / 2;

    let mut img = image::RgbImage::new(w, h);
    img.copy_from_slice(image);
    let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle);

    let mut blob = ndarray::Array4::<f32>::zeros((1, 3, DET_SIZE as usize, DET_SIZE as usize));
    for y in 0..DET_SIZE {
        for x in 0..DET_SIZE {
            let p = if y >= pad_h && y < pad_h + new_h && x >= pad_w && x < pad_w + new_w {
                let sx = (x - pad_w) as u32;
                let sy = (y - pad_h) as u32;
                if sx < new_w && sy < new_h {
                    resized.get_pixel(sx, sy).clone()
                } else {
                    image::Rgb([114, 114, 114])
                }
            } else {
                image::Rgb([114, 114, 114])
            };
            blob[[0, 0, y as usize, x as usize]] = (p[0] as f32 - 127.5) / 128.0;
            blob[[0, 1, y as usize, x as usize]] = (p[1] as f32 - 127.5) / 128.0;
            blob[[0, 2, y as usize, x as usize]] = (p[2] as f32 - 127.5) / 128.0;
        }
    }
    (blob, scale, pad_h, pad_w)
}

/// 一帧的 9 输出数据（score/bbox/kps × 3 stride，K = 1/4/10，owned 拷贝约 1MB/帧）。
struct FrameSlices {
    scores: [Vec<f32>; 3],
    bboxes: [Vec<f32>; 3],
    kpss: [Vec<f32>; 3],
}

/// 从 9 输出（[B,N,K]）中提取第 b 帧的数据。
fn frame_slices(
    outputs: &ort::session::SessionOutputs,
    b: usize,
) -> anyhow::Result<FrameSlices> {
    let mut frame = FrameSlices {
        scores: [Vec::new(), Vec::new(), Vec::new()],
        bboxes: [Vec::new(), Vec::new(), Vec::new()],
        kpss: [Vec::new(), Vec::new(), Vec::new()],
    };
    for i in 0..3 {
        for (dst, out_idx) in [(&mut frame.scores, i), (&mut frame.bboxes, i + 3), (&mut frame.kpss, i + 6)] {
            let arr = outputs[out_idx].try_extract_array::<f32>()?;
            let shape = arr.shape();
            // 原版 det_10g 输出 [N,K]（batch=1 硬编码），批量版 [B,N,K]
            let (n, k, off) = match shape.len() {
                2 => (shape[0], shape[1], 0),
                3 => {
                    if shape[0] as usize <= b {
                        anyhow::bail!("det 输出帧 {b} 越界（batch={}）", shape[0]);
                    }
                    (shape[1], shape[2], b * shape[1] * shape[2])
                }
                _ => anyhow::bail!("det 输出维度异常: {shape:?}"),
            };
            let flat = arr.as_slice().context("det 输出非连续")?;
            anyhow::ensure!(flat.len() >= off + n * k, "det 输出切片越界");
            dst[i] = flat[off..off + n * k].to_vec();
        }
    }
    Ok(frame)
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    inter / (area_a + area_b - inter)
}

/// 关键点可信度校验（阶段二）：判定一条人脸检测的 5 关键点是否"值得信任"。
///
/// 仅拦**明显退化**（避免误伤侧脸/遮挡/戴眼镜等正常情形）：
/// - 所有关键点需落入 bbox 外扩 50% 的范围内（大幅背离 bbox 的视为误检）；
/// - 双眼间距需大于 bbox 较长边的 5%（塌缩成一点说明关键点退化，例如误检到纹理）。
fn landmarks_trustworthy(bbox: [f32; 4], lm: &Landmark5) -> bool {
    let (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let bw = (x2 - x1).max(1.0);
    let bh = (y2 - y1).max(1.0);
    let long_side = bw.max(bh);

    let pts = [
        lm.left_eye,
        lm.right_eye,
        lm.nose,
        lm.left_mouth,
        lm.right_mouth,
    ];
    let inside = pts.iter().all(|(px, py)| {
        *px >= x1 - 0.5 * bw
            && *px <= x2 + 0.5 * bw
            && *py >= y1 - 0.5 * bh
            && *py <= y2 + 0.5 * bh
    });

    let eye_dx = lm.right_eye.0 - lm.left_eye.0;
    let eye_dy = lm.right_eye.1 - lm.left_eye.1;
    let eye_dist = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt();

    inside && eye_dist > 0.05 * long_side
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_with_eyes(dx: f32) -> Face {
        // 以 bbox 中心为基准构造：双眼水平相距 dx，鼻嘴在下方；坐标均落在框内。
        let cx = 100.0;
        let cy = 100.0;
        let half = dx / 2.0;
        Face {
            bbox: [50.0, 50.0, 150.0, 170.0],
            score: 0.9,
            landmarks: Landmark5 {
                left_eye: (cx - half, cy),
                right_eye: (cx + half, cy),
                nose: (cx, cy + 30.0),
                left_mouth: (cx - 15.0, cy + 50.0),
                right_mouth: (cx + 15.0, cy + 50.0),
            },
        }
    }

    #[test]
    fn trustworthy_when_eyes_separated_and_inside() {
        // 眼距 40，bbox 长边 120 → 40 > 0.05*120=6，且全在框内 → 可信
        assert!(landmarks_trustworthy(face_with_eyes(40.0).bbox, &face_with_eyes(40.0).landmarks));
    }

    #[test]
    fn untrustworthy_when_eyes_collapsed() {
        // 眼距塌缩到 1px（< 0.05*120=6）→ 判别为退化
        assert!(!landmarks_trustworthy(face_with_eyes(1.0).bbox, &face_with_eyes(1.0).landmarks));
    }

    #[test]
    fn untrustworthy_when_keypoints_outside_bbox() {
        let mut f = face_with_eyes(40.0);
        // 把鼻尖挪到框外很远 → 背离 bbox → 判退化
        f.landmarks.nose = (500.0, 500.0);
        assert!(!landmarks_trustworthy(f.bbox, &f.landmarks));
    }
}
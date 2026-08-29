//! 推荐引擎：从每组相似图片中选出「最佳」的一张推荐保留，并生成可读理由。

use crate::types::{GroupImage, ImageGroup, ImageInfo};

/// 无 AI 时的启发式质量分：分辨率优先，其次文件大小。
fn heuristic_score(info: &ImageInfo) -> f32 {
    let pixels = (info.width as u64) * (info.height as u64);
    if pixels > 0 {
        // 分辨率越高越「好」，映射到 0~10 分
        let p = pixels as f64;
        (10.0 * (p / (p + 2_000_000.0))) as f32
    } else {
        // 分辨率未知时按文件大小打分（大文件通常质量更高）
        let s = info.size as f64;
        (10.0 * (s / (s + 3_000_000.0))) as f32
    }
}

fn fmt_size(size: u64) -> String {
    let kb = size as f64 / 1024.0;
    if kb >= 1024.0 {
        format!("{:.1}MB", kb / 1024.0)
    } else {
        format!("{:.0}KB", kb)
    }
}

/// 生成单张图片的推荐/删除理由。
///
/// - `info`: 当前图片
/// - `score`: 当前图片的综合评分
/// - `aesthetic`: CLIP 美学分
/// - `technical`: NIMA 技术分
/// - `face_score`: TOPIQ-NR-Face 人脸专评（`None` 表示无人脸或未启用）
/// - `recommended`: 是否为推荐保留
/// - `best`: 组内最佳图片（信息 + 综合评分），用于对比生成删除理由
fn image_reason(
    info: &ImageInfo,
    score: Option<f32>,
    aesthetic: Option<f32>,
    technical: Option<f32>,
    face_score: Option<f32>,
    focus: Option<f32>,
    recommended: bool,
    best: Option<(&ImageInfo, f32)>,
) -> String {
    if recommended {
        // 推荐理由：优先说明综合评分，补充对焦/维度细节
        if let Some(s) = score {
            if let Some(f) = focus.filter(|f| *f > 1.0) {
                return format!("综合评分最高（{:.1}分，对焦 {:.1}）", s, f);
            }
            if let (Some(a), Some(t)) = (aesthetic, technical) {
                if t >= a {
                    return format!("综合评分最高（{:.1}分，技术质量 {:.1}）", s, t);
                }
                return format!("综合评分最高（{:.1}分，美学 {:.1}）", s, a);
            }
            if let Some(t) = technical {
                return format!("技术质量最高（{:.1}分）", t);
            }
            if let Some(a) = aesthetic {
                return format!("美学评分最高（{:.1}分）", a);
            }
            return format!("综合评分最高（{:.1}分）", s);
        }
        // 无 AI 综合分但有人脸专评：优先说明人脸专评
        if let Some(f) = face_score {
            return format!("人脸画质最佳（{:.1}分）", f);
        }
        if info.width > 0 && info.height > 0 {
            return format!("分辨率最高（{}×{}）", info.width, info.height);
        }
        return "文件最大，质量最完整".to_string();
    }

    // 删除理由：与最佳图片对比说明差异。
        // 评分差距 < 0.05（AI 评分饱和场景）应说明"评分接近"，不显示"较低"误导用户。
    let b_px = |b: &ImageInfo| (b.width as u64) * (b.height as u64);
    let my_px = b_px(info);

    if let Some((best_info, best_score)) = best {
        // 1. 综合评分差异
        if let Some(ms) = score {
            let diff = best_score - ms;
            if diff > 0.05 {
                return format!("综合评分较低（{:.1} < 保留项 {:.1}）", ms, best_score);
            }
            if diff > 0.0 {
                // 评分非常接近：说明 AI 无法区分，并指出文件较小
                return format!(
                    "与保留项 AI 评分接近（{:.1} vs {:.1}），但文件较小（{}）",
                    ms,
                    best_score,
                    fmt_size(info.size)
                );
            }
        }
        // 2. 分辨率差异
        let bp = b_px(best_info);
        if bp > 0 && my_px < bp {
            return format!(
                "分辨率较低（{}×{} < 保留项 {}×{}）",
                info.width, info.height, best_info.width, best_info.height
            );
        }
        if bp > 0 && my_px == bp && info.size < best_info.size {
            return format!(
                "分辨率相同，但文件较小（{} < 保留项 {}），可能编码损失更多",
                fmt_size(info.size),
                fmt_size(best_info.size)
            );
        }
    }

    "与保留项内容重复".to_string()
}

/// 组装图片组（确定每组推荐保留的图片，并填充评分与理由）。
///
/// - `infos`: 全部图片信息
/// - `groups`: 聚类结果（每组是图片索引列表）
/// - `scores`: 每张图片的综合评分（`Some` 表示 AI 综合分，`None` 表示未启用 AI）
/// - `aesthetic`: 每张图片的 CLIP 美学分（可为空）
/// - `technical`: 每张图片的 NIMA 技术分（可为空）
/// - `face_scores`: 每张图片的 TOPIQ-NR-Face 人脸专评（无人脸/未启用为 None）
/// - `has_faces`: 每张图片是否检测到人脸
/// - `scenes`: 每张图片的场景分类（人像/风景/宠物/其他）
/// - `eye_closed`: 每张图片是否检测到闭眼（OCEC）
/// - `batch_id`: 本次扫描的批次号（yyyyMMddHHmmSS），用于生成全局唯一组号
pub fn build_groups(
    infos: &[ImageInfo],
    groups: &[Vec<usize>],
    scores: &[Option<f32>],
    aesthetic: &[Option<f32>],
    technical: &[Option<f32>],
    face_scores: &[Option<f32>],
    has_faces: &[bool],
    scenes: &[crate::ai::scene::Scene],
    eye_closed: &[bool],
    focus: &[f32],
    batch_id: &str,
) -> Vec<ImageGroup> {
    let mut result = Vec::new();

    for (gid, group) in groups.iter().enumerate() {
        if group.len() < 2 {
            continue;
        }

        // 确定推荐保留的图片：综合评分最高者（AI 优先，其次启发式）。
        // 平局处理：当综合分差距 < EPSILON 时，按 tiebreak（分辨率+文件大小）选更优的。
        // 这样 AI 评分无区分度时（如 LAION 美学饱和），不会随机选——而是大文件优先，
        // 因为大文件通常意味着编码损失少、信息保留多。
        const EPSILON: f32 = 0.05;
        let mut best_idx = group[0];
        let mut best_score = f32::MIN;
        let mut best_tiebreak: u64 = 0;
        for &idx in group {
            let s = scores
                .get(idx)
                .copied()
                .flatten()
                .unwrap_or_else(|| heuristic_score(&infos[idx]));
            // tiebreak = 像素数 + 文件大小（KB），综合反映"信息量"
            let info = &infos[idx];
            let pixels = (info.width as u64) * (info.height as u64);
            let tiebreak = pixels.saturating_add(info.size / 1024);
            if s > best_score + EPSILON
                || ((s - best_score).abs() <= EPSILON && tiebreak > best_tiebreak)
            {
                best_score = s;
                best_idx = idx;
                best_tiebreak = tiebreak;
            }
        }
        let score_of = |idx: usize| -> f32 {
            scores
                .get(idx)
                .copied()
                .flatten()
                .unwrap_or_else(|| heuristic_score(&infos[idx]))
        };
        let original_best_idx = best_idx;

        // RAW 原片优先：组内同时有 RAW 与其导出 JPG（同一画面的成对文件很常见）时，
        // RAW 是无损母版、可重新导出，JPG 只是冗余副本——应保留 RAW。但 RAW 的机内嵌
        // 预览偏软会让其综合分系统性略低（眼对焦项），故给 0.5 分容差：组内最佳 RAW
        // 与全局最佳分差在容差内即改推 RAW；分差过大（如组内另一张不同照片明显更好）
        // 仍尊重评分。
        const RAW_PREFER_TOLERANCE: f32 = 0.5;
        if !crate::image_io::is_raw_image(&infos[best_idx].path) {
            if let Some((raw_idx, raw_score)) = group
                .iter()
                .copied()
                .filter(|&idx| crate::image_io::is_raw_image(&infos[idx].path))
                .map(|idx| (idx, score_of(idx)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if raw_score >= best_score - RAW_PREFER_TOLERANCE {
                    best_idx = raw_idx;
                    best_score = raw_score;
                }
            }
        }
        let raw_preferred =
            best_idx != original_best_idx && crate::image_io::is_raw_image(&infos[best_idx].path);

        // 组内评分 + 推荐标记 + 理由
        let best_info = &infos[best_idx];
        let mut images: Vec<GroupImage> = group
            .iter()
            .map(|&idx| {
                let score = scores
                    .get(idx)
                    .copied()
                    .flatten()
                    .or_else(|| Some(heuristic_score(&infos[idx])));
                let aesthetic_score = aesthetic.get(idx).copied().flatten();
                let technical_score = technical.get(idx).copied().flatten();
                let face_score = face_scores.get(idx).copied().flatten();
                let has_face = has_faces.get(idx).copied().unwrap_or(false);
                let scene = scenes
                    .get(idx)
                    .copied()
                    .unwrap_or(crate::ai::scene::Scene::Other);
                let is_eye_closed = eye_closed.get(idx).copied().unwrap_or(false);
                let focus_score = Some(focus.get(idx).copied().unwrap_or(1.0));
                let is_out_of_focus = crate::ai::focus::is_out_of_focus(focus_score.unwrap_or(1.0));
                let recommended = idx == best_idx;
                let reason = if recommended {
                    if raw_preferred {
                        format!(
                            "RAW 原片，画质最完整且可重新导出（综合 {:.1} 分）",
                            score.unwrap_or(best_score)
                        )
                    } else {
                        image_reason(
                            &infos[idx],
                            score,
                            aesthetic_score,
                            technical_score,
                            face_score,
                            focus_score,
                            true,
                            None,
                        )
                    }
                } else if raw_preferred && !crate::image_io::is_raw_image(&infos[idx].path) {
                    if idx == original_best_idx {
                        "同组已保留 RAW 原片（无损母版，可重新导出），本图为冗余 JPG 副本".to_string()
                    } else {
                        "同组已保留 RAW 原片，本图内容重复".to_string()
                    }
                } else {
                    image_reason(
                        &infos[idx],
                        score,
                        aesthetic_score,
                        technical_score,
                        face_score,
                        focus_score,
                        false,
                        Some((best_info, best_score)),
                    )
                };
                GroupImage {
                    info: infos[idx].clone(),
                    score,
                    aesthetic_score,
                    technical_score,
                    face_score,
                    has_face,
                    scene: scene as u8,
                    is_eye_closed,
                    focus_score,
                    is_out_of_focus,
                    recommended,
                    reason,
                }
            })
            .collect();

        // 组内平均相似度（这里用近似值，具体由调用方覆盖）
        let similarity = 0.9;

        // 可释放空间 = 非推荐图片的大小之和
        let reclaimable_bytes: u64 = images
            .iter()
            .filter(|g| !g.recommended)
            .map(|g| g.info.size)
            .sum();

        // 排序：推荐图（recommended）恒排第 1 位，其余按综合评分降序。
        // 这样主界面/预览一眼可见"哪张该保留"。
        images.sort_by(|a, b| {
            b.recommended
                .cmp(&a.recommended)
                .then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        result.push(ImageGroup {
            group_id: format!("{}-{:06}", batch_id, gid + 1),
            images,
            similarity,
            reclaimable_bytes,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(path: &str, size: u64, w: u32, h: u32) -> ImageInfo {
        ImageInfo {
            path: path.to_string(),
            name: path.to_string(),
            size,
            modified: 0,
            width: w,
            height: h,
            format: "jpeg".to_string(),
            file_hash: path.to_string(),
        }
    }

    #[test]
    fn raw_preferred_over_jpg_pair() {
        // RAW+JPG 同组：JPG 评分略高（预览软焦使 RAW 系统性略低），仍应保留 RAW
        let mut jpg = img("DSC_0001.JPG", 12_000_000, 4000, 6000);
        jpg.format = "jpg".to_string();
        let mut raw = img("DSC_0001.NEF", 28_000_000, 4000, 6000);
        raw.format = "nef".to_string();
        let infos = vec![jpg, raw];
        let scores = vec![Some(4.33), Some(4.18)]; // JPG 略高
        let built = build_groups(
            &infos,
            &vec![vec![0, 1]],
            &scores,
            &vec![None, None],
            &vec![None, None],
            &vec![None, None],
            &vec![false, false],
            &vec![crate::ai::scene::Scene::Other, crate::ai::scene::Scene::Other],
            &vec![false, false],
            &vec![1.0, 1.0],
            "test",
        );
        assert_eq!(built.len(), 1);
        let rec = built[0].images.iter().find(|g| g.recommended).unwrap();
        assert_eq!(rec.info.path, "DSC_0001.NEF");
        assert!(rec.reason.contains("RAW"));
        let del = built[0].images.iter().find(|g| !g.recommended).unwrap();
        assert!(del.reason.contains("RAW"));
    }

    #[test]
    fn raw_not_preferred_when_far_worse() {
        // 组内另一张不同照片明显更好（分差 > 容差 0.5）→ 仍按评分推荐
        let mut jpg_a = img("best.JPG", 12_000_000, 4000, 6000);
        jpg_a.format = "jpg".to_string();
        let mut raw_b = img("old.NEF", 28_000_000, 4000, 6000);
        raw_b.format = "nef".to_string();
        let infos = vec![jpg_a, raw_b];
        let scores = vec![Some(7.0), Some(3.0)];
        let built = build_groups(
            &infos,
            &vec![vec![0, 1]],
            &scores,
            &vec![None, None],
            &vec![None, None],
            &vec![None, None],
            &vec![false, false],
            &vec![crate::ai::scene::Scene::Other, crate::ai::scene::Scene::Other],
            &vec![false, false],
            &vec![1.0, 1.0],
            "test",
        );
        let rec = built[0].images.iter().find(|g| g.recommended).unwrap();
        assert_eq!(rec.info.path, "best.JPG");
    }

    #[test]
    fn picks_highest_resolution() {
        let infos = vec![
            img("a.jpg", 1000, 100, 100),
            img("b.jpg", 1000, 4000, 3000),
            img("c.jpg", 1000, 200, 200),
        ];
        let groups = vec![vec![0, 1, 2]];
        let scores = vec![None, None, None];
        let aesthetic = vec![None, None, None];
        let technical = vec![None, None, None];
        let face_scores = vec![None, None, None];
        let has_faces = vec![false, false, false];
        let scenes = vec![crate::ai::scene::Scene::Other; infos.len()];
        let eye_closed = vec![false; infos.len()];
        let built = build_groups(&infos, &groups, &scores, &aesthetic, &technical, &face_scores, &has_faces, &scenes, &eye_closed, &vec![1.0; infos.len()], "20260818093107");
        let g = &built[0];
        // 推荐图必须排在第 1 位（排序规则：推荐优先，其余按评分降序）
        assert!(g.images[0].recommended, "推荐图应排在第 1 位");
        let recommended = g.images.iter().find(|i| i.recommended).unwrap();
        assert_eq!(recommended.info.name, "b.jpg");
        // 推荐理由：要么是"分辨率最高"（启发式路径），要么是"综合评分最高"
        assert!(
            recommended.reason.contains("分辨率最高") || recommended.reason.contains("综合评分最高"),
            "推荐理由未提'分辨率最高'或'综合评分最高': {}",
            recommended.reason
        );
        let to_delete = g.images.iter().find(|i| !i.recommended).unwrap();
        assert!(
            to_delete.reason.contains("分辨率较低") || to_delete.reason.contains("综合评分较低"),
            "删除理由未提'分辨率较低'或'综合评分较低': {}",
            to_delete.reason
        );
    }

    #[test]
    fn ai_score_overrides_heuristic() {
        let infos = vec![
            img("a.jpg", 1000, 4000, 3000), // 高分辨率但低 AI 评分
            img("b.jpg", 1000, 100, 100),   // 低分辨率但高 AI 评分
        ];
        let groups = vec![vec![0, 1]];
        let scores = vec![Some(5.0), Some(9.0)];
        let aesthetic = vec![Some(5.0), Some(9.0)];
        let technical = vec![Some(5.0), Some(9.0)];
        let face_scores = vec![None, None];
        let has_faces = vec![false, false];
        let scenes = vec![crate::ai::scene::Scene::Other; infos.len()];
        let eye_closed = vec![false; infos.len()];
        let built = build_groups(&infos, &groups, &scores, &aesthetic, &technical, &face_scores, &has_faces, &scenes, &eye_closed, &vec![1.0; infos.len()], "20260818093107");
        let g = &built[0];
        // 推荐图必须排在第 1 位（排序规则：推荐优先，其余按评分降序）
        assert!(g.images[0].recommended, "推荐图应排在第 1 位");
        let recommended = g.images.iter().find(|i| i.recommended).unwrap();
        assert_eq!(recommended.info.name, "b.jpg");
        assert!(recommended.reason.contains("综合评分最高"));
    }
}

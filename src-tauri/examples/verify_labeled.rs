//! 闭眼标注集回归：按文件名前缀「组N-」分组，对每组跑完整 AI 链路，
//! 输出每张的分项与综合分，检验「睁眼-实焦」那张是否被推荐。
//! 是闭眼/眼对焦参数调整（`MESH_RAW_*`/`OCEC_VETO_MAX` 等）的**回归基准**，
//! 调参后必须重跑并对照历史结论（当前 7/7，组4 极端侧脸为已知不可解）。
//! 输入 folder：含「组N-xxx」命名图片的本机标注集目录（路径不入库，见 PRIVATE.local.md）。

use pixsweep_lib::ai::engine::AiEngine;
use pixsweep_lib::ai::preprocess;
use pixsweep_lib::image_io;
use pixsweep_lib::models_dir;
use image::GenericImageView;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone)]
struct Img {
    path: String,
    name: String,
    width: u32,
    height: u32,
    size: u64,
}

fn main() {
    let _ = env_logger::Builder::from_default_env().try_init();
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1).cloned() else {
        eprintln!("用法: cargo run --example verify_labeled -- <标注集目录>");
        eprintln!("（标注集为「组N-睁眼-实焦 / 组N-闭眼-*」命名的本机目录，路径见 PRIVATE.local.md）");
        std::process::exit(2);
    };

    let model_dir = {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        if manifest.join("topiq_nr.onnx").exists() { manifest } else { models_dir() }
    };
    let engine = AiEngine::new(&model_dir).expect("引擎初始化失败");
    println!(
        "[diag] 后端 {} | face {} | eye {} | scene {} | nr {} | iaa {}",
        engine.backend().label(),
        engine.face_scoring_available(),
        engine.eye_status_available(),
        engine.scene_scoring_available(),
        engine.has_topiq_nr(),
        engine.has_topiq_iia(),
    );

    // 收集并按「组N」前缀分组
    let mut groups: BTreeMap<String, Vec<Img>> = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for e in entries.flatten() {
            let p = e.path();
            let Some(ext) = p.extension().and_then(|s| s.to_str()) else { continue };
            if !matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg") { continue; }
            let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
            // 组号 = 「组」之后的数字
            let key = name
                .chars()
                .skip_while(|c| *c != '组')
                .skip(1)
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>();
            if key.is_empty() {
                println!("[diag] 跳过未按组命名的文件: {}", name);
                continue;
            }
            let path = p.to_string_lossy().to_string();
            let Ok(img) = image_io::load_image_oriented(&path) else { continue };
            let (width, height) = img.dimensions();
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            groups.entry(key).or_default().push(Img { path, name, width, height, size });
        }
    }
    println!("[diag] 共 {} 组", groups.len());

    let total_start = Instant::now();
    for (gkey, imgs) in &groups {
        // 组内排序：睁眼-实焦 的放前，便于对照；同时保留原条目
        let mut in_group = imgs.to_vec();
        in_group.sort_by_key(|i| {
            // 标签排序键：0=睁眼-实焦(应保留) 1=其他(应降级)
            if i.name.contains("睁眼") && i.name.contains("实焦") { 0 } else { 1 }
        });
        let paths: Vec<String> = in_group.iter().map(|i| i.path.clone()).collect();
        let widths: Vec<u32> = in_group.iter().map(|i| i.width).collect();
        let heights: Vec<u32> = in_group.iter().map(|i| i.height).collect();
        let sizes: Vec<u64> = in_group.iter().map(|i| i.size).collect();

        // 完整链路（与 score_groups_with_ai 等价，但无缓存/无 db）
        let g0 = Instant::now();
        let tech_batch = preprocess::images_to_batch_topiq(&paths).expect("TOPIQ 预处理");
        let tech = engine.topiq_nr_scores(&tech_batch).unwrap_or_default();
        let aes = engine.topiq_iia_scores(&tech_batch).unwrap_or_default();

        let (face, has_face) = engine.face_scores(&paths);
        let eye_open = engine.eye_open_probs(&paths, &has_face);
        let scenes = engine.scene_scores(&paths);
        let focus = engine.focus_scores(&paths, &has_face);
        let g1 = Instant::now();

        let face_vec: Vec<f32> = face.iter().map(|o| o.unwrap_or(0.0)).collect();
        let comp = engine
            .composite_scores(
                Some(&aes),
                Some(&focus),
                Some(&face_vec),
                &has_face,
                &scenes,
                &eye_open,
                &widths,
                &heights,
                &sizes,
            )
            .expect("综合评分");

        // 结果：期望 0 号（睁眼-实焦）综合分最高
        println!("\n[组{}] ({:.1}s, {} 张)", gkey, g1.elapsed().as_secs_f64(), in_group.len());
        for (j, im) in in_group.iter().enumerate() {
            let is_keep = im.name.contains("睁眼") && im.name.contains("实焦");
            println!(
                "  {}  {} | tech={:.2} aes={:.2} face={:.2} has_face={} eye={:.2} focus={:.2} 综合={:.2}",
                if is_keep { "[保留]" } else { "[降级]" },
                &im.name[..im.name.len().min(20)],
                tech.get(j).copied().unwrap_or(0.0),
                aes.get(j).copied().unwrap_or(0.0),
                face_vec.get(j).copied().unwrap_or(0.0),
                has_face.get(j).copied().unwrap_or(false),
                eye_open.get(j).copied().unwrap_or(0.0),
                focus.get(j).copied().unwrap_or(0.0),
                comp.get(j).copied().unwrap_or(0.0),
            );
        }
        // 比对：睁眼-实焦 这张在 comp 里的索引是否为最高
        let keep_idx = in_group.iter().position(|i| i.name.contains("睁眼") && i.name.contains("实焦"));
        if let Some(k) = keep_idx {
            let keep_score = comp.get(k).copied().unwrap_or(0.0);
            let max_in_group = comp.iter().copied().filter(|s| s.is_finite()).fold(0.0f32, f32::max);
            let correct = (keep_score - max_in_group).abs() < 1e-6;
            println!("  => 睁眼-实焦 综合分 {:.2}，本组最高 {:.2}，{}", keep_score, max_in_group,
                if correct { "✅ 正确推荐" } else { "❌ 未胜出" });
        }
        let _ = g0;
    }
    println!("\n[diag] 总耗时 {:.1}s", total_start.elapsed().as_secs_f64());
}

# PixSweep 计划功能 · 详细技术实现

> 与 `docs/PLANNED.md` 配套。本文给出每项计划功能的**具体实现路径**：涉及到哪些文件/函数、算法、验证方法、风险。文件路径相对 `src-tauri/src/`（Rust），`src/` 为前端，`scripts/` 为脚本。
> ⚠️ 约定：改 `composite_scores`/`build_groups` 必须同步 `commands.rs`、`examples/verify_ai.rs`、`examples/verify_full.rs`、`db/store.rs` 与相关测试。

---

## R1. 打包白名单补全人像/场景/闭眼模型（最简单，先做）

**问题**：`scripts/build_release.ps1` 的 `$neededModels` 只列了基础评分模型，漏掉 `insightface/`、`scene/`、`eye/` 子目录 → 发布包缺人像/闭眼/场景模型，功能空转。

**改法**（`scripts/build_release.ps1`）：
1. 复制基础 flat 模型（现有逻辑）。
2. 新增复制子目录模型到 `$destModels` 对应子目录：

```powershell
# insightface 子目录
$subModels = @{
    "insightface" = @("det_10g.onnx", "2d106det.onnx", "genderage.onnx")
    "scene"       = @("mobilenet_v3_large.onnx", "mobilenet_v3_large.data", "labels.txt")
    "eye"         = @("ocec_l.onnx")
}
foreach ($sub in $subModels.Keys) {
    $subDir = Join-Path $modelDir $sub
    if (Test-Path $subDir) {
        $destSub = Join-Path $destModels $sub
        New-Item -ItemType Directory -Path $destSub -Force | Out-Null
        foreach ($m in $subModels[$sub]) {
            $src = Join-Path $subDir $m
            if (Test-Path $src) { Copy-Item $src (Join-Path $destSub $m) }
            else { Write-Host "  警告: 缺少 $sub/$m" -ForegroundColor Yellow }
        }
    }
}
```
3. 顶层 `$neededModels` 也补上 `topiq_nr_face.onnx` / `topiq_nr_face.onnx.data`（当前缺 → 人脸专评空转）。

**验证**：解压发布包 → 设 `RUST_LOG=info` 跑含人像的目录 → 日志应见「InsightFace 人脸检测就绪」「OCEC 闭眼检测就绪」「MobileNetV3 场景分类就绪」，不再只 warn 跳过。

---

## A1. 修 eye ROI + 闭眼判罚聚合（已落地）

### 背景（现状）
`ai/eye.rs::sample_eye_rgb_internal` 原用 5 关键点算相似变换，把模板眼位 (38,51)/(75,51) 反投影回原图采样 24×40。真图验证发现 OCEC 对几乎每张人脸判"单眼 prob_open=0.00"（眼镜/旋转/偏脸），旧 `min(open)` 恒 0 → 全脸 ×0.5，无区分度。

### 根因（真图 A/B 定位）
1. **采样**：`det_10g` 5 关键点被眼镜/旋转带偏 → 单眼 ROI 采到皮肤/眼镜（`verify_bbox` 干净 ROI 图可见：0.00 眼采到皮肤、0.93 眼正中眼球）。
2. **聚合**：`min(左,右)` 太脆弱，任一眼噪声 0.00 就压垮整张脸。

### 已实施
**方案一（采样重锚定）**：`sample_eye_rgb_internal` 不投影模板，直接以 `face.landmarks.left_eye/right_eye`（原图坐标）为中心，沿眼线方向采 24×40；`inv_scale = 眼距/37` 保持尺寸与旧投影一致。删除 `crop_eye_window`。仅"模板投影采偏但关键点准"的情形受益。

**聚合改 max**：`eye_open_probs` 返回 `l.max(r)`（阶段三）；`composite_scores`/`eye_penalty` 只对 `open<0.5`（双眼都闭）降权。`AI_FACE_CACHE_SCHEMA`=3（旧 v2 缓存失效）。

### 验证（A/B 实测）
`verify_full` 对比改前（min）/改后（max）：

| 图 | 闭眼 L/R | max | 综合 改前→改后 |
|---|---|---|---|
| 竖屏-实焦-正脸（戴镜） | L=0.00 R=0.93 | 0.93 | 3.40→**6.79** |
| 竖屏-虚焦-侧脸 | L=0.00 R=0.00 | 0.00 | 3.30→3.31 |
| 宋宇芳（仰躺戴镜） | L=1.00 R=0.00 | 1.00 | 3.41→**6.81** |
| 邹存、谷琛（大脸） | L=0.01 R=0.01 | 0.01 | 3.16→3.17 |

- **通过**：睁眼（实焦/宋宇芳）综合明显高于虚焦；实焦>虚焦排序保留；单测全绿。
- **残余**：双眼都判闭（邹存、小脸合影）仍降权——OCEC 对"双眼都读低"的脸本就读低，属模型/人脸质量局限，非本次回归。

### 风险 / 遗留
- 关键点本身不准（眼镜/旋转）时，方案一救不回；若需彻底，考虑"A2 对齐后按固定模板位采样"或换更稳的闭眼模型。
- 左右眼交换对 eye 采样无实质影响（两眼都采、取 max，无需额外纠正），方案二不必要。

---

## P1. 人脸检测去重：一图只检一次（纯性能）

### 现状
`ai/engine.rs` 里 `face_scores`（671 行）与 `eye_open_probs`（517 行）**各自调用一次 `face_engine.detect()`**，同一图 det_10g 两遍（~50ms/遍）。

### 方案
一次扫描内缓存"图 → 检测结果"：
- 新增中间结构，如 `engine` 内一个 `HashMap<path, Vec<Face>>`（或按批次缓存），`face_scores` 先调用并缓存，`eye_open_probs` / 未来 `scene` / `get_face_landmarks` 直接复用。
- `commands.rs::score_groups_with_ai` 目前把人脸专评、闭眼分成两个独立 block，可合并为"先统一检测一次，再分别跑 TOPIQ-NR-Face 与 OCEC"。

### 改法要点
- `engine.rs` 加 `fn detect_faces_cached(&self, path) -> Option<Vec<Face>>`（带 `OnceCell`/`HashMap` 缓存，key = path）。
- `face_scores`、`eye_open_probs` 改为调用它。注意**线程安全**（`parking_lot`），并处理缓存淘汰（一次扫描内即可，不跨扫描持久化）。

### 验证
- 大目录真实照片集扫描耗时下降；分数不变（结果应与去重前完全一致，因 det_10g 确定性）。
- 用 `verify_full` 对比改前/改后输出一致。

---

## A2. 全仿射对齐（中优先级，需实证）

### 现状
`ai/insightface.rs::affine_params` 算相似变换（`Affine{cos_t,sin_t,inv_scale,src_center,tmpl_center}`），`tmpl_to_src` 用逆旋转变换。`align_face` 用它做反向 warp。

### 改法（6-DOF 全仿射）
- 用 5 点**最小二乘**求 `A(2×2)`、`b(2×1)`，使 `A·tmpl_i + b ≈ src_i`（src=检测到的 5 关键点，tmpl=112 模板 5 点）。
  - 因为有 5 点、共 10 个方程解 6 个未知数，是超定 → 正规方程最小二乘。
  - 模板点固定，`MᵀM`（3×3）与其逆可**预计算**一次（`template` 图固定），运行时只做矩阵乘。
- 替换：
  - `affine_params` → 返回 `[f32;6]`（a11,a12,b1,a21,a22,b2）。
  - `tmpl_to_src(tx,ty) = (a11*tx + a12*ty + b1, a21*tx + a22*ty + b2)`（线性映射，无需反三角函数）。
  - `align_face` 反向 warp 随之简化。
- `eye.rs::crop_eye_window` 若需同步，则复用同一 `A,b`。

**模板用哪个？** 用现有 `[(38,51),(75,51),(56,71),(41,92),(71,92)]`（112 坐标）即可。


### 验证（A/B，劣化即弃）
- `examples/verify_full.rs`：对比实焦/虚焦、侧脸、戴眼镜的**人脸专评分**与 `min(open)`。
- **通过标准**：① 实焦>虚焦 不劣化；② 侧脸/眼镜 crop 更"正"（`verify_bbox` 导出的 `align_face crop` 肉眼更规整）；③ 眼 ROI 若联动则更稳。
- **否决标准**：任一排序倒挂 / 人脸分明显退化 → 回退方案（git revoke）。

### 风险（高）
- 全仿射引入剪切，正常脸可能被微拉变形 → 评分抖动。det_10g 关键点若本身有噪（眼镜/侧脸），最小二乘会把噪放大到 shear。
- 这是对"实焦>虚焦"扰动最大的改动。**必须先做 A1，A2 单独隔离验证。**

---

## A3. 场景权重数据校准（中优先级）

### 数据
- 从真实照片集中，挑选多组"同一内容但不同质量/场景"的样本，人工标注"哪张更好"（正面：人像/风景/宠物 vs 反面）。
- 记录每张的 `aesthetic/technical/face_score/scene/heuristic` 五元组 + 标注（谁更好）。

### 方法
- 目标：调 `weighted_score` 的 `W_*` 权重，使"综合分更高者"贴合"标注更好者"。
- 用网格搜索 / 简单线性回归（或 scikit-learn 在 Python 侧离线跑，输出权重常量，再回填 `ai/engine.rs` 的 `W_*`）。
- 约束：权重单调性合理（技术分对清晰度最关键，人脸分只对人像档生效），归一化到和为 1。

### 验证
- 保留一组不参与训练的样本做盲测，统计"推荐即最好"准确率，与当前常量权重对比，需有提升。
- 权重回填后跑 `verify_full` 在代表性组上人工确认优先级符合直觉。

---

## P2/P3. 解码复用与体验（低优先级，思路）

- **P2 解码复用**：人脸专评、闭眼、场景三者各自 `load_image_oriented` 解码一次。可在 `score_groups_with_ai` 内先统一解码每张图（`HashMap<path, RgbImage>`），三个模型共享。注意内存（大目录一次性持有解码图会爆内存，需分批/限流）。
- **P3 体验**：`ScanProgress` 已按阶段推送；可加"当前阶段内已完成/总数"细粒度进度与当前文件名展示。是否做以你的排期为准。

---

## 附：常用验证命令

| 目的 | 命令 |
|------|------|
| 后端单测 | `cargo test`（Zig 工具链 + 非沙箱，env 见 FEATURES §9） |
| 真图评分/闭眼/场景 | `cargo run --example verify_full -- <图片...>` |
| 眼 ROI / 人脸对齐可视化 | `cargo run --example verify_bbox -- <图片...>`（导出 crop/ROI 到 %TEMP%） |
| 人脸框坐标 | `cargo run --example verify_landmarks -- <图片...>` |
| 前端类型检查 | `npx tsc --noEmit` |
| 前端测试 | `npx vitest run src/components/__tests__/GroupCard.test.tsx` |

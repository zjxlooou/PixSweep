# CODING_HISTORY — PixSweep 开发历史

> 本文档按时间线记录 PixSweep 从启动到现在的**需求变动**和**技术变更**，方便回溯
> "为什么代码是现在这个样子"。新功能/重构后应在此追加记录。

---

## 项目定位

**PixSweep**：Windows 本地图片去重桌面应用。扫描文件夹，用感知哈希（dhash + ahash）双哈希聚类找出重复/相似图片，再用「人像优先」的 AI 综合评分（人脸专评 / 闭眼 / 眼部对焦 / 场景 / 技术美学）推荐组内最佳照片，帮助清理重复图片、释放硬盘空间。

- 技术栈：Tauri 2 + Rust（后端）+ React/TypeScript（前端）+ ONNX Runtime
- 硬件约束：x86 + Win11 + 16G 内存 + 6G 显存，**不绑定特定 GPU 生态**（DirectML 全 GPU 通用）

---

## 时间线

### 2026-08-16 — 项目启动 + 核心架构 + 打包突破

**需求**：本地图片去重工具，扫描文件夹找相似图片、智能推荐最佳。

**完成**：
- 项目设计（`docs/DESIGN.md`）+ 完整实现（22 个 Rust 文件 + 12 个 TS 文件）
- **关键技术突破**：在无 MSVC 的 Windows 机器上编译 Tauri 应用
  - xwin 下载 Windows SDK + zig 作 C/C++ 编译器 + rust-lld 链接
  - 自研 `zigwrap`（zigcc/zigcxx/ziglib/zigrc）桥接 wrapper
- 核心链路：扫描 → pHash 初筛 → CLIP embedding → 聚类 → 评分 → 推荐

**关键 bug 修复**：
- Tauri localhost 拒绝连接 → 需 `custom-protocol` feature
- 大目录扫描卡死（9000+ 图片）
- 白屏：事件大 payload 经 evaluate_script 传输 → 改 invoke 返回
- serde flatten 导致前后端结构不匹配
- **沙箱 ACL 污染**：沙箱内编译会破坏 target 的 ACL，导致增量编译"拒绝访问" → 编译全程非沙箱

### 2026-08-17 — AI 功能落地 + CUDA → DirectML

**需求**：AI 评分（推荐组内最佳照片）。

**完成**：
- CLIP 推理验证（CPU 版）→ GPU 版（CUDA，RTX 2060 验证）
- 五大优化：组序号、点击预览、推荐/删除理由、NIMA 美学评分、CLI
- 文件日志系统（GUI 无控制台，日志落盘）

**技术变更**：
- **CUDA → DirectML**：发现 onnxruntime_providers_shared.dll 缺失导致 CUDA 静默回退 CPU，且 CUDA 生态（cuDNN 等）在无 MSVC 机器上获取困难。最终切换 DirectML（DirectX 12，NVIDIA/AMD/Intel 全通用，无 CUDA 依赖）。

### 2026-08-18 — 双维度评分 + 预览重构 + 回收站 + 增量扫描

**需求**：更精细的评分（技术 + 美学双维度）、更好用的预览、安全的删除。

**完成**：
- 摒弃 CUDA 生态 + 双维度评分（方案 A：CLIP 美学 + NIMA 技术）
- 批次号 + 组号系统（日志追溯）
- 预览组件彻底重构（三栏布局，参考 QQ 空间相册）
- 大量预览 bug 修复（切换组、JPEG 编码、缩放、fitScale、时大时小）
- 聚类误判修复：dhash + ahash 双哈希校验（AND 逻辑）
- **临时回收站改造**（替代系统回收站）：误删 1285 张照片后，实现独立隔离区
- 无限滚动 → 滑动窗口（防爆内存）
- 增量扫描 + AI 评分增量缓存
- 纯键盘操作

### 2026-08-19 — MCP + 测试 + 打包 + 交互优化

**需求**：外部 Agent 可操作应用（MCP）、完善测试、优化交互。

**完成**：
- **MCP server**（HTTP JSON-RPC，127.0.0.1:18765），12 个工具覆盖完整能力
- vitest 前端测试（18 个测试）+ test_e2e.sh 端到端脚本（8 阶段）
- 打包压缩优化（7-Zip mx=9）
- 窗口不可见 bug 修复（多显示器位置记忆到屏幕外）
- 交互优化：单条删除无确认、红框选中、右键菜单、双击预览

### 2026-08-20 — 模型升级 + EXIF + 确定性 + 清理重导出

**需求**：① EXIF 方向 bug 修复；② 更精细的评分模型（6G 显存门槛）；③ 代码清理整理。

**完成**：
- **EXIF 方向校正**：竖拍照片自动旋转（`image_io.rs`）
- **评分确定性修复**：DirectML 需 `parallel_execution(false)` + `memory_pattern(false)` + `intra_threads(1)`，否则同图时高时低
- **模型升级研究**（`docs/MODEL_UPGRADE_RESEARCH.md`）：调研 NR-IQA/Aesthetic SOTA
- **TOPIQ 集成**：技术评分 CLIP-IQA+ → TOPIQ-NR（KonIQ SRCC 0.930）；美学 LAION V1 → TOPIQ-IAA（AVA SRCC 0.791）
- **TOPIQ-NR 重新导出走 GPU**：原第三方 ONNX 因 pytorch 2.9.1 图结构 DirectML 初始化失败，用 pyiqa 官方权重 + torch 2.11 重导出后 DirectML 正常
- 清理：删除 node_modules、导出脚本、`.tmp_torch` 临时环境

### 2026-08-21~22 — 人像优先评分体系重构（type-first pipeline）

**需求**：评分从"技术 + 美学双维度"升级为「人像优先」体系——人脸、闭眼、眼部对焦、人像美学分主导；并加入对焦（清晰度）维度。

**完成**：
- **type-first 分支**：风景/其他 = 通用（对焦 + 美学）；宠物 = 定位眼部（暂无动物眼模型，回退整图对焦）；人像 = 人脸/关键点 → 闭眼 → 眼部对焦 → 人像美学。
- **对焦指标**（`ai/focus.rs`）：灰度拉普拉斯方差（模型无关），归一化到 `FOCUS_REF=1024`，人像取眼部 ROI、非人像取整图。
- **两级代理图**（`cache/proxy.rs`）：整图模型用 2048 代理图；人脸/眼/眼对焦用原图（降采样破坏小脸眼精度，OCEC 0.93→0.09）。
- **闭眼聚合改 `max(open_l, open_r)`**：只有双眼都闭才降权（单眼 ROI 采偏不再压垮整脸）。
- **Eye ROI 重锚定**：直接以检测眼关键点为中心、沿眼线方向采样。
- **场景前置**：先判场景类型，再按类型走对应评分分支；人像由 InsightFace 覆盖（ImageNet 无 person 类）。
- **UI 4 标签**：类型 / 闭眼 / 失焦 / 综合。
- **暂停/继续扫描**：哈希阶段分批并行（256/批）+ AI 评分各子阶段 chunk 边界检查暂停标志。
- **扫描排序稳定**：UnionFind 分量确定性排序（HashMap 遍历顺序随机 → 显式排序）。

### 2026-08-23 — 代码整理 + 仓库规范 + 发布

**需求**：整理代码、剔除无用文件、规范目录、整理脚本与文档、发 release。

**完成**：
- **删除死代码**：`get_face_landmarks` IPC + `FaceLandmarks` 类型（前端可视化已取消）、`pet_eye` 脚手架、`cluster_by_embeddings`/`cosine_similarity`（CLIP 聚类未接入）、`db` 旧 NIMA 单值缓存方法、`evict_cache`/`evict_proxy`、`image_dimensions`、`image_dimensions_oriented`、`topiq_face::model_exists`、`2d106det` 加载（未使用）、`quality_threshold`/`min_dimension` 设置字段（未实现）。
- **抽取重复**：`empty_scores` 占位评分、`removePaths`/`removeRestoredPaths` 合并。
- **仓库规范**：模型二进制（eye/scene）误入 git → `git rm --cached` + `.gitignore` 递归匹配 `**/*.onnx`/`**/*.data`/`**/*.bin`。
- **脚本**：新增 `scripts/build.ps1`（纯构建）+ `scripts/README.md`；`build_release.ps1` 补 `RUSTUP_TOOLCHAIN`、移除 `2d106det` 打包。
- **文档**：README / AGENT / FEATURES / TESTING / CLAUDE 同步当前架构。

### 2026-08-23 — 移除"暂停扫描"功能

**需求**：暂停扫描为兼顾暂停需要在哈希阶段分批并行（256/批）+ 各耗时循环边界轮询暂停标志，严重拖累扫描性能，移除该功能。

**完成**：
- 删除 `pause_scan` / `resume_scan` / `is_scan_paused` IPC 命令与 `scan_paused` 全局标志、`wait_if_paused` 轮询。
- `compute_hashes` 恢复为单次 `par_iter()` + 原子计数器推进度（去掉 256/批的分批屏障，rayon 持续占满核心，不再批间等待）。
- 前端删除暂停/继续按钮与状态，扫描中主按钮显示"扫描中…"禁用。
- README 功能表移除"暂停/继续"行。

### 2026-08-24 — AI 评分性能优化（并行预处理 + 双缓冲流水线）

**需求**：AI 评分阶段 GPU 利用率仅 ~2%、是扫描总耗时主要部分。根因：① 预处理逐张串行（GPU 等 CPU 解码）；② 模型 fix batch=1 逐张推理；③ 无流水线（decode → infer 完全串行）。要求优化不破坏评分确定性（同图两次评分逐位一致）。

**完成**（spec：`.scratch/ai-scoring-performance/spec.md`）：
- **阶段一（并行预处理）**：`ai/preprocess.rs` 三个 `images_to_batch_*` 统一走 rayon 并行解码/缩放/归一化共享实现（消除三处重复逻辑），rayon collect 保序 → 单张图计算与串行逐位一致。
- **阶段二（双缓冲流水线）**：`ai/engine.rs` 新增 `score_batch_scores(engine, paths, batch_size, progress)` 公共入口——producer 线程用 rayon 构建下一批 tensor，consumer 用现有单 session 逐张推理，`sync_channel(2)` 双缓冲让解码与 GPU 推理重叠。批内顺序/模型调用顺序/session 串行语义不变 → 确定性不变。`commands.rs::score_groups_with_ai` 与 `verify_ai` 均改调该入口（消除内联重复逻辑）。
- `verify_ai` 改为唯一性能基准 seam：单图完整流水线双跑做确定性命中断 + 输出 `score_batch_scores` 的 prep/infer/wall 分阶段计时。

**性能基线**（release，本机测试照片集 25 张（路径不入库），热代理缓存，首图技术分逐位一致 = 5.727092）：

| 指标 | 串行基线 | 阶段一 | 阶段二 |
|------|---------|--------|--------|
| 预处理 | 0.64s | 0.15s | 0.09s（完全重叠） |
| 推理 | 1.72s | 1.69s | 1.70s |
| 总计 wall | 2.55s | 2.03s（-20%） | **1.75s（-31%）** |
| 每张 | 102ms | 81ms | 70ms |

注：该目录是小图（600×800，0.48MP）；真实 24MP 原图的解码成本高一个量级（实测 ~625ms/张），双缓冲重叠收益会更大。

### 2026-08-25 — TOPIQ 动态 batch 重导出（官方 cfanet 权重）

**需求**：实机（NVIDIA 独显）观察 AI 评分 GPU/CPU 占用都极低——模型 fix batch=1 逐张推理是 latency-bound。用户选择放弃旧第三方权重，用官方 cfanet 权重重导出动态 batch ONNX（代价：评分值变化，需接受）。

**关键过程**：
- 现有 `topiq_nr.onnx` 输入声明静态 `[1,3,384,384]`，内部 transformer 的 Reshape/Unsqueeze/attention 布局 batch=1 硬编码遍布 → onnx 图手术不可行；onnx2torch 转回虽成功（补 Split/Resize v18 + 动态 ReduceMean converter，输出逐位一致 diff 2.4e-7），但 batch>1 同样报错 → 放弃保权重。
- 官方权重来源：HuggingFace `chaofengc/IQA-PyTorch-Weights`（`cfanet_nr_koniq_res50-9a73138b.pth` / `cfanet_iaa_ava_res50-3cd62bb3.pth`）。**IAA 必须显式传 `semantic_model_name='resnet50'`**（pyiqa 默认 swin，加载会 shape mismatch）。
- **torch 原生 CFANet 本来就支持 batch>1**（seq-first MultiheadAttention，L,N,C 布局），现有 onnx 的 batch=1 是当初 trace 导出固定的。唯一导出障碍：`forward_cross_attention` 里 `b,c,th,tw = feat.shape` 在动态 batch trace 时产生不可折叠 Shape/Gather → monkeypatch 静态空间尺寸版 forward（resnet50@384 末级特征固定 12×12）后导出成功。
- 导出脚本：`.scratch/ai-scoring-performance/export_dynamic.py`（opset 17，dynamic_axes batch）。
- **CUDA batch size 曲线**（25 张，NR+IAA）：batch=1: 1.81s / batch=4: 1.16s / **batch=16: 0.97s（最优）** / batch=25: 2.01s（非 2 次幂大 batch 劣化）→ 现有流水线 BATCH=16 恰好落在最优点，Rust 无需再调。

**完成**：
- 新模型单文件替换（无 `.data` 配对）：`topiq_nr.onnx`(177MB)、`topiq_iaa_res50.onnx`(277MB)；旧模型归档 `models-archive/topiq-batch1-20260825/`。打包白名单移除 `topiq_nr.onnx.data`。
- `engine.rs::topiq_nr_scores/topiq_iia_scores` 改为整批一次 run（输出 [N,1] / [N,10]，映射逻辑不变）。
- 打包脚本白名单去掉 topiq_nr.onnx.data；CLAUDE.md/AGENT.md 更新 batch 约定。
- **分数变化说明**：官方权重与旧第三方 onnx 不同源（同图 raw 差 ~5%），全部图片的技术/美学分变化、旧评分缓存作废（增量扫描会自动重算）；确定性本身不受影响（同图同分逐位可复现，实测首图技术分 5.504240 双跑一致）。
- 端到端（25 张热缓存）：推理 1.70s→1.31s，wall 1.75s→1.38s（每张 55ms）；叠加此前双缓冲优化，相对最初串行基线 2.55s→1.38s（**-46%**）。

---

### 2026-08-27 — 闭眼检测换检测器（方案 B）：MediaPipe 脸网格垂目信号 + 网格眼位修复 OCEC

**需求**：用户标注集（本机目录，7 组「睁眼-实焦」vs「闭眼/垂目」）基线只有 3/7 正确。上一轮已证实根因是任务与模型不匹配：OCEC 训练数据是眨眼式闭眼，对「垂目/低头看」姿态判 0.84~1.00（全开），调参/采样/权重扫描全部不可达 7/7。用户决策规则：垂目检测可行则换检测器（B），不可行则弱化闭眼惩罚（A）。

**调研结论（方案 B 可行）**：
- MediaPipe FaceLandmarker 478 点（含虹膜圆，Apache-2.0，yakhyo/mediapipe-face-mesh-onnx 的 ONNX 转换，4.6MB）输出虹膜 + 上下睑轮廓。
- 「睑缝垂直高 / 虹膜直径」是尺度无关的开度几何量：垂目时上睑下压 → 比值显著下降。
- 标注集实测 6 个闭眼组（含垂目组 2/4/5/6 与半闭眨眼组 7）几何分离全部正确；模型 ONNX 得来全不费工夫（无 blendshapes 也有虹膜点）。

**实施**：
- 新模型 `models/eye/face_landmarker.onnx`（可选，缺失自动退化为仅 OCEC）；打包白名单补录。
- `ai/eye.rs`：网格会话 + 方形旋转 ROI（中心=bbox 中心、边长=1.5×长边、角度=双眼连线近水平解）→ `mesh_result()` 返回归一化开度（锚点 raw≤0.10→0 / 0.42→0.5 / ≥0.65→1）与映射回原图的虹膜中心。
- **存量 bug 修复（意外收获）**：InsightFace 5 关键点的眼位实测系统性偏低约 10% 脸高（ROI 落在脸颊/鼻翼，OCEC 长期看到皮肤、睁眼图大量误判 0.00——导出 ROI PNG 直视确认）。现在 OCEC 的眼 ROI 改用网格虹膜中心（`detect_probs_at`），组2 睁眼图从 0.05 修复到 0.98。
- **融合规则**（`engine.rs::eye_open_probs`）：网格为主信号；OCEC 仅作「眨眼否决」——网格模棱两可（<0.85）且 OCEC 双眼都强判闭（min<0.2）时取较小值。网格为主是因为 OCEC 对刘海遮挡睁眼仍常有假闭误报（组1 实焦 0.15），min 全信它会颠倒组1。网格缺失时回退 OCEC 原 max 语义。
- MESH_SCORE_GATE=0.001（极低：仅 InsightFace 已确认人脸后才调用，且输出是软惩罚，极端侧脸「方向大致正确的弱信号」好过没有）。

**回归**：标注集 6/7 正确（基线 3/7；组1 7.63vs6.21、组2 7.79vs3.90、组3 7.84vs4.16、组5 7.43vs3.81、组6 7.53vs5.75、组7 7.90vs3.85）。`cargo test` 39 通过。

**已知未决**：组4（极端侧脸低头）两种信号都不可用（网格置信门拦截、关键点采样无网格眼位可修），且其对焦信号本身反向（闭眼图 10.00 > 睁眼图 4.67），眼信号即使修复也救不了该组，需另想办法（如人脸姿态辅助）。

**决策记录**：早期「只用 DirectML」决策已被「CUDA→DirectML→CPU 三级回退」取代（efaac91 加真检测防误报），架构演进表相应更新。

---

### 2026-08-27（晚）— FP16 化主力模型 + 移除 CLIP/LAION 后备（体积 -72%）

**需求**：四目标——体积、性能、人像美学可靠性、非人像美学可靠性。先建客观评测基准再动生产：
Wikimedia Commons 4 类（人像/狗/风景/食物）× 合成降级（blur/jpeg/quarter/dark）= 357 张「原图>降级」有序对，
以降级敏感度/Cohen's d/单调性/fp32↔fp16 一致性为无标注可靠性的客观代理。评测在 DirectML GPU 上跑
（`.scratch/model-research/`，REPORT.md 有全表）。

**落地改动**：
- **三主力 FP16**（onnxconverter_common，IO 保持 FP32，引擎零改动）：topiq_nr 177→88.7MB、
  topiq_iaa 277→138.8MB、topiq_nr_face 178→89.8MB；与 fp32 的 ρ≥0.9998、降级敏感度逐位一致、
  DML 上更快（29ms vs 38ms 等）。注意坑：转换默认写外部 `.data` 且引用路径跟随 onnx 文件名，
  生产重命名必须内嵌单文件（`convert_model_from_external_data`）。
- **移除 CLIP/LAION 后备**（-489MB）：clip-vit-b32-visual.onnx、clipiqa_model.onnx+.data、
  aesthetic_linear.bin 归档至 `models-archive/clip-removal-20260827/`。engine.rs 的 clip_session/
  clipiqa_session/aesthetic_head 及 extract_embeddings/aesthetic_scores/clipiqa_scores/has_clipiqa
  全部删除；`AestheticHead` 移除；技术后备链变为 TOPIQ-NR → NIMA，美学无后备；`backend` 字段改为
  「首个成功加载的评分模型会话的后端」。前端/TS/MCP 的 `clip_model_available` 字段删除。
- 发货模型体积 ~1.12GB → **~310MB（-72%）**。

**新架构候选结论**：hyperiqa（KonIQ，55MB fp16）全变体敏感度第一（d=1.20）但 512² 输入 GPU 118ms/张
且人像分数系统性偏低——适合「非人像场景的质量辅助信号」，未接入；maniqa/nima(AVA) 因 torch 2.13
导出层兼容问题止损未采用。

**目标3 发现（未落地，待第二轮）**：nr_face 对欠曝人像不扣分（盲区），而 topiq_nr 直接打人脸 crop
对 blur/jpeg/quarter 全部强敏感（+0.21~0.28）；两者 ρ=0.59 互补 → 可用 nr-on-face 作为 nr_face 的
互补/否决信号。

**回归**：cargo test 39 全过、tsc/前端构建过、verify_labeled 6/7 与 fp32 基线一致（组4 已知不可解），
综合分仅 fp16 尾数差（≤0.01）。

---

### 2026-08-27（晚·二）— 落地人像/非人像双融合（目标 3+4）

**需求**：用户拍板把研究中两项融合建议落地。

- **人像质量融合（目标 3）**：`face = nr_face ⊕ nr-on-face 50/50`——TOPIQ-NR 直接对
  对齐 512 crop（缩到 384）打技术分，与 nr_face 加权。标定依据：nr_face 的暗光盲区
  （dark45 敏感 -0.038）在 w=0.5 归零、平均降级敏感 ×3.6（+0.010→+0.036），
  保留一半人脸特化信号。`AI_FACE_CACHE_SCHEMA` 4→5。
- **非人像美学融合（目标 4）**：场景≠人像时 `美学 = IAA ⊕ HyperIQA 50/50`
  （hyperiqa_fp16 55MB 单文件 `models/hyperiqa.onnx`，512² raw [0,1] 输入，
  线性校准 `3.2251*h+3.2467` 对齐 IAA 值域后融合）。commands 在场景判定后新增
  融合段并重写回缓存。`AI_SCORES_CACHE_SCHEMA` 新增（v2），旧美学缓存一次性失效。
- 新增 `preprocess::images_to_batch_raw01_512 / face_crops_to_batch_topiq`、
  `engine.has_hyperiqa / hyperiqa_scores`、常量 `FACE_FUSION_NR_FACE_WEIGHT /
  HYPERIQA_CAL_A/B / HYPERIQA_FUSION_WEIGHT`。

**回归**：cargo test 39 ✓、tsc/前端 ✓、verify_labeled 6/7 保持（四组正确裕度健康，
组4 已知不可解不变）。综合分整体小幅上移（人像融合把部分被误压的人像分修正）。

---

### 2026-08-28 — 0.8.1：相机 RAW 多格式兼容（23 种）

**需求**：支持主流相机品牌原图格式（RW2/NEF/ARW/CR3/RAF/ORF/DNG 等）进入扫描→去重→评分→显示全链。

**选型**：rawler 0.7.2（纯 Rust，dnglab 的解码核心，LGPL-2.1 依赖）——零 C 编译依赖，契合无 MSVC 工具链；LibRaw 绑定因需编 C++ 被排除。

**实现**（唯一解码漏斗接入，全链自动生效）：
- `image_io::load_image_oriented` 增 RAW 分支：机内嵌预览优先（`full_image` > `preview_image` > `thumbnail_image`，相机端已完成去马赛克/白平衡/降噪，毫秒级），全无嵌入预览才回退全显影（demosaic→sRGB，RW2 实测 0.4s）；两条路径手动应用 EXIF orientation（官方 `Orientation::from_exif`）。
- `scanner/walker` 增 23 种扩展名 + 单测；`RAW_EXTENSIONS` 与 walker 清单两处需同步维护（已注明）。
- 缩略图（dataUrl 由后端生成）、AI 评分、代理、哈希、对焦全部走 `load_image_oriented`——RAW 无需任何单独适配。扫描阶段宽高 0×0 属瞬态，哈希阶段真解码后回填。

**验证**（样张来自 raw.pixls.us，代理下载；部分 LFS 慢件用 HEAD 限 45MB + 短超时规避卡死）：
- **RW2/NEF/ARW/CR3 四品牌 100% 通过**：四家均含机内嵌全尺寸预览（1920×1440~5088×3392），
  全显影回退 0.4~1.6s；四品牌全过生产漏斗 + verify_ai 完整评分链 + 确定性双跑逐位一致；
  Sony ARW 实测竖图方向旋转正确（预览 1920×1080 横 → 漏斗输出 1080×1920 竖）；
  Canon CR3 为 EOS R5 的 CRAW 压缩变体（ISO-BMFF 容器）也通过；scan_folder 正确收录；walker 扩展名单测过。
- RAF/ORF/DNG 等样张因 pixls LFS 慢件下载受限未逐一实测——覆盖率由 rawler（dnglab 同引擎）背书；
  个别变体解码失败时该文件被优雅跳过（hash=None，不参与分组评分，不崩溃）。

**版本**：0.8.1；发布包 429MB（较 0.7.3 的 999MB -57%）。

### 2026-08-28 — 统一前置代理（规格重做）

**需求**：代理生成规则统一为——满足任一条件（分辨率 >2K、文件 >2MB、RAW 原片）即把源文件转为一张 **<2K 且 <2MB 的 JPG 代理**，后续处理只读代理；代理图放**临时文件夹**；进入程序时工具栏"临时文件夹"按钮旁显示磁盘占用。

**实现**（`cache/proxy.rs` 重写，缓存版本 v3）：
- 触发：最长边 >2048 ∨ 文件 >2MiB ∨ `is_raw_image`；不触发的普通小图直接解码使用不落盘。
- 输出保证：最长边 ≤1920（<2K）+ JPEG 质量阶梯（95→88→80→70→60）取首个 <2MiB 档，全超限降边 1280 重试；极端噪声图单测覆盖。
- 存放：`app_data_dir()/quarantine/proxy/`（临时文件夹子目录）；旧版程序根 `proxy/` 首次访问自动清除。清空临时文件夹/清理缓存照常可清代理（可重建）。
- 消费端全统一：人脸检测/闭眼网格+OCEC/眼对焦/nr-on-face crop 原本走原图全分辨率，本次一并改走 `ai_proxy`（精度安全性有依据：对焦归一 1024/眼 ROI 归一 24×40、SCRFD letterbox 640、网格几何比例，均对分辨率不敏感）。**哈希与缩略图保持原图**（哈希值稳定性）。
- 新命令 `get_temp_folder_stats`（递归统计隔离区目录），Toolbar 按钮 `临时文件夹 · {占用}`，进程序/扫描结束/临时文件夹操作/清理缓存后刷新。

**验证**（测试集：三品牌 RAW+同名 JPG 成对 + 相似标注集，见 PRIVATE.local.md）：
- 规格断言 5/5（RW2 35MB、NEF 28MB、ARW 47MB、22.9MP JPG、10MB JPG）：代理均 <2K 且 <2MB，缓存命中 ~0.02s（原图解码 0.4s 的 1/20）。
- 闭眼标注集回归 **7/7 全对**（统一前基线 6/7，组4 极端侧脸在本轮也通过）。
- 成对一致性：NEF 技术分差 ≤0.02；RW2 ≤0.08；ARW 技术/美学 ≤0.08。ARW 综合分差 0.5~0.9 来自**既有分辨率启发式**（ARW 内嵌预览 1.7MP vs JPG 24MP），非代理引入。
- `cargo test` 42/42、`tsc` 0 错、vitest 25/25。

### 2026-08-29 — 性能与内存大排查（用户实测反馈）+ RAW 推荐语义

**用户反馈三问题**：①指纹阶段后 CPU<10%、GPU<5%，扫描极慢；②内存 14GB 不正常；③RAW+JPG 同组推荐删 RAW，不合直觉。

**MCP 实测复现**（350 张测试集，打包版 + 内存探针逐阶段落日志）：
- 慢因：人脸专评 37s/闭眼 25s/对焦——**逐张串行**，且 SCRFD 每张图全链重复检测 3 次；GPU 会话互斥下 CPU 解码不重叠。
- 内存：rayon 全局池每逻辑核一线程 × 24MP 解码缓冲无上限并发 + **HyperIQA 按 16 张批量推理单次 run 触碰 3.2GB**（fix batch=1 的既有坑，0.8.0 融合实现就错了）；探针逐阶段定位到具体行。

**修复**：
1. `image_io::heavy_pool()` 专用 6 线程池（== 解码信号量数），哈希/预处理/三阶段并行全部 install 进池；`load_image_oriented` 全局解码信号量（上限 6）双保险。
2. engine 三阶段（人脸/闭眼/对焦）rayon 并行化——单张结果只依赖自身，保序回填，确定性不变；GPU 会话 Mutex 串行。
3. 检测共享缓存：`detect_cache`（路径→最大脸），SCRFD 每张全链只跑一次，闭眼/对焦复用。
4. **哈希改从代理图计算**：全库每张图只做一次全分辨率解码（生成代理那次），哈希阶段不再全尺寸解码；尺寸走文件头/传感器探针（`header_dimensions`）。指纹 v2 版本前缀使旧哈希缓存一次性失效。
5. HyperIQA 改逐张推理：9.6s→2.1s，内存 +3.2GB→+66MB，且修正批量静默错误结果。
6. 推荐器 RAW 优先规则（容差 0.5）+ 显式理由文案；两个单测。
7. mimalloc 尝试治堆驻留——zig 链接 v2/v3 均失败，弃用。

**实测效果**（350 张）：
- 耗时：193s → ~100s（AI 链 143→62s；闭眼 25→4.2s；哈希+代理生成 80→39s；HyperIQA 9.6→2.1s）
- 内存：任务管理器视角全程 WS 峰值 ~5GB→**1.6GB 稳定**（HyperIQA 修复后无尾部跳变）
- 推荐：115 个 RAW+JPG 混合组 112 个推荐保留 RAW（3 个为 RAW 预览过差超出容差）
- 回归：cargo test 44/44（新增 2 个 RAW 推荐测试）、闭眼标注集 7/7

### 2026-08-28（续）— RAW 分辨率源口径

**需求**：评分中的大小/分辨率比较一律用源口径。文件大小审计确认已全走 `ImageInfo.size`（源文件），但 RAW 的宽高来自机内嵌预览解码结果——Sony ARW 预览仅 1.7MP，与同画面 24MP JPG 对比时分辨率启发式被低估（ARW 综合分差 0.5~0.9 的主因）。

**实现**：
- `image_io::raw_source_dimensions`：rawler `raw_image(dummy=true)` 探针模式（只解析容器与尺寸，不分配不解码像素，实测 1~57ms）；尺寸取 crop_area → active_area → 全幅，按 EXIF 方向转正（竖拍宽高互换），与解码显示口径一致。
- 扫描哈希写回阶段对 RAW 无条件覆盖 `info.width/height`（缓存命中路径同样生效，顺带修复旧缓存里的预览尺寸）；`verify_ai` 诊断 example 同步。

**验证**：四品牌探针全对（Panasonic 2.6MP 预览→17.1MP、Sony 1.7MP→22.9MP 竖拍 4000×6000 与 JPG 完全一致、Nikon/Canon 本就一致）；ARW 成对综合分差 0.5~0.9 → 0.11~0.20（残余为预览真实清晰度差异）；RW2 成对综合分差 0.5 → ≤0.01；确定性双跑一致；cargo test 42/42。

### 2026-08-29（晚）— 功能封板整理（命名 / 死代码 / 函数拆分 / 文档同步）

**需求**：功能改动封板，做全面整理——命名优化、删死代码与冗余、拆函数抽公共工具降嵌套、
文档随代码更新、梳理测试与打包脚本；临时文件留本地不入 git；全量测试通过后汇报。

**完成**：
- **examples 修复与梳理**：修复 `dump_aligned_crops` / `verify_bbox` 缺 `mut` 的编译错误
  （`InsightFaceEngine::load` 需要 `&mut self`，此前 `cargo check --all-targets` 在这两个
  脚本上必失败）；删除已完成使命的一次性脚本 `dump_aligned_crops.rs`（nr_face 重导出
  评测用，git 历史可找回）；新增 `examples/README.md` 按「功能回归 / 人工诊断 / 一次性
  探针」三类编目全部脚本；`verify_labeled` / `preview_check` 去掉"用完即删"过时标注
  （前者是闭眼标注集回归基准，长期保留）。
- **评分结果束**：`commands.rs` ↔ `build_groups` 之间的 8 元组收敛为
  `quality/recommender.rs::AiScoreBundle`（`empty(len)` 生成 AI 未启用占位），
  `build_groups` 签名从 11 参数降到 4，消除易错位置参数（呼应"易变签名"痛点）。
- **拆分长函数**：`score_groups_with_ai`（约 390 行）拆为缓存装载（`load_score_caches`
  / `load_face_caches`）、场景线程（`spawn_scene_thread`）、人脸阶段（`run_face_phase`）、
  闭眼阶段（`run_eye_phase`）、对焦阶段（`run_focus_phase`）、HyperIQA 融合
  （`fuse_hyperiqa_for_non_portrait`）、场景合并（`merge_scene_results`）；
  `AiEngine::new` 的 7 段重复装载收敛为 `load_optional_session`。
- **公共工具抽取**：`Face::largest`（4 处"取最大脸"重复）、`scene_input_tensor` + `argmax`
  （scene.rs 两份重复预处理/argmax）、`ai::mos_from_bins`（NIMA 与 TOPIQ-IAA 同一
  10-bin 加权平均公式）、`image_io::apply_exif` + `develop_raw_oriented`（RAW 显影+转正
  两份重复）、`chunk_paths`（扫描阶段 5 处取路径重复）。
- **注释纠偏**：清除残留 CLIP/LAION 表述（types / phash / recommender / commands /
  CODING_HISTORY 评分链）；`get_full_image` 注释 1600 → 实际 3072；内存日志 tag 与
  实际阶段对齐（原"hash+cluster done"打在评分后）；`det_10_batched` 拼写；
  `lib.rs` 日志路径与"DirectML 单后端"过时描述；`BatchTensors` 改 derive(Default)。
- **打包脚本**：`build_release.ps1` 在 zip 校验通过后自动删除解包目录（此前每次打包
  残留约 1GB 解包副本，dist-package 已积压 15G）。
- **文档同步**：`docs/DESIGN.md` 整体重写为当前架构（原稿停留在立项期的
  CLIP 聚类 / SQLite / 单 DirectML 时代）；`docs/TESTING.md`（vitest 命令、测试覆盖描述）、
  `scripts/README.md`（examples 指引）、`AGENT.md`（`dump_eye_roi` 已删改指 `verify_bbox`、
  标注集回归结论 7/7、AiScoreBundle 签名约定）同步更新。

**验证**：cargo check --all-targets 零警告（含全部 examples）；cargo test 46/46；
vitest 25/25；tsc 零错误；e2e PASSED。

**新坑记录**：本机 `cargo test` 默认并行构建存在竞态——example/bin 单元随机报
`E0462 found staticlib xxx.dll.lib` / `E0786 invalid metadata`（失败点漂移，删
`target/debug` 重建不能根除）；`cargo test -j 1` 串行全量稳定通过。日常验证用
`cargo test --lib`（46 个单测全在 lib），需要 example 链接验证时用 `-j 1`。

---

## 评分模型演进（重要）

| 阶段 | 技术评分 | 美学评分 | 时间 |
|------|---------|---------|------|
| 初版 | NIMA（技术） | NIMA（美学，同模型） | 08-17 |
| 双维度 | NIMA technical | CLIP + LAION V1 线性头 | 08-18 |
| CLIP-IQA | CLIP-IQA+（主）/ NIMA（后备） | CLIP + LAION V1 | 08-20 早 |
| TOPIQ | **TOPIQ-NR（主）**/ CLIP-IQA+ / NIMA | **TOPIQ-IAA（主）**/ LAION V1 | 08-20 晚 |
| 人像优先 | 技术/美学作基础分 + **对焦** + **人脸专评** + **闭眼** | 同左（人像由人脸分主导） | 08-22 |

**当前评分链**（type-first，权重见 `ai/engine.rs`）：
- 技术：TOPIQ-NR（ResNet50，KonIQ-10k）→ NIMA（MobileNet 后备；CLIP-IQA+ 已于 08-27 移除）
- 美学：TOPIQ-IAA（ResNet50，AVA）；非人像场景 ⊕ HyperIQA 50/50 融合（LAION 线性头已移除，美学无后备）
- 对焦：灰度拉普拉斯方差（模型无关，`ai/focus.rs`）
- 人脸专评：TOPIQ-NR-Face（ResNet50，CGFIQA-40k，需 InsightFace 检测）⊕ nr-on-face 50/50（08-27）
- 闭眼：**MediaPipe 脸网格垂目开度为主信号 + OCEC 眨眼否决**（`ai/eye.rs`，只有双眼都闭才降权）
- 去重核心：**感知哈希（dhash + ahash）双哈希**（不依赖任何 AI 模型）

## 架构演进（重要）

| 项 | 旧 | 新 | 原因 |
|----|----|----|------|
| 推理后端 | CUDA | **DirectML** | 无 CUDA 生态依赖，全 GPU 通用 |
| 推理后端（再演进） | DirectML | **CUDA→DirectML→CPU 三级回退**（08-26） | POC 实测 CUDA 最快（~4× CPU）；EP DLL 能加载 ≠ 有 NVIDIA GPU，需真检测（efaac91） |
| 闭眼检测 | OCEC 单信号 | **脸网格垂目 + OCEC 眨眼否决**（08-27） | OCEC 训练数据是眨眼式，垂目判全开；网格虹膜几何 6/6 分离 |
| 数据存储 | SQLite（rusqlite） | **JSON 文件** | 避免 cl.exe 编译 SQLite C |
| 编译器 | MSVC | **zig + xwin** | 用户机器无 Visual Studio |
| 大对象传输 | 事件 emit（evaluate_script） | **invoke 返回** | 大 payload 白屏 |
| 回收站 | 系统回收站 | **临时回收站（隔离区）** | 误删后恢复困难，独立管理 |

## 关键坑与教训（供后续开发者）

1. **`.onnx.data` 配对文件**：CLIP-IQA+ 和 TOPIQ-NR 的 ONNX 权重存外部 `.data` 文件，打包白名单必须显式列出（扩展名是 `.data` 不是 `.onnx`，会被静默漏包）。
2. **DirectML 的 Reshape op**：CLIP-IQA+ 因 `Reshape` 不兼容走 CPU EP；TOPIQ-NR 原 pytorch 2.9.1 导出失败，需 torch 2.11 重导出。
3. **沙箱 ACL 污染（历史）**：旧 agent 沙箱环境编译会污染 target ACL；现环境无沙箱，若遇"拒绝访问"删 `target/` 重建。
4. **路径格式**：Windows 原生程序用 `盘符:/` 形式而非 Git Bash 的 `/盘符/`。
5. **git 仓库**：本仓库远程为 `github.com:zjxlooou/PixSweep`，但模型（~960MB）与 `.tools/`（2.2G）不入库；删除大目录前先向用户确认范围。本机私人路径集中在 `PRIVATE.local.md`（不入库）。

---

*最后更新：2026-08-29*

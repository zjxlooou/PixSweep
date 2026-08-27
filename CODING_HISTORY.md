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

## 评分模型演进（重要）

| 阶段 | 技术评分 | 美学评分 | 时间 |
|------|---------|---------|------|
| 初版 | NIMA（技术） | NIMA（美学，同模型） | 08-17 |
| 双维度 | NIMA technical | CLIP + LAION V1 线性头 | 08-18 |
| CLIP-IQA | CLIP-IQA+（主）/ NIMA（后备） | CLIP + LAION V1 | 08-20 早 |
| TOPIQ | **TOPIQ-NR（主）**/ CLIP-IQA+ / NIMA | **TOPIQ-IAA（主）**/ LAION V1 | 08-20 晚 |
| 人像优先 | 技术/美学作基础分 + **对焦** + **人脸专评** + **闭眼** | 同左（人像由人脸分主导） | 08-22 |

**当前评分链**（type-first，权重见 `ai/engine.rs`）：
- 技术：TOPIQ-NR（ResNet50，KonIQ-10k，DirectML）→ CLIP-IQA+（CPU EP）→ NIMA（DirectML）
- 美学：TOPIQ-IAA（ResNet50，AVA，DirectML）→ CLIP ViT-B/32 + LAION 线性头
- 对焦：灰度拉普拉斯方差（模型无关，`ai/focus.rs`）
- 人脸专评：TOPIQ-NR-Face（ResNet50，CGFIQA-40k，需 InsightFace 检测）
- 闭眼：**MediaPipe 脸网格垂目开度为主信号 + OCEC 眨眼否决**（`ai/eye.rs`，只有双眼都闭才降权）
- 去重核心：**感知哈希（dhash + ahash）双哈希**（CLIP 仅作美学后备，不参与聚类）

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

*最后更新：2026-08-27*

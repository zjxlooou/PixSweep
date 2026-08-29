# GPU 利用率优化计划（2026-08-29）

> 背景：用户反馈扫描期间 GPU 利用率低（<5%）。经实测拆解，AI 评分链（350 张，
> 62s）中**人脸检测阶段占 27s（43%）**——SCRFD 640×640 逐张串行推理，且
> det_10g 官方 ONNX **batch=1 硬编码**（insightface 发布版未用 PR #1781 的
> dynamic-axes 修复重导出）。GPU 利用率低不是 GPU 不够快，而是小模型单张
> 推理时 kernel launch / 拷贝开销占比过高、且阶段间存在 GPU 空窗。
>
> 原则：**利用率是手段，墙钟时间是目标**。每步都跑基准回归（闭眼标注集
> 7/7、RAW+JPG 成对一致性、确定性双跑）后才允许合入。

## 实测瓶颈画像（350 张，RTX 5070，0.8.3）

| 阶段 | 耗时 | 占比 | GPU 形态 |
|---|---|---|---|
| 人脸专评（SCRFD+align+nr_face） | 26.9s | 43% | 640×640 单张串行，batch=1 |
| 场景分类（MobileNet 224） | 12.7s | 20% | 批 64 预处理并行、推理串行 |
| TOPIQ nr/iaa（动态 batch 16） | 7.4s | 12% | 已批量化，形态最好 |
| 闭眼（网格+OCEC，复用检测） | 4.2s | 7% | 逐张串行 |
| HyperIQA（512 逐张） | 2.1s | 3% | fix batch=1 |
| 对焦 | ~2s | 3% | 无模型（CPU） |

## 优化项（按 ROI 排序）

### M1 — SCRFD 动态 batch 重导出（预计 AI 链 62s → ~45s）

- 现状：insightface PR #1781 已修复 SCRFD 导出脚本的 dynamic_axes，但
  buffalo 包的 det_10g.onnx 从未用修复版重导出；HuggingFace 有现成重导出
  （`ceyxprime/scrfd_640_batched` 等）可先验证。
- 做法：获得/重导出 dynamic-batch det_10g → 每批 letterbox 后 8~16 张
  一次推理 → 现有 Rust NMS 逐图解码输出（输出 shape 多一个 batch 维）。
- 验收：bbox/keypoint 与单张版逐位一致（小数值容差内）；人脸专评 ≤12s。
- 风险：重导出模型的输出布局/关键点顺序与 det_10g 有差异——用基准图
  矩阵比对；不匹配就自己走 PR #1781 脚本重导出。

### M1 — SCRFD 动态 batch 重导出（✅ 已完成 2026-08-29，结论修正）

**做了什么**：
- 会话副本池：det_sessions ×2 各自 CUDA 流 + detect 并行（重活池 6 线程）。实测 4 副本会使后续 HyperIQA 减速 6×（机制未明，与 det 会话显存/arena 相关），**2 副本最优**：人脸 27.4→24.4s。
- det_10g 批量手术（`scripts/make_det_batched.py`，权重不变）：输入 batch 动态化 + 9 Transpose [2,3,0,1]→[0,2,3,1] + Reshape [-1,K]→[0,-1,K]。三重验证全过（单张与原模型逐位一致、同图批量一致、跨帧零泄漏——用"A 图+随机噪声帧"判定法）。Rust 侧 detect_batch 集成 + face_scores 批量管线。
- **生产实测后决定不随包发布批量模型**：标注集上检测 2.7× 提速（12.7→4.6s），但生产 face 阶段 30.8s（vs 副本池 24.4s）——face 阶段大头是 nr_face 逐张推理与代理读取，检测提速被稀释；且批量会话运行后 HyperIQA 同样减速 6×。代码保留（模型文件存在即启用），未来 M2/M3 改变格局后可重新评估。

**M1 净收益**：350 张 AI 链 69→65s（约 6%），闭眼 7/7、44 单测全绿。

### M2 — IO Binding + 固定形状输入（预计再 -10~15%）

- ort 2.0.0-rc.13 已支持 `session.create_binding()`（IoBinding）与
  `ep::CUDA::with_cuda_graph(true)`。
- IO Binding：为固定形状输入（TOPIQ 16×3×384×384、scene 64×224、
  SCRFD 16×640×640）预分配 GPU 输入/输出缓冲 + pinned host staging，
  消除每批的 H2D/D2H 往返。小模型拷贝开销占比高，收益明显。
- 风险：绑定错设备反而变慢（GitHub #10000 的教训）——每模型单独基准。

### M2 — nr_face 逐张推理消除（✅ 调研结论 2026-08-29）

- **图手术不可行**：topiq_nr_face 的注意力结构把 batch 缠进 head 维（{256,B,256}→GEMM {256,256}），机械改图无法动态 batch。
- **正解是 torch 重导出**（与 0.8.0 的 TOPIQ-NR/IAA 动态 batch 同路线，官方 cfanet 权重 + dynamic_axes）——列为后续 M2.5。
- **IO Binding 暂缓**：ort rc.13 支持 IoBinding，但 nr_face 逐张推理的每 run 开销（输出分配+拷贝）占比小，收益远小于 M4 的阶段重叠；避免在 rc API 上过度投入。

### M4 — 阶段流水线化（✅ 已完成 2026-08-29，AI 链 65→52s）

- **scene ∥ face 跨模型并发**：场景分类（MobileNet 会话）与人脸专评（SCRFD+nr_face 会话）计算独立，用 `std::thread::scope` 线程并发，场景结果写独立缓冲、人脸阶段结束后合并（人像覆盖语义不变）。最终场景分布与串行版逐位一致（{Portrait:325/327, Other:21}）。
- **顺带修了一个真实并发 bug**：场景/人脸并发时同一代理图可能被双生成，`fs::write` 半写状态下另一方 `image::open` 读到坏文件——代理落盘改为**临时文件 + 原子 rename**。
- 实测（350 张热缓存）：AI 链 65→52s；全程 69→52s（-25%，含 M1）。
- 后续可重叠的组合（收益递减）：hyperiqa ∥ eye/focus（~2s）。

### M2.5 — nr_face 动态 batch 重导出（✅ 已完成 2026-08-29）

- `export_face_dynamic.py`（0.8.0 管线同款）：pyiqa `topiq_nr` + CGFIQA-40k 权重
  （`topiq_nr_cgfiqa_res50-0a8b8e4f.pth`，HF），static_forward th=tw=16（512 输入末级
  16×16，256 token 与旧模型一致；12 会导致非整因子 adaptive pool 无法导出），
  torch.onnx.export dynamic batch（dynamo=False）。
- 导出即语义正确：与 pyiqa eager 推理 spearman=1.00000、maxdiff=0。
- **FP16**（keep_io_types）：与 fp32 spearman=1.0、maxdiff=0.0011 MOS；84.6MB（fp32 168.6MB）。
- Rust `face_quality_scores` 批量 8 张一次前向，旧 fix batch=1 模型自动逐张回退。
- 旧模型（语义不同的静态 batch 版）归档 `models-archive/topiq_nr_face-static-20260829/`。
- **重要发现：GPU 显存只有 6GB**。大激活会话（fp32 批量 nr_face / det 批量 / 4 副本）
  会把 VRAM 顶到 91%+，后续小模型（HyperIQA）分配落入 WDDM 共享内存 → 减速 6×。
  fp16 后缓解。**显存预算是本机一等约束**——新增/扩大会话前必须查 nvidia-smi。
- 实测：HyperIQA 12.7s→2.2s（VRAM 修复）；AI 链 53.7→40.0s。

### M3 — CUDA Graphs（固定形状场景消除 launch overhead，预计再 -10~20%）

- 适用：形状完全固定的推理（SCRFD 批 16、scene 批 64、hyperiqa 单张）。
  TOPIQ 动态 batch 最后一块需 pad 到 16。
- 限制：启用 CUDA Graphs 后该 session **不可多线程并发 Run**；形状一变
  要重新捕获。逐 session 评估，只给收益明确的模型开。
- 与 M1/M2 叠加。

### M4 — 阶段流水线化（消灭阶段间 GPU 空窗，预计再 -15~25%）

- 现状：整库跑完一个阶段才进下一个（topiq → scene → face → eye → ...），
  阶段边界 GPU 空闲。
- 做法：不同 session 允许并发 Run——把 scene（MobileNet）与 face（SCRFD）
  重叠；或按 chunk 组织"一张图的全部模型工作"，GPU 永远有活干。
- 依赖 M1（face 批量化）先落地，否则流水线救不了串行 SCRFD。

### M3 — CUDA Graphs（❌ 已尝试 2026-08-29，结论：当前栈不可行）

- ort 2.0.0-rc.13 的 `ep::CUDA::with_cuda_graph(true)` 实测：
  - 初始化期同步 warmup 可完成图捕获（无并发时 OK）；
  - **扫描期间 replay 与其他会话（topiq 流水线）并发 → 进程直接崩溃**（无异常可捕获，native abort）。
  - 与 M4 的多会话并发架构根本冲突；官方文档也要求图模式只支持固定形状 + 固定 I/O 指针（需完整 IO binding 管线，rc API 不暴露用户可写设备缓冲）。
- **重开条件**：ort 升级到正式版且支持设备侧缓冲写入，或改为单会话串行架构（放弃 M4）。
- 预期收益本就有限（launch 开销在 scene/eye/hyperIQA 阶段约 1-2s），优先级最低。

### M5 — TensorRT EP（✅ 研究完成 2026-08-29，结论：**暂不上**，理由与重开条件如下）

**支持现状（ort 2.0.0-rc.13）**：
- `ep::tensorrt` 模块存在，需 cargo feature `tensorrt`（ort-sys/tensorrt）→ ort 会改用
  **TensorRT 构建的 onnxruntime**（含 onnxruntime_providers_tensorrt.dll）。
- 但 **TensorRT 运行库本身（nvinfer*.dll 等，数百 MB）不在 ort 下载物里**——需要随包
  分发或用户安装 TensorRT SDK；当前包仅 DirectML + CUDA 两个 provider DLL。

**预期收益——对小模型场景为负或接近零**：
- 社区实测：小模型上 TRT EP 反而比 CUDA EP 慢 4×（rust-birdnet-onnx #18）；
  NVIDIA 论坛亦有 CUDA EP 追平 TRT 的讨论（小模型/动态形状场景 TRT 部分回退）。
  TRT 的全图融合收益主要在大模型/固定形状高频推理。
- 我们的模型全是小模型（最大 84.6MB fp16），且 AI 链实测大头已是预处理/代理读取
  与逐张同步开销，kernel 本身不是瓶颈（face 阶段 GPU util 已能到 30-90%）。

**成本——与本机约束正面冲突**：
1. **显存 6GB**：TRT 引擎+workspace 独占设备内存，叠加现有会话必顶爆 VRAM
   （刚修完 VRAM 耗尽减速问题）；
2. 包体 +300MB~1GB（TRT runtime）；
3. 每模型每 GPU 首次构建引擎分钟级（trt_engine_cache_enable 可缓存，但驱动升级
   即失效重建）；
4. 数值差异风险（YOLOv8 #22354 有先例），需要全部基准重新校准。

**重开条件**：模型规模显著增大（如换 1B+ 级美学模型）；固定形状需求稳定
（放弃动态 batch）；用户 GPU ≥12GB。满足后再做 1 天 spike：
feature `tensorrt` + `trt_engine_cache_enable` + `trt_fp16_enable` + 三个主力模型
基准对比（秩相关 + 墙钟），实测决定。

**当前更优路径**：瓶颈已转移至预处理/代理读取（CPU）与逐张同步——继续收益最大的是
scene 批量化（MobileNet 图手术与 det_10g 同路线，可行）与 nr_face/det 更大批量，
均为零新依赖。

### M5 — TensorRT EP（原始占位，见上）（可选大杀器，最后评估）

- 社区报告 ~2× 收益（TRT 全图融合 + fp16 kernel），代价：包体 +~100MB
  TRT DLL、每 GPU 首次构建引擎分钟级（`trt_engine_cache_enable` 可缓存）、
  精度可能有差异（YOLOv8 有先例）。
- 注意：小模型 + 动态形状场景 CUDA EP 有追平 TRT 的先例（NVIDIA 论坛）。
  **决策点：M1~M4 做完后基准仍不达标，才做 1 天 spike 实测定去留。**

## 不做的事

- 不盲换更小的检测模型（det_500m 等）：人脸关键点精度是闭眼/眼对焦的地基，
  精度回归风险大于性能收益。先榨现有模型的工程空间。
- 不上多进程/多 GPU：单 GPU 单进程内并发（多 session 多流）已够。

## 验收基线（每里程碑必须全绿）

1. 闭眼标注集 7/7；
2. RAW+JPG 成对技术分差 ≤0.08 / 美学 ≤0.02；
3. 确定性双跑逐位一致；
4. AI 评分链墙钟时间（350 张）。

## 参考来源

- ORT CUDA EP 并发/流：https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html 、https://github.com/microsoft/onnxruntime/issues/23319
- IO Binding：https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html 、https://onnxruntime.ai/docs/performance/device-tensor.html
- TensorRT EP：https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html 、https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/
- SCRFD batch=1 硬编码与重导出：https://github.com/deepinsight/insightface/issues/2643 、https://github.com/deepinsight/insightface/issues/2879 、https://huggingface.co/ceyxprime/scrfd_640_batched

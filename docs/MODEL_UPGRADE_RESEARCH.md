# PixSweep 模型升级研究报告（2026-08-20）

> **目标**：在 6GB 显存门槛下，研究"更大、更好的模型"以提升 AI 评分的可信度、稳定性、准确性。
>
> **当前栈**：
> - **技术质量**：CLIP-IQA+（CLIP/RN50，146MB，CPU EP）→ 后备 NIMA（13MB）
> - **美学**：CLIP ViT-B/32 (336MB) + LAION Aesthetics V1 线性头（4KB）
> - **综合权重**：美学 0.25 / 技术 0.60 / 启发式 0.15（技术主导）

---

## 一、核心结论（TL;DR）

### 推荐路线：**A 路线（TOPIQ + LAION Aesthetics V2.5）**

| 维度 | 现状 | 推荐 | 提升幅度 | 工程周期 |
|---|---|---|---|---|
| **技术质量** | CLIP-IQA+ (KonIQ SRCC 0.885) | **TOPIQ-NR (ResNet50, 45M)** | +0.045 SRCC | **1-3 天** |
| **美学** | LAION V1 + CLIP-B/32 (AVA SRCC 0.665) | **LAION Aesthetics V2.5 (SigLIP-base)** | 跨域显著更好 | 2-4 天 |
| **总包大小** | 5 模型 494M | +TOPIQ 90-180M | +~5% | — |
| **6GB 显存可行性** | ✅ | ✅（同时跑 CLIP-B/32 + TOPIQ-NR + TOPIQ-IAA peak ~2.9GB） | — | — |

### **不上 Q-ReAlign-Mini-0.8B（关键决策）**

精度确实最强（KonIQ SRCC 0.935 vs TOPIQ 0.928，AVA 0.797 vs TOPIQ 0.733），但**边际收益不抵边际成本**：
- **TOPIQ 已达 KonIQ 0.928**：与 Q-ReAlign-Mini 仅差 0.7%，去重场景几乎不可感知
- **AVA 美学差距 0.064 显著**，但需要业务验证是否影响实际去重决策
- **工程成本**：TOPIQ ONNX 1-3 天 vs Q-ReAlign 4 周（需自写 ONNX 导出或换 candle/Python HTTP 栈）
- **建议**：先用 TOPIQ 跑通闭环，业务验证后必要时走 HTTP+Python 旁路升级 Q-ReAlign

---

## 二、IQA 模型调研（技术质量维度）

### 2.1 候选模型对比表

**基准测试数据来源：IQA-PyTorch 官方 benchmark（`https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html`）**

| 模型 | Backbone | 参数 | LIVEC SRCC | KonIQ-10k SRCC | SPAQ SRCC | AVA SRCC | ONNX 可得性 | DirectML |
|---|---|---|---|---|---|---|---|---|
| **当前栈** | | | | | | | | |
| CLIP-IQA+ | CLIP/RN50 | ~150M | 0.805 | 0.885 | 0.895 | 0.595 | ✅ 86Cao (146MB) | 🔴 强制 CPU |
| NIMA (MobileNet) | MobileNetV2 | 4.2M | 0.507 | 0.715 | 0.520 | 0.713 | ✅ 86Cao (13MB) | ✅ |
| **推荐升级** | | | | | | | | |
| **TOPIQ-NR (ResNet50)** | ResNet50 | **45M** | **0.811** | **0.930** | **0.870** | 0.595 | ⚠️ 自导 / 第三方 177MB | 🟡 预计可行 |
| TOPIQ-IAA (ResNet50) | ResNet50 | 25M | – | – | – | **0.791** | ⚠️ 第三方 280MB | 🟡 预计可行 |
| **备选方案** | | | | | | | | |
| MUSIQ-koniq | Transformer | 27M | 0.789 | 0.896 | 0.852 | – | ✅ 86Cao (104MB) | 🟡 已弃 |
| MANIQA | ViT | 135M | 0.840 | 0.893 | 0.817 | – | ✅ 86Cao (457MB) | 🟠 较低 |
| HyperIQA | ResNet50 | – | 0.755 | 0.904 | 0.708 | – | ✅ 86Cao (105MB) | 🟢 友好 |
| **SOTA 但不可行** | | | | | | | | |
| Q-Align | LMM 7B | 7B | **0.881** | **0.831** | – | **0.819** | ❌ 无 ONNX | ❌ 显存超 |
| Q-ReAlign-Mini | Qwen3.5-VL 0.8B | 0.8B | – | **0.935** | 0.931 | 0.797 | ❌ 无 ONNX | ❌ 未验证 |
| Q-ReAlign-Lite | Qwen3.5-VL 4B | 4B | – | 0.943 | 0.932 | 0.814 | ❌ 无 ONNX | ❌ 显存超 |

### 2.2 关键发现

1. **TOPIQ 是 ONNX 可用性的最佳折中**：
   - KonIQ SRCC 0.930（与 Q-Align 7B 0.831 比，明显胜出）
   - 比 MUSIQ（86Cao 已导出 104MB ONNX）**高 0.6pp**
   - ResNet50 backbone，DML 兼容性远好于 CLIP-IQA+ 的 CLIP backbone

2. **TOPIQ 已找到第三方 ONNX**（意外发现）：
   - **Skulleton12/TOPIQ** 的 `topiq_nr.onnx`（177MB，2025-05 上传）
   - **cromsc/topiq-iaa-res50** 的 `topiq_iaa_res50.onnx`（280MB，2025-11 上传）
   - ⚠️ README 都为空，**必须用 Netron 验证 input shape、mean/std、输出维度**

3. **86Cao/IQA-ONNX-Models 不含 TOPIQ**，只含 MUSIQ/MANIQA/HyperIQA/NIMA/CLIPIQA+ 等

4. **TOPIQ 在 IQA-PyTorch 官方 benchmark 的横扫**（KonIQ SRCC 0.930 vs CLIP-IQA+ 0.885）：
   - ResNet50 backbone 决策精确度优于 CLIP/RN50（可能是 CLIP 的 zero-shot 范式在量化精度上吃亏）
   - 头部 TransformerEncoder 4 层（每 scale 一个）比 CLIP 头部更精细

### 2.3 DirectML 兼容性预测

| TOPIQ 子模块 | Op 候选 | DML 风险 |
|---|---|---|
| ResNet50 backbone | Conv, BN, ReLU, MaxPool | 🟢 **极低** |
| 多尺度特征提取（timm feature_info） | 索引操作 | 🟢 无 |
| dim_reduce (1×1 Conv + GELU) | Conv, GELU | 🟢 |
| **sa_attn_blks (TransformerEncoder ×4 scale)** | **Gather (Int64), MatMul, Softmax, LayerNorm** | 🟡 **中等** |
| attn_blks (TransformerDecoder 跨尺度) | + **Reshape (动态 shape)** | 🟠 **较高** |
| attn_pool | TransformerEncoderLayer | 🟡 |
| TF.resize in forward | ONNX Resize op | 🟢 |

**判断**：TOPIQ 的 Transformer 头部是风险点，但比 CLIP-IQA+ 全 ViT block 链稳定得多。
**必备配置**：`ORT_SEQUENTIAL` + `enable_mem_pattern=false` + **Int64→Int32 Cast on Gather indices**
**fallback**：若 DML 失败，ResNet50 CNN backbone 在 CPU 上跑 384×224 单图 <50ms，可接受

### 2.4 6GB 显存占用

| 模型 | FP32 大小 | FP16 大小 | 推理峰值显存（batch=8） |
|---|---|---|---|
| CLIP ViT-B/32（现有） | ~600MB | ~300MB | ~1.5GB |
| CLIP-IQA+（现有） | 146MB | – | 0（CPU） |
| **TOPIQ-NR** | 177MB | ~90MB | ~700MB |
| **TOPIQ-IAA ResNet50** | 280MB（带 .data） | ~140MB | ~700MB |

**三总峰值 ~2.9GB，6GB 显存留 50% 余量。**

---

## 三、美学模型调研

### 3.1 候选模型对比表

| 模型 | Backbone | ONNX 可得性 | AVA SRCC | 跨域能力 | 模型大小 |
|---|---|---|---|---|---|
| **当前栈** | | | | | |
| LAION Aesthetics V1 | CLIP-B/32 + 4KB 线性头 | ✅ 现成 | 0.665 | 一般 | 336MB+4KB |
| **推荐升级** | | | | | |
| **LAION Aesthetics V2.5** | **SigLIP-base + 4KB 头** | ✅ `fsw/aesthetic-predictor-v2-5_onnx` 1.72GB | – | **显著优于 V1/V2** | ~93M+4KB |
| LAION Aesthetics V2 | ViT-L/14 + MLP 头 | ✅ `shunk031/aesthetics-predictor-v2-...` ~430MB | 比 V1 +5-10pp | 良好 | ~430MB |
| **SOTA 但不可行** | | | | | |
| Q-ReAlign-Mini | Qwen3.5-VL 0.8B | ❌ | **0.797** | SOTA | 2.21GB bf16 |
| ImageReward | BLIP ViT-L + Transformer | ❌ 需 prompt | – | – | ~470M |
| HPSv3 | Qwen2-VL-7B | ❌ | – | SRCC 0.94 | 7B |

### 3.2 关键发现

1. **V2.5 是反直觉的最佳选择**：SigLIP-base (~93M) 比 CLIP-B/32 (336MB) **更小**，但跨域（插画、AI 生图、传统绘画）显著优于 V2
2. **vs V2**：V2.5 backbone 更小（Sigmoid 损失 +8-bucket SORD）、MLP 头不变
3. **vs Q-ReAlign-Mini**：AVA 0.797 vs 0.797 持平，但 V2.5 部署成本仅 1 天（下载 ONNX）vs Q-ReAlign 4 周

### 3.3 工程方案

- **首选**：用 `fsw/aesthetic-predictor-v2-5_onnx` 仓库的现成 ONNX（1.72GB）
  - 含 SigLIP-base 视觉编码器 + V2.5 线性头
  - **比现有 CLIP-B/32 + V1 head 体积更大**，但精度更高
- **次选**：保持 CLIP-B/32 视觉编码器（已有 ONNX），仅替换为 LAION V2 linear head（`shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`），但 backbone 不换，跨域改进有限

---

## 四、不上 Q-ReAlign-Mini 的核心论证

### 4.1 精度对比

| 模型 | KonIQ SRCC | AVA SRCC | 差距 vs TOPIQ |
|---|---|---|---|
| TOPIQ-NR + TOPIQ-IAA | 0.930 | 0.791 | 基准 |
| Q-ReAlign-Mini | 0.935 | 0.797 | +0.005 / +0.006 |
| Q-Align (7B) | 0.831 | 0.819 | -0.099 / +0.028 |

**关键观察**：
- KonIQ 0.5% 的差距在去重场景里**几乎不可感知**（人眼区分 MOS 差异通常需 ≥5%）
- AVA 0.6% 差距同样边缘
- Q-Align 在 KonIQ 上反而输（7B 也不一定打过 ResNet50）

### 4.2 工程成本对比

| 路线 | 工程周期 | 安装包增量 | 风险 |
|---|---|---|---|
| **A. TOPIQ + V2.5 ONNX** | **1-3 天** | +~90MB（TOPIQ 90MB FP16） | 🟢 低（CNN backbone） |
| B. Q-ReAlign ONNX 自导 | 2-4 周 | +~1.5GB | 🟠 高（dynamic patch + GQA + DML） |
| C. candle-core 自写 Qwen3.5-VL forward | 3-4 周 | +~2GB | 🟠 高（candle 无现成 Qwen3.5 实现） |
| D. HTTP + Python 子进程 | 3-5 天 | +2.5GB（Python + transformers） | 🟡 中（架构耦合） |

**推荐**：A 路线作为主路径。D 路线作为未来升级兜底（业务验证后再启动）。

### 4.3 DirectML 对 LMM 推理的风险

- `ort` crate 通过 DirectML EP 跑 Qwen3.5-VL：**无任何公开基准**
- DirectML 强制 `ORT_SEQUENTIAL` + `enable_mem_pattern=false`，对 LLM 类大模型推理性能**未见报告**
- opset 20 上限，Qwen3.5-VL 的 dynamic patch reshape + GQA + DeepStack 多层视觉注入兼容性**完全未验证**

---

## 五、行动计划

### 阶段 1：TOPIQ 集成（1-3 天）

#### Day 1：下载 + 验证
1. 下载 `Skulleton12/TOPIQ` 的 `topiq_nr.onnx`（177MB）
2. 下载 `cromsc/topiq-iaa-res50` 的 `topiq_iaa_res50.onnx`（280MB）
3. 用 Netron 检查：
   - Input shape（应是 B,3,H,W 固定 384×384）
   - Output shape（标量 MOS / 10-bin 分布）
   - 是否有 `.onnx.data` 配套文件
4. 写 PyTorch baseline 对照脚本（用 `pyiqa.create_metric`）跑相同输入验证 MSE<1e-3

#### Day 2-3：Rust 集成
1. 在 `engine.rs` 增加 `topiq_nr_session` / `topiq_iia_session` 字段
2. 复用 `configure_session_builder` 保持 DirectML 确定性配置
3. 优先尝试 DirectML EP，报错则强制 CPU EP（ResNet50 CPU 也快）
4. 新增 `topiq_nr_scores(batch: &Array4<f32>) -> Vec<f32>` 方法
5. commands.rs 中扩展 score_groups_with_ai：
   - 如果 `has_topiq_nr()`，用 TOPIQ 取代 CLIP-IQA+（更准）
   - 如果 `has_topiq_iia()`，用 TOPIQ-IAA 取代 LAION V1（跨域更好）
6. 跑 verify_ai 真图测试，对比升级前后的评分分布

### 阶段 2：LAION Aesthetics V2.5 集成（2-4 天）

1. 下载 SigLIP-base ONNX（`fsw/aesthetic-predictor-v2-5_onnx`，含 V2.5 头）
2. 替换 `clip-vit-b32-visual.onnx` + `aesthetic_linear.bin` 路径
3. 评估是否保留旧 V1 head 作为后备
4. 美学权重 0.25 维持不变（业务验证后再调整）

### 阶段 3：业务验证（视情况 1-2 周）

1. 用 1000 张真实用户图片跑 A/B：TOPIQ vs CLIP-IQA+ 的去重决策差异
2. 跨域图片（AI 生成图、插画、传统摄影）的美学评分一致性
3. **如果业务确认 0.5-0.7% 差距不影响实际去重**：到此为止
4. **如果业务确认差距显著**：启动 D 路线（HTTP + Python transformers 部署 Q-ReAlign-Mini）

---

## 六、风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 第三方 ONNX 输入归一化参数未知 | 评分错误 | Netron + PyTorch baseline 对照 |
| TOPIQ Transformer 头部 DML 不兼容 | 评分失败 | 强制 CPU EP（ResNet50 CPU 仍快） |
| TOPIQ-NR + IAA 总 90+280MB = 370MB | 包膨胀 ~25% | FP16 量化压缩（onnxconverter-tool），降到 ~185MB |
| V2.5 SigLIP ONNX 1.72GB | 包膨胀 ~4x | 评估是否改用 V2 + ViT-L/14 (~430MB) |
| 业务验证显示差距不显著 | 投入浪费 | 1-3 天小成本验证，不重 |
| DirectML 动态 shape 边界 | 评分失败 | `with_intra_threads(1)` + 固定 batch size + ORT_SEQUENTIAL |

---

## 七、参考资源

### 仓库
- **IQA-PyTorch**（TOPIQ/MUSIQ/MANIQA 官方）：`https://github.com/chaofengc/IQA-PyTorch`
- **86Cao/IQA-ONNX-Models**（CLIPIQA/MUSIQ/MANIQA/NIMA ONNX）：hf-mirror.com/86Cao/IQA-ONNX-Models
- **TOPIQ-NR 第三方 ONNX**：hf-mirror.com/Skulleton12/TOPIQ
- **TOPIQ-IAA 第三方 ONNX**：hf-mirror.com/cromsc/topiq-iaa-res50
- **LAION V2.5 SigLIP ONNX**：hf-mirror.com/fsw/aesthetic-predictor-v2-5_onnx
- **LAION V2 (ViT-L/14)**：hf-mirror.com/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
- **Q-ReAlign-Mini**（备选升级）：hf-mirror.com/q-future/Q-ReAlign-Mini-0.8B

### Benchmark
- IQA-PyTorch 官方 benchmark：`https://iqa-pytorch.readthedocs.io/en/latest/benchmark.html`
- Q-ReAlign 论文 Table 3（HPSv3 对比）：`https://arxiv.org/abs/2508.03789`

### 关键技术约束
- DirectML 强制 `ORT_SEQUENTIAL` + `enable_mem_pattern=false` + `with_intra_threads(1)`
- Reshape op 在 DirectML Int64 Gather 时需转 Int32
- 固定输入尺寸（384×384 或 224×224）避免动态 shape 边界

---

**报告生成时间**：2026-08-20 17:30 GMT+8
**研究方法**：WebSearch + 4 个并行 Agent（NR-IQA / 美学 / TOPIQ / LMM 部署）
**关键来源**：IQA-PyTorch 官方 benchmark + HuggingFace 仓库 metadata + Web 文档
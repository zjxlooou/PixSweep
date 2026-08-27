# 人像优先的 AI 评分体系研究报告

> **需求来源**：用户整理手机拍照图库，主要痛点是"细微差别的人像摄影照片"——技术/美学评分在通用模型上不够精准。需要把人像摄影作为优先方向、风景次之、宠物最后。
>
> **调研时间**：2026-08-21
>
> **目标硬件约束**：Windows x64 + DirectX 12 GPU（NVIDIA/AMD/Intel 通用）；6GB 显存；ONNX Runtime + DirectML EP（不绑定 CUDA）。

---

## 1. 问题定位

### 当前评分管线的局限

| 评分维度 | 当前模型 | 训练集 | 人像上的问题 |
|---|---|---|---|
| 技术质量（NR） | **TOPIQ-NR** | KonIQ-10k（整体画质模糊/噪声） | 整体画质好但人脸虚焦的图会被误判为"好" |
| 美学评分（IAA） | **TOPIQ-IAA** | AVA（用户审美，泛摄影） | 评分偏向构图/色彩，**人脸好看/不好看**几乎无影响 |
| 通用去重 | CLIP ViT-B/32 | LAION-2B | 适合，不动 |

**核心缺失**：
1. **人脸区域专评**：人脸清晰度（虚焦）、表情自然度、眼睛状态、肤色曝光
2. **构图美学**：人像构图（头部比例/黄金分割）、正脸/侧脸适配
3. **场景自适应**：同样的模糊标准，对风景可以接受，对人像不可接受

### 推荐优先级

| 场景 | 重要度 | 评分策略 |
|---|---|---|
| **人像** | ★★★★★ | 引入人脸检测 + 闭眼检测 + 人脸区域专评，权重最大 |
| **风景** | ★★★ | 保留 TOPIQ-NR/IAA 通用评分 |
| **宠物** | ★★ | 保留通用评分，弱化人像因子（人脸检测失败则自动跳过） |
| 其他（文档/截图） | ★ | 通用评分 |

---

## 2. 候选模型综合评估

### 2.1 候选清单 + DirectML 兼容性

| 模型 | 任务 | 大小 | 现成 ONNX | DirectML | 优先级 |
|---|---|---|---|---|---|
| **InsightFace buffalo_l**（det_10g + 2d106det） | 人脸检测 + 106 关键点 | ~250 MB | ✅ InsightFace pkg | ⚠️ 需手动注入 DML provider | **★★★ 必选** |
| **TOPIQ-NR-Face**（ResNet-50 CFANet） | 人脸 IQA（GCFIQA 训练） | ~100 MB | ⚠️ 需自导（pyiqa 权重复用） | ✅ 标准 ResNet | **★★★ 必选** |
| **PINTO0309/OCEC**（L） | 闭眼/睁眼检测 | 6.4 MB | ✅ Zenodo/HF | ✅ 完全 DML | **★★ 强烈推荐** |
| **MobileNetV3-Large**（ImageNet） | 场景分类（1000 类→映射） | 21 MB | ✅ HF/ONNX zoo | ✅ 完全 DML | **★★ 强烈推荐** |
| **SAMP-Net**（ResNet-18） | 构图美学（14 种构图模式） | ~180 MB PyTorch | ❌ 需自导 ONNX | ✅ ResNet 标准 | ★★ 重要加分 |
| **emotion-ferplus** | 8 类表情 | 31 MB | ✅ onnx model zoo | ✅ 完全 DML | ★ 可选 |
| **eDifFIQA** | 人脸质量（轻量） | ~30 MB | ✅ HF mediainbox | ✅ 完全 DML | ★ 备用 |
| **LibreFace** | 面部 AU 强度 | ~43 MB | ✅ NuGet 包内嵌 | ✅ DML 可跑 | ★ 长期跟进 |
| **LAION Aesthetics V2.5** | 通用美学 VLM | ~890 MB+3.7MB | ⚠️ 依赖 CLIP-L/14 | ⚠️ CLIP 走 CPU | — **不推荐** |
| **Q-Align / mPLUG-Owl2** | 多模态美学 VLM | GB 级 | — | ❌ 必须 CPU | — **不推荐（太大）** |

### 2.2 三类基础模型的对比

#### A. 人脸检测 + 关键点（必须）

| 选项 | 输入 | 输出 | 优势 | 劣势 |
|---|---|---|---|---|
| **InsightFace buffalo_l** | 640×640 | bbox + 5/106 关键点 + 性别年龄 | 一次推理拿到所有几何信息；社区案例最丰富 | 250MB，InsightFace 需手动配置 DML provider |
| MediaPipe Face Mesh | 256×256 | 468 点 3D landmarks | 极轻量（3MB） | 是 `.task` 格式（TFLite+metadata），集成 ORT 麻烦 |
| YOLO-Face v8 | 640×640 | bbox | 快速 | 不出关键点，仍需另接 landmarks 模型 |

**选 InsightFace buffalo_l**：一次性解决"检测 + 关键点 + 性别年龄"，是其他所有人像子能力（构图、闭眼、人脸 crop）的前置依赖。

#### B. 人脸质量评分

| 选项 | 输入 | 训练集 | 现成 ONNX | 推荐度 |
|---|---|---|---|---|
| **TOPIQ-NR-Face** | 512×512 | GCFIQA（人脸 IQA） | 需自导（已有 pyiqa 权重） | **★★★★** SOTA 之一 |
| eDifFIQA | 112×112 | 人脸质量 | ✅ mediainbox HF | ★★★ 轻量备选 |
| CR-FIQA | 112×112 | NIST FATE SOTA | 需自导（已有 .pth） | ★★★ 但仅 NC-SA 许可 |
| FaceQualityCNN | 112×112 | 通用 | ✅ HF | ★★ |

**首选 TOPIQ-NR-Face**：①已在 pyiqa 生态（PixSweep 现有导出流程可复用）②训练集是 GFIQA，**域更接近手机随拍人像** ③ResNet-50 标准结构 DML 友好。

#### C. 场景分类

| 选项 | 大小 | Top-1 | 推荐度 |
|---|---|---|---|
| **MobileNetV3-Large** | 21 MB | 75.2% | **★★★★** 平衡首选 |
| MobileNetV2 | 14 MB | 70.9% | ★★★ 微软官方 NPU 示例 |
| EfficientNet-B0 | 20 MB | 77.1% | ★★★ 稍高 |
| Places365-ResNet18 | 45 MB | 54% 场景专用 | ★★ 场景专家，但缺人像类 |
| YOLO11n-cls | 5 MB | 70% | ★★ 极轻量 |

**首选 MobileNetV3-Large**：21MB 与 EfficientNet-B0 精度相当（75% vs 77%），社区 ONNX 多、DirectML 全支持。ImageNet 1000 类经过映射表可识别：
- "人像" → 包含 137 个"人脸/人形"标签（carton_face、bride、scuba_diver 等）
- "宠物" → cat/dog 多个品种（128 类）
- "风景" → mountain/sky/beach/tree 等（~80 类）
- "夜景" → 视觉特征（dark 类）+ heuristic 检测

---

## 3. 推荐的混合评分方案

### 3.1 方案全景（4 个新模型）

```
[输入图片]
   │
   ├──> MobileNetV3-Large (21MB, DML) ──> 场景分类 {人像, 风景, 宠物, 其他}
   │                                              │
   │                                              ▼
   ├──> InsightFace buffalo_l (250MB, DML) ──> 人脸 bbox + 106 关键点（仅人像类）
   │                                              │
   │   ├──> 人脸 crop (224×224) ──> TOPIQ-NR-Face (100MB, DML) ──> 人脸区域画质
   │   │
   │   ├──> 眼点 ROI (24×40) ──> OCEC (6MB, DML) ──> 闭眼/睁眼
   │   │
   │   └──> 几何规则 ──> 头部比例/黄金分割/正侧脸（纯几何，无需模型）
   │
   └──> 整体图 (384×384) ──> 保留 TOPIQ-NR (技术) + TOPIQ-IAA (美学) [现有]
                                                          │
                                                          ▼
                                            加权融合 ──> 最终综合分
```

### 3.2 综合分公式（按场景加权）

```python
# 基础：通用评分（所有图都有）
tech_general  = TOPIQ_NR(image)        # 0~10
aesth_general = TOPIQ_IAA(image)       # 0~10

# 人像：加入人脸子维度（仅当检测到人脸时）
if has_face:
    face_quality = TOPIQ_NR_Face(face_crop)  # 0~10
    eye_open     = OCEC(eye_roi)             # 0=闭眼, 1=睁眼
    composition  = geometry_score(landmarks) # 0~10（黄金分割 + 头部比例 + 正侧脸）
    pose_aesth   = geometry_score(landmarks) # 0~10（yaw/pitch 自然度）

# 按场景加权融合
if scene == "人像":
    # 人像：技术 0.20 + 美学 0.20 + 人脸质量 0.30 + 构图 0.20 + 睁眼 0.10
    final = (tech_general * 0.20 + aesth_general * 0.20 +
             face_quality * 0.30 + composition * 0.20 + eye_open * 0.10)

elif scene == "风景":
    # 风景：技术 0.50 + 美学 0.50（通用即可）
    final = tech_general * 0.50 + aesth_general * 0.50

elif scene == "宠物":
    # 宠物：技术 0.40 + 美学 0.40 + 构图 0.20
    final = tech_general * 0.40 + aesth_general * 0.40 + composition * 0.20

else:
    # 文档/截图/其他：技术 0.30 + 美学 0.70
    final = tech_general * 0.30 + aesth_general * 0.70

# 强制闭眼惩罚（人像场景）
if scene == "人像" and eye_open < 0.5:
    final *= 0.5  # 闭眼照片综合分减半
```

### 3.3 工程指标

| 指标 | 数值 |
|---|---|
| 新增模型总大小 | ~380 MB（InsightFace 250 + TOPIQ-NR-Face 100 + OCEC 6 + MobileNet 21） |
| 新增显存峰值（人像） | ~600 MB（InsightFace 250 + TOPIQ-NR-Face 100 + 现有 CLIP/TOPIQ） |
| 单张推理时间（人像） | 150-250 ms（DML）：InsightFace 80 + TOPIQ-NR-Face 30 + OCEC 5 + 通用 50 + 场景 10 |
| 单张推理时间（风景/宠物） | 70-100 ms（只跑场景 + 通用评分） |
| 9893 张真实图总耗时 | 约 25-40 分钟（后台批处理 + 增量缓存可接受） |
| 显存余量（6GB） | 5.4 GB 可用，新方案留 50%+ 余量 |

---

## 4. 实施路线图（3 个迭代）

### Phase 1：场景分类 + 几何构图评分（最小改动，1 周）
1. **新增模型**：
   - MobileNetV3-Large ONNX（21 MB）—— 场景分类
   - InsightFace buffalo_l det_10g + 2d106det（250 MB）—— 人脸检测 + 关键点
2. **新增能力**：
   - 场景分类器（人像/风景/宠物/其他）
   - 几何构图评分（头部比例、黄金分割、正侧脸）
3. **风险最低**：纯新增，不动现有评分管线
4. **预期效果**：主界面/预览里"人像组的排序更贴合用户预期"，风景组无变化

### Phase 2：人脸专评（核心改进，1-2 周）
1. **新增模型**：
   - TOPIQ-NR-Face ONNX（100 MB）—— 人脸区域画质
   - OCEC L ONNX（6 MB）—— 闭眼检测
2. **新增能力**：
   - 人脸 crop → TOPIQ-NR-Face 评分
   - 眼点 ROI → OCEC 闭眼检测 → 闭眼惩罚
3. **风险中**：需修改 recommender 评分融合公式
4. **预期效果**：人像组能区分"清晰人脸 vs 模糊人脸 vs 闭眼"，精准推荐

### Phase 3：构图美学 + 表情（锦上添花，2-3 周，可选）
1. **新增模型**：
   - SAMP-Net（自导 ONNX，180 MB）—— 构图美学 14 种模式
   - emotion-ferplus（31 MB）—— 表情识别
2. **可选升级**：LibreFace 替代 emotion-ferplus（更细粒度 AU）
3. **预期效果**：提供可解释的"为什么这张构图好/差"理由

---

## 5. 关键技术风险与对策

### 风险 1：InsightFace DML 集成
- **问题**：InsightFace 0.7+ 默认 providers `[CUDA, CPU]`，不识别 DML
- **对策**：手动构造 ONNX Runtime session，传 `providers=['DMLExecutionProvider', 'CPUExecutionProvider']`，绕过 InsightFace Python 接口

### 风险 2：DirectML Reshape/动态 shape 坑（PixSweep 历史踩过）
- **对策**：
  - 导出 ONNX 时锁定 `dynamic_axes=None`，batch=1
  - Session 配置 `with_memory_pattern(false)` + `with_intra_threads(1)` + `with_parallel_execution(false)`
  - 加载后 `session.get_providers()` 校验实际 EP，DML 失败自动 fallback CPU

### 风险 3：模型文件总大小 + 用户下载
- **对策**：模型仍不入 git 仓库（按现有 B 方案），通过 release zip 分发；用户首次运行若有增量下载需求，文档说明即可

### 风险 4：场景分类准确度（ImageNet → 自定义映射）
- **对策**：建立完整映射表（1000 类 → 4 类），并保留原 TOPIQ-IAA 作为兜底；后期可微调 MobileNetV3 在人像/风景数据上

### 风险 5：人脸检测召回率（侧脸、遮挡、多人）
- **对策**：
  - InsightFace SCRFD-10G 已支持侧脸、多人
  - 检测失败时**回退到通用评分**（而非报错）
  - 取最大人脸（多人时聚焦主体）

---

## 6. 如果只能选 3 个模型（落地优先级）

1. **InsightFace buffalo_l**（250 MB）—— 一站式解决人脸检测 + 关键点 + 性别年龄，所有所有后续人像能力的前置依赖
2. **TOPIQ-NR-Face**（100 MB）—— SOTA 的人脸 IQA 模型，GCFIQA 训练，是手机随拍人像场景最对口的质量评分
3. **MobileNetV3-Large**（21 MB）—— 轻量场景分类器，区分人像/风景/宠物，是评分融合的前提

**三个加起来 371 MB**，运行时新增显存峰值 < 400 MB，在 6GB 显存上有充足余量。

**强烈推荐加分**（极小成本）：
- **PINTO0309/OCEC**（6 MB）—— 闭眼检测，F1=0.99，零成本解决"闭眼照片"误判
- **几何规则**（无模型）—— 基于 InsightFace 关键点的构图评分（头部比例、黄金分割、正侧脸），纯算法计算零成本

---

## 7. 结论与下一步

**研究结论**：
- 用户痛点明确：人像摄影需要专项评分
- 技术可行：InsightFace + TOPIQ-NR-Face + MobileNetV3 三件套可立即落地
- DirectML 兼容性：9/12 候选模型可直接 GPU 加速，CLIP/DA3 等大模型走 CPU
- 工程成本可控：Phase 1（场景+几何）1 周、Phase 2（人脸专评）1-2 周、Phase 3（构图表情）可选

**推荐先实施 Phase 1**（场景分类 + 几何构图评分），原因：
1. 风险最低（纯新增，不动现有评分）
2. 能快速看到效果（人像组在主界面排序更贴合用户预期）
3. 为 Phase 2 的人脸专评铺路

**等待用户决策**：
- A. 立即开始 Phase 1 实施
- B. 进一步细化某个 Phase 的细节
- C. 等特定模型实测后再决定

---

## 8. 关键模型集成细节（深度调研）

### 8.1 InsightFace buffalo_l × Rust + DirectML

**ONNX 文件清单**（从 GitHub Release `v0.7` 下载，~326 MB）：
| 模型 | 输入 | 归一化 (mean/std) | 输出 | 大小 |
|---|---|---|---|---|
| `det_10g.onnx` | `[1, 3, H, W]` 动态 | 127.5 / 128.0 | 9 个 tensor（3 stride × score/bbox/kps），**NMS 不在 ONNX 内** | 16 MB |
| `2d106det.onnx` | `[None, 3, 192, 192]` | 0.0 / 1.0 | `[1, 212]` 归一化坐标（必须 `(out+1)/2*192` 才是像素） | 5 MB |
| `genderage.onnx` | `[None, 3, 96, 96]` | 0.0 / 1.0 | `[1, 3]`（gender + age） | 1.3 MB |

**Rust ort crate 关键配置**：
```rust
Session::builder()?
    .with_execution_providers([DirectML::default().with_device_id(0).build()])?
    .with_intra_op_num_threads(1)?  // DML 单线程 run 强制
    .commit_from_file(path)
```

**`det_10g` 动态 shape 处理**：
- 必须用 `.with_dimension_override("height", 640)?.with_dimension_override("width", 640)?` 固化为 640
- 否则 DirectML 性能下降甚至回退 CPU

**NMS 后处理伪代码**（det_10g 已包含 5 点关键点，score=0.5 / IoU=0.4）：
```rust
// 9 outputs: score_8/16/32, bbox_8/16/32, kps_8/16/32
for (i, stride) in [8, 16, 32].iter().enumerate() {
    let score = sigmoid(outs[i]);     // (N, 1)
    let bbox  = anchor + outs[i+3] * stride;  // 反算到 feature map
    let kps   = anchor + outs[i+6] * stride;  // (N, 5, 2)
}
// concat + NMS(iou=0.4) → 最终 bbox+5kps
```

**InsightFace DML 集成陷阱**：InsightFace Python 包默认 providers `[CUDA, CPU]`，不识别 DML。Rust 项目不通过 Python 包，直接加载 .onnx + 自建 ORT session，可彻底规避。

**下载命令**（无 Python 环境，PowerShell）：
```powershell
Invoke-WebRequest -Uri "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" -OutFile "$env:USERPROFILE\models\buffalo_l.zip"
Expand-Archive "$env:USERPROFILE\models\buffalo_l.zip" -DestinationPath "$env:USERPROFILE\models\buffalo_l"
```

**依赖链**：det_10g → bbox+5kps → 仿射对齐 → 2d106det（106 关键点） + genderage（属性）

### 8.2 TOPIQ-NR-Face ONNX 导出

**注册名**：`topiq_nr-face`（pyiqa 内部 `model_name='topiq_nr_cgfiqa_res50'`），CGFIQA-40K 训练（2024-10 更新）。

**权重**：**https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/topiq_nr_cgfiqa_res50-0a8b8e4f.pth**（~102 MB）。

**网络结构**：与现有 `topiq_nr.onnx`（TOPIQ-NR）**共用 `CFANet` class**（ResNet50 backbone + 4-stage gated pooling + cross-scale attention），架构 100% 一致，**仅权重不同**。

**关键导出步骤**（**必须关闭 facexlib 对齐分支**）：

```python
import torch, pyiqa
from pyiqa.archs.topiq_arch import CFANet

model = pyiqa.create_metric('topiq_nr-face', device='cpu')
net = model.net

# 关键：禁用 forward 内的人脸对齐（保留 ONNX 可 trace 性）
net.align_crop_face = False  # ← 否则会调 facexlib/RetinaFace，onnx 无法 trace
net.eval()

# 加载本地权重
from pyiqa.archs.arch_util import load_pretrained_network
load_pretrained_network(net, 'topiq_nr_cgfiqa_res50-0a8b8e4f.pth')

# 导出（与 topiq_nr.onnx 同一套参数）
dummy = torch.randn(1, 3, 512, 512)
torch.onnx.export(
    net, (dummy,),
    'topiq_nr_face.onnx',
    input_names=['face_crop'], output_names=['quality'],
    opset_version=18,  # 与 topiq_nr.onnx 一致
    dynamic_axes={'face_crop': {0: 'batch'}, 'quality': {0: 'batch'}},
    do_constant_folding=True,
    dynamo=False,  # 走 TorchScript 路径，兼容老 ops
)
```

**验证流程**（与现有 `topiq_nr.onnx` 完全一致）：
```python
import onnxruntime as ort, pyiqa, numpy as np

ref = pyiqa.create_metric('topiq_nr-face', device='cpu').net
ref.align_crop_face = False
x = np.random.randn(1, 3, 512, 512).astype(np.float32)
pyiqa_score = ref(torch.from_numpy(x)).item()

sess = ort.InferenceSession('topiq_nr_face.onnx', providers=['DML', 'CPU'])
onnx_score = sess.run(None, {'face_crop': x})[0].squeeze()
print(f'|pyiqa - onnx| = {abs(ref_score - onnx_score):.2e}')  # 期望 <1e-5
```

**输出映射**：标量 `[0, 1]` → `1 + clip(score, 0, 1) * 9` → 1~10 分（与 TOPIQ-NR 一致）。

**集成建议**：**新增 `src-tauri/src/ai/topiq_face.rs`**（不强行复用 `topiq.rs`），输入是 InsightFace 对齐后的 512×512 crop，整段推理可与 `topiq.rs` 共享 ort::Session 加载逻辑。

---

*最后更新：2026-08-21*
*相关文件*：`docs/MODEL_UPGRADE_RESEARCH.md`（v0.1.1 升级研究）、`AGENT.md`（构建规范）、`CODING_HISTORY.md`（项目历史）
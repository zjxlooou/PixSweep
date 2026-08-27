# 模型目录说明

本目录存放 PixSweep 的 AI 模型文件（ONNX 格式 + 外部权重）。

> **模型文件不入 git 仓库**（体积约 960MB），通过发布包 `dist-package/PixSweep-v*.zip` 分发。
> 解压发布包后，将其中 `models/` 目录整体放到本目录即可运行。

## 模型清单

### 通用模型（`models/` 顶层，10 文件 7 模型）

| 文件 | 模型 | 角色 | 后端 |
|------|------|------|------|
| `topiq_nr.onnx` + `topiq_nr.onnx.data` | TOPIQ-NR | 技术主评分（KonIQ-10k） | DirectML |
| `topiq_iaa_res50.onnx` | TOPIQ-IAA | 美学主评分（AVA） | DirectML |
| `topiq_nr_face.onnx` + `topiq_nr_face.onnx.data` | TOPIQ-NR-Face | 人脸专评（CGFIQA-40k，人像主导） | DirectML |
| `clipiqa_model.onnx` + `clipiqa_model.onnx.data` | CLIP-IQA+ | 技术后备 | CPU |
| `nima-technical.onnx` | NIMA | 技术二级后备 | DirectML |
| `clip-vit-b32-visual.onnx` | CLIP ViT-B/32 | 美学后备（LAION 头依赖其 embedding） | DirectML |
| `aesthetic_linear.bin` | LAION 线性头 | 美学后备 | — |

### 子目录模型

| 路径 | 模型 | 角色 |
|------|------|------|
| `insightface/det_10g.onnx` | SCRFD-10G | 人脸检测 + 5 关键点（`2d106det`/`genderage` 未使用） |
| `scene/mobilenet_v3_large.onnx` + `.data` + `labels.txt` | MobileNetV3-Large | 场景分类（人像/风景/宠物/其他） |
| `eye/ocec_l.onnx` | OCEC | 闭眼检测 |

## 配对文件说明

以下模型是**配对文件**（图结构 + 外部权重），缺一不可：

- `topiq_nr.onnx` + `topiq_nr.onnx.data`（TOPIQ-NR）
- `topiq_nr_face.onnx` + `topiq_nr_face.onnx.data`（TOPIQ-NR-Face）
- `clipiqa_model.onnx` + `clipiqa_model.onnx.data`（CLIP-IQA+）

打包时 `.onnx.data` 扩展名是 `.data`，按扩展名扫描会漏掉，需在 `scripts/build_release.ps1` 的 `$neededModels` 中显式列出。

## 获取方式

1. **推荐**：从发布包 `dist-package/PixSweep-v*.zip` 解压获取（已含全部模型，开箱即用）。
2. **重新获取**：各模型来源见上表；TOPIQ-NR 的重新导出方法见 `CODING_HISTORY.md` 与 `AGENT.md`。

详细评分体系与后端说明见 `docs/DESIGN.md`。

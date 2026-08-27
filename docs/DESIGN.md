# PixSweep - 图片智能去重应用设计方案

> Windows 11 桌面应用，使用 Rust + Tauri 2 开发，本地 GPU（ONNX Runtime DirectML / DirectX 12，NVIDIA / AMD / Intel 全系通用）跑 AI 模型，扫描文件夹找出相似图片并智能推荐最佳图片，帮助用户清理重复图片以减少硬盘占用。

---

## 1. 需求概述

### 1.1 核心功能

| 功能 | 描述 |
|------|------|
| 文件夹扫描 | 支持选择一个或多个文件夹，递归扫描所有图片文件 |
| 相似图片检测 | 基于 AI 视觉模型提取图片特征，对比相似度并聚类分组 |
| 智能推荐 | AI 评估每组图片的质量（清晰度/构图/色彩），推荐一张最佳图片保留 |
| 删除操作 | 用户可单组操作或一键批量删除非推荐图片，删除走回收站可恢复 |
| 可视化界面 | 图形化操作，缩略图预览，进度展示，评分标注 |

### 1.2 目标平台

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 11 (x86_64) |
| CPU | x86_64，至少 8GB 内存 |
| GPU | 支持 DirectML 的 GPU（NVIDIA / AMD / Intel，GTX 10 系及更新或同等架构） |
| 显存 | 至少 6GB VRAM |
| 驱动 | 对应厂商驱动（DirectML 走 DirectX 12，Win11 自带，无需 CUDA Toolkit） |

### 1.3 支持的图片格式

```
JPEG (.jpg .jpeg)
PNG (.png)
WebP (.webp)
BMP (.bmp)
TIFF (.tiff .tif)
HEIC (.heic .heif) — 需额外解码支持
```

---

## 2. 技术选型

### 2.1 整体技术栈

| 层级 | 技术 | 理由 |
|------|------|------|
| 语言 | **Rust** | 性能接近 C++，内存安全，无 GC 停顿，编译为原生 Windows 可执行文件 |
| GUI 框架 | **Tauri 2.0** | Rust 后端 + Web 前端 (React)，二进制体积小 (~10MB)，原生窗口，IPC 通信高效 |
| 前端 | React 18 + TypeScript + Tailwind CSS | 组件化开发，图片网格/缩略图展示成熟方案丰富 |
| AI 推理 | **ONNX Runtime + DirectML EP** | `ort` crate 调用，全 GPU 加速（NVIDIA/AMD/Intel）；CLIP-IQA+ 因 Reshape op 不兼容走 CPU EP |
| 图像处理 | `image` + `imageproc` crate | Rust 生态标准图像库，支持主流格式解码/缩放/裁剪 |
| 数据存储 | JSON 文件（`serde_json`，纯 Rust 无 C 编译依赖） | 缓存图片元信息和 embedding，实现增量扫描 |
| 文件遍历 | `walkdir` crate | 高性能递归目录遍历 |
| 文件删除 | `trash` crate | 跨平台安全删除到回收站，不直接永久删除 |
| 并发 | `tokio` + `rayon` | tokio 管理 async IO，rayon 管理 CPU 并行（图像解码/预处理） |

### 2.2 为什么选 Tauri 而非 egui/iced？

| 对比项 | Tauri 2.0 | egui | iced |
|--------|-----------|------|------|
| UI 渲染 | Web (Chromium/WebView2) | 即时模式绘制 | 保留模式 wgpu |
| 图片网格 | CSS Grid/Flexbox，成熟 | 需要手动布局，体验差 | 支持但生态弱 |
| 缩略图懒加载 | 原生支持 (Intersection Observer) | 需自行实现 | 需自行实现 |
| 二进制大小 | ~10-15MB (含 WebView2) | ~5MB | ~8MB |
| 开发效率 | 高 (React 生态) | 中 | 中 |
| Windows 适配 | 原生 (WebView2 预装于 Win11) | 良好 | 良好 |

结论：对于需要展示大量图片缩略图、网格布局、交互动画的应用，**Tauri + Web 前端** 是最佳选择。Win11 预装 WebView2 运行时，无需额外安装。

### 2.3 AI 模型选择

| 用途 | 模型 | 大小 | 输入 | 输出 | 为什么 |
|------|------|------|------|------|--------|
| 相似度检测 | **CLIP ViT-B/32** (OpenAI) | ~350MB (ONNX) | 224x224 RGB | 512-dim embedding | 业界标准视觉特征提取器，对相似图片/构图/内容变化感知强，零样本能力强 |
| 技术质量评估 | **TOPIQ-NR** (ResNet50) | ~177MB (ONNX) | 384x384 RGB | 0~1 质量分 | KonIQ-10k SRCC 0.930，显著优于 CLIP-IQA+ (0.885)；CNN backbone，CPU/GPU 均可 |
| 美学评估 | **TOPIQ-IAA** (ResNet50) | ~280MB (ONNX) | 384x384 RGB | 10-bin 分布 (1-10分) | AVA SRCC 0.791，显著优于 LAION V1 (0.665)；与 TOPIQ-NR 同 backbone |
| 技术质量后备 | **CLIP-IQA+** (CLIP/RN50 零样本) | ~146MB (ONNX) | 224x224 RGB | 0~1 质量分 | TOPIQ-NR 不可用时的后备；CLIP 架构在 DirectML 上需 CPU EP |
| 技术质量二级后备 | **NIMA** (Neural Image Assessment) | ~13MB (ONNX) | 224x224 RGB | 10-bin 分布 (1-10分) | 模型轻量推理快，最终兜底 |
| 美学后备 | **CLIP + LAION-Aesthetics 线性头** | ~350MB + 2KB | 224x224 RGB | 1~10 分 | TOPIQ-IAA 不可用时回退：CLIP embedding 接 LAION 美学线性头 |

模型均导出为 ONNX 格式，通过 ONNX Runtime 的 **DirectML Execution Provider**（DirectX 12，全显卡通用）推理，完全脱离 CUDA 生态。除 CLIP-IQA+ 因 `Reshape` op 与 DirectML 不兼容而走 CPU EP 外，其余模型（CLIP / TOPIQ-NR / TOPIQ-IAA / NIMA）全部走 DirectML。会话统一配置顺序执行 + 关闭内存模式 + 单线程 intra-op 以保证评分可复现。

---

## 3. 系统架构

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React + TS)                      │
│   Tauri Webview · 文件夹选择 · 扫描进度 · 分组展示 · 删除操作  │
└──────────────────────────┬──────────────────────────────────┘
                           │ Tauri IPC (invoke / event)
┌──────────────────────────┴──────────────────────────────────┐
│                   Backend (Rust Core)                         │
│                                                               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Scanner  │→ │ Feature   │→ │ Cluster  │→ │ Quality    │  │
│  │ Module   │  │ Extractor │  │ Engine   │  │ Assessor   │  │
│  └──────────┘  └───────────┘  └──────────┘  └────────────┘  │
│                                                               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │
│  │ File Ops │  │ Thumb     │  │ Async    │  │ Tauri IPC  │  │
│  │ (recycle)│  │ Cache     │  │ Runtime  │  │ Commands   │  │
│  └──────────┘  └───────────┘  └──────────┘  └────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │ ort crate
┌──────────────────────────┴──────────────────────────────────┐
│         AI Inference (ONNX Runtime + DirectML)              │
│                                                               │
│   CLIP ViT-B/32 (语义去重)    TOPIQ-NR (技术质量, DirectML)   │
│   TOPIQ-IAA (美学, DirectML)  CLIP-IQA+/NIMA (技术后备)        │
│   aesthetic_linear (美学后备)                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ rusqlite
┌──────────────────────────┴──────────────────────────────────┐
│              Storage (SQLite)                                  │
│   文件元信息 · CLIP embedding 缓存 · NIMA 评分 · 聚类结果      │
│   增量扫描：跳过已处理文件 (基于路径+修改时间 hash)            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模块职责

#### Scanner (扫描模块)
- 使用 `walkdir` 递归遍历指定文件夹
- 过滤支持的图片文件扩展名
- 提取文件元信息：路径、大小、修改时间
- 计算 `path + mtime` 的 hash 作为文件指纹，用于增量扫描
- 查询 SQLite 缓存，跳过已处理且未修改的文件

#### Feature Extractor (特征提取模块)
- 使用 `image` crate 解码图片
- 预处理：resize 到 224x224，归一化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- 批量组装 tensor (batch_size=64)
- 通过 ONNX Runtime DirectML EP 执行 CLIP forward pass
- 输出 512 维 embedding，L2 归一化后存入 SQLite

#### Cluster Engine (聚类引擎)
- 计算所有 embedding 的余弦相似度矩阵 (N x N)
- 应用相似度阈值 (默认 0.92，可配置 0.85~0.98)
- 使用 Union-Find (并查集) 算法找出连通分量作为图片组
- 仅保留组内成员 >= 2 的组

#### Quality Assessor (质量评估模块)
- 双维度评分：技术质量（清晰度/失真/噪点）+ 美学质量（构图/观感）
- 技术质量：优先 TOPIQ-NR（ResNet50，KonIQ-10k，1~10 分）；不可用时回退 CLIP-IQA+（CLIP/RN50 零样本），再回退 NIMA（10-bin 分布加权平均）
- 美学质量：优先 TOPIQ-IAA（ResNet50，AVA，10-bin 分布加权平均）；不可用时回退 CLIP ViT-B/32 embedding 经 `aesthetic_linear.bin` 线性头
- 综合分 = 技术分与美学分加权，每组综合分最高的图片标记为 "推荐保留"

#### File Ops (文件操作模块)
- 使用 `trash` crate 将文件移至 Windows 回收站
- 支持单文件删除和批量删除
- 删除前校验文件是否存在、是否有写入权限
- 返回操作结果（成功/失败列表）

#### Thumbnail Cache (缩略图缓存)
- 首次扫描时生成缩略图 (256x256 JPEG, quality=85)
- 缓存到 `%LOCALAPPDATA%/com.pixsweep.app/thumbnails/`
- LRU 策略管理缓存大小 (上限 500MB)
- 通过 Tauri 自定义协议 (`asset://`) 提供给前端

#### Tauri IPC (前后端通信)
- `invoke` 命令：前端调用后端函数 (如 `start_scan`, `delete_files`)
- `event` 推送：后端向前端推送进度 (如 `scan_progress`, `group_ready`)
- 使用 `tauri::async_runtime` 管理异步任务

---

## 4. AI 推理流水线

> **注**：本章为早期设计（SQLite 缓存、CLIP 聚类等已改为 JSON / 双哈希）。当前评分与闭眼链路以 `AGENT.md`「架构」节和 `src-tauri/src/ai/` 代码为准；闭眼检测现为「MediaPipe 脸网格垂目开度为主 + OCEC 眨眼否决」双信号（2026-08-27，详见 `CODING_HISTORY.md`）。

### 4.1 处理流程

```
[扫描目录] → [图片预处理] → [CLIP特征提取] → [相似度矩阵] → [聚类分组]
                                                              ↓
[结果持久化] ← [推荐决策] ← [双维度AI评分: 技术(TOPIQ-NR/CLIP-IQA+/NIMA) + 美学(TOPIQ-IAA/CLIP)]
```

### 4.2 详细步骤

#### Step 1: 目录扫描
```
输入: Vec<PathBuf> (用户选择的文件夹列表)
输出: Vec<ImageMeta> { path, size, modified, file_hash }
```
- `walkdir` 递归遍历
- 过滤扩展名: jpg/jpeg/png/webp/bmp/tiff/heic
- 计算 `(path + mtime)` 的 blake3 hash 作为文件指纹
- 查 SQLite 缓存: 若 hash 已存在且有 embedding，直接复用，跳过推理

#### Step 2: 图片预处理
```
输入: ImageMeta
输出: Tensor [C=3, H=224, W=224] (CHW格式, NCHW batch维度后续拼)
```
- `image` crate 解码 (支持 EXIF orientation 自动旋转)
- resize 到 224x224 (Lanczos 采样)
- 归一化: (pixel / 255.0 - mean) / std
- 转为 CHW tensor
- **并行**: 使用 rayon 并行解码，预填充 batch queue

#### Step 3: CLIP 特征提取 (GPU)
```
输入: Batch<Tensor [3, 224, 224]>  (batch_size=64)
输出: Vec<[f32; 512]>  (L2归一化后的embedding)
```
- ONNX Runtime session 配置（统一保证可复现）：
  - CLIP / TOPIQ-NR / TOPIQ-IAA / NIMA：DirectML EP（DirectX 12，全 GPU 通用）
  - CLIP-IQA+：CPU EP（Reshape op 与 DirectML 不兼容）
  - Graph optimization: Level3
  - 顺序执行 + 关闭内存模式 + 单线程 intra-op（保证评分确定性，修复同一张图分数时高时低）
- CLIP ViT-B/32 visual encoder ONNX 模型
- 输入: `pixel_values` [N, 3, 224, 224]
- 输出: `image_embeds` [N, 512]
- L2 归一化后存入 SQLite `embeddings` 表

#### Step 4a: 余弦相似度矩阵
```
输入: Vec<[f32; 512]>  (N 张图片的embedding)
输出: [[f32; N]; N]  (NxN相似度矩阵)
```
- embedding 已 L2 归一化 → 余弦相似度 = 矩阵乘法 A * A^T
- 使用 `ndarray` crate 矩阵运算
- 仅计算上三角 (对称矩阵)
- 阈值过滤: sim > threshold (默认 0.92)

#### Step 4b: 聚类分组
```
输入: Vec<(usize, usize, f32)>  (相似图片对的索引和相似度)
输出: Vec<Vec<usize>>  (图片组的索引列表)
```
- Union-Find (并查集) 算法
- 遍历所有相似对，union 两个图片的索引
- 最终找出所有连通分量 (size >= 2)
- 每个连通分量即为一个 "相似图片组"

#### Step 5: 双维度 AI 评分 (GPU / DirectML)
```
输入: 每组图片路径
输出: (综合分, 美学分, 技术分) 各 f32 (1.0~10.0)
```
- **美学分**：TOPIQ-IAA（ResNet50，AVA）输出 10-bin softmax 分布 → 加权平均 1~10；
  TOPIQ-IAA 不可用时回退 CLIP ViT-B/32 embedding → LAON-Aesthetics 线性头 → 1~10
- **技术分（主）**：TOPIQ-NR（ResNet50，KonIQ-10k）输出 0~1 → 映射到 1~10；
  TOPIQ-NR 不可用时回退 CLIP-IQA+（CLIP/RN50 零样本），再回退 NIMA（MobileNet-v2，10-bin 分布加权平均）
- 预处理：居中裁剪正方形 → resize（TOPIQ 384×384，CLIP/NIMA 224×224）→ ImageNet / CLIP 归一化
- 推理后端：优先 DirectML（DirectX 12），失败回退 CPU；会话统一配置为
  顺序执行 + 关闭内存模式 + 单线程 intra-op，保证同一张图评分可复现
- 仅对相似组内的图片评分（非组内图片无需评分）

#### Step 6: 推荐决策
```
输入: Vec<ImageGroup> { images: Vec<(ImageMeta, f32 /*综合分*/, f32 /*美学分*/, f32 /*技术分*/)> }
输出: Vec<Recommendation> { keep: ImageMeta, delete: Vec<ImageMeta> }
```
- 每组选综合评分最高的图片作为 "推荐保留"
- 综合分 = 美学 × 0.25 + 技术 × 0.60 + 启发式(分辨率) × 0.15
- 其余标记为 "建议删除"
- 可选规则: 当评分差距 < 0.5 时，优先保留文件更大的 (通常分辨率更高)

### 4.3 显存预算

| 项目 | 显存占用 |
|------|----------|
| CLIP ViT-B/32 权重 (FP16) | ~175 MB |
| NIMA 权重 (FP32) | ~14 MB |
| ONNX Runtime 推理缓冲 (batch=64) | ~1,200 MB |
| ONNX Runtime / DirectML 运行时 | ~300 MB |
| **总计** | **~1.7 GB** |

6GB VRAM 绰绰有余，剩余 ~4.3GB 空间。可将 batch_size 提升至 128 进一步加速。

### 4.4 性能预估

| 阶段 | 耗时 (RTX 2060, ~4000 图片) | 说明 |
|------|---------------------------|------|
| 目录扫描 | ~3s | walkdir + 文件元信息提取 |
| 图片解码+预处理 | ~30s | rayon 8线程并行解码 |
| CLIP 推理 | ~60s | batch=64, GPU 推理 |
| 相似度矩阵 | ~2s | ndarray 矩阵乘法 |
| 聚类 | <1s | Union-Find |
| NIMA 推理 | ~45s | 仅对组内图片推理 (~200张) |
| **总计** | **~2.5 分钟** | 首次全量扫描 |
| 增量扫描 | **<10s** | 跳过已有 embedding 的文件 |

---

## 5. UI 设计

### 5.1 界面布局

```
┌──────────────────────────────────────────────────────────────────┐
│  [PixSweep]  [文件夹路径输入框]  [+添加] [开始扫描] [设置]  │  ← 顶部工具栏
├──────────────────────────────────────────────────────────────────┤
│  扫描进度 ████████████░░░░░░  1,247/1,832 图片  CLIP推理中 68%  │  ← 进度条
├──────────────────────────────────────────────────────────────────┤
│  相似图片分组 (12 组 · 可节省 2.3 GB)                  [全删]   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ [推荐] IMG_0231    IMG_0229    IMG_0230                      │  │  ← 分组卡片
│  │  8.7分             7.1分       6.5分                          │  │
│  │  3.2MB             2.8MB       3.0MB                          │  │
│  │  相似度: 94.2%    可节省: 5.8MB                               │  │
│  │  [删除其余2张]  [手动选择]                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ [推荐] DSC_4412    DSC_4413    DSC_4411    DSC_4414          │  │
│  │  9.1分             7.8分       7.2分       6.9分              │
│  │  相似度: 91.8%    可节省: 11.6MB                              │
│  │  [删除其余3张]  [手动选择]                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─ 第3组 (5张相似) ─ 点击展开 ─────────────────────────────┐     │  ← 折叠组
│  └────────────────────────────────────────────────────────────┘     │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│  扫描结果: 1,832张·12组·38张可删  可节省: 2.3GB                   │  ← 底部统计
│                                          [一键删除全部] [导出报告]  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 交互流程

```
用户选择文件夹 → 点击"开始扫描"
    ↓
进度条实时更新 (扫描→解码(含EXIF方向校正)→CLIP推理→聚类→双维度AI评分: 技术+美学)
    ↓
分组卡片逐个出现 (后端 event 推送 group_ready)
    ↓
每组: 推荐图片(绿色高亮+评分) vs 待删图片(红色标记+评分)
    ↓
用户操作:
  ├─ 单组: 点击"删除其余N张" → 确认弹窗 → 删除到回收站
  ├─ 单组: 点击"手动选择" → 切换保留/删除标记 → 确认删除
  └─ 全局: 点击"一键删除全部" → 确认弹窗(列出所有待删文件) → 批量删除
    ↓
删除完成 → 更新UI → 显示释放空间统计
```

### 5.3 设置面板

| 设置项 | 默认值 | 范围 | 说明 |
|--------|--------|------|------|
| 相似度阈值 | 0.92 | 0.85~0.98 | 越高越严格，仅找出几乎相同的图片 |
| 批量推理大小 | 64 | 16/32/64/128 | 根据显存调节 |
| FP16 半精度 | 开启 | 开/关 | 关闭可略微提升精度 |
| 删除方式 | 回收站 | 回收站/永久删除 | 永久删除需二次确认 |
| 缩略图大小 | 256px | 128/192/256/384 | 影响预览质量和缓存大小 |
| GPU 设备 | GPU 0 | 自动检测 | 多显卡时选择 |
| 增量扫描 | 开启 | 开/关 | 跳过已处理文件 |

### 5.4 安全机制

- **所有删除默认走回收站**：使用 Windows Shell API (SHFileOperation / IFileOperation)，文件可从回收站恢复
- **删除前确认弹窗**：列出所有待删文件路径和总大小，需用户勾选确认
- **文件锁定检测**：删除前检查文件是否被其他进程占用
- **操作日志**：所有删除操作记录到日志文件 (`%LOCALAPPDATA%/com.pixsweep.app/logs/`)
- **导出报告**：可导出 CSV/HTML 报告，记录保留/删除决策和评分

---

## 6. 项目结构

```
image-dedup-ai/
├── src-tauri/                          # Rust 后端
│   ├── Cargo.toml
│   ├── tauri.conf.json                 # Tauri 配置
│   ├── build.rs
│   ├── models/                         # ONNX 模型文件
│   │   ├── clip-vit-b32-visual.onnx    # CLIP 视觉编码器（去重核心 + 美学后备，~350MB）
│   │   ├── topiq_nr.onnx               # TOPIQ-NR 技术质量评分（主用，ResNet50，~177MB）
│   │   ├── topiq_iaa_res50.onnx        # TOPIQ-IAA 美学评分（主用，ResNet50，~280MB）
│   │   ├── clipiqa_model.onnx          # CLIP-IQA+ 技术后备（图结构 ~0.4MB）
│   │   ├── clipiqa_model.onnx.data      # CLIP-IQA+ 外部权重（~146MB，与 .onnx 配对）
│   │   ├── nima-technical.onnx         # NIMA 技术二级后备
│   │   └── aesthetic_linear.bin        # LAION 美学线性头（后备）
│   └── src/
│       ├── main.rs                      # 入口 + Tauri 启动
│       ├── lib.rs                       # 模块导出
│       ├── commands.rs                  # Tauri IPC 命令定义
│       ├── image_io.rs                  # 图像解码 + EXIF 方向校正（自动旋转竖拍照片）
│       ├── scanner/
│       │   ├── mod.rs
│       │   └── walker.rs               # 目录遍历 + 文件过滤
│       ├── ai/
│       │   ├── mod.rs
│       │   ├── engine.rs                # ONNX Runtime 会话管理（DirectML + CPU 兜底）
│       │   ├── clip.rs                  # CLIP 推理封装（去重 embedding）
│       │   ├── topiq.rs                 # TOPIQ-NR/IAA 推理封装（主评分）
│       │   ├── nima.rs                  # NIMA 技术评分封装（后备）
│       │   └── preprocess.rs           # 图像预处理 (decode/resize/normalize)
│       ├── cluster/
│       │   ├── mod.rs
│       │   ├── similarity.rs            # 余弦相似度矩阵
│       │   └── unionfind.rs            # Union-Find 聚类
│       ├── quality/
│       │   ├── mod.rs
│       │   └── recommender.rs          # 推荐引擎
│       ├── fileops/
│       │   ├── mod.rs
│       │   └── trash.rs               # 回收站删除
│       ├── cache/
│       │   ├── mod.rs
│       │   ├── thumbnail.rs            # 缩略图生成+缓存
│       │   └── lru.rs                  # LRU 缓存管理
│       └── db/
│           ├── mod.rs
│           ├── schema.sql              # SQLite 表结构
│           └── store.rs                # 数据库操作
│
├── src/                                # React 前端
│   ├── main.tsx                        # 入口
│   ├── App.tsx                         # 主应用
│   ├── components/
│   │   ├── Toolbar.tsx                 # 顶部工具栏
│   │   ├── ProgressBar.tsx             # 扫描进度条
│   │   ├── GroupCard.tsx               # 相似图片分组卡片
│   │   ├── ImageThumbnail.tsx          # 缩略图组件
│   │   ├── QualityBadge.tsx            # 质量评分徽章
│   │   ├── DeleteConfirm.tsx           # 删除确认弹窗
│   │   ├── StatsBar.tsx                # 底部统计栏
│   │   └── SettingsPanel.tsx           # 设置面板
│   ├── hooks/
│   │   ├── useScan.ts                  # 扫描状态管理
│   │   └── useGroups.ts               # 分组数据管理
│   ├── types/
│   │   └── index.ts                    # TypeScript 类型定义
│   └── styles/
│       └── global.css                  # 全局样式
│
├── scripts/
│   ├── export_clip_onnx.py             # 导出 CLIP 为 ONNX
│   └── export_nima_onnx.py            # 导出 NIMA 为 ONNX
│
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── DESIGN.md                           # 本文档
```

---

## 7. 数据库设计

### 7.1 SQLite 表结构

```sql
-- 图片文件元信息
CREATE TABLE images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL UNIQUE,
    file_size   INTEGER NOT NULL,
    file_hash   TEXT NOT NULL,          -- blake3(path + mtime)
    modified    INTEGER NOT NULL,       -- Unix timestamp
    width       INTEGER,
    height      INTEGER,
    format      TEXT,                   -- jpeg/png/webp/...
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    updated_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- CLIP embedding 缓存
CREATE TABLE embeddings (
    image_id    INTEGER PRIMARY KEY REFERENCES images(id),
    embedding   BLOB NOT NULL,          -- 512 x f32 = 2048 bytes
    model_name  TEXT NOT NULL,          -- "clip-vit-b32"
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- NIMA 质量评分
CREATE TABLE quality_scores (
    image_id    INTEGER PRIMARY KEY REFERENCES images(id),
    score       REAL NOT NULL,          -- 1.0 ~ 10.0
    distribution BLOB,                  -- 10 x f32 原始分布
    model_name  TEXT NOT NULL,          -- "nima-technical"
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- 聚类结果 (每次扫描一个 session)
CREATE TABLE scan_sessions (
    id          TEXT PRIMARY KEY,       -- UUID
    folder_paths TEXT NOT NULL,         -- JSON array
    threshold   REAL NOT NULL,
    total_images INTEGER NOT NULL,
    total_groups INTEGER NOT NULL,
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE groups (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES scan_sessions(id),
    group_index INTEGER NOT NULL,
    image_ids   TEXT NOT NULL,           -- JSON array of image IDs
    keep_id     INTEGER REFERENCES images(id),
    similarity  REAL NOT NULL            -- 组内平均相似度
);

-- 索引
CREATE INDEX idx_images_hash ON images(file_hash);
CREATE INDEX idx_images_path ON images(file_path);
```

---

## 8. 核心 Rust 代码骨架

### 8.1 Cargo.toml 依赖

```toml
[dependencies]
# Tauri
tauri = { version = "2", features = ["dialog", "fs"] }

# AI 推理（feature = "ai"，默认启用；directml feature 启用 ort/directml）
ort = { version = "2", features = ["half"], optional = true }
ndarray = "0.17"
half = "2"                    # FP16 支持

# 图像处理
image = "0.25"
imageproc = "0.25"
kamadak-exif = "0.6"         # EXIF 方向旋转

# 文件系统
walkdir = "2"
trash = "5"
blake3 = "1"

# 数据库（改用 JSON 文件存储，见 §3 数据存储，无 rusqlite / SQLite C 编译依赖）

# 并发
tokio = { version = "1", features = ["full"] }
rayon = "1"

# 序列化
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# 工具
uuid = { version = "1", features = ["v4"] }
log = "0.4"
env_logger = "0.11"
anyhow = "1"
thiserror = "2"
```

### 8.2 ONNX Runtime 会话管理

```rust
use ort::{Environment, Session, SessionBuilder};
use ort::execution_providers::{DirectMLExecutionProvider, CPUExecutionProvider};

pub struct AIEngine {
    clip_session: Session,     // DirectML EP
    nima_session: Session,     // DirectML EP（技术评分后备）
    clipiqa_session: Session,  // CPU EP（CLIP-IQA+ 的 Reshape op 与 DirectML 不兼容）
}

impl AIEngine {
    /// 统一会话配置：顺序执行 + 关闭内存模式 + 单线程 intra-op
    /// 以保证评分确定性（修复同一张图分数时高时低的问题）
    fn build_session(path: &Path, force_cpu: bool) -> anyhow::Result<Session> {
        let mut builder = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?
            .with_intra_threads(1)?;
        let ep = if force_cpu {
            CPUExecutionProvider::default().build()
        } else {
            DirectMLExecutionProvider::default().build()
        };
        builder = builder.with_execution_providers([ep])?;
        Ok(builder.with_model_from_file(path)?)
    }

    pub fn new(model_dir: &Path) -> anyhow::Result<Self> {
        let clip_session = Self::build_session(
            &model_dir.join("clip-vit-b32-visual.onnx"), false)?;
        let nima_session = Self::build_session(
            &model_dir.join("nima-technical.onnx"), false)?;
        let clipiqa_session = Self::build_session(
            &model_dir.join("clipiqa_model.onnx"), true)?; // 强制 CPU EP
        Ok(Self { clip_session, nima_session, clipiqa_session })
    }

    /// 批量提取 CLIP embedding
    pub fn extract_clip_embeddings(
        &self,
        batch: &Array4<f32>,  // [N, 3, 224, 224]
    ) -> anyhow::Result<Array2<f32>> {  // [N, 512]
        let outputs = self.clip_session.run(ort::inputs![
            "pixel_values" => batch.view(),
        ]?)?;

        let embeddings = outputs["image_embeds"]
            .try_extract_tensor::<f32>()?
            .view()
            .to_owned();

        // L2 归一化
        Ok(normalize_l2(&embeddings))
    }

    /// NIMA 美学评分
    pub fn assess_quality(
        &self,
        batch: &Array4<f32>,  // [N, 3, 224, 224]
    ) -> anyhow::Result<Vec<f32>> {  // [N] scores 1.0~10.0
        let outputs = self.nima_session.run(ort::inputs![
            "input_1" => batch.view(),
        ]?)?;

        let dist = outputs["activation_55"]
            .try_extract_tensor::<f32>()?
            .view()
            .to_owned();  // [N, 10]

        // 加权均值: score = sum(bin_i * (i+1)) / sum(bin_i)
        let scores: Vec<f32> = dist.outer_iter().map(|row| {
            let sum: f32 = row.sum();
            if sum > 0.0 {
                (0..10).map(|i| row[i] * (i as f32 + 1.0)).sum::<f32>() / sum
            } else {
                5.0 // 默认中间分
            }
        }).collect();

        Ok(scores)
    }
}
```

### 8.3 Tauri IPC 命令

```rust
#[tauri::command]
async fn start_scan(
    state: tauri::State<'_, AppState>,
    app: tauri::AppHandle,
    folders: Vec<String>,
    threshold: f32,
) -> Result<String, String> {
    let session_id = Uuid::new_v4().to_string();

    // 在后台异步执行扫描
    let app_handle = app.clone();
    let db = state.db.clone();
    let ai = state.ai.clone();

    tokio::spawn(async move {
        let total = scan_folders(&folders);

        // 1. 扫描
        for (i, meta) in total.iter().enumerate() {
            let _ = app_handle.emit("scan_progress", ScanProgress {
                current: i + 1,
                total: total.len(),
                phase: "scanning".into(),
                current_file: meta.file_name().to_string_lossy().to_string(),
            });
        }

        // 2. CLIP 特征提取 (批量)
        let embeddings = extract_features_batched(&ai, &total, &app_handle).await;

        // 3. 聚类
        let groups = cluster_embeddings(&embeddings, threshold);

        // 4. NIMA 评分 (仅组内图片)
        let recommendations = assess_groups(&ai, &groups, &total).await;

        // 5. 持久化
        db.save_session(&session_id, &folders, &recommendations);

        // 6. 推送结果
        let _ = app_handle.emit("scan_complete", ScanResult {
            session_id,
            groups: recommendations,
        });
    });

    Ok(session_id)
}

#[tauri::command]
async fn delete_files(
    _state: tauri::State<'_, AppState>,
    file_paths: Vec<String>,
    permanent: bool,
) -> Result<DeleteResult, String> {
    let mut deleted = Vec::new();
    let mut failed = Vec::new();

    for path in file_paths {
        match trash::delete(&path) {
            Ok(_) => deleted.push(path),
            Err(e) => failed.push((path, e.to_string())),
        }
    }

    Ok(DeleteResult { deleted, failed })
}
```

### 8.4 相似度聚类

```rust
use ndarray::Array2;

pub fn compute_similarity_matrix(embeddings: &Array2<f32>) -> Array2<f32> {
    // embeddings: [N, 512], already L2-normalized
    // cosine similarity = A * A^T
    embeddings.dot(&embeddings.t())
}

pub fn cluster_images(
    embeddings: &Array2<f32>,
    threshold: f32,
) -> Vec<Vec<usize>> {
    let n = embeddings.nrows();
    let sim_matrix = compute_similarity_matrix(embeddings);

    // Union-Find
    let mut uf = UnionFind::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            if sim_matrix[[i, j]] > threshold {
                uf.union(i, j);
            }
        }
    }

    // 提取组 (size >= 2)
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf.find(i);
        groups.entry(root).or_default().push(i);
    }

    groups.into_values()
        .filter(|g| g.len() >= 2)
        .collect()
}
```

---

## 9. AI 模型获取与管理

### 9.1 AI 模型清单与获取方式

所有模型为 ONNX 格式，随应用内置（打包进 `dist-package/*.zip`，离线可用），不再现场导出。

| 模型文件 | 用途 | 获取方式 | 大小 |
|----------|------|----------|------|
| `clip-vit-b32-visual.onnx` | CLIP ViT-B/32 视觉编码器（语义 + 美学） | 社区导出 ONNX | ~336 MB |
| `clipiqa_model.onnx` + `clipiqa_model.onnx.data` | CLIP-IQA+ 技术质量评分（主用，图结构 + 外部权重，二者必须同时打包） | hf-mirror `86Cao/IQA-ONNX-Models` | ~146 MB |
| `nima-technical.onnx` | NIMA 技术质量评分（CLIP-IQA+ 不可用时的后备） | 社区导出 ONNX | ~13 MB |
| `aesthetic_linear.bin` | LAION 美学线性头权重 | 随仓库提供 | ~4 KB |

> 注：早期设计通过 PyTorch 现场导出（`export_clip_onnx.py` / `export_nima_onnx.py`），
> 现改为直接下载成熟 ONNX 模型，导出脚本已移除。模型文件因体积大被 `.gitignore` 忽略，
> 需本地放置于 `src-tauri/models/`（或由打包流程复制）。

---

## 10. 构建与分发

### 10.1 构建步骤

```bash
# 1. 导出 ONNX 模型 (需要 Python + PyTorch)
python scripts/export_clip_onnx.py
python scripts/export_nima_onnx.py

# 2. 安装前端依赖
npm install

# 3. 构建 Release 版本
npm run tauri build

# 输出:
# src-tauri/target/release/pixsweep.exe
# src-tauri/target/release/bundle/msi/PixSweep_0.1.0_x64.msi  (安装包)
```

### 10.2 分发要求

| 项目 | 说明 |
|------|------|
| 安装包大小 | ~15MB (Tauri) + ~380MB (ONNX 模型) ≈ 400MB |
| 运行时依赖 | WebView2 (Win11 预装) · 对应 GPU 厂商驱动（DirectML 走 DirectX 12） |
| AI 推理 | ONNX Runtime DirectML（随应用内置，无需额外安装 CUDA） |
| 模型分发 | 首次启动时自动下载 (CDN) 或随安装包内置 |

### 10.3 安装包选项

- **精简版** (~15MB)：不含模型，首次启动下载模型
- **完整版** (~400MB)：包含模型，离线可用

---

## 11. 实施计划

### Phase 1: 核心管线 (MVP)

| 任务 | 预计工时 | 产出 |
|------|----------|------|
| 项目脚手架 (Tauri + React) | 0.5天 | 可运行的空壳应用 |
| 获取 CLIP + NIMA + CLIP-IQA+ ONNX 模型（下载） | 0.5天 | models/ 目录 |
| Scanner 模块 (walkdir + 文件过滤) | 0.5天 | 图片列表 |
| 图像预处理 (decode + resize + normalize) | 1天 | 预处理 pipeline |
| ONNX Runtime 集成 + CLIP 推理 | 1.5天 | embedding 提取 |
| 相似度矩阵 + Union-Find 聚类 | 0.5天 | 分组结果 |
| NIMA 质量评分 | 0.5天 | 每组推荐 |
| SQLite 缓存 + 增量扫描 | 1天 | 数据库层 |

### Phase 2: UI 与交互

| 任务 | 预计工时 | 产出 |
|------|----------|------|
| 前端框架 (工具栏 + 进度条 + 分组卡片) | 1.5天 | 基础 UI |
| 缩略图生成 + LRU 缓存 + asset 协议 | 1天 | 图片预览 |
| Tauri IPC 命令 (scan/delete/settings) | 1天 | 前后端打通 |
| 删除操作 (回收站) + 确认弹窗 | 0.5天 | 删除功能 |
| 批量删除 + 操作日志 | 0.5天 | 批量操作 |
| 设置面板 | 0.5天 | 可配置项 |

### Phase 3: 打磨与分发

| 任务 | 预计工时 | 产出 |
|------|----------|------|
| 错误处理 + 边界情况 (大图/损坏文件/权限) | 1天 | 健壮性 |
| 性能优化 (batch 调优 + 内存管理) | 1天 | 性能调优 |
| MSI 安装包构建 | 0.5天 | 分发包 |
| 测试 (功能 + 性能 + 不同 GPU) | 2天 | 质量保证 |
| 文档 + 用户指南 | 0.5天 | README |

**总预计工时: ~16 天 (单人)**

---

## 12. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| ONNX Runtime DirectML 兼容性 | 中 | 已验证全系 GPU；仅 CLIP-IQA+ 因 Reshape op 走 CPU EP；TOPIQ-NR/IAA 用 torch 2.11 重导出后 DirectML 正常 |
| 大图片集 OOM | 中 | 流式处理 + 增量扫描 + embedding 仅存磁盘不全部常驻内存 |
| HEIC 格式解码 | 低 | 使用 `heif` crate 或调用 Windows HEIC 解码器 |
| 模型下载失败 | 中 | 内置模型到安装包 / 提供 CDN + 镜像下载 |
| 误删文件 | 高 | 默认回收站删除 + 确认弹窗 + 操作日志 + 导出报告 |
| 多显示器 DPI | 低 | Tauri WebView2 原生支持 DPI 缩放 |
| GPU 被其他程序占用 | 中 | 检测显存占用，不足时降低 batch_size 或回退 CPU |

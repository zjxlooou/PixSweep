# PixSweep - 图片智能去重应用设计文档

> Windows 桌面应用：Rust + Tauri 2 后端 + React/TS 前端，本地 GPU（ONNX Runtime，
> CUDA → DirectML → CPU 三级回退）跑 AI 模型，扫描文件夹找出相似/重复图片并按
> 「人像优先」智能推荐最佳图片，帮助用户清理重复图片以减少硬盘占用。
>
> **本文档描述当前架构（2026-08-29 与代码同步）**。历史演进与决策依据见
> `CODING_HISTORY.md`；Agent 协作规范与踩坑清单见 `AGENT.md`。

---

## 1. 需求概述

| 功能 | 描述 |
|------|------|
| 文件夹扫描 | 支持一个或多个文件夹，递归扫描图片（含 23 种相机 RAW 格式） |
| 相似图片检测 | dhash + ahash **双哈希**指纹聚类，过滤渐变图误判 |
| 智能推荐 | AI 逐组评分（人像优先），综合分最高者推荐保留；RAW+JPG 成对时优先保留 RAW 母版 |
| 删除操作 | 批量删除非推荐图片；默认进应用内**临时回收站**（可恢复），可永久删除 |
| 可视化界面 | 图形化操作，缩略图 + 滚轮缩放预览，进度展示，评分标注 |
| 缓存管理 | 代理图/缩略图/评分缓存/日志分类展示占用，可勾选清理（移入系统回收站） |

### 目标平台

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 (x86_64) |
| 硬件 | 无强制 GPU 要求——无 N 卡/无 GPU 时自动回退 CPU 推理 |
| GPU（可选） | NVIDIA（CUDA）或任意支持 DirectX 12 的显卡（DirectML） |
| 显存（可选） | ≥5GB 体验最佳；并发参数按显存/内存自动收缩 |

### 支持的图片格式

- 常规：JPEG / PNG / WebP / BMP / TIFF / GIF / HEIC / HEIF
- 相机 RAW（rawler 解码，机内嵌预览优先）：RW2 / NEF / NRW / ARW / SRW / CR2 / CR3 / CRW /
  RAF / ORF / PEF / PTX / DNG / RAW / RWL / X3F / 3FR / ERF / MRW / IIQ / GPR / KDC / DCR

---

## 2. 技术选型

| 层级 | 技术 | 理由 |
|------|------|------|
| 语言 | **Rust** | 性能接近 C++，内存安全，编译为原生 Windows 可执行文件 |
| GUI 框架 | **Tauri 2** | Rust 后端 + Web 前端 (React)，体积小、IPC 高效，Win11 自带 WebView2 |
| 前端 | React 18 + TypeScript | 组件化开发，图片网格/缩略图展示成熟 |
| AI 推理 | **ONNX Runtime（ort crate）** | CUDA → DirectML → CPU 三级回退；会话配置保证评分确定性 |
| RAW 解码 | **rawler 0.7**（纯 Rust，dnglab 核心） | 无 C 依赖；机内嵌预览毫秒级，全显影秒级 |
| 图像处理 | `image` 0.25 | 主流格式解码/缩放/裁剪 |
| 数据存储 | **JSON 文件**（`serde_json`） | 纯 Rust 无 C 编译依赖（不用 SQLite）；哈希 + 评分双缓存支持增量扫描 |
| 文件遍历 | `walkdir` | 高性能递归目录遍历 |
| 文件删除 | `trash` + 自研隔离区 | 缓存清理移入系统回收站；图片删除走应用内临时回收站（可恢复） |
| 并发 | `tokio` + `rayon`（专用重活池） | tokio 管 async；大缓冲并行跑固定线程数的 `heavy_pool`（内存护栏） |

### AI 模型清单（当前）

| 用途 | 模型 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 技术质量（主） | **TOPIQ-NR** (ResNet50, FP16) | 384×384 CHW | 0~1 → 1~10 | KonIQ-10k SRCC 0.930；动态 batch |
| 美学（主） | **TOPIQ-IAA** (ResNet50, FP16) | 384×384 CHW | 10-bin → 1~10 | AVA SRCC 0.791；动态 batch |
| 非人像美学融合 | **HyperIQA** (ResNet50, FP16, 可选) | 512×512 CHW | 0~1 | 对风景/宠物/食物降级敏感度强；**必须逐张推理**（fix batch=1） |
| 技术质量后备 | **NIMA** (MobileNet, 可选) | 224×224 NHWC | 10-bin → 1~10 | TOPIQ-NR 缺失时的兜底 |
| 人脸专评 | **TOPIQ-NR-Face** (ResNet50, FP16) | 512×512 对齐人脸 | 0~1 → 1~10 | CGFIQA-40k；与 nr-on-face 50/50 融合修暗光盲区 |
| 人脸检测 | **SCRFD det_10g** (buffalo_l) | 640×640 letterbox | bbox + 5 关键点 | 会话副本池按显存分档（6GB 卡 2 副本） |
| 场景分类 | **MobileNetV3-Large** | 224×224 [0,1] | ImageNet 1000 → 人像/宠物/风景/其他 | 人像由人脸检测覆盖，本模型只管无脸图 |
| 闭眼检测 | **OCEC** + **MediaPipe 脸网格** | 24×40 眼 ROI / 256×256 | 开眼概率 | 脸网格垂目开度为主信号，OCEC 眨眼否决 |

- 三主力模型为 **FP16**（IO 保持 FP32），体积减半、GPU 更快；CPU EP 无原生 fp16 核会慢数倍。
- 模型**缺失不报错**，仅告警并跳过对应能力（技术后备链 TOPIQ-NR → NIMA；美学无后备）。
- CLIP/LAION 美学后备已于 2026-08-27 移除（主模型健康时零参与评分，-489MB）。

### 会话配置（评分确定性）

所有会话统一：`with_parallel_execution(false)` + `with_memory_pattern(false)` +
`with_intra_threads(1)` + `GraphOptimizationLevel::Level3`，保证同一张图分数可复现。
CUDA 会话额外用 `SameAsRequested` arena 策略，防止 arena 翻倍预留挤占 6GB 显存
（大激活会话把后续小模型拖慢 6× 的根因，见 `docs/GPU_PERF_PLAN.md`）。

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                Frontend (React + TS, Tauri Webview)           │
│   文件夹选择 · 扫描进度 · 分组卡片 · 预览缩放 · 设置/缓存面板  │
└──────────────────────────┬───────────────────────────────────┘
                           │ Tauri IPC (invoke / event)
┌──────────────────────────┴───────────────────────────────────┐
│                    Backend (Rust, src-tauri/src)               │
│                                                                │
│  scanner/walker → hashing/phash → cluster (双哈希+UnionFind)   │
│        ↓                                                       │
│  cache/proxy (统一前置代理) → ai/engine (评分编排)              │
│        ↓                                                       │
│  quality/recommender (build_groups + 推荐理由)                  │
│                                                                │
│  fileops/trash (隔离区) · cache/thumbnail · db/store (JSON)    │
│  commands.rs (IPC 命令) · mcp.rs (HTTP JSON-RPC, :18765)       │
│  ai/hardware (核心数/内存/显存画像 → 动态并发参数)              │
└──────────────────────────┬───────────────────────────────────┘
                           │ ort crate
┌──────────────────────────┴───────────────────────────────────┐
│        AI Inference (ONNX Runtime, CUDA→DirectML→CPU)          │
│   TOPIQ-NR/IAA/HyperIQA/NIMA · nr_face · SCRFD · MobileNetV3   │
│   OCEC 闭眼 · MediaPipe 脸网格                                  │
└──────────────────────────────────────────────────────────────┘
```

### 后端模块职责

- `commands.rs` — Tauri IPC 命令层 + 扫描编排（`run_scan`）：扫描 → 哈希 → 聚类 →
  AI 评分 → 分组 → 结果落 state（大对象 invoke 拉取，事件只推小摘要）。
- `ai/engine.rs` — 核心：可选模型会话装载（缺失跳过）、`composite_scores` 综合评分、
  双缓冲批量评分流水线、场景∥人脸并发、闭眼/对焦批处理。
- `ai/insightface.rs` — SCRFD 检测（副本池 + 可选动态 batch）+ 5 关键点仿射对齐。
- `ai/eye.rs` — 闭眼双信号融合（脸网格垂目开度为主 + OCEC 眨眼否决）。
- `ai/focus.rs` — 拉普拉斯方差对焦分（整图/眼部 ROI 两口径）。
- `ai/scene.rs` — MobileNetV3 场景分类（ImageNet 子集映射）。
- `ai/hardware.rs` — 硬件画像：核心数定重活线程数，显存/内存定 SCRFD 副本数。
- `ai/preprocess.rs` — 各模型输入 tensor（居中裁剪/缩放/归一化，rayon 重活池并行）。
- `quality/recommender.rs` — `AiScoreBundle`（评分结果束）+ `build_groups`（推荐 +
  平局 tie-break + RAW 容差优先 + 理由文案）。
- `cache/proxy.rs` — 统一前置代理（见 §4.3）。
- `db/store.rs` — JSON 缓存（哈希/尺寸 + 美学/技术分 + 人脸/场景/闭眼/对焦），带
  schema 版本号防旧缓存静默复用。
- `fileops/trash.rs` — 临时回收站（复制→校验→索引→删原件，支持恢复/清空）。
- `mcp.rs` — 本机 HTTP JSON-RPC（`127.0.0.1:18765/mcp`），供外部 Agent 端到端测试。

---

## 4. 核心流程

### 4.1 扫描主链路（`run_scan`）

```
扫描文件夹(walkdir) → 感知哈希(dhash+ahash, 增量缓存) → 聚类(双哈希+并查集)
  → AI 评分（人像优先，见 4.2） → build_groups 分组推荐 → 事件通知 + invoke 拉取
```

- 文件指纹 = blake3("pixsweep-fp-v2" + path + size + mtime)；指纹算法数据源改代理图时
  版本前缀 +1，旧缓存一次性失效。
- 聚类：dhash 相似度 ≥ 阈值（默认 0.92）**且** ahash ≥ 0.80 双条件；无特征图
  （纯色/渐变）跳过配对。
- RAW 的宽高在哈希阶段被 `raw_source_dimensions` 覆盖为**传感器原生尺寸**（dummy 探针
  毫秒级，EXIF 转正含竖拍互换），保证分辨率启发式不被机内嵌小预览低估。

### 4.2 AI 评分链路（人像优先）

```
每张图（增量模式先查两级缓存）：
  1. TOPIQ-IAA 美学 + TOPIQ-NR/NIMA 技术 —— 双缓冲流水线批量推理
  2. 场景分类(MobileNetV3) ∥ 人脸检测+专评(TOPIQ-NR-Face) —— M4 并发（不同 session）
  3. 闭眼检测（仅有人脸图）：脸网格垂目开度为主，OCEC 双眼强判闭时否决
  4. 对焦：人像=眼部对焦（眼 ROI 锐度），非人像=整图对焦
  5. 非人像美学融合：TOPIQ-IAA ⊕ HyperIQA 50/50（必须逐张）
  6. composite_scores 融合：
     人像   = 人脸专评 0.55 + 眼部对焦 0.30 + 启发式 0.15（闭眼连续降权）
     风景   = 美学 0.40 + 对焦 0.50 + 启发式 0.10
     宠物   = 美学 0.45 + 对焦 0.45 + 启发式 0.10
     其他   = 美学 0.25 + 对焦 0.60 + 启发式 0.15
```

推荐语义：组内综合分最高者保留（分差 <0.05 时按"像素+文件大小" tie-break）；
同组有 RAW 与其导出 JPG 时，RAW 分差在 0.5 容差内即改推 RAW（无损母版）。
**大小对比一律用源文件大小**（`ImageInfo.size`），代理图只参与内容分析（结构性保证见 §4.3）。

### 4.3 统一前置代理（`cache/proxy.rs`）

- **触发**（任一）：最长边 > 2048 ∨ 源文件 > 2MiB ∨ RAW。
- **输出**：EXIF 转正、最长边 ≤ 1920、JPEG < 2MiB（质量阶梯 95→60，全超限降 1280 重试）；
  原子落盘（临时文件 + rename）。
- **存放**：临时文件夹 `app_data_dir()/quarantine/proxy/`（工具栏"临时文件夹"按钮显示
  整个隔离区占用）。
- **消费端**：全部 AI 路径与感知哈希走 `ai_proxy`（只返回像素不返回元数据）；缩略图
  与 RAW 预览走自有路径（哈希稳定 / RAW 预览要原生分辨率）。
- **内存护栏**：解码/代理/预处理等大缓冲并行跑 `image_io::heavy_pool()`（线程数 =
  `hardware::decode_threads()`），配合 RAII 解码信号量限制同时驻留的全分辨率缓冲。

### 4.4 RAW 双口径

- **AI 分析口径**：机内嵌预览（full > preview > thumbnail，毫秒级）→ 统一代理。
- **预览查看口径**：全显影（demosaic → sRGB，传感器原生分辨率，0.4~1.6s，结果落盘
  `full_preview/` 缓存）——机内嵌预览放大后远不如同画面 JPG 清晰。
- 两条路径均手动应用 EXIF orientation（`image_io::apply_exif`）。

---

## 5. 数据与缓存

### 5.1 JSON 缓存（`db/store.rs`）

```jsonc
// pixsweep-cache.json（程序根目录）
{
  "records": {
    "<file_hash>": {
      "path": "...", "size": 0, "modified": 0, "width": 0, "height": 0,
      "dhash": 0, "ahash": 0,
      "aesthetic_score": null, "technical_score": null, "ai_scores_schema": 2,
      "ai_face_cache": { "schema_version": 5, "has_face": true, "face_score": 6.5,
                          "scene": 1, "eye_open": 0.9, "focus_score": 7.0 }
    }
  }
}
```

- 增量扫描按 `file_hash`（内容指纹）命中；schema 版本不符视为未缓存，防止旧语义
  静默复用（评分 v2 = HyperIQA 融合；人脸 v5 = nr-on-face 融合）。

### 5.2 运行期文件布局（程序根目录，exe 同级）

```
pixsweep.log           # 文件日志（Info）
pixsweep-cache.json    # 哈希 + 评分缓存
thumbnails/            # 256px 缩略图（v2 前缀）
full_preview/          # RAW 全显影预览缓存
quarantine/            # 临时回收站（files/ + index.json）
quarantine/proxy/      # 统一代理图缓存（v3 前缀）
logs/                  # 删除操作日志
```

"清理缓存"面板按 类型（代理/缩略图/评分缓存/日志/隔离区）展示占用，清理动作把文件
移入**系统回收站**（可恢复，非永久删）。

---

## 6. 打包与分发

- `scripts/build_release.ps1`：npm build → cargo release（自动注入 zig+xwin 工具链）→
  按白名单复制 exe / 3 个 DLL（DirectML、CUDA provider、providers_shared）/ 模型 →
  7-Zip mx=7 压缩 → 校验 zip 清单 → 清理解包目录。
- 产物：`dist-package/PixSweep-vX.Y.Z.zip`（约 430MB），解压即用，离线运行。
- 模型不入 git（`.gitignore`），以发布 zip 分发；打包白名单必须显式列出全部
  `.onnx.data` 配对权重（ORT 校验外部数据引用跟随 onnx 文件名）。
- 数据目录 = 程序根目录（便携运行，不写 `%LOCALAPPDATA%`）。

---

## 7. 测试与验证

| 层级 | 命令 | 覆盖 |
|------|------|------|
| Rust 单测 | `cargo test` | 哈希/聚类/推荐/评分公式/缓存 schema/闭眼映射/硬件分档等 46 项 |
| 前端组件 | `npx vitest run` | 删除确认弹窗、分组卡片、formatBytes 等 25 项 |
| 类型检查 | `npx tsc --noEmit` | 前后端共享类型 |
| 端到端 | `bash scripts/test_e2e.sh` | 启动 app+MCP → 扫描 → 分组 → 删除/恢复/清空 → 导出 |
| 真图回归 | `cargo run --example verify_*` | 各 AI 能力 + 标注集（见 `src-tauri/examples/README.md`） |

详见 `docs/TESTING.md`。

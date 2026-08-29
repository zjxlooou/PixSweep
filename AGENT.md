# AGENT.md — PixSweep Agent 规范

> 本仓库 agent 规章的唯一权威入口。以下约束违反会导致编译失败、数据丢失或返工。

## 项目速览

- **定位**：Windows 本地图片去重桌面应用（Tauri 2 + Rust 后端 + React/TS 前端），核心卖点「人像优先」AI 评分
- **主目录**：本仓库唯一工作目录，不另起副本（本机绝对路径见 `PRIVATE.local.md`，不入库）

## 构建环境（违反则无法编译）

用户机器**没有 Visual Studio / MSVC**，Rust/C++ 编译依赖 `.tools/` 工具链：`xwin-sdk`（Windows SDK）、`zig 0.16`（替代 cl.exe/lib.exe/rc.exe）、`zigwrap`（桥接 wrapper）。

**编译命令模板**（`ROOT` 为本仓库根目录的 Windows 风格路径，本机真实值见 `PRIVATE.local.md`；务必完整复制环境变量）：

```bash
ROOT="<本仓库根目录>"
export PATH="$HOME/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin:$ROOT/.tools/zig-bin/zig-x86_64-windows-0.16.0:$PATH"
export RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-msvc
export CC="$ROOT/.tools/zigwrap/zigcc.exe"
export CXX="$ROOT/.tools/zigwrap/zigcxx.exe"
export AR="$ROOT/.tools/zigwrap/ziglib.exe"
export RC="$ROOT/.tools/zigwrap/zigrc.exe"
export ZIG_GLOBAL_CACHE_DIR="$ROOT/.tools/zig-cache"
export ZIG_LOCAL_CACHE_DIR="$ROOT/.tools/zig-cache/local"
cd "$ROOT/src-tauri" && cargo build --release
```

**硬性规则**：
1. **路径格式**：传给 Windows 原生程序（rustc/cargo/python/zig/xwin）的路径必须用 `盘符:/目录/...` 形式，禁用 Git Bash 的 `/盘符/目录/...`（会被误解析成 `盘符:\盘符\目录\...`）。Git Bash 内置工具（ls/cat/grep）才可用 `/d/` 风格。
2. cargo 命令静默退出时，检查是否设置了 `RUSTUP_TOOLCHAIN`（rustup proxy 无此变量会 no-op）。
3. **构建顺序铁律**：`npm run build` **必须先于** `cargo build --release`——Tauri 编译时嵌入 `dist/`，顺序颠倒打包出的 exe 用的是旧前端。

## 常用命令

| 目的 | 命令 |
|------|------|
| 后端单测 | `cargo test`（需同上环境变量） |
| 前端组件测试 | `npm test`（vitest，`src/**/__tests__/*.test.tsx`） |
| 类型检查 | `npx tsc --noEmit` |
| 端到端测试 | `bash scripts/test_e2e.sh` |
| 后端编译 | `cargo build --release`（见上文环境模板） |
| 真图验证 | `cargo run --example verify_ai -- <照片目录> 8` |
| 构建 exe | `powershell -ExecutionPolicy Bypass -File scripts/build.ps1` |
| 打包发布 | `powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1` → `dist-package/PixSweep-vX.Y.Z.zip`（手工等价：复制 exe/DLL/模型后 `.tools/7zip/7za.exe -mx=9`） |

## 真图验证（examples）

验证 Rust 侧 AI 链路，传真实照片目录（**只读**），必须给 Windows 风格路径：

```bash
cargo run --example verify_ai -- <照片目录> 8   # 全链路评分
cargo run --example verify_face -- <目录>                    # 人脸检测
cargo run --example verify_eye -- <目录>                     # 闭眼检测
cargo run --example verify_scene -- <目录>                   # 场景分类
```

照片目录等本机私人路径见 `PRIVATE.local.md`（不入库）。

注意：example 没初始化 logger 会吞 warn，排查先拿返回值或加 `env_logger`；`verify_ai` 恒设 `has_face=false`、`eye_open=1.0`（纯整图链路），**测不到人脸/闭眼/眼部对焦**，测这些须走完整链路（如 `score_groups_with_ai` 或 `verify_labeled`）。

## 架构

```
前端 src/ (React+TS) ──Tauri IPC (invoke/event)── 后端 src-tauri/src/ (Rust)
扫描(walkdir) → dhash+ahash 双哈希聚类 → AI 综合评分（人像优先） → 每组推荐最佳
```

后端模块（`src-tauri/src/`）：

- `commands.rs` — IPC 命令层，扫描编排，调用 `composite_scores` 得到评分送入 `build_groups`
- `ai/engine.rs` — 核心：加载全部 ONNX session（三级回退 CUDA→DirectML→CPU）、`composite_scores` 组合评分（场景权重重 + 闭眼惩罚）
- `ai/topiq.rs` / `ai/topiq_face.rs` / `ai/nima.rs` — 各推理封装；`ai/preprocess.rs` 图像预处理
- `ai/insightface.rs` — 人脸检测（buffalo_l `det_10g` 输入名 `input.1` + 手动 NMS）+ `align_face`（5 关键点→112 模板）
- `ai/eye.rs` — 闭眼检测双信号：MediaPipe 脸网格（垂目开度，主信号）+ OCEC（眨眼否决），网格虹膜中心回喂 OCEC 做 ROI
- `ai/scene.rs` + `ai/scene_map.rs` — MobileNetV3 场景分类（人像/风景/宠物/其他）
- `quality/recommender.rs` — `build_groups` 组装分组 + 推荐理由
- `scanner/` `hashing/` `cluster/` `db/` `fileops/` `cache/` `mcp.rs` — 遍历/哈希/聚类/JSON 缓存/回收站/缩略图/MCP server
- `types.rs`（Rust）/ `src/types/index.ts`（TS）— 前后端共享类型

新功能按上述模块归位，不另起顶层目录。

### AI 评分链路（人像优先）

```
每张图（type-first）：
  InsightFace 人脸检测 → 有脸 => 人脸专评 (TOPIQ-NR-Face) 主导
  闭眼双信号           → 脸网格垂目开度为主；OCEC 仅在"网格模棱两可且双眼强判闭"时眨眼否决
  MobileNetV3 场景分类  → 人像/风景/宠物/其他 不同权重
  对焦判断（拉普拉斯方差）→ 人像=眼部对焦，非人像=整图对焦
  TOPIQ-NR (技术) + TOPIQ-IAA (美学) + 分辨率启发式 → 基础分
  composite_scores 融合 → 综合分（face 主导人像；无脸由对焦主导）
```

### 推理后端分配

- **三级回退链 CUDA→DirectML→CPU** 用于所有模型（CUDA 仅在检测到真实 NVIDIA GPU 时启用，见 `build_session`）。
- 会话统一 `with_parallel_execution(false)` + `with_memory_pattern(false)` + `with_intra_threads(1)` 保证评分确定性（否则同一张图分数时高时低）。
- 模型**缺失不报错**，仅 `log::warn` 并跳过对应能力（真机测要核对日志）。
- CLIP/LAION 美学后备已移除（2026-08-27）：技术后备链 = TOPIQ-NR → NIMA；美学无后备。

### RAW 相机格式（0.8.1）

- **解码**：rawler 0.7（纯 Rust，dnglab 核心，LGPL-2.1 依赖）。唯一入口在 `image_io::load_image_oriented` 的 RAW 分支——优先机内嵌预览（`full_image` > `preview_image` > `thumbnail_image`，毫秒级，相机端已去马赛克/白平衡），全无嵌入预览才回退全显影（demosaic，实测 RW2 0.4s）。两条路径手动应用 EXIF orientation（官方 `Orientation::from_exif`）。
- **扩展名清单**：`image_io::RAW_EXTENSIONS` 与 `scanner/walker.rs::SUPPORTED_RAW_EXTENSIONS` 两处保持一致（新增格式要同步改）。23 种：RW2/NEF/NRW/ARW/SRW/CR2/CR3/CRW/RAF/ORF/PEF/PTX/DNG/RAW/RWL/X3F/3FR/ERF/MRW/IIQ/GPR/KDC/DCR。
- **全链自动生效**：缩略图/AI 评分/代理/哈希/对焦都走 `load_image_oriented`，RAW 无需各自适配。
- **分辨率源口径**（2026-08-28）：RAW 的宽高在扫描哈希阶段被 `image_io::raw_source_dimensions` 覆盖为传感器原生尺寸（rawler `raw_image(dummy=true)` 探针，毫秒级，crop→active→全幅，EXIF 转正含竖拍宽高互换）——机内嵌预览往往远小于传感器（Sony 1080p 预览 vs 24MP），不覆盖会低估分辨率启发式；缓存命中路径同样覆盖，顺带修复旧缓存记录。验证工具 `examples/raw_dims_check.rs`。
- 已实测解码：Panasonic RW2（嵌入预览 1920×1440 + 全显影 0.4s）。探针工具：`examples/probe_raw_decode.rs`。

### 统一前置代理（2026-08-28，`cache/proxy.rs`）

- **触发**（任一）：最长边 > 2048（2K）｜ 源文件 > 2MiB ｜ RAW 原片。不触发的普通小图直接解码用，不落盘。
- **输出**：EXIF 转正、最长边 ≤ 1920（<2K）、JPEG 体积 < 2MiB（质量阶梯 95→60，全超限再降到 1280 重试）。缓存版本 v3。
- **存放**：临时文件夹 `app_data_dir()/quarantine/proxy/`（工具栏"临时文件夹"按钮显示整个隔离区占用，含代理）。旧版程序根 `proxy/` 目录在首次访问时代理模块自动删除。
- **消费端全统一**：全部 AI 路径（整图评分/场景/人脸检测/闭眼网格+OCEC/眼对焦/nr-on-face crop）都走 `ai_proxy`；**哈希与缩略图除外**（哈希值稳定性、缩略图自有快路径）。
- **大小对比基准**：代理只参与图像内容分析（指纹/AI/对焦）；一切文件大小比较（综合分启发式、推荐 tie-break、理由文案）用源文件大小 `ImageInfo.size`（walker 阶段 `fs::metadata` 写入）。`ai_proxy` 只返回像素不返回元数据，即结构性保证。
- **分辨率口径同理**：RAW 的宽高为传感器原生（`raw_source_dimensions` 覆盖），非预览尺寸；代理/解码尺寸只服务内容分析。
- **重活并发上限**：全部大缓冲并行工作（解码/代理/检测预处理）跑在 `image_io::heavy_pool()` 专用 6 线程池（== 解码信号量数），绝不用 rayon 全局池（每逻辑核一线程会并发解码，直接把工作集推到 10GB+）。mimalloc 全局分配器与 zig 工具链不兼容（v2/v3 均链接失败），已弃用——内存靠并发上限控制。
- **精度依据**：对焦整图归一 1024、眼 ROI 归一 24×40 再算拉普拉斯方差，SCRFD letterbox 640×640，闭眼网格用几何比例——对输入分辨率不敏感。闭眼标注集回归 **7/7**（统一前基线 6/7）；RAW 与同画面 JPG 技术分差 ≤0.08、美学 ≤0.02。
- 验证工具：`examples/proxy_check.rs`（触发判定 + 双断言 + 缓存命中计时）。

### 推荐语义：RAW 优先（2026-08-28）

组内同时有 RAW 与其导出 JPG 时保留 RAW（无损母版，JPG 是冗余副本）。RAW 因机内嵌预览偏软综合分系统性略低，故给 0.5 分容差（`RAW_PREFER_TOLERANCE`）：组内最佳 RAW 与全局最佳分差在容差内即改推 RAW；分差过大仍尊重评分。理由文案显式说明 RAW 保留原因。

### 易变签名

`composite_scores`（engine.rs）与 `build_groups`（quality/recommender.rs）签名改动频繁（前者已从纯 slice 演化为 Option 混合）。改前先查定义与全部调用点（`commands.rs`、`examples/verify_*.rs`、测试），改后逐一同步。

## 模型与打包

- 模型在 `src-tauri/models/`（**FP16 精度**，~310MB 通用评分模型 + `insightface/` + `scene/` + `eye/` 子目录），**不入 git，不可删**（删除需重新下载），通过发布 zip 分发。文件清单以磁盘为准（`ls src-tauri/models/`）；`models/eye/face_landmarker.onnx`（4.6MB 脸网格）是可选信号，缺失自动退化为仅 OCEC。
- **外部数据配对文件坑**：部分模型是 `*.onnx` + `*.onnx.data` 成对存放（当前仅 `topiq_nr_face`）。**ORT 校验 .data 引用路径跟随 onnx 文件名**——重命名 onnx 必须同步改内部引用或干脆内嵌为单文件（`onnx.external_data_helper.convert_model_from_external_data`）。打包白名单 `scripts/build_release.ps1` 的 `$neededModels` 必须显式列出全部 `.onnx.data`，否则发布版静默缺件。
- **精度**：2026-08-27 起三主力模型为 FP16（IO 保持 FP32，引擎零改动）。实测与 FP32 的 ρ≥0.9998、GPU 更快；**CPU EP 无原生 fp16 核会慢数倍，性能结论只在 GPU 上有效**。
- **美学融合（可选，hyperiqa.onnx 55MB）**：场景≠人像时 美学=TOPIQ-IAA ⊕ HyperIQA 50/50（线性校准 `HYPERIQA_CAL_*`）。人像偏置重故人像不启用。**必须逐张推理**（fix batch=1）：批量喂入不仅静默出错，其密集层中间张量随 batch 膨胀——batch=16 单次 run 实测触碰 3.2GB 主机内存（2026-08-29 排查定位，曾致扫描内存峰值 14GB）。
- **人像融合**：face = nr_face ⊕ nr-on-face 50/50（`FACE_FUSION_NR_FACE_WEIGHT`），修 nr_face 的暗光盲区（欠曝人像不再反向加分）；face 缓存 schema v5、评分缓存 v2 联动失效。
- **batch 维度**：TOPIQ-NR/IAA 为**动态 batch**（整批一次推理）；其余评分模型仍 **fix batch=1**，必须逐张推理（批量输入静默失败）。SCRFD 人脸检测：`det_10g_batched.onnx`（`scripts/make_det_batched.py` 图手术生成，权重不变）存在即启用批量检测，**默认不随包发布**（生产链路实测无净收益且拖慢 HyperIQA，见 GPU_PERF_PLAN.md）；缺失时走 2 副本会话池并行逐张。
- `models-archive/` 存弃用模型存档（MUSIQ、CLIP 对、FP32 三巨头），不参与打包、不入库。
- 临时 Python 模型验证脚本用完即删，**不要 `git add` 进提交**。

## 数据安全

- 本仓库是 git 仓库（远程 `github.com:zjxlooou/PixSweep`）。提交作者为 **ZCode**（邮箱占位 `noreply@zcode.local`），不加 co-author。
- ✅ 可安全删除（本地源可重建）：`node_modules/`、`dist/`、`target/`
- ❌ 不可删（无简单恢复/会破坏构建）：`.tools/`（2.2G 工具链）、`src-tauri/models/`、`dist-package/*.zip`
- **删除大目录/大量文件前，必须先向用户确认范围。**
- 数据目录：**程序根目录**（exe 同级，`app_data_dir()` 返回 exe 所在目录，不写 `%LOCALAPPDATA%`）。
- 用户照片测试源与标注集：本机私人目录（见 `PRIVATE.local.md`，只读）。
- **隐私约定**：本机私人路径、用户名、机器信息统一记录在 `PRIVATE.local.md`（gitignored）；仓库内文档与代码一律用占位符，禁止提交真实路径。

## 近期坑（必读，避免再踩）

### EXIF Orientation 必须用 decoder.orientation()

`image::metadata::Orientation::from_exif_chunk` 要求偏移 0 就是 TIFF magic，但 JPEG 文件头以 `FF D8`(SOI) 开头，永远不工作。要用 `ImageReader::into_decoder()?.orientation()`。所有 AI 预处理、人脸检测、缩略图统一走 `image_io::load_image_oriented` → 一处修复全局生效。

### InsightFace SCRFD-10G 检测的 2 个隐藏 bug

1. **每个 grid cell 有 2 个 anchor**：输出 shape `[N,1/4/10]` 中 N = `h_grid × w_grid × 2`。代码必须用 `(cy*w_grid+cx)*2+anchor` 双层循环，按 1 anchor/grid 索引会完全错位。
2. **5 关键点输出顺序为 `(right_eye, left_eye, nose, right_mouth, left_mouth)`**——不是直觉顺序。用错会导致 `align_face` 镜像/错位、眼睛 ROI 采错 → 闭眼误判。

都在 `src-tauri/src/ai/insightface.rs::detect`。

### InsightFace 5 关键点眼位系统性偏低（~10% 脸高）

det_10g 的「眼睛」关键点实测落在真实眼睛下方约 0.1×脸高（bbox 相对高度 0.45~0.50，应为 ~0.35），直接以其为中心的小窗口采样会采到脸颊。**眼 ROI 一律改用脸网格虹膜中心**（`EyeDetector::mesh_result` 返回 `*_eye_src`，经逆仿射映射回原图）；关键点只用于角度/bbox。诊断手法：`dump_eye_roi` 导出 ROI PNG 目视。

### 闭眼检测：OCEC 只认眨眼式，垂目靠脸网格

OCEC（训练数据=眨眼式闭眼）对「垂目/低头看」判全开（任务不匹配，调参无解），且对刘海遮挡的睁眼常有假闭误报。现行融合（`engine.rs::eye_open_probs`）：**脸网格垂目开度为主信号**（睑缝高/虹膜直径，尺度无关），OCEC 仅在「网格 <0.85 且双眼 min(prob_open) <0.2」时眨眼否决；网格缺失回退 OCEC 原 max 语义。锚点/门限常量（`MESH_RAW_*`/`OCEC_VETO_MAX`/`MESH_VETO_BAND`）如需调整，回归基准是 `verify_labeled`（标注集 6/7，组4 极端侧脸为已知不可解）。

### fix batch=1 模型批量推理的报错特征

把 N 张输入拼成 `[N,...]` 一次 run 会报 `Got invalid dimensions for input: xxx`。看到此报错先怀疑 fix batch 模型被整批喂入，修法是循环逐张推理。

## 关键决策记录（避免重复踩坑）

| 决策 | 结论 | 原因 |
|------|------|------|
| 推理后端 | CUDA→DirectML→CPU 三级回退 | POC 实测 CUDA 最快（~4× CPU）；但 EP DLL 能加载 ≠ 有 NVIDIA GPU，需真实检测；不绑定单一硬件生态 |
| 权重精度 | 主力三模型 FP16（IO 保持 FP32） | 实测 ρ≥0.9998、体积减半、GPU 更快；CPU EP 慢数倍故仅 GPU 有效 |
| CLIP/LAION 后备 | 移除（2026-08-27） | 主模型健康时零参与评分；-489MB；技术后备保留 NIMA，美学无后备 |
| 人像质量 | nr_face ⊕ nr-on-face 50/50 | 357 张基准：nr_face 暗光盲区归零、敏感 ×3.6，保留一半特化信号 |
| 非人像美学 | IAA ⊕ HyperIQA 50/50（仅场景≠人像） | hyperiqa 对风景/宠物/食物降级敏感度第一（d=1.20）；人像偏置重不用于人像 |
| 存储 | JSON（非 SQLite） | 避免 C 编译依赖 |
| 技术评分主模型 | TOPIQ-NR（非 CLIP-IQA+） | KonIQ SRCC 0.930 > 0.885 |
| 美学评分主模型 | TOPIQ-IAA（非 LAION V1） | AVA SRCC 0.791 > 0.665 |
| 大对象传输 | invoke 返回（非事件 emit） | 大 payload 白屏 |
| 回收站 | 临时隔离区（非系统回收站） | 误删后恢复可控 |

## 文档维护

- 架构变更（模型更换、后端切换、存储方式变更）后，**必须同步** `docs/DESIGN.md`、`README.md`，并在 `CODING_HISTORY.md` 追加记录。
- 真实配置以 `src-tauri/Cargo.toml`、`src-tauri/tauri.conf.json` 为准。

| 文件 | 内容 |
|------|------|
| `CODING_HISTORY.md` | 需求变动与技术历史时间线 |
| `docs/DESIGN.md` | 架构设计（部分过时，以代码为准） |
| `docs/PORTRAIT_RATING_RESEARCH.md` | 人像评分公式（§3.2 已落地） |
| `docs/TESTING.md` | 测试指南 |

## 代码风格

- Rust：模块级用 `//!`、公开项用 `///` 文档注释（缺注释会被视为未整理）。
- 前后端共享类型集中在 `src-tauri/src/types.rs`（Rust）/ `src/types/index.ts`（TS），不要在 commands.rs 里内联定义。

## Engineering skills 约定

- **Issue tracker**：issues 与 specs 以本地 markdown 存放在 `.scratch/<feature-slug>/`（spec 为其下 `spec.md`）。详见 `docs/agents/issue-tracker.md`。
- **Domain docs**：单 context 仓库惯例为根目录 `CONTEXT.md` + `docs/adr/`；文件尚不存在时静默继续。详见 `docs/agents/domain.md`。

---

*最后更新：2026-08-27*

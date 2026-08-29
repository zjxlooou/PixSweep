# PixSweep 现有功能说明

> 本文档描述 PixSweep **当前已实现、可运行**的功能（2026-08-29 与代码同步）。以 `src-tauri/src/` 与 `src/` 实际代码为准；架构细节见 `docs/DESIGN.md`，人像评分公式见 `docs/PORTRAIT_RATING_RESEARCH.md` §3.2。

## 1. 产品定位

Windows 本地图片去重桌面应用（Tauri 2 + Rust 后端 + React/TS 前端）。**核心卖点是「人像优先」的 AI 评分体系**：扫描文件夹 → 相似图片聚类 → AI 综合评分 → 每组推荐"最好的一张"。

- 前端 `src/`（React + TS）──Tauri IPC（invoke/event）── 后端 `src-tauri/src/`（Rust）
- 流程：`扫描(walkdir)` → `phash 双哈希去重初筛` → `相似聚类` → `AI 综合评分（人像优先）` → `每组推荐最佳`

## 2. 核心流程（一次扫描做了什么）

### 2.1 扫描文件夹
`scanner/walker.rs::scan_folders`。递归遍历文件夹，过滤出图片格式（jpeg/png/webp/bmp/gif/tiff/heic）。返回图片 `ImageInfo` 列表（路径/文件名/大小/修改时间/格式，宽高此时未知）。

### 2.2 感知哈希去重初筛 / 聚类
`hashing/phash.rs` 计算两种感知哈希，`cluster/similarity.rs::cluster_by_hash` 用 Union-Find 聚类：

- **dhash**（对压缩/重编码稳健）+ **ahash**（对平滑渐变图敏感）**双条件**同时满足才归组。
- 过滤"无特征"图（popcount 过低的纯色/渐变图），跳过其配对，避免误判。
- `dhash_similarity >= similarity_threshold`（默认 0.92，可在设置里调）**且** `ahash_similarity >= 0.80`。
- 返回组内成员 ≥ 2 的图片索引分组。

> 去重聚类 = phash 双哈希（dhash + ahash），不依赖任何 AI 模型（CLIP 路线已于 2026-08-27 移除）。

### 2.3 增量缓存（跳过未变化文件）
`db/store.rs` 用 JSON 文件（程序根目录 `pixsweep-cache.json`，exe 同级）缓存每张图的结果，支持增量扫描：

- **文件指纹** `file_hash = blake3(path + size + mtime)`。文件未变 → 命中缓存。
- **哈希/尺寸缓存**：命中则复用 `dhash`/`ahash`/`width`/`height`，跳过读取文件（增量扫描核心加速点）。
- **AI 评分缓存**：
  - `aesthetic_score` + `technical_score`（美学/技术分）。
  - `AiFaceCache`（`has_face` / `face_score` / `scene` / `eye_open`）+ `schema_version`。人脸/场景/闭眼结果与美学/技术分**独立判定缓存**：美术分命中并不代表人脸分命中，反之亦然；只有缓存命中的维度才跳过推理。
- `schema_version` 使**语义变化**能可靠失效缓存（如阶段二把闭眼从 bool 改为连续概率，旧缓存据此重算）。
- 非增量模式仍会写缓存，使下一次增量重扫受益。

### 2.4 AI 综合评分（人像优先）
`ai/engine.rs` 加载全部 ONNX session，三级回退 **CUDA → DirectML → CPU**。对每组图片做：

**每张图，按顺序：**

1. **InsightFace 人脸检测**（`ai/insightface.rs` SCRFD-10G `det_10g`，输入 640×640）：
   - bbox + 5 关键点。**2-anchor per grid cell** 索引（`(cy*w_grid+cx)*2+anchor`）。
   - 手动 NMS（IoU 0.4）。关键点可信度校验（`landmarks_trustworthy`：关键点落 bbox 外扩 50% + 眼距 > 5% 长边），仅拦明显退化人脸。
   - 5 关键点顺序经真实照片反向校准（左右眼交换的映射，用于人脸对齐/评分——**为排名正确而保留**）。
2. **人脸专评**（有脸才跑）：`ai/engine.rs::face_scores` → InsightFace 检测（批量会话或副本池）→ `ai/topiq_face.rs` `TOPIQ-NR-Face` 对齐人脸 crop（512×512，`align_face` 相似变换，动态 batch 分批 8）→ ⊕ nr-on-face（TOPIQ-NR 对同一 crop 打技术分）**50/50 融合**（修 nr_face 暗光盲区）。输出 1~10 人脸专评分。
3. **闭眼检测**（有脸才跑）：`ai/eye.rs` **双信号**——MediaPipe 脸网格（`face_landmarker.onnx`，478 点含虹膜）垂目开度为主信号，OCEC（`ocec_l.onnx`）仅在"网格模棱两可且双眼强判闭"时眨眼否决；网格缺失回退仅 OCEC。聚合为 `max(open_l, open_r)`（至少一眼开，双眼都判闭才降权）。
4. **场景分类**（`ai/scene.rs` MobileNetV3-Large，ImageNet 1000 类 → 人像/宠物/风景/其他）：**人像不靠本分类器**（ImageNet 无 person 类），由人脸检测覆盖（有脸 → 人像）；本分类器只产出无脸图的宠物/风景。与人脸专评**并发执行**（M4，不同 session）。
5. **技术分** TOPIQ-NR（ResNet50，KonIQ-10k，主用，动态 batch）→ NIMA（二级后备），映射 1~10。
6. **美学分** TOPIQ-IAA（ResNet50，AVA，主用，动态 batch），映射 1~10；**非人像**再 ⊕ HyperIQA 50/50 融合（hyperiqa 对风景/宠物/食物降级敏感度更强；**必须逐张推理**，批量输入会静默出错且内存膨胀）。
6b. **对焦分**（`ai/focus.rs` 拉普拉斯方差）：人像=眼部对焦（眼 ROI 锐度），非人像=整图对焦。
7. **综合分** `ai/engine.rs::composite_scores` 融合：

   - 权重分场景档（对焦替代"技术"成为主维度）：
     - **人像**（有人脸且人脸分有效）：人脸 0.55 + 眼部对焦 0.30 + 启发式 0.15（人脸主导，整图美学不参与）。
     - **风景**：美学 0.40 + 对焦 0.50 + 启发式 0.10。
     - **宠物**：美学 0.45 + 对焦 0.45 + 启发式 0.10。
     - **其他**：美学 0.25 + 对焦 0.60 + 启发式 0.15。
   - **闭眼降权**（阶段三）：`max(open) >= 0.5`（至少一眼开）不罚；`< 0.5`（双眼都闭）平滑降到 `0.5`（全闭取 0.5，近阈值几乎不罚）——替代旧的"任一眼闭即 ×0.5"硬切换。
   - 启发式：分辨率/文件大小越高分越高。

### 2.5 每组推荐"最好的一张"
`quality/recommender.rs::build_groups`（输入为 `AiScoreBundle` 评分结果束）：
- 组内综合分最高者标记为**推荐保留**；平局按 tie-break（像素数 + 文件大小）取更优。
- **RAW 优先**：组内同时有 RAW 与其导出 JPG 时，RAW 分差在 0.5 容差内（`RAW_PREFER_TOLERANCE`）即改推 RAW（无损母版，可重新导出）；分差过大仍尊重评分。理由文案显式说明。
- 生成**推荐/删除理由**（"综合评分最高（…）" / "分辨率较低（… < 保留项 …）"等）。
- 组内排序：推荐图恒排第 1，其余按综合分降序。
- 计算可释放空间（非推荐图大小之和）。

## 3. 前端界面（`src/`）

| 组件 | 功能 |
|------|------|
| `App.tsx` | 主流程：扫描进度（阶段：扫描/哈希/聚类/评分/完成）、结果展示 |
| `Toolbar.tsx` | 文件夹选择、启动扫描、AI 开关、临时文件夹（显示磁盘占用）、设置入口 |
| `ProgressBar.tsx` | 扫描/删除进度（分阶段） |
| `GroupCard.tsx` | 每组卡片：相似度、图片数、评分徽标（人脸/闭眼/宠物/风景/综合/美学/技术）、推荐理由、手动选择（红框标记）、右键菜单（删除当前/删除其他）、批量删除 |
| `ImageThumbnail.tsx` | 缩略图懒加载（IntersectionObserver + 共享实例） |
| `PreviewModal.tsx` | 双击全屏预览（三栏：主图 + 底部缩略图导航 + 右侧评分/理由信息栏）；滚轮直接缩放（以鼠标为锚点）、拖拽平移、左右切换 |
| `SettingsPanel.tsx` | 相似度阈值、AI 开关、删除方式、增量扫描、MCP 开关、缓存清理 |
| `TrashBinModal.tsx` | 临时回收站（隔离区）：列出/恢复/清空/在资源管理器打开 |
| `DeleteConfirmModal.tsx` | 批量删除确认 |
| `StatsBar.tsx` | 底部统计：总数/组数/可释放空间 |
| `ScoreHelpModal.tsx` | 评分标准说明弹窗 |

**评分徽标**（`GroupCard`）：推荐 / 人像（有人脸）/ 闭眼 / 宠物 / 风景 / 人脸分 / 综合分 / 美学分 / 技术分 / 删除。

> 注：曾在缩略图上叠加"人脸框 + 关键点红点"的可视化 **不在此功能范围**（经确认取消，不纳入）。

## 4. 数据与删除

- **删除**：`fileops/trash.rs`，默认移入**临时文件夹（隔离区）**（可恢复），可永久删除；异步执行，进度事件推送。删除记录日志。
- **临时文件夹（隔离区）**：删除的图片移入程序根目录 `quarantine/`，可列出/恢复/清空。
- **数据目录**：**程序根目录**（exe 同级：缓存 JSON `pixsweep-cache.json`、日志 `pixsweep.log`、临时文件夹）。

## 5. 设置项（`AppSettings`）

| 字段 | 含义 | 默认 |
|------|------|------|
| `similarity_threshold` | dhash 相似度阈值 | 0.92 |
| `ai_enabled` | 是否启用 AI 推理 | true |
| `permanent_delete` | 是否永久删除（否则移入临时文件夹） | false |
| `incremental` | 是否增量扫描 | true |
| `mcp_enabled` | MCP server 开关 | false |

## 6. MCP server（`mcp.rs`）

HTTP JSON-RPC（127.0.0.1:18765），供外部 AI Agent 操作应用（启动扫描/取结果等）。设置面板可启停（热切换），启动时也可用 `--mcp` 参数开启。

## 7. 系统信息（`commands.rs::get_system_info`）

返回 GPU 可用性（真实推理后端：CUDA / DirectML / CPU）、GPU 名称、TOPIQ-NR 模型是否可用、数据目录。触发引擎初始化以保证与扫描进度条显示一致。

## 8. 推理后端与模型

- **三级回退**：CUDA（NVIDIA，驱动级检测 `nvcuda.dll` 设备数，EP DLL 能加载 ≠ 有 N 卡）→ DirectML（NVIDIA/AMD/Intel 通用）→ CPU。CUDA 会话用 `SameAsRequested` arena 策略防显存翻倍预留。
- **会话一致性**：所有评分模型 `with_parallel_execution(false)` + `with_memory_pattern(false)` + `with_intra_threads(1)`，保证评分确定性（消除浮点求和顺序带来的抖动）。
- **batch 维度**：TOPIQ-NR/IAA/NR-Face 为**动态 batch**（整批推理，Rust 侧分批 8/16）；OCEC/scene/MobileNet/HyperIQA 为 **fix batch=1**，必须逐张推理（批量输入静默失败）。SCRFD 批量模型存在即启用，默认不随包发布（走 2 副本会话池并行）。
- **模型缺失不报错**：仅 `log::warn` 跳过对应能力，对应维度评分退化/回退。
- **模型清单**（`src-tauri/models/`，FP16 约 310MB + insightface/scene/eye 子目录，不入 git，随发布 zip 分发）：
  - 通用：`topiq_nr.onnx`、`topiq_iaa_res50.onnx`、`nima-technical.onnx`、`hyperiqa.onnx`、`topiq_nr_face.onnx(+.data)`（CLIP/LAION 已移除）
  - `insightface/`：`det_10g.onnx`（`2d106det`/`genderage` 未使用，不打包）
  - `scene/`：`mobilenet_v3_large.onnx(+.data)`、`labels.txt`
  - `eye/`：`ocec_l.onnx`、`face_landmarker.onnx`（可选垂目信号，缺则仅 OCEC）

## 9. 构建与打包

- **无 MSVC，用 Zig 工具链**（`.tools/`：xwin SDK + zig 0.16 + zigwrap）。Rust 编译必须**非沙箱**，并设置 `CC/CXX/AR/RC` + `ZIG_GLOBAL/LOCAL_CACHE_DIR` + `RUSTUP_TOOLCHAIN`。
- **构建顺序铁律**：`npm run build` **必须先于** `cargo build --release`（Tauri 嵌入 `dist/`，顺序颠倒会用旧前端）。
- `scripts/build_release.ps1` 打包为 zip（exe + DirectML/onnxruntime DLL + `$neededModels` 列出的模型），用 7-Zip 最大压缩。
  - `$subModels` 显式列出 `insightface/det_10g`、`scene/`（mobilenet + labels）、`eye/ocec_l` 子目录模型。
- 诊断 examples：`verify_ai` / `verify_face` / `verify_eye` / `verify_scene` / `verify_orient` / `verify_bbox` / `verify_full` / `verify_landmarks`。

## 10. 关键约定 / 易错点

- **EXIF 方向**：必须用各格式 decoder 的 `.orientation()`（`image_io::load_image_oriented`），不能用 `Orientation::from_exif_chunk(整个文件头)`（JPEG 以 `FF D8` 开头，无 TIFF magic）。
- **InsightFace det_10g 两个坑**：① 每个 grid cell 有 **2 个 anchor**，N = `h_grid × w_grid × 2`（不是单 anchor）；② 5 关键点实际顺序与直觉/文档相反，libr 库内的映射经真实照片校准，**不要随意改**。
- **改动 `composite_scores` / `AiScoreBundle` / `build_groups`** 必须同步所有调用点与测试（`commands.rs`、`verify_ai.rs`、`verify_full.rs`、测试、`db/store.rs` 缓存字段）。
- **故障排查**：日志在程序根目录 `pixsweep.log`；example 未初始化 logger 会吞 `warn`，排查先拿返回值。

## 11. 近期已落地改进（含在当前版本）

- **阶段一：增量缓存人脸/场景/闭眼**——重扫不再重跑人脸/场景/闭眼（`AiFaceCache` + `schema_version`）；修复 `score_groups_with_ai` 中 `s.get(i)`→`s.get(idx)` 索引错位（含未入组唯一图时会写错推荐分）。
- **阶段二：闭眼连续降权**——`composite_scores` 改为接收 `eye_open` 连续概率，分段降权（spec 字面 `0.5+0.5*open` 会误伤睁眼，故改分段）。
- **阶段三：eye ROI 采样重锚定（方案一）**——`sample_eye_rgb_internal` 直接以检测到的眼关键点为中心、沿眼线方向采 24×40 ROI，替代"5 点模板投影"；删除 `crop_eye_window`。救回"模板投影采偏"的眼（实焦-正脸 R 0→0.93、宋宇芳 L 0→1.00）。
- **阶段三：闭眼聚合改 `max`**——`eye_open = max(open_l, open_r)`，只有双眼都判闭才降权。实测单眼 ROI 采到皮肤/眼镜致 0.00 时（戴镜/偏脸），旧 `min` 会把整张睁眼花压成 ×0.5；改 `max` 后不再压垮。`AI_FACE_CACHE_SCHEMA`=3。
- **阶段三：闭眼聚合实测**——竖屏-实焦-正脸综合 3.40→**6.79**、宋宇芳 3.41→**6.81**（睁眼不再被压）；竖屏-虚焦-侧脸保持 ~3.3（降权）；实焦>虚焦排序保留。
- **阶段二：关键点可信度校验**——`landmarks_trustworthy`，仅拦明显退化人脸。
- **阶段二：权重集中管理**——`W_FACE_*` / `W_LAND_*` / `W_PET_*` 提为模块级常量。
- **阶段二：`is_closed(open)` 统一闭眼判定**——消除 `<=0.5`/`<0.5` 边界矛盾；删除 `eye.rs` 旧硬阈值死代码。
- **0.8.1 相机 RAW 兼容**——23 种 RAW（rawler 解码，机内嵌预览优先）；RAW 分辨率按传感器源口径参与启发式（`raw_source_dimensions` dummy 探针）。
- **0.8.2/0.8.3 稳定性与体验**——闭眼检测改双信号（脸网格垂目 + OCEC 眨眼否决，标注集 7/7）；RAW 预览全显影落盘缓存；预览滚轮直接缩放并以鼠标为锚点；扫描后内存释放（解码信号量 + 重活池 + EmptyWorkingSet）。
- **0.8.2 统一前置代理**——>2K / >2MB / RAW 一律生成 <2K 且 <2MB 的 JPEG 代理（临时文件夹 `quarantine/proxy/`），AI 链路全走代理；文件大小/分辨率对比保持源口径（结构性保证）；临时文件夹按钮显示磁盘占用；缓存清理面板（移入系统回收站）。
- **0.8.4 GPU 优化与硬件画像**——SCRFD 会话副本池（副本数按显存/内存分档）、nr_face FP16 动态 batch 重导出、场景∥人脸并发、TOPIQ 双缓冲流水线（AI 链 143s→40s）；硬件探测（核心/内存/显存）动态定重活线程数与副本数，启动打日志。

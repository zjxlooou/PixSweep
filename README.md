# PixSweep

智能图片去重桌面应用 — 扫描文件夹、找出相似图片、推荐每组最佳的一张，一键清理释放空间。

> 基于 Tauri 2 + Rust + React + ONNX Runtime (DirectML)，Windows 11 直接可用。

---

## 📦 快速开始

1. [Releases](../../releases) 下载 `PixSweep-v*.zip`
2. 解压到任意目录
3. 双击 `PixSweep.exe`

**离线运行**，所有依赖（ONNX Runtime + AI 模型 + DLL）已打包。

---

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 多格式扫描 | jpg / png / webp / bmp / gif / tiff / heic + 相机 RAW 23 种（RW2 / NEF / ARW / CR2 / CR3 / RAF / ORF / DNG 等，rawler 解码） |
| 双哈希聚类 | dhash + ahash，过滤渐变图误判 |
| 人像优先评分 | 人脸专评 / 闭眼（垂目+眨眼双信号）/ 眼部对焦 / 场景分级（TOPIQ + InsightFace + MediaPipe 脸网格 + OCEC，本地推理） |
| 对焦检测 | 拉普拉斯方差清晰度指标，标出失焦照片 |
| GPU 加速 | CUDA → DirectML → CPU 三级回退（NVIDIA / AMD / Intel 全支持） |
| 增量扫描 | 文件指纹缓存，二次扫描秒级完成 |
| 临时文件夹 | 删除可恢复，不污染系统回收站 |
| 缓存清理 | 代理图 / 缩略图 / 日志等按类型清理（移入系统回收站） |
| 全键盘操作 | 方向键 / Enter / Y / N / Esc |
| CSV 导出 | 扫描结果一键导出 |

---

## ⌨️ 键盘快捷键

| 场景 | 按键 | 动作 |
|------|------|------|
| 预览 | `↑` / `↓` | 上一组 / 下一组 |
| 预览 | `←` / `→` | 组内上一张 / 下一张 |
| 预览 | `Enter` | 删除本组非推荐照片 |
| 预览 | `Esc` | 退出预览 |
| 删除确认 | `Y` / `Enter` | 确认 |
| 删除确认 | `N` / `Esc` | 取消 |
| 任意弹窗 | `Esc` | 关闭 |

---

## 🛠️ 开发者

### 环境

| 项目 | 要求 |
|------|------|
| OS | Windows 10 / 11 (x86_64) |
| Rust | stable 1.75+ |
| Node.js | 18+ |
| GPU | 任意支持 DirectX 12 的 GPU |

> 无 MSVC 工具链也可编译：项目自带 `.tools/`（xwin SDK + zig）。

### 从源码构建

```bash
# 1. 克隆
git clone https://github.com/yourname/PixSweep.git
cd PixSweep
npm install

# 2. 准备 AI 模型（可选，无模型则仅哈希去重）
#    模型为 ONNX 格式，需自行下载/导出到 src-tauri/models/，清单见下方「打包说明」。
#    也可直接使用 dist-package/ 下已打包好的发布包（含全部模型）。

# 3. 构建
npm run build                              # 先构建前端
cd src-tauri && cargo build --release      # 再编译后端（嵌入前端）

# 4. 运行
./src-tauri/target/release/pixsweep.exe
```

### 打包发布

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1
# 输出：dist-package/PixSweep-v*.zip（7-Zip mx=9 最大压缩，~218 MB）
```

打包说明：脚本内置 7-Zip（`.tools/7zip/7za.exe`）`-tzip -mx=9` 最大压缩；仅打包代码引用的模型，未引用的自动跳过。详见 `docs/TESTING.md`。

### 测试

```bash
# 前端组件（vitest 4.x + jsdom，需在 pixsweep 子目录运行）
cd PixSweep && npx vitest run           # 25 个测试

# 后端单元测试（cargo test）
cd src-tauri && cargo test              # 37 个测试

# 端到端测试（Git Bash，含扫描/设置/删除/回收站/恢复/清空/导出）
cd PixSweep && bash scripts/test_e2e.sh # 8 阶段全 PASS
```

完整测试指南见 [`docs/TESTING.md`](./TESTING.md)。

压缩说明：优先使用项目自带 7-Zip（`.tools/7zip/7za.exe`）`-tzip -mx=9` 最大压缩；仅打包代码引用的模型（FP16 精度，共 6 个通用评分文件：`topiq_nr.onnx`（技术主评分）/ `topiq_iaa_res50.onnx`（美学主评分）/ `nima-technical.onnx`（二级后备）/ `hyperiqa.onnx`（非人像美学融合）/ `topiq_nr_face.onnx` + `topiq_nr_face.onnx.data`（人脸专评，配对文件），另有 `scene/`、`eye/`、`insightface/` 子目录模型），未引用的模型自动跳过。

---

## 📁 项目结构

```
PixSweep/
├── src/                     # 前端（React + TypeScript + Vitest）
│   ├── App.tsx
│   ├── api.ts               # Tauri IPC 封装（含 MCP 调用）
│   ├── components/          # UI 组件
│   │   ├── GroupCard.tsx
│   │   ├── PreviewModal.tsx
│   │   ├── SettingsPanel.tsx
│   │   ├── TrashBinModal.tsx
│   │   └── __tests__/       # 组件测试（vitest）
│   ├── hooks/               # useScan / useDelete
│   ├── types/               # 共享类型 + __tests__
│   └── test/setup.ts        # vitest 全局 mock（jsdom + Tauri API）
├── src-tauri/               # 后端（Rust + Tauri 2）
│   ├── src/
│   │   ├── lib.rs           # Tauri Builder + MCP 启动钩子
│   │   ├── main.rs          # 入口（隐藏控制台窗口）
│   │   ├── commands.rs      # IPC 命令层
│   │   ├── mcp.rs           # MCP HTTP JSON-RPC server（127.0.0.1:18765）
│   │   ├── state.rs         # 共享状态（含 MCP runtime 句柄）
│   │   ├── types.rs         # AppSettings + 类型
│   │   ├── scanner/         # 文件遍历（walker.rs）
│   │   ├── hashing/         # dhash + ahash（phash.rs）
│   │   ├── cluster/         # 相似度聚类（UnionFind）
│   │   ├── ai/              # ONNX 推理（TOPIQ + NIMA + InsightFace + 场景/闭眼/脸网格）
│   │   ├── quality/         # 质量推荐（recommender.rs）
│   │   ├── db/              # JSON 缓存（store.rs）
│   │   ├── fileops/         # 临时回收站（trash.rs）
│   │   └── cache/           # 缩略图缓存
│   ├── models/              # AI 模型（gitignore；topiq/nima/insightface/scene/eye）
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── capabilities/        # Tauri 2 ACL 权限
├── test_assets/             # 端到端测试样本（14 张 / 0.59 MB / 6 格式）
│   ├── _generate.ps1        # 样本生成脚本（PowerShell + System.Drawing）
│   └── *.png/.jpg/.bmp/.gif/.tif
├── scripts/
│   ├── build_release.ps1    # 打包脚本（7-Zip mx=9 + 模型白名单）
│   ├── test_e2e.sh          # 端到端测试（Git Bash，8 阶段）
│   └── generate_icons.py    # Tauri 图标生成
├── .tools/                  # 本地工具链（gitignore）
│   ├── zigwrap/             # zig 编译桥（无 MSVC 替代）
│   ├── zig-cache/
│   └── 7zip/                # 7-Zip 独立命令行（mx=9 压缩）
├── dist-package/            # 打包产物（gitignore）
├── docs/
│   ├── DESIGN.md            # 详细设计文档
│   └── TESTING.md           # 测试指南（8 层测试）
├── package.json
├── vite.config.ts
└── vitest.config.ts
```

---

## 🤖 MCP 自动化测试接口

PixSweep 内置 **MCP (Model Context Protocol) server**，让 AI Agent 可远程操作完整功能。

### 启动

```bash
# 命令行强制启动
PixSweep.exe --mcp

# 或设置面板：勾选 MCP 服务（默认关闭）
```

### 连接

```
URL: http://127.0.0.1:18765/mcp
协议: JSON-RPC 2.0（POST）
```

### 14 个工具

| 工具 | 用途 |
|------|------|
| `get_system_info` | GPU / 模型 / 数据目录 |
| `get_settings` / `set_settings` | 读写应用设置 |
| `start_scan` | 同步扫描，返回完整结果 |
| `get_scan_result` | 获取最近扫描结果 |
| `delete_files` | 删除文件（临时文件夹 / 永久） |
| `list_trash` / `restore_trash_item` / `restore_all_trash` / `clear_trash` | 临时文件夹 |
| `clear_cache` | 清空扫描缓存 |
| `get_cache_summary` / `cleanup_cache` | 缓存清理（查询摘要 / 移入系统回收站） |
| `export_report` | 导出 CSV |

### 调用示例

```bash
curl -X POST http://127.0.0.1:18765/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

curl -X POST http://127.0.0.1:18765/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"start_scan","arguments":{"folders":["E:/photos"]}}}'
```

> Agent 通过 MCP 的操作与 GUI **共用同一份状态**——MCP 触发的扫描/删除，界面实时同步。

---

## 🧪 测试

详细测试指南：[TESTING.md](./TESTING.md)。

```bash
# 快速回归
cd src-tauri && cargo test
npx vitest run
npm run build

# 完整端到端（Git Bash，覆盖扫描/分组/设置/删除/回收站/恢复/清空/导出）
bash scripts/test_e2e.sh
```

---

## 📐 架构

```
┌──────────── 前端 (React) ────────────┐
│  Toolbar / GroupCard / PreviewModal    │
│  DeleteConfirm / TrashBin / Settings   │
└────────────────┬───────────────────────┘
                 │ Tauri IPC (invoke + events)
┌────────────────▼───────────────────────┐
│            后端 (Rust)                  │
│  commands.rs ← IPC 入口                 │
│      ├→ scanner/walker   遍历文件夹      │
│      ├→ hashing/phash    计算 dhash+ahash │
│      ├→ cluster/         聚类相似图片      │
│      ├→ ai/engine        ONNX 推理（GPU）  │
│      ├→ quality/         推荐最佳图片      │
│      ├→ db/store         JSON 缓存        │
│      ├→ fileops/trash    临时回收站        │
│      └→ mcp/             HTTP JSON-RPC    │
└────────────────────────────────────────┘
```

### 关键技术决策

| 决策 | 原因 |
|------|------|
| DirectML（而非 CUDA） | 全系 GPU 支持，无需 NVIDIA 工具链 |
| JSON 缓存（而非 SQLite） | 纯 Rust 无 C 编译依赖 |
| 应用内置隔离区 | 元数据完整可控，支持跨盘恢复 |
| 滑动窗口渲染 | 固定 120 组 DOM，防内存爆炸 |
| xwin + zig 工具链 | 无 Visual Studio 也可编译 |

---

## 📄 许可证

MIT License

## 🙏 致谢

- [Tauri](https://tauri.app/) — 桌面应用框架
- [ONNX Runtime](https://onnxruntime.ai/) — ML 推理引擎
- [InsightFace](https://github.com/deepinsight/insightface) — 人脸检测
- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe) — 人脸网格（闭眼检测）
- [TOPIQ](https://arxiv.org/abs/2308.03060) — 技术 + 美学质量评估（主评分模型）
- [NIMA](https://github.com/idealo/image-quality-assessment) — 质量评估（后备）
- [image](https://github.com/image-rs/image) — Rust 图像库

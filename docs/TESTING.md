# PixSweep 测试指南

本文档指导测试 PixSweep：Rust 单测 → 前端组件 → 端到端集成 → 打包。

**每次修改代码后，按"快速回归"或"完整回归"执行对应测试。**

---

## 📋 测试层级

| # | 层级 | 命令 | 耗时 |
|---|------|------|------|
| 1 | Rust 单元测试 | `cd src-tauri && cargo test` | ~30s |
| 2 | 前端组件测试 | `npx vitest run`（jsdom，无需 GUI） | ~5s |
| 3 | TS 类型检查 | `npx tsc --noEmit` | ~3s |
| 4 | 前端构建 | `npm run build` | ~2s |
| 5 | 后端编译 | `cd src-tauri && cargo build --release` | ~2min |
| 6 | 应用启动 | 运行 exe，检查日志无 ERROR | ~5s |
| 7 | 端到端测试 | `bash scripts/test_e2e.sh`（Git Bash） | ~30s |
| 8 | 打包 | `pwsh -File scripts/build_release.ps1` | ~3min |

- **快速回归**：1 → 2 → 3 → 4
- **完整回归**（上传前 / 改核心逻辑）：1 → 7 全部

---

## 1. Rust 单元测试

```bash
cd src-tauri && cargo test
```

覆盖：感知哈希（hashing）、双哈希聚类（cluster）、TOPIQ/NIMA 评分（ai/topiq、ai/nima）、推荐（quality）。

**预期**：`test result: ok. 37 passed; 0 failed`

## 2. 前端组件测试

```bash
# 必须在项目子目录运行（vitest 4.x 在外层 workspace 根找不到 jsdom）
cd PixSweep && npx vitest run
```

覆盖 DeleteConfirmModal（键盘）、GroupCard（渲染）、formatBytes（纯函数）。

**预期**：`Test Files 3 passed / Tests 25 passed`

**注意**（vitest 4.x）：组件测试文件首行加 `// @vitest-environment jsdom`（vitest 4 已弃用 `environmentMatchGlobs`）。

## 3. TS 类型检查

```bash
npx tsc --noEmit
```

退出码 0，无输出。

## 4. 后端编译

```bash
cd src-tauri && cargo build --release
```

**预期**：`Finished release profile in ~2min`

⚠️ Tauri 编译时嵌入 `dist/` —— **前端构建必须在后端编译之前**。

## 5. 应用启动

```bash
./src-tauri/target/release/pixsweep.exe --mcp &
sleep 4
tail -5 "$(dirname "$(readlink -f ./src-tauri/target/release/pixsweep.exe 2>/dev/null || echo ./src-tauri/target/release/pixsweep.exe)")/pixsweep.log"
```

**预期日志**：`[MCP] 服务已启动: http://127.0.0.1:18765/mcp`

## 6. 端到端测试（推荐）

```bash
# 在 Git Bash / WSL 运行
bash scripts/test_e2e.sh
```

脚本自动完成：

1. 复制 `test_assets/` 样本到 `$TEMP/pixsweep_e2e_test/scan_input/`
2. 启动 PixSweep + 等待 MCP ready（并清空回收站保证干净起点）
3. 通过 MCP 调用 `start_scan` 扫描
4. 验证分组结果（≥3 组 + sunset 跨格式识别）
5. **设置功能测试**（`get_settings` / `set_settings` 读写回环）：
   - 设置字段完整性（5 个字段齐全）
   - 相似度阈值调整（0.92 → 0.8 → 恢复 0.92）
   - AI 质量评分开关（关闭 → 开启）
   - 增量扫描开关（关闭 → 开启）
   - 增量扫描行为验证（开启后命中缓存二次扫描仍返回完整 14 张）
   - 恢复默认设置（防止污染后续阶段）
6. **功能链路测试**：
   - `delete_files` 单文件删除（验证移入回收站、原文件消失）
   - `list_trash` 回收站计数
   - `restore_trash_item` 单文件恢复（验证回到原路径）
   - `delete_files` 批量删除 3 个文件
   - `restore_all_trash` 全部恢复（验证 3 个都回来）
   - `clear_trash` 清空回收站（验证永久删除、回收站为空）
7. 导出 CSV 报告
8. 清理临时目录和 PixSweep 进程（`taskkill /F /IM`，无残留）

**预期输出**：

```
====== TEST PASSED ======
```

### 重生成测试样本

如果修改了样本设计：

```bash
pwsh -File test_assets/_generate.ps1
```

---

## 🚨 常见陷阱

### 1. 前端未嵌入 exe

**症状**：改前端后运行 exe 看不到变化。

**原因**：Tauri 编译时嵌入 `dist/`。

**修复**：
```bash
npm run build                              # 先
cd src-tauri && cargo build --release      # 后
```

### 2. 增量编译"拒绝访问"

**症状**：增量编译报"拒绝访问"。

**修复**：删除 `target/` 后全量重建。（历史注：旧 agent 沙箱环境会污染 target 的 ACL 导致此症状；现环境无沙箱，若重现多为文件占用或权限问题。）

### 3. 路径格式

**症状**：Windows 程序报"路径不存在"。

**修复**：传给 Windows 原生程序用 `盘符:/目录/...` 形式（如 `D:/proj`），**不要** `/d/...`（会被误解析成 `D:\d\...`）。

### 4. PowerShell 5.1 启动 GUI

**症状**：从 `.ps1 -File` 启动 PixSweep 进程立即退出。

**原因**：PS 5.1 启动 GUI 子进程在脚本结束后回收。

**修复**：用项目提供的 `test_e2e.sh`（Git Bash 启动），或先在交互 PS 里启动 PixSweep。

### 5. Git Bash 中 taskkill 参数

**症状**：`taskkill //F //IM` 报 `无效参数/选项 - '//F'`。

**原因**：Git Bash 把 `//F` 当作路径转换，不会传给 Windows 的 taskkill。

**修复**：用单斜杠 `taskkill /F /IM PixSweep.exe`（Git Bash 中可直接用）。

### 7. 无 MSVC 环境编译

**症状**：首次 cargo build 报 `linker 'link.exe' not found` / `vswhom-sys` 编译失败。

**修复**：项目 `.tools/zigwrap/` 已提供 zig 编译器替代 MSVC，需在 `cargo build` 前设置环境变量：

```bash
ROOT='<本仓库根目录，Windows 风格>'   # 本机真实值见 PRIVATE.local.md（不入库）
export CC="$ROOT/.tools/zigwrap/zigcc.exe"
export CXX="$ROOT/.tools/zigwrap/zigcxx.exe"
export AR="$ROOT/.tools/zigwrap/ziglib.exe"
export RC="$ROOT/.tools/zigwrap/zigrc.exe"
export ZIG_GLOBAL_CACHE_DIR="$ROOT/.tools/zig-cache"
export ZIG_LOCAL_CACHE_DIR="$ROOT/.tools/zig-cache/local"
export PATH="$HOME/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin:$PATH"
```

### 8. 安全软件清理无签名 exe

**症状**：刚编译的 `pixsweep.exe` 启动失败（os error 5），几秒后 exe 文件被删除；含 exe 的 zip 也会被清理。

**根因**：公司 EDR / 终端管控策略按数字签名清理无签名可执行文件。

**修复**：将仓库整个目录（本机路径见 PRIVATE.local.md）加入安全软件排除列表。**确认排除后再运行 build_release.ps1**。

### 6. MCP 调用偶发空响应

**症状**：`mcp_call` 返回空/`INVALID_JSON`，重试后恢复；同一序列中相邻调用一个成功一个失败。

**原因**：Windows 快速连续 TCP 连接 + 手写 HTTP server（每连接一线程，响应头 `Connection: close`）偶发连接被拒；另注意 server 响应中 JSON 文本带 `\` 转义，**不要**用 `grep '"incremental": false'` 直接匹配外层响应（匹配不到），应经 `mcp_text` 提取后再解析。

**修复**（已内置 `test_e2e.sh`）：
- `mcp_call` 使用 `-H "Connection: close"` + `--retry-connrefused --retry 2` + 响应检查重试 3 次
- 断言基于**最终状态**（文件存在 / `list_trash` 计数）而非响应文本
- 浮点断言用值比较（`0.8`，勿写 `0.80`——f32 序列化无尾零）

---

## ⚠️ Agent 改代码后必读

**AI Agent 修改代码后，必须按"完整回归"流程执行 1-7，所有测试通过后才能标记任务完成。**

报告格式：

```
✅ Rust 单元测试: 37 passed
✅ 前端组件测试: 25 passed
✅ TS 类型检查: 通过
✅ 前端构建: 成功 (index-XXXX.js)
✅ 后端编译: 成功
✅ 应用启动: 正常
✅ 端到端测试: PASSED
```

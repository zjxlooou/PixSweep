# 脚本目录说明

PixSweep 的所有构建、打包、测试、模型导出脚本集中在此目录。

> 运行 PowerShell 脚本：`powershell -ExecutionPolicy Bypass -File scripts/<脚本>.ps1`
> 运行 Python 脚本：`python scripts/<脚本>.py`
> 运行 Shell 脚本：`bash scripts/<脚本>.sh`

## 构建

| 脚本 | 作用 | 用法 |
|------|------|------|
| [`build.ps1`](build.ps1) | 构建可执行文件（前端 build + 后端 cargo build），不打包 | `powershell -ExecutionPolicy Bypass -File scripts/build.ps1`（加 `-Debug` 出 debug 版） |
| [`build_release.ps1`](build_release.ps1) | 构建 + 打包为离线可运行 zip（含 exe/DLL/模型） | `powershell -ExecutionPolicy Bypass -File scripts/build_release.ps1` |

两者都会自动检测 `.tools/` 工具链并注入 Zig + xwin 环境（无 MSVC 机器专用）。

## 测试

| 脚本 | 作用 | 用法 |
|------|------|------|
| [`test_e2e.sh`](test_e2e.sh) | 端到端测试：启动 app + MCP → 扫描 → 分组 → 删除/恢复/清空 → 导出 | `bash scripts/test_e2e.sh`（需先 `build.ps1` 出 release exe） |

其余测试直接用命令（见 [`docs/TESTING.md`](../docs/TESTING.md)）：

- 后端单测：`cargo test`（需 Zig 环境，非沙箱）
- 前端组件测试：`npm test`（vitest）
- 类型检查：`npx tsc --noEmit`

## 模型导出 / 资源生成

| 脚本 | 作用 | 用法 |
|------|------|------|
| [`export_onnx.py`](export_onnx.py) | 把 pyiqa / HuggingFace 的 PyTorch 模型导出为 ONNX（含 `.onnx.data` 外部权重） | `python scripts/export_onnx.py topiq_nr` / `topiq_nr_face` / `topiq_iaa`，或 `custom` 传权重 URL |
| [`generate_icons.py`](generate_icons.py) | 生成应用图标（纯标准库，无需 PIL） | `python scripts/generate_icons.py` |

## 真图验证（examples）

验证 Rust 侧 AI 链路，传真实照片目录（只读），见 [`AGENT.md`](../AGENT.md) 的「真图验证」一节：

```bash
cargo run --example verify_ai -- <照片目录> 8   # 全链路评分
cargo run --example verify_face -- <目录>                    # 人脸检测
cargo run --example verify_eye -- <目录>                     # 闭眼检测
cargo run --example verify_scene -- <目录>                   # 场景分类
cargo run --example verify_focus -- <图片...>                # 对焦校准
```

> `src-tauri/examples/` 下还有 `verify_full`（全维度诊断）、`verify_bbox`（人脸框/眼 ROI 可视化）、
> `verify_orient`（EXIF 方向）、`verify_landmarks`（人脸关键点）等诊断脚本。

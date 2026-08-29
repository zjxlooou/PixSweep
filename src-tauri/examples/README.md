# 验证与诊断脚本（examples）

用 `cargo run --example <名字> -- <参数>` 运行（需 AGENT.md 的 zig 工具链环境变量）。
传给脚本的**真实照片目录一律只读**，且必须用 Windows 风格路径（`盘符:/目录`）；
本机私人路径（标注集、测试照片库）见仓库根 `PRIVATE.local.md`（不入库）。

> example 若未初始化 logger 会吞 warn 日志，排查问题先看返回值或加 `env_logger`。
> `verify_ai` 恒设 `has_face=false`、`eye_open=1.0`（纯整图链路），**测不到人脸/闭眼/
> 眼部对焦**——测这些须走完整链路（`verify_labeled`）或全维度诊断（`verify_full`）。

## 功能回归（改相关代码后应跑）

| 脚本 | 验证内容 | 用法 |
|------|----------|------|
| `verify_ai` | 全链路评分：模型加载 + 三级回退后端 + 同图两次打分一致性 | `-- <照片目录> 8` |
| `verify_face` | InsightFace 人脸检测 + TOPIQ-NR-Face 人脸专评 | `-- <目录> [最大张数]` |
| `verify_eye` | 闭眼检测（OCEC + 脸网格） | `-- <目录> [最大张数]` |
| `verify_scene` | MobileNetV3 场景分类（不含人脸覆盖） | `-- <目录> [最大张数]` |
| `verify_focus` | 对焦指标（拉普拉斯方差 + 1~10 分），校准 `focus.rs` 阈值 | `-- <图片...>` |
| `verify_labeled` | **闭眼标注集回归基准**（「组N-」命名目录，当前 7/7），调闭眼/眼对焦参数后必须重跑 | `-- <标注集目录>` |
| `verify_orient` | EXIF Orientation 处理（竖拍旋转） | `-- <图片路径>` |
| `verify_full` | 全维度逐张诊断（技术/美学/人脸/场景/闭眼/对焦/综合） | `-- <图片...>` |
| `proxy_check` | 统一前置代理：触发判定 + <2K 且 <2MB 双断言 + 缓存命中计时 | `-- <图片路径>...` |
| `raw_dims_check` | RAW 源口径分辨率探针（传感器原生 vs 嵌入预览） | `-- <RAW 路径>...` |
| `preview_check` | RAW 全显影预览（分辨率 + EXIF + 耗时） | `-- <RAW 路径>...` |

## 人工诊断（输出图片到临时目录目视核对）

| 脚本 | 用途 |
|------|------|
| `verify_bbox` | 人脸框/关键点/眼 ROI 可视化（检测误判排查；产物在系统临时目录） |
| `verify_landmarks` | 最大脸 bbox + 5 关键点坐标 sanity 校验 |

## 一次性研究探针（RAW 解码调研期产物，保留备查）

| 脚本 | 用途 |
|------|------|
| `probe_raw_decode` | 多品牌 RAW 解码探针（嵌入预览可用性 + 全显影耗时） |
| `raw_funnel_check` | 生产解码漏斗 `load_image_oriented` 的 RAW 分支验证 |
| `scan_check` | 生产扫描入口 `scan_folder` 对 RAW 的收录验证 |

## 历史约定

- **一次性脚本用完即删，勿提交**（本目录现存脚本均长期保留）。
- 闭眼标注集回归结论与调参锚点见 `AGENT.md`「闭眼检测」一节。

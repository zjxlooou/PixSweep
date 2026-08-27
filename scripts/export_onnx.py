#!/usr/bin/env python3
"""
PixSweep ONNX 模型导出工具

## 用途
把 pyiqa / HuggingFace 上的 PyTorch 模型一键导出成 ONNX + 验证误差 <1e-5。
大幅降低模型升级成本（之前每次升级要重建 2GB Python 环境手动导）。

## 用法
    # 导出已实现的模型（默认流程）
    python scripts/export_onnx.py topiq_nr        # 导出 TOPIQ-NR（技术质量）
    python scripts/export_onnx.py topiq_nr_face   # 导出 TOPIQ-NR-Face（人脸专评）
    python scripts/export_onnx.py topiq_iaa       # 导出 TOPIQ-IAA（美学）

    # 自定义模型（提供权重 URL + pyiqa 注册名）
    python scripts/export_onnx.py custom \\
        --name my_model \\
        --weight-url https://huggingface.co/.../model.pth \\
        --input-shape 1,3,384,384 \\
        --input-name image \\
        --output-name quality

## 输出
- src-tauri/models/{name}.onnx（图结构）
- src-tauri/models/{name}.onnx.data（外部权重，配对文件，缺一不可）
- 自动用 onnxruntime CUDA EP + pyiqa PyTorch CUDA 推理对比，输出 |err| 值

## 环境要求
- torch 2.11+（含 CUDA 12.6 wheel）
- pyiqa 0.1.16+
- onnxruntime 1.18+
- onnxscript（torch.onnx.export 需要）
- 已 pip install 到目标目录

## 设计要点
- **所有模型都走固定 batch=1 导出**（避免 DirectML 动态 shape 坑）
- **opset=18**（DirectML EP 1.15.2 上限 20，18 是安全值）
- **dynamo=False**（走 TorchScript 路径，避开 dynamo 偶发 bug）
- **CUDA 验证**：ONNX 用 onnxruntime-gpu + CUDA EP，PyTorch 用 CUDA，误差 <1e-5 才算通过
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# 项目的"标准模型"清单（pyiqa 注册名 → 权重 URL + 导出配置）
# 新模型加这里 + 在 constants 注明来源
STANDARD_MODELS = {
    "topiq_nr": {
        "pyiqa_name": "topiq_nr",
        "weight_url": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_nr_koniq_res50-9a73138b.pth",
        "input_shape": (1, 3, 384, 384),
        "input_name": "image",
        "output_name": "quality",
        "align_crop_face": False,  # 关闭 facexlib，ONNX 可 trace
        "description": "TOPIQ-NR 技术质量（KonIQ-10k 训练）",
    },
    "topiq_nr_face": {
        "pyiqa_name": "topiq_nr-face",
        "weight_url": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/topiq_nr_cgfiqa_res50-0a8b8e4f.pth",
        "input_shape": (1, 3, 512, 512),
        "input_name": "face_crop",
        "output_name": "quality",
        "align_crop_face": False,
        "description": "TOPIQ-NR-Face 人脸专评（CGFIQA-40k 训练）",
    },
    "topiq_iaa": {
        "pyiqa_name": "topiq_iaa",
        "weight_url": "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/cfanet_iaa_ava_res50-0a2c4d2f.pth",
        "input_shape": (1, 3, 384, 384),
        "input_name": "image",
        "output_name": "quality_distribution",
        "align_crop_face": False,
        "description": "TOPIQ-IAA 美学（AVA 训练，10-bin softmax 分布）",
    },
}

DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "src-tauri" / "models"


def detect_pytorch_env() -> dict:
    """检测当前 Python 环境的 torch / pyiqa / onnxruntime 是否可用。"""
    info = {"torch": None, "cuda": False, "pyiqa": None, "onnxruntime": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import pyiqa
        info["pyiqa"] = pyiqa.__version__
    except ImportError:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    return info


def download_weight(url: str, cache_dir: Path) -> Path:
    """下载权重文件到 cache_dir（用 curl，断点续传 + 重试）。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 从 URL 提取文件名
    fname = url.rsplit("/", 1)[-1]
    target = cache_dir / fname
    if target.exists() and target.stat().st_size > 1024:
        print(f"  [cache] 命中 {target.name} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
        return target

    print(f"  [download] {url}")
    subprocess.run(
        ["curl", "-L", "-C", "-", "-k", "--http1.1",
         "-o", str(target), "--retry", "2", "--connect-timeout", "30", "-#", url],
        check=True,
    )
    print(f"  [done] {target.name} ({target.stat().st_size / 1024 / 1024:.1f} MB)")
    return target


def load_pyiqa_model(name: str):
    """用 pyiqa 加载 PyTorch 模型（不下载，假设权重已就位）。"""
    import pyiqa
    model = pyiqa.create_metric(name, device="cpu")
    net = model.net if hasattr(model, "net") else model
    net.eval().cpu()
    return net


def export_to_onnx(net, weight_path: Path, output_path: Path,
                   input_shape: Tuple[int, int, int, int],
                   input_name: str, output_name: str,
                   opset: int = 18) -> Path:
    """用 torch.onnx.export 把 PyTorch 模型导出为 ONNX。"""
    import torch
    # 加载权重
    from pyiqa.archs.arch_util import load_pretrained_network
    load_pretrained_network(net, str(weight_path), weight_keys="params")

    # 关键：部分模型（topiq_nr-face）有 align_crop_face 分支会调 facexlib
    if hasattr(net, "align_crop_face"):
        net.align_crop_face = False
    net.eval()

    dummy = torch.randn(*input_shape)
    print(f"  [export] {input_shape} → {output_path}")
    torch.onnx.export(
        net, (dummy,),
        str(output_path),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes=None,  # 固定 batch=1，避开 DirectML 动态 shape 坑
        do_constant_folding=True,
        dynamo=False,  # 走 TorchScript 路径
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  [export] ONNX 导出成功 ({size_mb:.1f} MB)")
    return output_path


def verify_onnx(net, onnx_path: Path, input_shape: Tuple[int, int, int, int],
                use_cuda: bool = True, tol: float = 1e-5) -> float:
    """用 onnxruntime 加载 + 跑同一张 dummy 图，对比 PyTorch 输出。"""
    import numpy as np
    import onnxruntime as ort
    import torch

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    used_providers = sess.get_providers()
    print(f"  [verify] ORT providers: {used_providers}")

    x_np = np.random.randn(*input_shape).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: x_np})[0]
    onnx_score = onnx_out.flatten()[0]

    x_t = torch.from_numpy(x_np)
    with torch.no_grad():
        if use_cuda and torch.cuda.is_available():
            x_t = x_t.cuda()
            net_cuda = net.cuda() if not next(net.parameters()).is_cuda else net
            py_out = net_cuda(x_t).flatten()[0].cpu().item()
        else:
            py_out = net(x_t).flatten()[0].item()

    err = abs(onnx_score - py_out)
    status = "✅" if err < tol else "❌"
    print(f"  [verify] onnx={onnx_score:.6f}  pytorch={py_out:.6f}  |err|={err:.2e}  {status}")
    if err >= tol:
        raise RuntimeError(f"ONNX 验证失败：|err|={err:.2e} >= {tol}")
    return err


def add_to_gitignore(onnx_name: str, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
    """确保 .onnx 和 .onnx.data 都在 .gitignore 里（不入库）。"""
    gitignore = models_dir.parent.parent / ".gitignore"
    if not gitignore.exists():
        return
    content = gitignore.read_text(encoding="utf-8")
    # 模型文件已统配，加个提醒注释
    if onnx_name not in content and "src-tauri/models/*.onnx" not in content:
        content += f"\n# {onnx_name}（已默认被 src-tauri/models/*.onnx 忽略）\n"
        gitignore.write_text(content, encoding="utf-8")


def export_one(name: str, override: dict = None) -> None:
    """导出单个模型到 src-tauri/models/。"""
    config = dict(STANDARD_MODELS.get(name, {}))
    if override:
        config.update(override)
    if not config:
        print(f"[error] 未知模型 '{name}'。可用：{list(STANDARD_MODELS.keys())}")
        sys.exit(1)

    print(f"\n=== 导出 {name}: {config['description']} ===")
    env = detect_pytorch_env()
    print(f"  [env] torch={env['torch']}  cuda={env['cuda']}  pyiqa={env['pyiqa']}  ort={env['onnxruntime']}")
    if not env["torch"] or not env["pyiqa"]:
        print("  [error] 缺少 torch 或 pyiqa，请先：")
        print("    pip install torch torchvision pyiqa onnxscript onnxruntime-gpu")
        sys.exit(1)

    cache_dir = Path.home() / ".cache" / "pixsweep-models"
    weight_path = download_weight(config["weight_url"], cache_dir)

    out_name = config.get("out_name", name)
    out_path = DEFAULT_MODELS_DIR / f"{out_name}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = load_pyiqa_model(config["pyiqa_name"])
    export_to_onnx(
        net, weight_path, out_path,
        input_shape=config["input_shape"],
        input_name=config["input_name"],
        output_name=config["output_name"],
    )
    err = verify_onnx(net, out_path, config["input_shape"], use_cuda=env["cuda"])
    add_to_gitignore(out_name)

    print(f"\n✅ {name} 导出完成")
    print(f"   ONNX: {out_path}")
    print(f"   配对: {out_path.with_suffix('.onnx.data')}")
    print(f"   误差: {err:.2e}")


def main():
    parser = argparse.ArgumentParser(description="PixSweep ONNX 模型导出工具")
    parser.add_argument("name", help="模型名（标准清单或 'custom'）")
    parser.add_argument("--weight-url", help="custom 模式：权重 URL")
    parser.add_argument("--pyiqa-name", help="custom 模式：pyiqa 注册名")
    parser.add_argument("--input-shape", help="custom 模式：输入 shape，如 1,3,384,384")
    parser.add_argument("--input-name", default="image", help="custom 模式：输入张量名")
    parser.add_argument("--output-name", default="quality", help="custom 模式：输出张量名")
    parser.add_argument("--out-name", help="输出文件名（默认同 name）")
    args = parser.parse_args()

    if args.name == "custom":
        if not (args.weight_url and args.pyiqa_name and args.input_shape):
            print("[error] custom 模式必须指定 --weight-url, --pyiqa-name, --input-shape")
            sys.exit(1)
        config = {
            "pyiqa_name": args.pyiqa_name,
            "weight_url": args.weight_url,
            "input_shape": tuple(int(x) for x in args.input_shape.split(",")),
            "input_name": args.input_name,
            "output_name": args.output_name,
            "out_name": args.out_name or args.pyiqa_name.replace("-", "_"),
            "description": f"custom model {args.pyiqa_name}",
        }
        export_one(args.pyiqa_name, config)
    else:
        export_one(args.name)


if __name__ == "__main__":
    main()
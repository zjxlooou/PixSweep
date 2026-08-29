# -*- coding: utf-8 -*-
"""把 InsightFace det_10g.onnx 改造为动态 batch（权重不变）。

依据 insightface PR #1781 的导出修复（官方 buffalo 包从未重导出）：
  1. 输入 batch 维 1 -> 动态 'N'
  2. 9 个 Transpose perm [2,3,0,1] -> [0,2,3,1]（batch 保持最外层）
  3. 9 个 Reshape [-1,K] -> [0,-1,K]（3 个共享 initializer，逐节点判断）
输出变为 [B, anchors, K]（每 stride 锚点数不变：12800/3200/800）。

验证基线（改后必须全过）：
  - batch=1 与原模型逐位一致
  - 同图 ×N 每帧一致
  - 帧 A 输出与同伴帧内容零相关（防跨帧泄漏）
用法: python make_det_batched.py <src.onnx> <dst.onnx>
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper


def main(src: str, dst: str) -> None:
    m = onnx.load(src)
    g = m.graph
    g.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    g.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")

    n_t = n_r = 0
    for n in g.node:
        if n.op_type == "Transpose" and list(n.attribute[0].ints) == [2, 3, 0, 1]:
            del n.attribute[0].ints[:]
            n.attribute[0].ints.extend([0, 2, 3, 1])
            n_t += 1

    inits = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Reshape":
            init = inits.get(n.input[1])
            if init is None:
                continue
            arr = numpy_helper.to_array(init)
            if arr.ndim == 1 and len(arr) == 2 and arr[0] == -1:
                new = np.concatenate([[0], arr]).astype(np.int64)
                init.CopyFrom(numpy_helper.from_array(new, init.name))
                n_r += 1

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"ok: {n_t} transposes, {n_r} reshapes -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

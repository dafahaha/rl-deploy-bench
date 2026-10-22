"""Temporary CI diagnostic: reproduce the verdict test and emit the actual
mse/cosine values as GitHub Actions annotations (visible without login)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import numpy as np
import torch

from test_core import SimplePolicy
from rl_deploy_bench.exporter.onnx_export import export_to_onnx
from rl_deploy_bench.quantizer.int8 import dynamic_quantize, evaluate_quantization

print("ORT_THREADS env:", os.environ.get("RL_DEPLOY_BENCH_ORT_THREADS"))
import onnxruntime as ort
print("onnxruntime:", ort.__version__)
import onnx
print("onnx:", onnx.__version__)
print("numpy:", np.__version__, "torch:", torch.__version__)

torch.manual_seed(0)
policy = SimplePolicy()
policy.eval()

for ns in (20, 100):
    d = tempfile.mkdtemp()
    onnx_path = os.path.join(d, "test.onnx")
    export_to_onnx(policy, (4,), onnx_path)
    q = dynamic_quantize(onnx_path)
    r = evaluate_quantization(onnx_path, q, (4,), num_samples=ns, seed=0)
    line = (
        f"ns={ns} mse={r['action_mse']:.6e} cos={r['cosine_similarity']:.8f} "
        f"mse_ok={r['mse_within_threshold']} cos_ok={r['cosine_within_threshold']} "
        f"verdict={r['verdict']}"
    )
    print(line)
    print(f"::warning title=CI verdict values::{line}")

    # Emit the quantized graph node op types so we can see the structure.
    import onnx
    from onnx import numpy_helper
    m = onnx.load(q)
    ops = [n.op_type for n in m.graph.node]
    from collections import Counter
    summ = dict(Counter(ops))
    print(f"::warning title=ns{ns} graph::{summ}")

    # Compact: only the weight scale shapes/ranges (per-channel vs per-tensor
    # and the actual scale values drive the quantized weight error).
    scale_info = []
    for init in m.graph.initializer:
        if init.name.endswith("weight_scale"):
            arr = numpy_helper.to_array(init)
            scale_info.append(
                f"{init.name.split('.')[-2]}:n{len(arr)}:{float(arr.min()):.4g}..{float(arr.max()):.4g}"
            )
    print(f"::warning title=ns{ns} weight_scales::{' | '.join(scale_info)}")

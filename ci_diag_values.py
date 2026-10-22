"""Temporary CI diagnostic: reproduce the verdict test and emit the actual
mse/cosine values AND per-sample actions as GitHub Actions annotations."""
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
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

import onnxruntime as ort
import onnx
print("::warning title=versions::"
      f"ort={ort.__version__} onnx={onnx.__version__} "
      f"np={np.__version__} torch={torch.__version__} "
      f"ORT_THREADS={os.environ.get('RL_DEPLOY_BENCH_ORT_THREADS')}")

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
        f"verdict={r['verdict']}"
    )
    print(f"::warning title=CI verdict values::{line}")

    # Re-run the first 5 observations directly and emit orig vs quant actions.
    orig_inf = OnnxRuntimeInference(onnx_path)
    quant_inf = OnnxRuntimeInference(q)
    obs = np.random.RandomState(0).randn(5, 4).astype(np.float32)
    detail = []
    for i, o in enumerate(obs):
        a = orig_inf.infer(o).actions[0]
        b = quant_inf.infer(o).actions[0]
        detail.append(f"{i}:{np.round(a,4).tolist()}/{np.round(b,4).tolist()}")
    print(f"::warning title=ns{ns} first5::{' '.join(detail)}")

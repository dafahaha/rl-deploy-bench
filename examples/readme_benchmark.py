"""Reproducible benchmark that generates the README "Benchmark Example Output" table.

This is NOT illustrative numbers: every row below comes from a real run of the
toolkit on a small synthetic MLP policy (4-dim obs -> 2-dim action, same
architecture as demo.py). On a CPU-only machine INT8 is typically *slower*
than FP32 (MatMulInteger overhead without VNNI) -- the table shows whatever
the hardware actually measures.

Usage:
    python examples/readme_benchmark.py

Notes on honesty:
- Static INT8 here is calibrated on random-normal observations (no Gymnasium
  env / SB3 policy available in this run). Environment-based calibration
  needs `pip install rl-deploy-bench[sb3] gymnasium`; see
  examples/end_to_end_benchmark.py.
- Latency numbers are machine- and load-dependent; re-run this script on your
  own hardware before quoting them as your own results.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

from rl_deploy_bench.benchmark.accuracy import (  # noqa: E402
    compare_actions,
    generate_test_observations,
)
from rl_deploy_bench.benchmark.latency import benchmark_latency  # noqa: E402
from rl_deploy_bench.exporter import export_to_onnx  # noqa: E402
from rl_deploy_bench.quantizer import (  # noqa: E402
    convert_onnx_to_fp16,
    dynamic_quantize,
)
from rl_deploy_bench.quantizer.int8 import static_quantize  # noqa: E402
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference  # noqa: E402
from rl_deploy_bench.utils.platform import detect_platform  # noqa: E402


class DemoPolicy(nn.Module):
    """Same small MLP used by demo.py (4-dim obs, 2-dim action)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    obs_shape = (4,)
    work = tempfile.mkdtemp(prefix="rdb_readme_bench_")

    platform = detect_platform()
    print(f"Platform: {platform.os} {platform.arch}")
    print(f"Python: {sys.version.split()[0]}")
    if platform.has_nvidia_gpu:
        print(f"GPU: {platform.gpu_name}")
    else:
        print("GPU: none detected (CPU-only run)")
    print()

    policy = DemoPolicy()
    policy.eval()

    fp32 = export_to_onnx(policy, obs_shape, os.path.join(work, "fp32.onnx"))
    fp16 = convert_onnx_to_fp16(fp32, os.path.join(work, "fp16.onnx"))
    int8_dyn = dynamic_quantize(fp32, os.path.join(work, "int8_dyn.onnx"))
    # Random-normal calibration (NOT environment data; see module docstring).
    calib = np.random.RandomState(42).randn(500, *obs_shape).astype(np.float32)
    int8_stat = static_quantize(
        fp32,
        obs_shape,
        os.path.join(work, "int8_stat.onnx"),
        calibration_observations=calib,
        calibration_samples=500,
    )

    models = [
        ("FP32 (Original)", fp32),
        ("FP16", fp16),
        ("INT8 Dynamic", int8_dyn),
        ("INT8 Static (random-norm calib.)", int8_stat),
    ]

    inferences, results = [], []
    for name, path in models:
        # Pin ORT to a single intra-op thread: on a loaded multi-core laptop the
        # thread-pool scheduling hiccups inflate mean latency above P95 for the
        # INT8 rows. Single-threaded measurements are quieter and comparable
        # across models.
        inf = OnnxRuntimeInference(path, num_threads=1)
        inferences.append(inf)
        # Throwaway pass: absorbs ONNX Runtime session JIT / process warmup so
        # the official numbers are not inflated by first-run scheduling hiccups.
        benchmark_latency(inf, obs_shape, num_warmup=100, num_runs=200)
        res = benchmark_latency(inf, obs_shape, num_warmup=100, num_runs=1000, seed=42)
        results.append(res)
        print(
            f"  {name:34s} mean={res.latency.mean_ms:.3f}ms "
            f"P50={res.latency.p50_ms:.3f}ms P95={res.latency.p95_ms:.3f}ms "
            f"FPS={res.latency.throughput_fps:.1f}"
        )

    # Accuracy: every model vs FP32 baseline.
    test_obs = generate_test_observations(obs_shape, num_samples=500, seed=0)
    fp32_actions = np.array([inferences[0].infer(o).actions[0] for o in test_obs])
    mses, verdicts = [0.0], ["baseline"]
    for i in range(1, len(models)):
        q_actions = np.array([inferences[i].infer(o).actions[0] for o in test_obs])
        acc = compare_actions(fp32_actions, q_actions, test_obs)
        mses.append(acc.action_mse)
        # Mirror evaluate_quantization's gate thresholds for the verdict label.
        mse_ok = acc.action_mse < 0.01
        cos_ok = acc.action_cosine_similarity > 0.99
        verdicts.append(
            "pass"
            if (mse_ok and cos_ok)
            else ("fail" if (not mse_ok and not cos_ok) else "caution")
        )

    print()
    print("Model                              Mean(ms)  P95(ms)   FPS       Action MSE   Verdict")
    print("-" * 86)
    for i, (name, _) in enumerate(models):
        lat = results[i].latency
        print(
            f"{name:35s} {lat.mean_ms:<9.3f} {lat.p95_ms:<9.3f} "
            f"{lat.throughput_fps:<9.1f} {mses[i]:<12.8f} {verdicts[i]}"
        )


if __name__ == "__main__":
    main()

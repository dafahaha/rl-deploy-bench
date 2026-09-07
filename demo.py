"""RL-Deploy-Bench Quick Demo — 30 seconds to see all core features.

This script demonstrates the complete workflow without any external
dependencies beyond the core package. It creates a simple policy,
exports it, quantizes it, benchmarks it, and generates a report.

Usage:
    python demo.py

No GPU or trained model required — everything runs on CPU.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Add src to path if running from repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import torch.nn as nn

from rl_deploy_bench import __version__
from rl_deploy_bench.exporter import export_to_onnx, export_to_torchscript
from rl_deploy_bench.quantizer import (
    convert_onnx_to_fp16,
    dynamic_quantize,
    evaluate_fp16_impact,
    evaluate_quantization,
)
from rl_deploy_bench.benchmark.latency import benchmark_latency
from rl_deploy_bench.benchmark.accuracy import compare_actions, generate_test_observations
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
from rl_deploy_bench.reporter.markdown import generate_markdown_report
from rl_deploy_bench.reporter.html import generate_html_report
from rl_deploy_bench.utils.platform import detect_platform


class DemoPolicy(nn.Module):
    """Simple RL policy for demonstration (4-dim obs, 2-dim action)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main():
    output_dir = os.path.join(tempfile.gettempdir(), "rl_deploy_bench_demo")
    os.makedirs(output_dir, exist_ok=True)
    obs_shape = (4,)

    print_header(f"RL-Deploy-Bench v{__version__} — Quick Demo")
    print(f"  Output directory: {output_dir}")

    # Platform info
    platform = detect_platform()
    print(f"  Platform: {platform.os} {platform.arch}")
    if platform.has_nvidia_gpu:
        print(f"  GPU: {platform.gpu_name}")
    print(f"  CPU: {platform.cpu_count} cores, {platform.total_memory_gb} GB RAM")

    # Step 1: Create policy
    print_header("Step 1: Create RL Policy")
    policy = DemoPolicy()
    policy.eval()
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: MLP(4→128→64→2), {total_params:,} parameters")

    # Step 2: Export to ONNX and TorchScript
    print_header("Step 2: Export Model (ONNX + TorchScript)")
    onnx_path = os.path.join(output_dir, "policy.onnx")
    onnx_path = export_to_onnx(policy, obs_shape, onnx_path)
    print(f"  ONNX: {os.path.getsize(onnx_path):,} bytes")

    ts_path = os.path.join(output_dir, "policy.pt")
    ts_path = export_to_torchscript(policy, obs_shape, ts_path)
    print(f"  TorchScript: {os.path.getsize(ts_path):,} bytes")

    # Step 3: Quantize (FP16 + INT8)
    print_header("Step 3: Quantize (FP16 + INT8 Dynamic)")

    fp16_path = convert_onnx_to_fp16(onnx_path, os.path.join(output_dir, "policy_fp16.onnx"))
    fp16_eval = evaluate_fp16_impact(onnx_path, fp16_path, obs_shape, num_samples=200)
    print(f"  FP16: {os.path.getsize(fp16_path):,} bytes, "
          f"MSE={fp16_eval['action_mse']:.2e}, "
          f"verdict={fp16_eval['verdict']}")

    int8_path = dynamic_quantize(onnx_path, os.path.join(output_dir, "policy_int8.onnx"))
    int8_eval = evaluate_quantization(onnx_path, int8_path, obs_shape, num_samples=200)
    print(f"  INT8: {os.path.getsize(int8_path):,} bytes, "
          f"MSE={int8_eval['action_mse']:.2e}, "
          f"verdict={int8_eval['verdict']}")

    # Step 4: Benchmark all models
    print_header("Step 4: Benchmark Latency & Throughput")
    models = [
        ("FP32", onnx_path),
        ("FP16", fp16_path),
        ("INT8", int8_path),
    ]
    results = []
    inferences = []

    for name, path in models:
        inf = OnnxRuntimeInference(path)
        inferences.append(inf)
        result = benchmark_latency(inf, obs_shape, num_warmup=20, num_runs=200)
        results.append(result)
        print(f"  {name:6s}: mean={result.latency.mean_ms:.3f}ms, "
              f"P95={result.latency.p95_ms:.3f}ms, "
              f"FPS={result.latency.throughput_fps:,.0f}")

    # Step 5: Accuracy comparison
    print_header("Step 5: Accuracy Comparison (vs FP32)")
    test_obs = generate_test_observations(obs_shape, num_samples=300)
    fp32_actions = np.array([inferences[0].infer(o).actions[0] for o in test_obs])

    accuracy_results = [None]  # FP32 is baseline
    for i in range(1, len(models)):
        quant_actions = np.array([inferences[i].infer(o).actions[0] for o in test_obs])
        acc = compare_actions(fp32_actions, quant_actions, test_obs)
        accuracy_results.append(acc)
        print(f"  {models[i][0]:6s}: MSE={acc.action_mse:.2e}, "
              f"Cosine={acc.action_cosine_similarity:.6f}, "
              f"MaxErr={acc.action_max_error:.6f}")

    # Step 6: Generate reports
    print_header("Step 6: Generate Reports")
    model_names = [m[0] for m in models]
    model_paths = [m[1] for m in models]

    md_path = os.path.join(output_dir, "demo_report.md")
    md_path = generate_markdown_report(
        md_path, results, model_names,
        accuracy_results=accuracy_results,
        model_paths=model_paths,
        platform_info=platform,
        title="RL-Deploy-Bench Demo Report",
    )
    print(f"  Markdown: {md_path}")

    html_path = os.path.join(output_dir, "demo_report.html")
    html_path = generate_html_report(
        html_path, results, model_names,
        accuracy_results=accuracy_results,
        platform_info=platform,
        title="RL-Deploy-Bench Demo Report",
    )
    print(f"  HTML: {html_path}")

    # Summary
    print_header("Demo Complete!")
    print(f"  All outputs saved to: {output_dir}")
    print(f"  Open {html_path} in your browser for interactive charts.")
    print(f"\n  Next steps:")
    print(f"    1. Try with your own trained RL model")
    print(f"    2. Use 'rl-deploy-bench calibrate' for environment-based INT8 calibration")
    print(f"    3. Check out examples/ for more workflows")
    print(f"    4. Star the repo: https://github.com/dafahaha/rl-deploy-bench")
    print("=" * 65)


if __name__ == "__main__":
    main()

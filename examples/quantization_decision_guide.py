"""Quantization Decision Guide - Choose the optimal precision for your deployment.

This example demonstrates how to systematically evaluate FP32, FP16,
INT8 dynamic, and INT8 static quantization, and choose the best option
based on your hardware, latency requirements, and accuracy tolerance.

Decision Framework:
1. Start with FP32 (baseline)
2. Try FP16 first - usually near-lossless, good speedup on GPU
3. Try INT8 dynamic - no calibration needed, good for small models
4. Try INT8 static - best accuracy with environment calibration
5. Choose the lowest precision that meets your accuracy threshold

Usage:
    python examples/quantization_decision_guide.py

Requirements:
    pip install gymnasium
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn

from rl_deploy_bench.exporter.onnx_export import export_to_onnx
from rl_deploy_bench.benchmark.calibration import (
    EnvironmentCalibrationGenerator,
    CalibrationConfig,
)
from rl_deploy_bench.benchmark.latency import benchmark_latency
from rl_deploy_bench.benchmark.accuracy import compare_actions, generate_test_observations
from rl_deploy_bench.quantizer.int8 import (
    dynamic_quantize,
    static_quantize_with_dataset,
    evaluate_quantization,
    compare_model_sizes,
)
from rl_deploy_bench.quantizer.fp16 import (
    convert_onnx_to_fp16,
    evaluate_fp16_impact,
    get_fp16_supported_gpus,
)
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
from rl_deploy_bench.utils.platform import detect_platform


class PendulumPolicy(nn.Module):
    """PPO-style policy for Pendulum-v1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x) * 2.0


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    obs_shape = (3,)
    env_name = "Pendulum-v1"

    # ============================================================
    # Step 0: Platform Analysis
    # ============================================================
    print_section("Step 0: Platform Analysis")
    platform = detect_platform()
    print(f"  OS: {platform.os} {platform.arch}")
    print(f"  CPU: {platform.cpu_count} cores, {platform.total_memory_gb} GB RAM")
    if platform.has_nvidia_gpu:
        print(f"  GPU: {platform.gpu_name}")
        print(f"  FP16 support: Likely YES (check GPU architecture)")
    else:
        print(f"  GPU: None (CPU-only deployment)")
        print(f"  FP16 support: Limited (CPU FP16 is often slower)")

    print(f"\n  GPUs with native FP16 support:")
    for gpu in get_fp16_supported_gpus()[:5]:
        print(f"    - {gpu}")

    # ============================================================
    # Step 1: Create and export FP32 baseline
    # ============================================================
    print_section("Step 1: FP32 Baseline")
    policy = PendulumPolicy()
    policy.eval()

    fp32_path = os.path.join(output_dir, "decision_fp32.onnx")
    fp32_path = export_to_onnx(policy, obs_shape, fp32_path)
    print(f"  Exported FP32 model: {fp32_path}")

    fp32_inf = OnnxRuntimeInference(fp32_path)
    fp32_bench = benchmark_latency(fp32_inf, obs_shape, num_warmup=30, num_runs=300)
    print(f"  FP32 Mean Latency: {fp32_bench.latency.mean_ms:.3f} ms")
    print(f"  FP32 Throughput: {fp32_bench.latency.throughput_fps:.1f} FPS")

    # ============================================================
    # Step 2: FP16 Evaluation
    # ============================================================
    print_section("Step 2: FP16 Evaluation")
    fp16_path = convert_onnx_to_fp16(fp32_path)
    print(f"  Converted to FP16: {fp16_path}")

    fp16_result = evaluate_fp16_impact(fp32_path, fp16_path, obs_shape, num_samples=300)
    print(f"  Verdict: {fp16_result['verdict'].upper()}")
    print(f"  Action MSE: {fp16_result['action_mse']:.10f}")
    print(f"  Cosine Similarity: {fp16_result['cosine_similarity']:.8f}")
    print(f"  Size: {fp16_result['size_comparison']['quantized_size_mb']:.3f} MB "
          f"(-{fp16_result['size_comparison']['size_reduction_pct']:.1f}%)")

    fp16_inf = OnnxRuntimeInference(fp16_path)
    fp16_bench = benchmark_latency(fp16_inf, obs_shape, num_warmup=30, num_runs=300)
    print(f"  FP16 Mean Latency: {fp16_bench.latency.mean_ms:.3f} ms "
          f"({(fp16_bench.latency.mean_ms / fp32_bench.latency.mean_ms - 1) * 100:+.1f}% vs FP32)")

    # ============================================================
    # Step 3: INT8 Dynamic Quantization
    # ============================================================
    print_section("Step 3: INT8 Dynamic Quantization")
    int8_dyn_path = dynamic_quantize(fp32_path, os.path.join(output_dir, "decision_int8_dynamic.onnx"))
    print(f"  Dynamic INT8: {int8_dyn_path}")

    dyn_result = evaluate_quantization(fp32_path, int8_dyn_path, obs_shape, num_samples=300)
    print(f"  Verdict: {dyn_result['verdict'].upper()}")
    print(f"  Action MSE: {dyn_result['action_mse']:.10f}")
    print(f"  Cosine Similarity: {dyn_result['cosine_similarity']:.8f}")
    print(f"  Size reduction: {dyn_result['size_comparison']['size_reduction_pct']:.1f}%")

    dyn_inf = OnnxRuntimeInference(int8_dyn_path)
    dyn_bench = benchmark_latency(dyn_inf, obs_shape, num_warmup=30, num_runs=300)
    print(f"  Dynamic INT8 Latency: {dyn_bench.latency.mean_ms:.3f} ms")

    # ============================================================
    # Step 4: INT8 Static Quantization (with calibration)
    # ============================================================
    print_section("Step 4: INT8 Static Quantization (Environment Calibration)")
    print("  Generating calibration data from Pendulum-v1...")
    calib_config = CalibrationConfig(num_samples=300, collection_strategy="random", seed=42)
    generator = EnvironmentCalibrationGenerator(env_name, config=calib_config)
    dataset = generator.generate()
    print(f"  Collected {len(dataset)} calibration samples")

    int8_static_path = static_quantize_with_dataset(
        fp32_path, dataset, os.path.join(output_dir, "decision_int8_static.onnx")
    )
    print(f"  Static INT8 (calibrated): {int8_static_path}")

    static_result = evaluate_quantization(fp32_path, int8_static_path, obs_shape, num_samples=300)
    print(f"  Verdict: {static_result['verdict'].upper()}")
    print(f"  Action MSE: {static_result['action_mse']:.10f}")
    print(f"  Cosine Similarity: {static_result['cosine_similarity']:.8f}")
    print(f"  Size reduction: {static_result['size_comparison']['size_reduction_pct']:.1f}%")

    static_inf = OnnxRuntimeInference(int8_static_path)
    static_bench = benchmark_latency(static_inf, obs_shape, num_warmup=30, num_runs=300)
    print(f"  Static INT8 Latency: {static_bench.latency.mean_ms:.3f} ms")

    # ============================================================
    # Step 5: Decision Summary
    # ============================================================
    print_section("Step 5: Quantization Decision Summary")

    results = [
        ("FP32 (Baseline)", fp32_bench, None, 1.0, "Reference"),
        ("FP16", fp16_bench, fp16_result, fp16_result['size_comparison']['compression_ratio'], fp16_result['verdict']),
        ("INT8 Dynamic", dyn_bench, dyn_result, dyn_result['size_comparison']['compression_ratio'], dyn_result['verdict']),
        ("INT8 Static (Calibrated)", static_bench, static_result, static_result['size_comparison']['compression_ratio'], static_result['verdict']),
    ]

    print(f"\n  {'Model':<28} {'Latency(ms)':<12} {'FPS':<10} {'MSE':<14} {'Size Ratio':<12} {'Verdict'}")
    print(f"  {'-'*90}")
    for name, bench, acc, ratio, verdict in results:
        mse = f"{acc['action_mse']:.2e}" if acc else "0 (baseline)"
        print(f"  {name:<28} {bench.latency.mean_ms:<12.3f} {bench.latency.throughput_fps:<10.1f} {mse:<14} {ratio:<12.2f}x {verdict}")

    # Decision logic
    print("\n  Recommendation:")
    print("  " + "-" * 60)

    # Find the best option
    best = None
    for name, bench, acc, ratio, verdict in results[1:]:  # Skip FP32 baseline
        if verdict == "pass":
            if best is None or bench.latency.mean_ms < best[1].latency.mean_ms:
                best = (name, bench, acc, ratio, verdict)

    if best:
        print(f"  -> RECOMMENDED: {best[0]}")
        print(f"     - Latency: {best[1].latency.mean_ms:.3f} ms "
              f"({(best[1].latency.mean_ms / fp32_bench.latency.mean_ms - 1) * 100:+.1f}% vs FP32)")
        print(f"     - Accuracy: MSE={best[2]['action_mse']:.2e}, "
              f"Cosine={best[2]['cosine_similarity']:.6f}")
        print(f"     - Size: {best[3]:.2f}x compression")
    else:
        print("  -> RECOMMENDED: FP32 (no quantized variant meets accuracy threshold)")
        print("     - Consider: more calibration data, selective quantization, or FP16 with op_blocklist")

    print("\n  Decision Rules:")
    print("    1. Always try FP16 first - near-lossless, good GPU speedup")
    print("    2. INT8 dynamic is best for small models (no calibration overhead)")
    print("    3. INT8 static with environment calibration gives best INT8 accuracy")
    print("    4. On CPU-only, FP16 may be slower (no native FP16 units)")
    print("    5. Always verify on actual environment rollouts, not just MSE")
    print("    6. If accuracy is marginal, try increasing calibration samples")

    print("\n" + "=" * 70)
    print("  Quantization Decision Guide Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Quickstart example: end-to-end deployment benchmark for an SB3 PPO model.

This example demonstrates the complete workflow:
1. Train a simple PPO model (or load an existing one)
2. Export to ONNX
3. Quantize to INT8
4. Benchmark latency and throughput
5. Compare accuracy between FP32 and INT8
6. Generate a report

Usage:
    python examples/quickstart.py
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from rl_deploy_bench.benchmark.accuracy import compare_actions, generate_test_observations
from rl_deploy_bench.benchmark.latency import benchmark_latency
from rl_deploy_bench.exporter.sb3 import export_sb3_model, load_sb3_model, verify_sb3_export
from rl_deploy_bench.monitor import create_monitor
from rl_deploy_bench.quantizer.int8 import compare_model_sizes, dynamic_quantize, get_model_size_mb
from rl_deploy_bench.reporter.html import generate_html_report
from rl_deploy_bench.reporter.markdown import generate_markdown_report
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
from rl_deploy_bench.utils.platform import detect_platform


def main():
    # Configuration
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    env_name = "CartPole-v1"
    algo = "PPO"
    model_path = os.path.join(output_dir, "ppo_cartpole.zip")
    onnx_path = os.path.join(output_dir, "ppo_cartpole.onnx")
    int8_path = os.path.join(output_dir, "ppo_cartpole_int8.onnx")
    report_md = os.path.join(output_dir, "benchmark_report.md")
    report_html = os.path.join(output_dir, "benchmark_report.html")

    print("=" * 60)
    print("RL-Deploy-Bench Quickstart Example")
    print("=" * 60)

    # Step 1: Train or load model
    print("\n[Step 1] Training/Loading PPO model on CartPole-v1...")
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO

        if os.path.exists(model_path):
            print(f"  Loading existing model: {model_path}")
            model = PPO.load(model_path)
        else:
            print("  Training new model (10000 steps)...")
            env = gym.make(env_name)
            model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)
            model.learn(total_timesteps=10000)
            model.save(model_path)
            env.close()
            print(f"  Model saved: {model_path}")
    except ImportError:
        print("  stable-baselines3 not installed. Install with: pip install stable-baselines3 gymnasium")
        print("  Skipping training, using a dummy model for demonstration...")
        model = None

    if model is None:
        print("\nERROR: This example requires stable-baselines3 and gymnasium.")
        print("Install with: pip install stable-baselines3 gymnasium")
        return

    # Get observation shape
    obs_shape = model.observation_space.shape
    print(f"  Observation shape: {obs_shape}")
    print(f"  Action space: {model.action_space}")

    # Step 2: Export to ONNX
    print("\n[Step 2] Exporting model to ONNX...")
    onnx_path = export_sb3_model(model, onnx_path)
    print(f"  Exported: {onnx_path}")
    print(f"  Size: {get_model_size_mb(onnx_path):.2f} MB")

    # Verify export
    print("  Verifying export...")
    verify_result = verify_sb3_export(onnx_path, model, num_samples=100)
    print(f"  Verification: {'PASSED' if verify_result['passed'] else 'WARNING'}")
    print(f"  Max absolute difference: {verify_result['max_abs_diff']:.6f}")

    # Step 3: Quantize to INT8
    print("\n[Step 3] Quantizing to INT8 (dynamic quantization)...")
    int8_path = dynamic_quantize(onnx_path, int8_path)
    print(f"  Quantized: {int8_path}")
    size_info = compare_model_sizes(onnx_path, int8_path)
    print(f"  Size: {size_info['quantized_size_mb']:.2f} MB "
          f"(-{size_info['size_reduction_pct']:.1f}%, {size_info['compression_ratio']:.2f}x compression)")

    # Step 4: Benchmark both models
    print("\n[Step 4] Benchmarking FP32 model...")
    fp32_inf = OnnxRuntimeInference(onnx_path)
    print(f"  Providers: {fp32_inf.session.get_providers()}")

    monitor = None
    try:
        monitor = create_monitor()
        print("  System monitor: enabled")
    except Exception:
        print("  System monitor: disabled (no GPU detected)")

    fp32_bench = benchmark_latency(
        fp32_inf, obs_shape, num_warmup=50, num_runs=500, monitor=monitor
    )
    print(f"  Mean latency: {fp32_bench.latency.mean_ms:.3f} ms")
    print(f"  P95 latency: {fp32_bench.latency.p95_ms:.3f} ms")
    print(f"  Throughput: {fp32_bench.latency.throughput_fps:.1f} FPS")

    print("\n  Benchmarking INT8 model...")
    int8_inf = OnnxRuntimeInference(int8_path)
    int8_bench = benchmark_latency(
        int8_inf, obs_shape, num_warmup=50, num_runs=500, monitor=None
    )
    print(f"  Mean latency: {int8_bench.latency.mean_ms:.3f} ms")
    print(f"  P95 latency: {int8_bench.latency.p95_ms:.3f} ms")
    print(f"  Throughput: {int8_bench.latency.throughput_fps:.1f} FPS")

    # Step 5: Compare accuracy
    print("\n[Step 5] Comparing action accuracy (FP32 vs INT8)...")
    observations = generate_test_observations(obs_shape, num_samples=1000)

    fp32_actions = []
    int8_actions = []
    for obs in observations:
        fp32_actions.append(fp32_inf.infer(obs).actions[0])
        int8_actions.append(int8_inf.infer(obs).actions[0])
    fp32_actions = np.array(fp32_actions)
    int8_actions = np.array(int8_actions)

    acc_result = compare_actions(fp32_actions, int8_actions, observations)
    print(f"  Action MSE: {acc_result.action_mse:.6f}")
    print(f"  Action MAE: {acc_result.action_mae:.6f}")
    print(f"  Max error: {acc_result.action_max_error:.6f}")
    print(f"  Cosine similarity: {acc_result.action_cosine_similarity:.6f}")
    print(f"  Relative error: {acc_result.action_relative_error:.4f}")

    # Step 6: Generate reports
    print("\n[Step 6] Generating reports...")
    platform_info = detect_platform()

    md_path = generate_markdown_report(
        report_md,
        [fp32_bench, int8_bench],
        ["FP32 (Original)", "INT8 (Quantized)"],
        accuracy_results=[None, acc_result],
        model_paths=[onnx_path, int8_path],
        platform_info=platform_info,
    )
    print(f"  Markdown report: {md_path}")

    html_path = generate_html_report(
        report_html,
        [fp32_bench, int8_bench],
        ["FP32 (Original)", "INT8 (Quantized)"],
        accuracy_results=[None, acc_result],
        platform_info=platform_info,
    )
    print(f"  HTML report: {html_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model: PPO on {env_name}")
    print(f"  Observation shape: {obs_shape}")
    print(f"  FP32 size: {size_info['original_size_mb']:.2f} MB")
    print(f"  INT8 size: {size_info['quantized_size_mb']:.2f} MB "
          f"(-{size_info['size_reduction_pct']:.1f}%)")
    print(f"  FP32 latency: {fp32_bench.latency.mean_ms:.3f} ms (P95: {fp32_bench.latency.p95_ms:.3f} ms)")
    print(f"  INT8 latency: {int8_bench.latency.mean_ms:.3f} ms (P95: {int8_bench.latency.p95_ms:.3f} ms)")
    print(f"  Action MSE (INT8 vs FP32): {acc_result.action_mse:.6f}")
    print(f"  Reports: {report_md}, {report_html}")
    print("=" * 60)


if __name__ == "__main__":
    main()

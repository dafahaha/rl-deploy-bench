"""End-to-end deployment benchmark example for RL models.

This example demonstrates the complete workflow:
1. Train (or load) a PPO model on Pendulum-v1
2. Export to ONNX
3. Generate calibration data from the environment
4. Quantize to INT8 (dynamic and static)
5. Benchmark latency and throughput
6. Compare action accuracy across models
7. Generate comprehensive Markdown and HTML reports

Usage:
    python examples/end_to_end_benchmark.py

Requirements:
    pip install stable-baselines3 gymnasium
    (If SB3 is not available, uses a pre-trained heuristic policy)
"""

from __future__ import annotations

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn

from rl_deploy_bench.exporter.onnx_export import export_to_onnx, verify_onnx_export
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
from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
from rl_deploy_bench.reporter.markdown import generate_markdown_report
from rl_deploy_bench.reporter.html import generate_html_report
from rl_deploy_bench.utils.platform import detect_platform


class PendulumPPOPolicy(nn.Module):
    """Simple MLP policy for Pendulum-v1.

    Architecture similar to what SB3 PPO would train:
    - Observation: 3 dims (cos(theta), sin(theta), theta_dot)
    - Action: 1 dim (torque in [-2, 2])
    - Two hidden layers of 64 units with ReLU
    - Tanh output scaled by 2.0
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x) * 2.0  # Pendulum action range is [-2, 2]


def train_ppo_policy(env_name: str = "Pendulum-v1", total_timesteps: int = 20000):
    """Train a PPO policy using Stable Baselines3 if available.

    Falls back to a randomly initialized (but deterministic) policy if SB3
    is not installed.
    """
    try:
        import gymnasium as gym
        from stable_baselines3 import PPO

        print(f"  Training PPO on {env_name} ({total_timesteps} steps)...")
        env = gym.make(env_name)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )
        model.learn(total_timesteps=total_timesteps)
        env.close()

        # Extract the policy network
        policy = model.policy
        print("  PPO training complete!")
        return policy, model

    except ImportError:
        print("  SB3 not available, using pre-defined architecture with random weights")
        policy = PendulumPPOPolicy()
        return policy, None


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("RL-Deploy-Bench: End-to-End Deployment Benchmark")
    print("=" * 70)

    obs_shape = (3,)  # Pendulum-v1 observation
    env_name = "Pendulum-v1"

    # Step 1: Train/load policy
    print("\n[Step 1/8] Training/loading policy...")
    policy, sb3_model = train_ppo_policy(env_name, total_timesteps=20000)
    policy.eval()

    # Step 2: Export to ONNX
    print("\n[Step 2/8] Exporting policy to ONNX...")
    fp32_path = os.path.join(output_dir, "pendulum_ppo_fp32.onnx")

    if sb3_model is not None:
        from rl_deploy_bench.exporter.sb3 import export_sb3_model, verify_sb3_export
        fp32_path = export_sb3_model(sb3_model, fp32_path)
        verify_result = verify_sb3_export(fp32_path, sb3_model, num_samples=100)
    else:
        fp32_path = export_to_onnx(policy, obs_shape, fp32_path)
        verify_result = verify_onnx_export(fp32_path, policy, obs_shape)

    print(f"  Exported: {fp32_path}")
    print(f"  Verification: {'PASSED' if verify_result['passed'] else 'WARNING'}")
    print(f"  Max diff: {verify_result['max_abs_diff']:.8f}")

    # Step 3: Generate calibration data
    print("\n[Step 3/8] Generating calibration data from environment...")
    calib_config = CalibrationConfig(
        num_samples=500,
        collection_strategy="random",
        seed=42,
    )

    if sb3_model is not None:
        from rl_deploy_bench.benchmark.calibration import SB3PolicyCalibrationGenerator
        generator = SB3PolicyCalibrationGenerator(env_name, sb3_model, config=calib_config)
    else:
        generator = EnvironmentCalibrationGenerator(env_name, config=calib_config)

    dataset = generator.generate()
    print(f"  Collected {len(dataset)} calibration samples")
    print(f"  Episodes: {dataset.collection_stats['episodes_completed']}")

    calib_path = os.path.join(output_dir, "pendulum_calibration.npz")
    dataset.save(calib_path)
    print(f"  Saved: {calib_path}")

    # Step 4: Dynamic quantization
    print("\n[Step 4/8] Dynamic INT8 quantization...")
    dynamic_path = os.path.join(output_dir, "pendulum_ppo_int8_dynamic.onnx")
    dynamic_path = dynamic_quantize(fp32_path, dynamic_path)
    dyn_size = compare_model_sizes(fp32_path, dynamic_path)
    print(f"  Quantized: {dynamic_path}")
    print(f"  Size: {dyn_size['quantized_size_mb']:.2f} MB "
          f"(-{dyn_size['size_reduction_pct']:.1f}%)")

    # Step 5: Static quantization with calibration
    print("\n[Step 5/8] Static INT8 quantization (with environment calibration)...")
    static_path = os.path.join(output_dir, "pendulum_ppo_int8_static.onnx")
    static_path = static_quantize_with_dataset(
        fp32_path, dataset, static_path, input_name="observation"
    )
    stat_size = compare_model_sizes(fp32_path, static_path)
    print(f"  Quantized: {static_path}")
    print(f"  Size: {stat_size['quantized_size_mb']:.2f} MB "
          f"(-{stat_size['size_reduction_pct']:.1f}%)")

    # Step 6: Benchmark all models
    print("\n[Step 6/8] Benchmarking all models...")
    models = [
        ("FP32 (Original)", fp32_path),
        ("INT8 Dynamic", dynamic_path),
        ("INT8 Static (Calibrated)", static_path),
    ]

    benchmark_results = []
    inference_engines = []

    for name, path in models:
        print(f"  Benchmarking {name}...")
        inf = OnnxRuntimeInference(path)
        inference_engines.append(inf)
        result = benchmark_latency(
            inf, obs_shape, num_warmup=50, num_runs=500, monitor=None
        )
        benchmark_results.append(result)
        print(f"    Mean: {result.latency.mean_ms:.3f} ms, "
              f"P95: {result.latency.p95_ms:.3f} ms, "
              f"Throughput: {result.latency.throughput_fps:.1f} FPS")

    # Step 7: Accuracy comparison
    print("\n[Step 7/8] Comparing action accuracy...")
    test_obs = generate_test_observations(obs_shape, num_samples=500)

    fp32_actions = np.array([inference_engines[0].infer(o).actions[0] for o in test_obs])
    accuracy_results = [None]  # FP32 is baseline, no comparison

    for i in range(1, len(models)):
        name = models[i][0]
        quant_actions = np.array([inference_engines[i].infer(o).actions[0] for o in test_obs])
        acc = compare_actions(fp32_actions, quant_actions, test_obs)
        accuracy_results.append(acc)
        print(f"  {name}: MSE={acc.action_mse:.8f}, "
              f"Cosine={acc.action_cosine_similarity:.8f}, "
              f"MaxErr={acc.action_max_error:.6f}")

    # Step 8: Generate reports
    print("\n[Step 8/8] Generating reports...")
    platform_info = detect_platform()
    model_names = [m[0] for m in models]
    model_paths = [m[1] for m in models]

    md_path = os.path.join(output_dir, "end_to_end_benchmark_report.md")
    md_path = generate_markdown_report(
        md_path,
        benchmark_results,
        model_names,
        accuracy_results=accuracy_results,
        model_paths=model_paths,
        platform_info=platform_info,
        title="Pendulum PPO Deployment Benchmark Report",
    )
    print(f"  Markdown: {md_path}")

    html_path = os.path.join(output_dir, "end_to_end_benchmark_report.html")
    html_path = generate_html_report(
        html_path,
        benchmark_results,
        model_names,
        accuracy_results=accuracy_results,
        platform_info=platform_info,
        title="Pendulum PPO Deployment Benchmark Report",
    )
    print(f"  HTML: {html_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("END-TO-END BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"\n  Environment: {env_name}")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Platform: {platform_info.os} {platform_info.arch}")
    if platform_info.has_nvidia_gpu:
        print(f"  GPU: {platform_info.gpu_name}")
    print()

    print(f"  {'Model':<30} {'Mean(ms)':<10} {'P95(ms)':<10} {'FPS':<10} {'MSE':<12}")
    print(f"  {'-'*72}")
    for i, (name, _) in enumerate(models):
        lat = benchmark_results[i].latency
        mse = accuracy_results[i].action_mse if accuracy_results[i] else 0.0
        print(f"  {name:<30} {lat.mean_ms:<10.3f} {lat.p95_ms:<10.3f} "
              f"{lat.throughput_fps:<10.1f} {mse:<12.8f}")

    print(f"\n  Reports:")
    print(f"    - Markdown: {md_path}")
    print(f"    - HTML: {html_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

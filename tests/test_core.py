"""Unit tests for rl-deploy-bench core modules.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ============================================================
# Test fixtures
# ============================================================

class SimplePolicy(nn.Module):
    """Simple MLP policy for testing."""
    def __init__(self, obs_dim=4, hidden_dim=32, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


@pytest.fixture
def policy():
    """Create a simple policy."""
    p = SimplePolicy()
    p.eval()
    return p


@pytest.fixture
def obs_shape():
    return (4,)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ============================================================
# Platform detection tests
# ============================================================

class TestPlatformDetection:
    def test_detect_platform_returns_info(self):
        from rl_deploy_bench.utils.platform import detect_platform
        info = detect_platform()
        assert info.os is not None
        assert info.arch is not None
        assert info.python_version is not None
        assert info.cpu_count > 0
        assert info.total_memory_gb > 0

    def test_get_monitor_backend_returns_string(self):
        from rl_deploy_bench.utils.platform import detect_platform, get_monitor_backend
        info = detect_platform()
        backend = get_monitor_backend(info)
        assert backend in ("nvidia", "jetson", "cpu")


# ============================================================
# Model export tests
# ============================================================

class TestModelExport:
    def test_export_to_onnx_creates_file(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        output_path = os.path.join(tmp_dir, "test.onnx")
        result = export_to_onnx(policy, obs_shape, output_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0

    def test_export_verification_passes(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx, verify_onnx_export
        output_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, output_path)
        result = verify_onnx_export(output_path, policy, obs_shape)
        assert bool(result["passed"]) is True
        assert result["max_abs_diff"] < 1e-4

    def test_export_with_action_bounds(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        output_path = os.path.join(tmp_dir, "test_bounds.onnx")
        low = np.array([-2.0, -1.0])
        high = np.array([2.0, 1.0])
        result = export_to_onnx(policy, obs_shape, output_path, action_low=low, action_high=high)
        assert os.path.exists(result)


# ============================================================
# Inference runtime tests
# ============================================================

class TestOnnxRuntimeInference:
    def test_inference_returns_correct_shape(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        inference = OnnxRuntimeInference(onnx_path)
        obs = np.random.randn(1, *obs_shape).astype(np.float32)
        result = inference.infer(obs)

        assert result.actions.shape == (1, 2)
        assert result.latency_ms > 0

    def test_inference_matches_pytorch(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        inference = OnnxRuntimeInference(onnx_path)
        obs = np.random.randn(1, *obs_shape).astype(np.float32)

        with torch.no_grad():
            torch_output = policy(torch.tensor(obs)).numpy()

        onnx_output = inference.infer(obs).actions
        assert np.allclose(torch_output, onnx_output, atol=1e-4)

    def test_warmup_does_not_crash(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        inference.warmup(num_runs=5, observation_shape=obs_shape)


# ============================================================
# Latency benchmark tests
# ============================================================

class TestLatencyBenchmark:
    def test_benchmark_returns_stats(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
        from rl_deploy_bench.benchmark.latency import benchmark_latency

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)

        result = benchmark_latency(
            inference, obs_shape, num_warmup=10, num_runs=50, monitor=None
        )

        assert result.latency.num_runs == 50
        assert result.latency.mean_ms > 0
        assert result.latency.p50_ms > 0
        assert result.latency.p95_ms >= result.latency.p50_ms
        assert result.latency.p99_ms >= result.latency.p95_ms
        assert result.latency.throughput_fps > 0
        assert len(result.latency.latencies_ms) == 50


# ============================================================
# Accuracy comparison tests
# ============================================================

class TestAccuracyComparison:
    def test_compare_identical_actions(self):
        from rl_deploy_bench.benchmark.accuracy import compare_actions
        actions = np.random.randn(100, 2).astype(np.float32)
        result = compare_actions(actions, actions)
        assert result.action_mse == pytest.approx(0.0, abs=1e-7)
        assert result.action_cosine_similarity == pytest.approx(1.0, abs=1e-5)

    def test_compare_different_actions(self):
        from rl_deploy_bench.benchmark.accuracy import compare_actions
        orig = np.random.randn(100, 2).astype(np.float32)
        deployed = orig + 0.1
        result = compare_actions(orig, deployed)
        assert result.action_mse > 0
        assert result.action_max_error > 0

    def test_generate_test_observations(self):
        from rl_deploy_bench.benchmark.accuracy import generate_test_observations
        obs = generate_test_observations((4,), num_samples=50)
        assert obs.shape == (50, 4)
        assert obs.dtype == np.float32


# ============================================================
# Quantization tests
# ============================================================

class TestQuantization:
    def test_dynamic_quantize_creates_file(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import dynamic_quantize, get_model_size_mb

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        quantized_path = os.path.join(tmp_dir, "test_int8.onnx")
        quantized_path = dynamic_quantize(onnx_path, quantized_path)
        assert os.path.exists(quantized_path)
        assert os.path.getsize(quantized_path) > 0

    def test_quantized_model_runs_inference(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import dynamic_quantize
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        quantized_path = dynamic_quantize(onnx_path)

        inference = OnnxRuntimeInference(quantized_path)
        obs = np.random.randn(1, *obs_shape).astype(np.float32)
        result = inference.infer(obs)
        assert result.actions.shape == (1, 2)

    def test_evaluate_quantization(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import dynamic_quantize, evaluate_quantization

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        quantized_path = dynamic_quantize(onnx_path)

        result = evaluate_quantization(onnx_path, quantized_path, obs_shape, num_samples=100)
        assert "verdict" in result
        assert "recommendation" in result
        assert "action_mse" in result
        assert "cosine_similarity" in result
        assert "size_comparison" in result
        assert result["verdict"] in ("pass", "caution", "fail")


# ============================================================
# Report generation tests
# ============================================================

class TestReportGeneration:
    def test_markdown_report_generated(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.reporter.markdown import generate_markdown_report
        from rl_deploy_bench.utils.platform import detect_platform

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.md")
        platform_info = detect_platform()
        output = generate_markdown_report(report_path, [result], ["Test Model"], platform_info=platform_info)

        assert os.path.exists(output)
        with open(output, "r") as f:
            content = f.read()
        assert "Latency" in content
        assert "Throughput" in content

    def test_html_report_generated(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.reporter.html import generate_html_report
        from rl_deploy_bench.utils.platform import detect_platform

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.html")
        platform_info = detect_platform()
        output = generate_html_report(report_path, [result], ["Test Model"], platform_info=platform_info)

        assert os.path.exists(output)
        with open(output, "r") as f:
            content = f.read()
        assert "<html" in content
        assert "plotly" in content.lower()


# ============================================================
# Calibration data tests
# ============================================================

class TestCalibrationData:
    def test_generate_calibration_from_env(self):
        from rl_deploy_bench.benchmark.calibration import (
            EnvironmentCalibrationGenerator,
            CalibrationConfig,
        )
        config = CalibrationConfig(num_samples=50, collection_strategy="random", seed=42)
        generator = EnvironmentCalibrationGenerator("CartPole-v1", config=config)
        dataset = generator.generate()

        assert len(dataset) == 50
        assert dataset.observations.shape == (50, 4)
        assert dataset.env_name == "CartPole-v1"
        assert "episodes_completed" in dataset.collection_stats

    def test_calibration_dataset_save_load(self, tmp_dir):
        from rl_deploy_bench.benchmark.calibration import (
            EnvironmentCalibrationGenerator,
            CalibrationConfig,
            CalibrationDataset,
        )
        config = CalibrationConfig(num_samples=30, collection_strategy="random", seed=42)
        generator = EnvironmentCalibrationGenerator("CartPole-v1", config=config)
        dataset = generator.generate()

        path = os.path.join(tmp_dir, "calib.npz")
        saved_path = dataset.save(path)
        assert os.path.exists(saved_path)

        loaded = CalibrationDataset.load(saved_path)
        assert len(loaded) == len(dataset)
        assert np.allclose(loaded.observations, dataset.observations)

    def test_calibration_statistics(self):
        from rl_deploy_bench.benchmark.calibration import (
            EnvironmentCalibrationGenerator,
            CalibrationConfig,
        )
        config = CalibrationConfig(num_samples=50, collection_strategy="random", seed=42)
        generator = EnvironmentCalibrationGenerator("CartPole-v1", config=config)
        dataset = generator.generate()
        stats = dataset.get_statistics()

        assert stats["num_samples"] == 50
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "per_dimension_mean" in stats


# ============================================================
# TensorRT fallback tests
# ============================================================

class TestTensorRTFallback:
    def test_is_tensorrt_available_returns_bool(self):
        from rl_deploy_bench.runtime.tensorrt_runtime import is_tensorrt_available
        result = is_tensorrt_available()
        assert isinstance(result, bool)

    def test_require_tensorrt_raises_if_unavailable(self):
        from rl_deploy_bench.runtime.tensorrt_runtime import (
            is_tensorrt_available,
            require_tensorrt,
        )
        if not is_tensorrt_available():
            with pytest.raises(ImportError) as exc_info:
                require_tensorrt()
            assert "TensorRT" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

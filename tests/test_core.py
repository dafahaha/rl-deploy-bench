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
        import onnxruntime as ort

        from rl_deploy_bench.exporter.onnx_export import export_to_onnx

        output_path = os.path.join(tmp_dir, "test_bounds.onnx")
        low = np.array([-2.0, -1.0])
        high = np.array([2.0, 1.0])
        result = export_to_onnx(policy, obs_shape, output_path, action_low=low, action_high=high)
        assert os.path.exists(result)

        # Numeric assertions (N2): ONNX Runtime outputs must (a) lie within the
        # requested bounds and (b) match the PyTorch policy that applies the same
        # tanh->[low, high] unscaling.
        session = ort.InferenceSession(result, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        obs = np.random.randn(8, *obs_shape).astype(np.float32)
        onnx_out = session.run(None, {input_name: obs})[0]

        assert np.all(onnx_out >= low - 1e-5), f"output below low: {onnx_out.min(axis=0)}"
        assert np.all(onnx_out <= high + 1e-5), f"output above high: {onnx_out.max(axis=0)}"

        # PyTorch reference with the same bounds applied.
        from rl_deploy_bench.exporter.onnx_export import OnnxablePolicy

        wrapped = OnnxablePolicy(policy)
        wrapped.set_action_bounds(low, high)
        wrapped.eval()
        with torch.no_grad():
            torch_out = wrapped(torch.tensor(obs)).numpy()
        assert np.allclose(torch_out, onnx_out, atol=1e-4)

    def test_export_with_action_bounds_dynamic_batch(self, policy, obs_shape, tmp_dir):
        """M4 regression: bounds must expand to the runtime batch size, not be
        traced from the batch=1 dummy input."""
        import onnxruntime as ort

        from rl_deploy_bench.exporter.onnx_export import (
            OnnxablePolicy,
            export_to_onnx,
            verify_onnx_export,
        )

        output_path = os.path.join(tmp_dir, "test_bounds_dyn.onnx")
        low = np.array([-2.0, -1.0])
        high = np.array([2.0, 1.0])
        export_to_onnx(policy, obs_shape, output_path, action_low=low, action_high=high)

        # verify_onnx_export must be told the bounds (M2) and must now pass.
        verify = verify_onnx_export(
            output_path, policy, obs_shape, action_low=low, action_high=high
        )
        assert bool(verify["passed"]) is True, verify
        assert verify["max_abs_diff"] < 1e-4

        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        wrapped = OnnxablePolicy(policy)
        wrapped.set_action_bounds(low, high)
        wrapped.eval()

        for batch in (1, 4):
            obs = np.random.randn(batch, *obs_shape).astype(np.float32)
            onnx_out = session.run(None, {input_name: obs})[0]
            with torch.no_grad():
                torch_out = wrapped(torch.tensor(obs)).numpy()
            assert onnx_out.shape == (batch, 2)
            assert np.allclose(torch_out, onnx_out, atol=1e-4), f"batch={batch} mismatch"
            assert np.all(onnx_out >= low - 1e-5)
            assert np.all(onnx_out <= high + 1e-5)


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
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)

        result = benchmark_latency(inference, obs_shape, num_warmup=10, num_runs=50, monitor=None)

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

    def test_static_quantize_with_random_data(self, policy, obs_shape, tmp_dir):
        """N3 / M1 regression: static_quantize() with the random calibration
        path must not crash with 'Invalid rank' and must produce a loadable,
        runnable INT8 model."""
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import static_quantize
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        quantized_path = os.path.join(tmp_dir, "test_static.onnx")
        quantized_path = static_quantize(
            onnx_path, obs_shape, quantized_path, calibration_samples=8
        )
        assert os.path.exists(quantized_path)
        assert os.path.getsize(quantized_path) > 0

        inference = OnnxRuntimeInference(quantized_path)
        obs = np.random.randn(1, *obs_shape).astype(np.float32)
        result = inference.infer(obs)
        assert result.actions.shape == (1, 2)

    def test_fp16_conversion_roundtrip(self, policy, obs_shape, tmp_dir):
        """N6: FP16 conversion must produce a loadable model that runs and
        stays close to FP32."""
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.fp16 import convert_onnx_to_fp16
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        fp16_path = os.path.join(tmp_dir, "test_fp16.onnx")
        fp16_path = convert_onnx_to_fp16(onnx_path, fp16_path)

        assert os.path.exists(fp16_path)
        assert os.path.getsize(fp16_path) > 0

        orig = OnnxRuntimeInference(onnx_path)
        half = OnnxRuntimeInference(fp16_path)
        obs = np.random.randn(4, *obs_shape).astype(np.float32)
        a = orig.infer(obs).actions
        b = half.infer(obs).actions
        assert b.shape == a.shape
        # FP16 on CPU keeps IO in FP32; outputs should be very close.
        assert np.allclose(a, b, atol=1e-2)


# ============================================================
# Report generation tests
# ============================================================


class TestReportGeneration:
    def test_markdown_report_generated(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.reporter.markdown import generate_markdown_report
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
        from rl_deploy_bench.utils.platform import detect_platform

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.md")
        platform_info = detect_platform()
        output = generate_markdown_report(
            report_path, [result], ["Test Model"], platform_info=platform_info
        )

        assert os.path.exists(output)
        with open(output, "r") as f:
            content = f.read()
        assert "Latency" in content
        assert "Throughput" in content

    def test_html_report_generated(self, policy, obs_shape, tmp_dir):
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.reporter.html import generate_html_report
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference
        from rl_deploy_bench.utils.platform import detect_platform

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.html")
        platform_info = detect_platform()
        output = generate_html_report(
            report_path, [result], ["Test Model"], platform_info=platform_info
        )

        assert os.path.exists(output)
        with open(output, "r") as f:
            content = f.read()
        assert "<html" in content
        assert "plotly" in content.lower()

    def test_html_report_escapes_model_name_and_title(self, policy, obs_shape, tmp_dir):
        """S1 regression: user-controlled model names/titles must be HTML-escaped."""
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.reporter.html import generate_html_report
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.html")
        evil = "<script>alert(1)</script>"
        output = generate_html_report(report_path, [result], [evil], title=evil, platform_info=None)
        with open(output, "r", encoding="utf-8") as f:
            content = f.read()
        # Raw script tag must not appear; the escaped form must.
        assert "<script>alert(1)</script>" not in content
        assert "&lt;script&gt;" in content


# ============================================================
# Calibration data tests
# ============================================================


class TestCalibrationData:
    def test_generate_calibration_from_env(self):
        from rl_deploy_bench.benchmark.calibration import (
            CalibrationConfig,
            EnvironmentCalibrationGenerator,
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
            CalibrationConfig,
            CalibrationDataset,
            EnvironmentCalibrationGenerator,
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
            CalibrationConfig,
            EnvironmentCalibrationGenerator,
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
# CLI integration tests
# ============================================================


class TestCLICompare:
    def test_compare_html_output_is_real_html(self, policy, obs_shape, tmp_dir):
        """M3 regression: `compare ... --output report.html` must write an HTML
        document, not Markdown content disguised with a .html suffix."""
        from typer.testing import CliRunner

        from rl_deploy_bench.cli import app
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import dynamic_quantize

        fp32 = os.path.join(tmp_dir, "fp32.onnx")
        export_to_onnx(policy, obs_shape, fp32)
        quant = dynamic_quantize(fp32, os.path.join(tmp_dir, "quant.onnx"))

        runner = CliRunner()
        report = os.path.join(tmp_dir, "report.html")
        result = runner.invoke(
            app,
            [
                "compare",
                fp32,
                quant,
                "--obs-shape",
                ",".join(str(s) for s in obs_shape),
                "--num-samples",
                "20",
                "--output",
                report,
            ],
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(report)
        with open(report, "r", encoding="utf-8") as f:
            content = f.read()
        assert content.lstrip().lower().startswith("<!doctype html") or "<html" in content
        # A real HTML report contains plotly; a markdown report would not.
        assert "plotly" in content.lower()


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

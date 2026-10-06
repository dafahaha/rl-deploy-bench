"""Unit tests for rl-deploy-bench core modules.

Run with: pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

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
        # M4: exporting must not emit a torch.jit.TracerWarning. A regression to
        # the old Python-level `if low.shape[0] == 1` form gets traced as a
        # constant (the dummy input has batch=1) and warns here.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            export_to_onnx(policy, obs_shape, output_path, action_low=low, action_high=high)
        tracer = [w for w in caught if issubclass(w.category, torch.jit.TracerWarning)]
        assert not tracer, [str(w.message) for w in tracer]

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

    def test_benchmark_rejects_nonpositive_num_runs(self, obs_shape):
        """R3-吹毛-4: num_runs<=0 would yield NaN percentiles; reject early."""
        from rl_deploy_bench.benchmark.latency import benchmark_latency

        # The check runs before the inference object is touched, so a dummy is
        # enough.
        with pytest.raises(ValueError):
            benchmark_latency(object(), obs_shape, num_warmup=0, num_runs=0)

    def test_benchmark_monitor_snapshot_failure_still_stops_monitor(self, obs_shape):
        """R4-建议-1: if an interval monitor.snapshot() raises, the benchmark
        must not skip monitor.stop() (which releases NVML/Jetson handles).
        Snapshot errors are best-effort: that sample is dropped and the run
        still completes."""
        from types import SimpleNamespace

        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.monitor.base import BaseMonitor

        class _FakeInference:
            def warmup(self, num_runs, observation_shape):
                pass

            def infer(self, obs):
                result = SimpleNamespace()
                result.latency_ms = 1.0
                return result

            def get_provider_info(self):
                return {}

        class _ExplodingMonitor(BaseMonitor):
            def __init__(self):
                self.stopped = False

            def start(self):
                pass

            def stop(self):
                self.stopped = True

            def snapshot(self):
                raise RuntimeError("snapshot blew up")

        monitor = _ExplodingMonitor()
        result = benchmark_latency(
            _FakeInference(),
            obs_shape,
            num_warmup=0,
            num_runs=3,
            monitor=monitor,
            monitor_interval_ms=0.0,
        )
        # The run completed despite every interval (and final) snapshot failing.
        assert len(result.latency.latencies_ms) == 3
        # Crucially: stop() was still called even though snapshots blew up.
        assert monitor.stopped is True

    def test_benchmark_stops_monitor_when_infer_raises(self, obs_shape):
        """R4-建议-1: the try/finally must release the monitor even when
        inference.infer() raises mid-benchmark; the error still propagates."""
        from types import SimpleNamespace

        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.monitor.base import BaseMonitor

        class _ExplodingInference:
            def __init__(self):
                self.calls = 0

            def warmup(self, num_runs, observation_shape):
                pass

            def infer(self, obs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("infer blew up")
                result = SimpleNamespace()
                result.latency_ms = 1.0
                return result

            def get_provider_info(self):
                return {}

        class _RecordingMonitor(BaseMonitor):
            def __init__(self):
                self.stopped = False

            def start(self):
                pass

            def stop(self):
                self.stopped = True

            def snapshot(self):
                return SimpleNamespace()

        monitor = _RecordingMonitor()
        with pytest.raises(RuntimeError, match="infer blew up"):
            benchmark_latency(
                _ExplodingInference(),
                obs_shape,
                num_warmup=0,
                num_runs=5,
                monitor=monitor,
                monitor_interval_ms=0.0,
            )
        # Error propagated, but the monitor was still stopped.
        assert monitor.stopped is True


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

    def test_static_quantize_respects_custom_input_name(self, policy, obs_shape, tmp_dir):
        """R3-应改-1: the random-calibration path must accept the model's real
        input name, matching static_quantize_with_dataset. A model exported as
        input 'obs' must calibrate against 'obs', not the hardcoded
        'observation'."""
        from rl_deploy_bench.exporter.onnx_export import ExportConfig, export_to_onnx
        from rl_deploy_bench.quantizer.int8 import static_quantize
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test_obs.onnx")
        export_to_onnx(policy, obs_shape, onnx_path, config=ExportConfig(input_names=("obs",)))

        quantized_path = os.path.join(tmp_dir, "test_obs_static.onnx")
        quantized_path = static_quantize(
            onnx_path, obs_shape, quantized_path, calibration_samples=8, input_name="obs"
        )
        assert os.path.exists(quantized_path)

        inference = OnnxRuntimeInference(quantized_path)
        obs = np.random.randn(1, *obs_shape).astype(np.float32)
        result = inference.infer(obs)
        assert result.actions.shape == (1, 2)

    def test_shape_inferred_temp_model_cleans_up_when_save_fails(
        self, policy, obs_shape, tmp_dir, monkeypatch
    ):
        """R3-建议-3: if onnx.save raises inside the shared helper, the temp
        file it created must be unlinked before the error propagates."""
        import onnx

        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.int8 import _shape_inferred_temp_model

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        recorded = {}

        def boom(_model, path):
            recorded["path"] = path
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(onnx, "save", boom)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            _shape_inferred_temp_model(onnx_path)
        assert "path" in recorded
        assert not os.path.exists(recorded["path"])

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

    def test_fp16_manual_fallback_method2_runs(self, policy, obs_shape, tmp_dir, monkeypatch):
        """R-建议-4: when onnxruntime's float16 converter is unavailable, the
        manual Method-2 fallback must still emit a topologically valid, runnable
        model (validated by onnx.checker at the end of conversion)."""
        import onnxruntime as ort

        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.fp16 import convert_onnx_to_fp16

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        # Force `from onnxruntime.transformers.float16 import ...` to raise
        # ImportError by leaving a None entry in sys.modules.
        monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", None)

        fp16_path = os.path.join(tmp_dir, "test_fp16_m2.onnx")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = convert_onnx_to_fp16(onnx_path, fp16_path)
        assert any("manual FP16 conversion" in str(w.message) for w in caught)

        assert os.path.exists(out)
        # Must load AND run in onnxruntime, not merely pass onnx.checker.
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        obs = np.random.randn(4, *obs_shape).astype(np.float32)
        half = sess.run(None, {in_name: obs})[0]

        ref = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        full = ref.run(None, {in_name: obs})[0]
        assert half.shape == full.shape
        assert np.allclose(half, full, atol=1e-2)

    def test_fp16_manual_fallback_warns_about_op_blocklist(
        self, policy, obs_shape, tmp_dir, monkeypatch
    ):
        """R3-建议-1: the manual Method-2 fallback cannot honor op_blocklist;
        it must warn explicitly instead of silently ignoring the setting."""
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.fp16 import FP16Config, convert_onnx_to_fp16

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)

        # Force the manual fallback (see test_fp16_manual_fallback_method2_runs).
        monkeypatch.setitem(sys.modules, "onnxruntime.transformers.float16", None)

        fp16_path = os.path.join(tmp_dir, "test_fp16_bl.onnx")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_onnx_to_fp16(onnx_path, fp16_path, config=FP16Config(op_blocklist=("Softmax",)))

        msgs = [str(w.message) for w in caught]
        assert any("op_blocklist" in m and "ignored" in m for m in msgs), msgs

    def test_evaluate_fp16_impact_respects_cosine_threshold(self, policy, obs_shape, tmp_dir):
        """R4-建议-3: evaluate_fp16_impact must expose cosine_threshold (matching
        evaluate_quantization) and actually consult it when gating the verdict,
        instead of hardcoding 0.99."""
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.quantizer.fp16 import convert_onnx_to_fp16, evaluate_fp16_impact

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        fp16_path = convert_onnx_to_fp16(onnx_path, os.path.join(tmp_dir, "test_fp16.onnx"))

        # Default gate: FP16 keeps IO in FP32, so cosine similarity is very high.
        default = evaluate_fp16_impact(onnx_path, fp16_path, obs_shape, num_samples=32)
        assert default["cosine_threshold"] == 0.99
        assert default["cosine_within_threshold"] is True

        # An impossible cosine threshold must flip only the cosine gate, leaving
        # the MSE gate unchanged for the same model pair.
        strict = evaluate_fp16_impact(
            onnx_path, fp16_path, obs_shape, num_samples=32, cosine_threshold=1.5
        )
        assert strict["cosine_threshold"] == 1.5
        assert strict["cosine_within_threshold"] is False
        assert strict["mse_within_threshold"] is default["mse_within_threshold"]


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

    def test_html_summary_table_has_p90(self, policy, obs_shape, tmp_dir):
        """R-建议-1: the embedded HTML summary table must include a P90 column,
        matching the Markdown table and the percentile bar chart."""
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.reporter.html import generate_html_report
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.html")
        output = generate_html_report(report_path, [result], ["M"], platform_info=None)
        with open(output, "r", encoding="utf-8") as f:
            content = f.read()
        assert "<th>P90 (ms)</th>" in content

    def test_markdown_report_escapes_pipe_in_name(self, policy, obs_shape, tmp_dir):
        """R-吹毛-2: a model name containing '|' must be escaped so it cannot
        shift Markdown table columns."""
        from rl_deploy_bench.benchmark.latency import benchmark_latency
        from rl_deploy_bench.exporter.onnx_export import export_to_onnx
        from rl_deploy_bench.reporter.markdown import generate_markdown_report
        from rl_deploy_bench.runtime.onnx_runtime import OnnxRuntimeInference

        onnx_path = os.path.join(tmp_dir, "test.onnx")
        export_to_onnx(policy, obs_shape, onnx_path)
        inference = OnnxRuntimeInference(onnx_path)
        result = benchmark_latency(inference, obs_shape, num_warmup=5, num_runs=20)

        report_path = os.path.join(tmp_dir, "report.md")
        output = generate_markdown_report(report_path, [result], ["a|b|c"])
        with open(output, "r", encoding="utf-8") as f:
            content = f.read()
        # The escaped form must appear; the name cannot split the row into extra
        # columns.
        assert r"a\|b\|c" in content

    def test_latency_distribution_data_includes_p90(self):
        """R3-吹毛-2: the plot-data helper must expose p90 like every other
        percentile, positioned between p50 and p95."""
        from types import SimpleNamespace

        from rl_deploy_bench.reporter.markdown import generate_latency_distribution_data

        result = SimpleNamespace(
            latency=SimpleNamespace(
                latencies_ms=[1.0, 2.0],
                p50_ms=1.0,
                p90_ms=1.8,
                p95_ms=1.9,
                p99_ms=2.0,
                mean_ms=1.5,
            )
        )
        data = generate_latency_distribution_data([result], ["M"])
        assert data["M"]["p90"] == 1.8
        keys = list(data["M"])
        assert keys.index("p90") == keys.index("p50") + 1
        assert keys.index("p95") == keys.index("p90") + 1


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
# Stable Baselines3 export tests (guarded; skipped without the [sb3] extra)
# ============================================================


class TestSB3Export:
    def test_sb3_onnx_matches_predict(self, tmp_dir):
        """R-建议-2: SB3 export must NOT double-unscale continuous actions.

        ``policy._predict`` already returns actions mapped into the action space;
        re-applying the tanh-unscale produced ~2x-deviating ONNX output. This
        test trains a tiny PPO on Pendulum-v1 and checks ONNX output matches
        ``model.predict`` tightly. Skipped automatically when SB3 is absent.
        """
        pytest.importorskip("stable_baselines3")
        import warnings

        import onnxruntime as ort
        from stable_baselines3 import PPO

        from rl_deploy_bench.exporter.sb3 import export_sb3_model, verify_sb3_export

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PPO("MlpPolicy", "Pendulum-v1", verbose=0, seed=0, device="cpu")
            model.learn(total_timesteps=500)

        obs_shape = model.observation_space.shape
        onnx_path = os.path.join(tmp_dir, "sb3.onnx")
        export_sb3_model(model, onnx_path)

        rng = np.random.default_rng(0)
        test_obs = rng.standard_normal((16, *obs_shape)).astype(np.float32)
        pred = np.array([model.predict(o, deterministic=True)[0] for o in test_obs])

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_a = sess.run(None, {sess.get_inputs()[0].name: test_obs})[0]

        # A second unscale would give ~0.1+ deviation; tight match confirms the
        # action was returned as-is.
        assert np.allclose(onnx_a, pred, atol=1e-5), float(np.max(np.abs(onnx_a - pred)))

        v = verify_sb3_export(onnx_path, model, num_samples=16, atol=1e-5)
        assert bool(v["passed"]) is True, v


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

# RL-Deploy-Bench API Reference

## Table of Contents

- [Platform Detection](#platform-detection)
- [Model Export](#model-export)
- [Inference Runtime](#inference-runtime)
- [Benchmarking](#benchmarking)
- [Quantization](#quantization)
- [Calibration Data](#calibration-data)
- [Report Generation](#report-generation)
- [System Monitoring](#system-monitoring)

---

## Platform Detection

### `detect_platform() -> PlatformInfo`

Detect current platform and hardware capabilities.

```python
from rl_deploy_bench.utils.platform import detect_platform, get_monitor_backend

info = detect_platform()
print(f"OS: {info.os}, GPU: {info.gpu_name}")
backend = get_monitor_backend(info)  # 'nvidia', 'jetson', or 'cpu'
```

**PlatformInfo fields:**
- `os`: Operating system name
- `arch`: Architecture (e.g., 'AMD64', 'aarch64')
- `python_version`: Python version string
- `has_nvidia_gpu`: Boolean
- `is_jetson`: Boolean
- `gpu_name`: GPU model name (if available)
- `gpu_count`: Number of GPUs
- `cpu_count`: Number of CPU cores
- `total_memory_gb`: Total system memory in GB

---

## Model Export

### `export_to_onnx(policy, observation_shape, output_path, config=None, action_low=None, action_high=None) -> str`

Export a generic PyTorch policy to ONNX format.

```python
from rl_deploy_bench.exporter import export_to_onnx, ExportConfig

config = ExportConfig(opset_version=17, dynamic_batch=True)
onnx_path = export_to_onnx(
    policy, observation_shape=(4,), output_path="model.onnx", config=config
)
```

### `export_sb3_model(sb3_model, output_path, config=None) -> str`

Export a Stable Baselines3 model to ONNX.

```python
from rl_deploy_bench.exporter import export_sb3_model, load_sb3_model

model = load_sb3_model("ppo_model.zip", algo="PPO")
onnx_path = export_sb3_model(model, "ppo.onnx")
```

### `verify_onnx_export(onnx_path, policy, observation_shape, atol=1e-4, action_low=None, action_high=None) -> dict`

Verify that ONNX export matches PyTorch output.

If the model was exported with `action_low`/`action_high` (continuous-action
bounds), you **must pass the same bounds here**: the verification-side
wrapper applies the identical tanh-unscaling, otherwise the two sides differ by
an affine transform and verification spuriously fails.

**Returns:** `{"passed": bool, "max_abs_diff": float, "mean_abs_diff": float, ...}`

---

## Inference Runtime

### `OnnxRuntimeInference(model_path, providers=None)`

ONNX Runtime inference engine.

```python
from rl_deploy_bench.runtime import OnnxRuntimeInference

inference = OnnxRuntimeInference("model.onnx")
result = inference.infer(observation)  # InferenceResult(actions, latency_ms)
inference.warmup(num_runs=10)
```

**Methods:**
- `infer(observation) -> InferenceResult`: Run single inference
- `infer_batch(observations) -> InferenceResult`: Run batch inference
- `warmup(num_runs=10, observation_shape=None)`: Warm up the session
- `get_provider_info() -> dict`: Get active provider info

### `TensorRTEngine(engine_path=None)`

TensorRT inference engine (requires TensorRT installation).

```python
from rl_deploy_bench.runtime import TensorRTEngine, TensorRTConfig, is_tensorrt_available

if is_tensorrt_available():
    engine = TensorRTEngine()
    engine.build_from_onnx("model.onnx", "model.engine", precision="fp16")
    actions, latency = engine.infer(observation)
```

---

## Benchmarking

### `benchmark_latency(inference, observation_shape, num_warmup=50, num_runs=500, batch_size=1, monitor=None, monitor_interval_ms=100.0, seed=None) -> BenchmarkResult`

Run latency and throughput benchmark.

`monitor_interval_ms` sets the snapshot interval when a `monitor` is supplied;
`seed` fixes the synthetic observations (measured wall-clock latencies still vary
with system load).

```python
from rl_deploy_bench.benchmark import benchmark_latency

result = benchmark_latency(inference, observation_shape=(4,), num_runs=500, seed=7)
print(f"P95: {result.latency.p95_ms:.3f} ms")
print(f"Throughput: {result.latency.throughput_fps:.1f} FPS")
```

`throughput_fps` is **pure inference compute throughput**: it sums the per-run
`session.run` latency only and excludes Python scheduling, tensor assembly, and
monitor snapshot overhead, so it is systematically higher than a real end-to-end
deployment FPS.

**LatencyStats fields:**
- `mean_ms`, `std_ms`, `min_ms`, `max_ms`
- `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms`
- `throughput_fps`
- `latencies_ms`: List of all latency measurements

### `compare_actions(actions_original, actions_deployed, observations=None) -> AccuracyComparisonResult`

Compare actions from original vs deployed model (RL-specific metrics).

```python
from rl_deploy_bench.benchmark import compare_actions

result = compare_actions(original_actions, quantized_actions)
print(f"Action MSE: {result.action_mse:.8f}")
print(f"Cosine similarity: {result.action_cosine_similarity:.6f}")
```

**AccuracyComparisonResult fields:**
- `action_mse`, `action_mae`, `action_max_error`
- `action_cosine_similarity`, `action_relative_error`
- `per_dimension_mse`: List of per-dimension MSE values

---

## Quantization

### `dynamic_quantize(onnx_model_path, output_path=None, config=None) -> str`

Apply dynamic INT8 quantization (no calibration needed).

```python
from rl_deploy_bench.quantizer import dynamic_quantize

quantized_path = dynamic_quantize("model.onnx")
```

### `static_quantize(onnx_model_path, observation_shape, output_path=None, config=None, calibration_samples=100, calibration_observations=None, input_name="observation") -> str`

Apply static INT8 quantization using **synthetic calibration data**.

This is the convenient alternative to `static_quantize_with_dataset`: when you
do not have environment rollouts on hand, it synthesizes `calibration_samples`
random Gaussian observations (or uses `calibration_observations` when supplied)
to estimate activation ranges. For best accuracy on RL policies prefer
`static_quantize_with_dataset` with observations collected from the real
environment — random noise is only a stand-in.

`input_name` must match the ONNX model's actual input name. The exporter's
default input name is `"observation"`, which is also this parameter's default.
If the model was exported with a custom name (e.g.
`ExportConfig(input_names=("obs",))`), pass the same name here; a mismatch
surfaces as an onnxruntime "missing inputs" error during calibration rather
than at quantization time.

```python
from rl_deploy_bench.quantizer import static_quantize

quantized_path = static_quantize(
    "model.onnx", observation_shape=(4,), calibration_samples=100
)
# A model exported with input name "obs" must calibrate against "obs":
quantized_path = static_quantize(
    "model.onnx", observation_shape=(4,), input_name="obs"
)
```

### `static_quantize_with_dataset(onnx_model_path, calibration_dataset, output_path=None, config=None, input_name="observation") -> str`

Apply static INT8 quantization with calibration data from a CalibrationDataset.

```python
from rl_deploy_bench.quantizer import static_quantize_with_dataset

quantized_path = static_quantize_with_dataset(
    "model.onnx", calibration_dataset, input_name="observation"
)
```

### `evaluate_quantization(original_model_path, quantized_model_path, observation_shape, num_samples=500, mse_threshold=0.01, cosine_threshold=0.99) -> dict`

Evaluate quantization impact with automated recommendation.

`cosine_threshold` defaults to `0.99`, on the same pass/fail semantics as
`evaluate_fp16_impact` below (the two evaluators share identical MSE / cosine
gates).

```python
from rl_deploy_bench.quantizer import evaluate_quantization

result = evaluate_quantization("model.onnx", "model_int8.onnx", observation_shape=(4,))
print(f"Verdict: {result['verdict']}")  # 'pass', 'caution', or 'fail'
print(f"Recommendation: {result['recommendation']}")
```

### `quantize_and_evaluate(onnx_model_path, observation_shape, output_dir=None, mode="dynamic", calibration_dataset=None, num_eval_samples=500, mse_threshold=0.01) -> dict`

One-click quantization and evaluation. Quantizes the ONNX model, evaluates the
accuracy impact, and returns the evaluation dictionary (same shape as
`evaluate_quantization`).

### FP16

#### `FP16Config(keep_io_in_fp32=True, convert_weights=True, op_blocklist=())`

Configuration for ONNX → FP16 conversion.

- `keep_io_in_fp32`: keep model inputs/outputs in FP32 (Cast nodes inserted
  around the FP16 compute graph)
- `convert_weights`: convert FP32 initializer weights to FP16
- `op_blocklist`: tuple of op types to keep in FP32 (e.g. `("Softmax", "Exp")`)
  — honored by the onnxruntime converter; the manual fallback warns and ignores
  it

#### `convert_onnx_to_fp16(onnx_model_path, output_path=None, config=None) -> str`

Convert an FP32 ONNX model to FP16. Prefers onnxruntime's `convert_float_to_float16`
and falls back to a manual rewrite (validated with `onnx.checker`) when that
converter is unavailable.

```python
from rl_deploy_bench.quantizer import FP16Config, convert_onnx_to_fp16

fp16_path = convert_onnx_to_fp16("model.onnx")
fp16_path = convert_onnx_to_fp16(
    "model.onnx", config=FP16Config(op_blocklist=("Softmax",))
)
```

#### `evaluate_fp16_impact(original_model_path, fp16_model_path, observation_shape, num_samples=500, mse_threshold=0.01, cosine_threshold=0.99) -> dict`

Evaluate FP16 conversion impact with an automated pass/caution/fail verdict,
same metrics and thresholds as `evaluate_quantization` (`cosine_threshold`
defaults to `0.99` so the two evaluators share pass/fail semantics).

```python
from rl_deploy_bench.quantizer import evaluate_fp16_impact

result = evaluate_fp16_impact("model.onnx", "model_fp16.onnx", observation_shape=(4,))
print(result["verdict"], result["recommendation"])
```

---

## Calibration Data

### `EnvironmentCalibrationGenerator(env_name, policy=None, config=None)`

Generate calibration data from a Gymnasium environment.

```python
from rl_deploy_bench.benchmark import (
    EnvironmentCalibrationGenerator, CalibrationConfig
)

config = CalibrationConfig(num_samples=500, collection_strategy="policy")
generator = EnvironmentCalibrationGenerator("Pendulum-v1", policy=policy_fn, config=config)
dataset = generator.generate()
dataset.save("calibration.npz")
```

### `SB3PolicyCalibrationGenerator(env_name, sb3_model, config=None, deterministic=True)`

Calibration generator using a Stable Baselines3 trained policy.

### `CalibrationDataset`

Dataset of calibration observations with save/load support.

```python
from rl_deploy_bench.benchmark import CalibrationDataset

dataset = CalibrationDataset.load("calibration.npz")
stats = dataset.get_statistics()
```

**Methods:**
- `save(path) -> str`: Save to .npz
- `load(path) -> CalibrationDataset`: Class method to load
- `get_statistics() -> dict`: Get dataset statistics

---

## Report Generation

### `generate_markdown_report(output_path, benchmark_results, model_names, accuracy_results=None, model_paths=None, platform_info=None, title="...") -> str`

Generate a Markdown benchmark report.

```python
from rl_deploy_bench.reporter import generate_markdown_report

report_path = generate_markdown_report(
    "report.md",
    [fp32_result, int8_result],
    ["FP32", "INT8"],
    accuracy_results=[None, int8_accuracy],
    platform_info=platform_info,
)
```

### `generate_html_report(output_path, benchmark_results, model_names, accuracy_results=None, platform_info=None, title="...") -> str`

Generate an interactive HTML report with Plotly charts.

---

## System Monitoring

### `create_monitor(backend=None, gpu_index=0) -> BaseMonitor`

Create the appropriate monitor for the current platform (auto-detected).

```python
from rl_deploy_bench.monitor import create_monitor

monitor = create_monitor()  # Auto-detects NVIDIA GPU, Jetson, or CPU
with monitor:
    # run benchmark
    snapshot = monitor.snapshot()  # SystemMetrics
```

**Monitor implementations:**
- `NvidiaGPUMonitor`: Uses pynvml for x86 NVIDIA GPUs
- `JetsonMonitor`: Uses jetson-stats for NVIDIA Jetson
- `CPUMonitor`: Uses psutil for CPU-only systems

**SystemMetrics fields:**
- `gpu_utilization`, `gpu_memory_used_mb`, `gpu_memory_total_mb`
- `gpu_power_w`, `gpu_temperature_c`
- `cpu_utilization`, `cpu_memory_used_mb`, `cpu_memory_total_mb`

---

## CLI Usage

```bash
# Show platform info
rl-deploy-bench info

# Export SB3 model
rl-deploy-bench export model.zip --output model.onnx --algo PPO

# Quantize
rl-deploy-bench quantize model.onnx --mode dynamic

# Benchmark
rl-deploy-bench benchmark model.onnx --obs-shape 4 --num-runs 500 --output report.md

# Compare FP32 vs INT8
rl-deploy-bench compare model.onnx model_int8.onnx --obs-shape 4 --output comparison.html
```

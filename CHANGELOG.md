# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-10-07

### Fixed
- **`static_quantize()` random-calibration path crashed**: `CalibrationDataReader.get_next()` returned observations without a leading batch dimension, which ONNX Runtime's static quantizer rejected with `Invalid rank: Got 1 Expected 2`. Calibration samples now carry the batch dim (`obs[np.newaxis]`), matching `_DatasetReader`.
- **`verify_onnx_export()` false-failure with action bounds**: the verification-side `OnnxablePolicy` was built without `action_low`/`action_high`, so PyTorch produced raw tanh outputs while the ONNX graph applied the affine unscaling (max diff ~0.43 on Pendulum-style bounds). `verify_onnx_export()` now accepts and applies `action_low`/`action_high`.
- **CLI `compare --output report.html` wrote Markdown content into a `.html` file**: the `compare` command now branches on the output extension exactly like `benchmark`, producing real HTML when the path ends in `.html`.
- **ONNX export TracerWarning / broken dynamic batch with action bounds**: `OnnxablePolicy.forward()` contained a Python-level `if low.shape[0] == 1 and action.shape[0] > 1`, which `torch.onnx.export` traced as a constant from the batch=1 dummy input. Bounds are now unconditionally `expand()`ed to the runtime batch size; batch=1 and batch>1 inference both match PyTorch.
- **HTML report HTML-injection / XSS**: `model_names`, `title`, and `platform_info` strings are now passed through `html.escape()` before being interpolated into the report.
- **FP16 conversion silently swallowed all exceptions**: `except (ImportError, Exception): pass` masked real converter failures. Now only `ImportError` (converter unavailable) warns and falls back to the manual path; any other converter error is re-raised as `RuntimeError`.
- **`np.load(allow_pickle=True)` on calibration datasets**: datasets are saved as plain `.npz` arrays; `allow_pickle=False` avoids deserializing arbitrary objects.
- **CLI `compare` divide-by-zero on degenerate models**: percentage-change cells now render `n/a` when the denominator latency/throughput is zero instead of raising `ZeroDivisionError`.
- **Markdown report title injection**: the top-level `# {title}` now goes through the same pipe-escaping/newline-folding as model names, so a newline in the title can no longer inject an arbitrary Markdown section at column 0.
- **NVML session leak on `NvidiaGPUMonitor.start()` failure**: when `nvmlInit()` succeeded but `nvmlDeviceGetHandleByIndex()` raised, the handle error path now calls `pynvml.nvmlShutdown()` before re-raising, so the NVML session is not left open.

### Changed
- `rl_deploy_bench.__version__` is now `1.1.0`, matching `pyproject.toml`.
- Removed phantom `jinja2` dependency from `dependencies` (HTML reports are f-string generated; Jinja2 was never imported).
- TensorRT backend now caches and reuses device buffers across `infer()` calls instead of `cuda.mem_alloc`/`free`-ing on every call; buffers are released in `close()`/`__del__`.
- `benchmark_latency()` accepts an optional `seed` for reproducible benchmark observation data (default behavior unchanged when omitted).
- Markdown report latency table now includes a P90 column, matching the CLI table and HTML chart.
- Top-level `rl_deploy_bench` package now re-exports the public Python API (`export_to_onnx`, `dynamic_quantize`, `static_quantize`, `convert_onnx_to_fp16`, `benchmark_latency`, `generate_html_report`, `OnnxRuntimeInference`, etc.) as shown in the README.
- TorchScript non-ASCII path fallback now copies to a unique temporary file (cleaned up after load) instead of a fixed shared path.
- `TorchScriptInference` now implements the unified inference duck-interface: `infer()` returns `InferenceResult` (`.actions` / `.latency_ms`) and `get_provider_info()` exists, so it can be passed to `benchmark_latency()` and the accuracy/quantization evaluators like `OnnxRuntimeInference`.
- TorchScript export (`torch.jit.trace`/`freeze`/`optimize_for_inference`) is documented as targeting **legacy PyTorch** deployments: on Python 3.14+ PyTorch emits `FutureWarning`; the forward direction is `torch.export` / `torch.compile`, and the ONNX export path is recommended for new projects.
- Removed the `/sys/class/gpio/export` heuristic from Jetson detection (it exists on many non-Jetson Linux systems).
- API documentation now covers the TorchScript export/inference surface (`export_to_torchscript`, `verify_torchscript_export`, `TorchScriptInference`, `compare_onnx_torchscript`) and the `OnnxRuntimeInference(provider_options=...)` argument.
- CI no longer runs `mypy src/` as a dead step (it currently reports ~48 pre-existing typing errors); the `[tool.mypy]` config remains available for local opt-in runs.
- `torchscript_export` now imports `time` at module top instead of inside `TorchScriptInference.infer()`.

## [1.0.0] - 2026-08-31

### Added
- **TorchScript export and inference**: Export PyTorch models to TorchScript for Python-free deployment (LibTorch C++ runtime, mobile, embedded). Includes TorchScriptInference runtime and ONNX vs TorchScript output comparison.
- **FP16 quantization**: Convert ONNX models to FP16 precision with automatic Cast node insertion. FP16 provides near-lossless accuracy with significant speedup on GPUs with native FP16 support (Jetson Xavier/Orin, RTX 20-series+). Includes FP16 impact evaluation and supported GPU list.
- **Quantization Decision Guide example**: 3rd example demonstrating systematic evaluation of FP32/FP16/INT8-dynamic/INT8-static and automated recommendation based on latency, accuracy, and size.
- **CLI `calibrate` command**: Generate calibration data from Gymnasium environments for static quantization.
- **CLI `quantize --calibration-file`**: Support environment-based calibration datasets for static quantization.

### Changed
- Version bumped to 1.0.0 (first stable release)
- Improved static quantization with environment-based calibration datasets
- Enhanced CLI output with richer formatting and better error messages

### Fixed
- TorchScript model loading with non-ASCII (Chinese) paths on Windows
- HTML/Markdown report crash when accuracy_results contains None values
- ONNX export with dynamic batch dimension
- Quantization temporary file path issues

## [0.2.0] - 2026-08-31

### Added
- **Environment calibration data generator**: Generate realistic calibration data from Gymnasium environments for better static quantization results
  - `EnvironmentCalibrationGenerator`: Random or policy-guided data collection
  - `SB3PolicyCalibrationGenerator`: Use trained SB3 models for policy-guided collection
  - `CalibrationDataset`: Save/load calibration datasets with statistics
  - 3 collection strategies: random, policy, mixed
- **Static quantization with calibration datasets**: `static_quantize_with_dataset()` uses environment-based calibration data
- **Quantization impact evaluation**: `evaluate_quantization()` automatically assesses quantization quality with pass/caution/fail verdict and recommendations
- **One-click quantization and evaluation**: `quantize_and_evaluate()`
- **TensorRT inference backend framework**: ONNX to TensorRT Engine conversion with FP16/INT8 support, Jetson DLA support, graceful fallback when TensorRT not installed
- **End-to-end benchmark example**: Complete 8-step workflow (train → export → calibrate → quantize → benchmark → compare → report)
- **CLI `calibrate` command**: Generate calibration data from Gymnasium environments
- **CLI `quantize` command**: Now supports `--calibration-file` for environment-based static quantization
- **Unit test suite**: 22 tests covering all core modules
- **GitHub Actions CI**: Automated testing and linting on Python 3.10/3.11/3.12
- **API documentation**: Complete API reference in API.md
- **Contributing guidelines**: CONTRIBUTING.md
- **MIT License**

### Changed
- Improved static quantization to support environment-based calibration datasets
- Enhanced CLI with richer output and better error messages

### Fixed
- HTML report crash when accuracy_results contains None values
- Markdown report crash when accuracy_results contains None values
- ONNX export with dynamic batch dimension
- Quantization temporary file path issues

## [0.1.0] - 2026-08-31

### Added
- Initial release
- **Platform detection**: Auto-detect x86 NVIDIA GPU, Jetson, and CPU-only platforms
- **System monitoring**: Abstract monitor interface with NVIDIA GPU (pynvml), Jetson (jetson-stats), and CPU (psutil) backends
- **Model export**: Generic PyTorch to ONNX export with verification, Stable Baselines3 dedicated export
- **Inference runtime**: ONNX Runtime backend with automatic provider selection
- **Latency benchmark**: P50/P90/P95/P99 latency, throughput, system metrics collection
- **RL-specific accuracy comparison**: Action MSE/MAE/Max Error, Cosine Similarity, Relative Error, per-dimension MSE
- **INT8 quantization**: Dynamic and static quantization with calibration data reader
- **Report generation**: Markdown and interactive HTML reports with Plotly charts
- **CLI**: 5 commands (info, export, quantize, benchmark, compare)
- **Cross-platform support**: x86 NVIDIA GPU + Jetson + CPU-only

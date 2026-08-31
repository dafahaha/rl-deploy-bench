# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

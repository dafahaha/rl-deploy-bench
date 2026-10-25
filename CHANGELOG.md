# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **README install section pointed at a nonexistent PyPI package**: the first install command was `pip install rl-deploy-bench`, but the package has never been published to PyPI (the PyPI JSON API and the simple index both return 404 for `rl-deploy-bench`), while the roadmap had checked off "v1.1 — PyPI release". The install section now says so explicitly and gives the working source-install path (`git clone https://github.com/dafahaha/rl-deploy-bench.git` → `pip install .` / `pip install -e .`, with the extras rewritten as `pip install ".[nvidia]"` / `.[jetson]` / `.[sb3]`). The roadmap entry is split: v1.1 keeps the checked items that actually shipped (CLI, demo scripts, templates) as a source/GitHub release, and a separate unchecked "v1.x — PyPI release" row tracks the not-yet-done work. `examples/readme_benchmark.py` docstring updated to match.
- **README benchmark note misstated the demo policy size**: the "How to read this table" note called the demo policy "~85k params"; the actual 4→128→64→2 MLP (`demo.py` / `examples/readme_benchmark.py` `DemoPolicy`) has 9,026 parameters (Linear 4·128+128=640, 128·64+64=8,256, 64·2+2=130; total 9,026), verified by `sum(p.numel() for p in policy.parameters())`. The note now reads "9,026 params: 4→128→64→2 MLP".

### Changed
- **TorchScript export crashed on non-ASCII Windows output paths**: `torch.jit.save` raised a misleading "Parent directory ... does not exist" even after `makedirs` succeeded. The write side now saves to a uniquely named ASCII temp file and `shutil.move`s it onto the target (symmetric with the existing load-side fallback); the optimize step loads through the same safe helper.
- **CLI `compare` printed no verdict**: it now calls `evaluate_quantization` and prints the pass/caution/fail verdict and recommendation in the summary instead of raw MSE/cosine alone.
- **CLI static quantization hardcoded input name `observation`**: `quantize --mode static` now reads the model's actual ONNX input name by default and accepts `--input-name` to override, so custom input names (e.g. `obs`) calibrate against the correct feed.
- **FP16 manual fallback `onnx.checker` had no direct guard**: the checker call is now asserted by a test, and a checker failure propagates instead of being saved unvalidated.
- **HTML report claimed self-contained but loaded Plotly from the CDN**: Plotly is now inlined (`include_plotlyjs=True`) so charts render offline.
- **Zero-vector / constant-action edge cases in `compare_actions`**: two identical all-zero actions now give cosine 1.0 (not 0.0); zero-variance constant actions with exact match give relative_error 0.0 (not inf).
- **TorchScript optimization failure was swallowed by a bare `except: pass`**: it now emits a warning and keeps the unoptimized model.
- **Non-ASCII TorchScript load fallback triggered only on the literal "No such file or directory" message**: it now retries via the ASCII temp copy on any RuntimeError for a non-ASCII path.
- **`num_episodes=0` / empty calibration dataset delivered silently**: both now emit a warning.
- **Flaky pass/caution verdict in CI**: `evaluate_quantization` / `evaluate_fp16_impact` drew their synthetic test observations from the global numpy RNG without a seed, so a near-threshold run could move the MSE/cosine gates and flip the verdict between runs. Both evaluators now accept a `seed` argument (default 0) and `generate_test_observations` supports a private seeded generator; the test `policy` fixture seeds torch as well. The MSE/cosine values and verdict are now byte-for-byte reproducible across runs and Python versions.
- **Intermittent failures from CPU scheduling on the shared runner**: added `tests/conftest.py`, which seeds numpy/torch and pins torch to a single thread, and the CI workflow now sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS=1`. This removes thread-pool contention on the 2-core runner as a source of non-determinism.
- **Remaining unseeded paths in the `compare` command**: the manual inference observations and both latency benchmarks were still drawn from the global RNG. They are now seeded (seed=0), so the entire `compare` command — including the printed table and standalone verdict — is reproducible.
- **ONNX Runtime's own thread pool still used all cores**: unlike the BLAS/torch threads pinned above, the per-session intra/inter-op pools defaulted to the runner's core count, so multi-threaded INT8 MatMul/QDQ reductions could still change the action output and flip the verdict. `OnnxRuntimeInference` now accepts `num_threads` and reads `RL_DEPLOY_BENCH_ORT_THREADS`; the conftest and CI pin it to 1.
- **Root cause of the flaky INT8 verdict: VPMADDUBSW saturation on non-VNNI x86 CPUs**. Dynamic quantization emits a uint8-activation × int8-weight `MatMulInteger` (QOperator). On x86-64 CPUs without VNNI (e.g. AMD Zen 3 / EPYC 7763, older AVX2), the U8S8 kernel uses the `VPMADDUBSW` instruction, whose adjacent-pair sum saturates (clamps) to int16 and silently corrupts the int32 accumulation. A byte-identical model therefore gave action MSE 1.8e-6 (PASS) on VNNI/AVX512 runners but 3.4e-3 (CAUTION) on EPYC 7763; the failure drifted across the Python matrix because matrix jobs land on different physical hosts. `QuantizationConfig.reduce_range` now defaults to `True` (7-bit weights), which keeps the worst-case `VPMADDUBSW` pair (255×64×2 = 32640) inside int16 so the result is correct on every x86. The seed/thread fixes above are retained as defense in depth. Reference: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- **Additional hardening from the independent adversarial review**: the three surviving mutations (test gaps) are now closed — the Markdown latency table row, the inlined Plotly bundle, and the standalone CLI verdict are each asserted directly; the CLI `compare` reuses its computed accuracy result instead of re-running the evaluation; the one-click `quantize_and_evaluate` forwards the model's real input name; the manual FP16 rewrite raises a clear error when an output has no producing node; the SB3 auto-infer narrows its exception to load/format mismatches; the Jetson monitor falls back to CPU when `jtop` is absent; the TorchScript warmup derives its shape from the captured input; and the Markdown platform fields are pipe-escaped. Python 3.13 is now in the CI matrix and classifiers. Test count: 79.

### Changed
- Verdict wording (`evaluate_quantization` / `evaluate_fp16_impact`) and README/API now state the boundary: a pass/caution/fail verdict reflects action deviation on synthetic observations only, not closed-loop task success; real-environment rollouts are required before deployment.
- README platform table marks TensorRT as **experimental** (standalone framework, not yet wired into the unified benchmark duck-interface) instead of a certified ✅.
- FutureWarning attribution corrected: the legacy `torch.jit` warning is driven by the PyTorch version (observed on PyTorch 2.14 / CPython 3.13), not gated on Python 3.14.
- `evaluate_quantization_impact` (superseded binary-MSE gate) is marked deprecated in its docstring; kept for backward compatibility.

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

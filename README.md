<div align="center">

# RL-Deploy-Bench

**Cross-platform RL model deployment & performance benchmarking toolkit**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red)](https://pytorch.org/)
[![CI](https://github.com/dafahaha/rl-deploy-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/dafahaha/rl-deploy-bench/actions)

**Export → Quantize → Benchmark → Compare → Report** — in one toolkit

[Quick Start](#-quick-start) · [Documentation](API.md) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

---

## 🎯 Why RL-Deploy-Bench?

Reinforcement learning models have unique deployment challenges that generic ML tools ignore:

- **Action-level accuracy matters** — small action deviations can cause policy failure
- **Observation distributions are environment-specific** — random calibration data produces poor quantization
- **Edge deployment is common** — Jetson, Raspberry Pi, embedded GPUs need cross-platform tools
- **Latency is critical** — real-time control requires sub-millisecond inference

RL-Deploy-Bench is built **specifically for RL deployment**, with RL-specific metrics, environment-based calibration, and cross-platform support from day one.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-format Export** | ONNX, TorchScript, Stable Baselines3 dedicated export |
| **3-level Quantization** | FP16, INT8 Dynamic, INT8 Static (with environment calibration) |
| **RL-specific Metrics** | Action MSE/MAE/Max Error, Cosine Similarity, Per-dimension analysis |
| **Latency Benchmark** | P50/P90/P95/P99 percentiles, throughput, system metrics (GPU/CPU/memory) |
| **Cross-platform** | x86 NVIDIA GPU, NVIDIA Jetson, CPU-only — auto-detection |
| **Auto Evaluation** | Quantization impact assessment with pass/caution/fail verdict & recommendations |
| **Rich Reports** | Markdown + interactive HTML with Plotly charts (6 visualization types) |
| **CLI & Python API** | 6 CLI commands + full Python API for integration |
| **Calibration Generator** | Gymnasium environment data collection for better static quantization |

## 🚀 Quick Start

### Install

> **Status — not on PyPI yet.** The `rl-deploy-bench` package is **not
> published to PyPI** (the PyPI JSON API and the simple index both return 404
> for `rl-deploy-bench`). Until a PyPI release is cut, install from the source
> repository:

```powershell
git clone https://github.com/dafahaha/rl-deploy-bench.git
cd rl-deploy-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install .
```

For an editable / development install, replace the last line with:

```powershell
pip install -e .
```

Optional extras (GPU / Jetson / Stable Baselines3 support):

```powershell
pip install ".[nvidia]"     # NVIDIA GPU monitoring (pynvml)
pip install ".[jetson]"     # Jetson monitoring (jetson-stats)
pip install ".[sb3]"        # Stable Baselines3 + Gymnasium support
```

### 3-Step Workflow

```bash
# 1. Export your trained RL model to ONNX
rl-deploy-bench export model.zip --output policy.onnx --algo PPO

# 2. Quantize to INT8 with environment calibration
rl-deploy-bench calibrate Pendulum-v1 --output calib.npz --num-samples 500
rl-deploy-bench quantize policy.onnx --mode static --calibration-file calib.npz

# 3. Benchmark & compare — generates full HTML report
rl-deploy-bench compare policy.onnx policy_int8.onnx --obs-shape 3 --output report.html
```

### One-Command Demo

```bash
python demo.py
```

No GPU or trained model required — runs a complete 6-step workflow in 30 seconds and generates an interactive HTML report.

### Python API

```python
from rl_deploy_bench import (
    export_to_onnx, dynamic_quantize, benchmark_latency,
    evaluate_quantization, generate_html_report, OnnxRuntimeInference
)

# Export
onnx_path = export_to_onnx(model, observation_shape=(4,))

# Quantize
int8_path = dynamic_quantize(onnx_path)

# Benchmark
inference = OnnxRuntimeInference(int8_path)
result = benchmark_latency(inference, (4,), num_runs=500)

# Evaluate accuracy impact
eval_result = evaluate_quantization(onnx_path, int8_path, (4,))
print(f"Verdict: {eval_result['verdict']}")  # pass / caution / fail

# Generate report
generate_html_report("report.html", [result], ["INT8 Model"])
```

## 📊 Benchmark Example Output

> **Real measured output — not illustrative numbers.** The table below is produced
> by running `python examples/readme_benchmark.py`, which drives this toolkit's
> own export → quantize → benchmark pipeline on a small synthetic MLP policy
> (4→128→64→2, the same architecture as `demo.py`). Raw run log:
> [`docs/readme_benchmark_output.txt`](docs/readme_benchmark_output.txt).
> Measured on Windows (Python 3.13.16, ONNX Runtime 1.31.0, single intra-op
> thread to reduce scheduling jitter) on 2026-10-26. Latency numbers are
> machine- and load-dependent — re-run the script on your own hardware before
> quoting them as your own results.

```
Model                              Mean(ms)  P95(ms)   FPS       Action MSE   Verdict
--------------------------------------------------------------------------------------
FP32 (Original)                     0.019     0.037     51486.1   0.00000000   baseline
FP16                                0.047     0.085     21465.2   0.00000000   pass
INT8 Dynamic                        0.064     0.130     15676.3   0.00000250   pass
INT8 Static (random-norm calib.)    0.044     0.109     22781.1   0.00000548   pass
```

> **How to read this table (honesty notes)**:
> - On this machine the quantized variants are *slower* than FP32. The policy is
>   tiny (9,026 params: 4→128→64→2 MLP), so INT8 MatMulInteger and manual FP16
>   conversion overhead dominates; speedups are expected only on larger models
>   or GPU execution. P95 ≥ Mean ≥ P50 holds in every row, as it must for a
>   real latency distribution.
> - "INT8 Static (random-norm calib.)" is calibrated on random-normal
>   observations, **not** environment data. True environment-based calibration
>   (Pendulum-v1) needs the `[sb3]` + gymnasium extras; see
>   `examples/end_to_end_benchmark.py`.

> **Throughput (FPS) 口径**: 表中 FPS 为纯推理算力吞吐（仅累计每次 `session.run` 的内部时延），**不含** Python 调度、数据拼装与监控快照等循环开销，因此高于真实端到端链路 FPS，跨模型比较时口径一致。

## 🖥️ Supported Platforms

| Platform | Monitoring | ONNX Runtime | TensorRT | FP16 | INT8 |
|----------|-----------|--------------|----------|------|------|
| **x86 NVIDIA GPU** | ✅ pynvml | ✅ CUDA/CPU | ⚠️ Experimental | ✅ | ✅ |
| **NVIDIA Jetson** | ✅ jetson-stats | ✅ CPU/CUDA | ⚠️ Experimental | ✅ | ✅ |
| **CPU-only** | ✅ psutil | ✅ CPU | ❌ | ⚠️ Slow | ✅ |
| **Apple Silicon** | ⚠️ Limited | ⚠️ CoreML (via ORT auto-EP, not explicitly adapted) | ❌ | ✅ | ⚠️ |

> **TensorRT maturity**: the TensorRT backend is an **experimental standalone
> framework**, not a certified integration. It does not yet implement the
> unified benchmark duck-interface (`warmup()` / `get_provider_info()`), so a
> TensorRT engine cannot be fed to `benchmark_latency` alongside ONNX /
> TorchScript for an apples-to-apples comparison, and it has no GPU-tested
> benchmark here. Full integration is tracked for **v2.0** on the roadmap.

## 🔄 Quantization Decision Guide

```
Start with FP32 baseline
    │
    ▼
Try FP16 first
    ├─ pass → Use FP16 (best speed/accuracy tradeoff on GPU)
    └─ fail → Try INT8
              │
              ▼
         Try INT8 Dynamic
              ├─ pass → Use INT8 Dynamic (no calibration needed)
              └─ fail → Try INT8 Static
                        │
                        ▼
                   Try INT8 Static + Environment Calibration
                        ├─ pass → Use INT8 Static (best INT8 accuracy)
                        └─ fail → Stay with FP32 or selective quantization
```

> **Verdict scope (read before trusting a `pass`)**: the pass/caution/fail
> verdict is computed on **synthetic random-normal observations**, not on
> environment rollouts. It only means the action MSE / cosine-similarity gates
> pass on those synthetic inputs; it does **not** represent closed-loop task
> success or preserved policy performance. A `pass` does not guarantee the
> deployed policy still reaches the task goal — always re-validate with
> real-environment rollouts before deploying.

## 📁 Project Structure

```
rl_deploy_bench/
├── cli.py                  # 6 CLI commands
├── utils/
│   └── platform.py         # Auto platform detection
├── monitor/                # System monitoring (3 backends)
│   ├── nvidia_gpu.py
│   ├── jetson.py
│   └── cpu.py
├── exporter/               # Model export (ONNX, TorchScript, SB3)
│   ├── onnx_export.py
│   ├── torchscript_export.py
│   └── sb3.py
├── runtime/                # Inference runtimes
│   ├── onnx_runtime.py
│   └── tensorrt_runtime.py
├── benchmark/              # Benchmarking & calibration
│   ├── latency.py
│   ├── accuracy.py
│   └── calibration.py
├── quantizer/              # Quantization (FP16, INT8)
│   ├── fp16.py
│   └── int8.py
└── reporter/               # Report generation
    ├── markdown.py
    └── html.py
```

> **Note on TorchScript export**: `export_to_torchscript()` builds on the
> legacy `torch.jit.trace`/`script`/`freeze` stack. It targets legacy PyTorch
> deployments (LibTorch C++, mobile, embedded). Recent PyTorch versions (2.9+;
> observed on PyTorch 2.14 already on CPython 3.13) emit a `FutureWarning`
> ("torch.jit ... may break, switch to torch.compile or torch.export"); the
> warning is driven by the **PyTorch version**, not the Python version, so it
> appears well before Python 3.14. The current implementation still works, but
> the forward direction is `torch.export` / `torch.compile` rather than further
> investment in TorchScript. New projects should prefer the ONNX export path.

## 🗺️ Roadmap

- [x] v1.0 — Core: ONNX/TorchScript export, FP16/INT8 quantization, latency benchmark, accuracy comparison, reports
- [x] v1.1 — improved CLI, demo scripts, community templates (source/GitHub release; **not yet on PyPI**)
- [ ] v1.x — PyPI release (tracked; install from source until then)
- [ ] v1.2 — ManiSkill/Gibson environment integration, multi-GPU benchmarking
- [ ] v2.0 — TensorRT full integration, distributed benchmarking, model zoo

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a Pull Request

## Related research notes

A static, auditable write-up of the deployment pitfalls fixed in this toolkit
(calibration dry episodes, action-bounds affine mismatch, static-quantization
rank errors, swallowed FP16 exceptions, monitor leaks, report injection) lives
in the companion notes site:
[`dafahaha/embodied-offpolicy-study` · `docs/`](https://github.com/dafahaha/embodied-offpolicy-study/tree/main/docs).

## 📝 Citation

If you use RL-Deploy-Bench in your research, please cite:

```bibtex
@software{rl_deploy_bench,
  author = {dafahaha},
  title = {RL-Deploy-Bench: Cross-platform RL model deployment and performance benchmarking toolkit},
  year = {2026},
  url = {https://github.com/dafahaha/rl-deploy-bench}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built for RL researchers and engineers deploying models to the edge.**

[⭐ Star this repo](https://github.com/dafahaha/rl-deploy-bench) · [🐛 Report an issue](https://github.com/dafahaha/rl-deploy-bench/issues)

</div>

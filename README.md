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

```bash
pip install rl-deploy-bench

# Optional extras
pip install rl-deploy-bench[nvidia]    # NVIDIA GPU monitoring
pip install rl-deploy-bench[jetson]    # Jetson monitoring
pip install rl-deploy-bench[sb3]       # Stable Baselines3 support
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

```
Model                          Mean(ms)   P95(ms)    FPS        Action MSE   Verdict
------------------------------------------------------------------------------------
FP32 (Original)                0.073      0.071      13699.8    0.00000000   baseline
FP16                           0.068      0.065      14705.9    0.00000001   pass
INT8 Dynamic                   0.346      0.188      2892.6     0.00000851   pass
INT8 Static (Calibrated)       0.410      0.286      2436.7     0.00001319   pass
```

## 🖥️ Supported Platforms

| Platform | Monitoring | ONNX Runtime | TensorRT | FP16 | INT8 |
|----------|-----------|--------------|----------|------|------|
| **x86 NVIDIA GPU** | ✅ pynvml | ✅ CUDA/CPU | ✅ Framework | ✅ | ✅ |
| **NVIDIA Jetson** | ✅ jetson-stats | ✅ CPU/CUDA | ✅ Pre-installed | ✅ | ✅ |
| **CPU-only** | ✅ psutil | ✅ CPU | ❌ | ⚠️ Slow | ✅ |
| **Apple Silicon** | ⚠️ Limited | ✅ CoreML | ❌ | ✅ | ⚠️ |

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

## 🗺️ Roadmap

- [x] v1.0 — Core: ONNX/TorchScript export, FP16/INT8 quantization, latency benchmark, accuracy comparison, reports
- [x] v1.1 — PyPI release, improved CLI, demo scripts, community templates
- [ ] v1.2 — ManiSkill/Gibson environment integration, multi-GPU benchmarking
- [ ] v2.0 — TensorRT full integration, distributed benchmarking, model zoo

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a Pull Request

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

[⭐ Star this repo](https://github.com/dafahaha/rl-deploy-bench) · [🐛 Report an issue](https://github.com/dafahaha/rl-deploy-bench/issues) · [💬 Start a discussion](https://github.com/dafahaha/rl-deploy-bench/discussions)

</div>

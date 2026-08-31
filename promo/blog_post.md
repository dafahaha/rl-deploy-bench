# 为什么我做了一个专门给RL模型用的部署基准工具？

> 从一个痛点出发，到一个完整的开源项目

## 背景：RL部署的"最后一公里"难题

做强化学习的同学可能都有过这样的经历：在仿真环境里训练了一个效果不错的策略，等到真正要部署到机器人、Jetson或者边缘设备上时，问题就来了：

1. **模型导出各种坑** — PyTorch模型转ONNX，动态shape、控制流、observation normalization，每一步都可能踩雷
2. **量化不知道效果如何** — INT8量化后策略会不会崩？通用ML工具只看分类准确率，RL需要看action-level的偏差
3. **校准数据用随机噪声** — 通用量化工具用随机数据做校准，但RL模型的观测分布是环境相关的，随机数据校准出来的模型根本不能用
4. **延迟测试不专业** — 只测mean latency，不看P99，不采集GPU/CPU指标，不知道瓶颈在哪
5. **跨平台痛苦** — x86 GPU、Jetson、CPU-only，每个平台的监控和推理后端都不一样

我找了一圈，发现**没有一个工具是专门为RL模型部署设计的**。现有的工具要么是通用ML基准（只看分类/检测），要么是LLM部署工具（和RL完全不相关），要么是某个特定框架的导出脚本（不完整）。

于是我决定自己做一个：**RL-Deploy-Bench**。

## RL-Deploy-Bench是什么？

一句话：**专门为强化学习模型设计的跨平台部署与性能基准测试工具包**。

核心工作流就5步：

```
Export → Quantize → Benchmark → Compare → Report
```

### 1. 多格式导出

支持ONNX、TorchScript、Stable Baselines3专用导出，带自动验证（导出后自动对比PyTorch输出，确保导出正确）。

```python
from rl_deploy_bench import export_to_onnx, verify_onnx_export

onnx_path = export_to_onnx(model, observation_shape=(4,))
result = verify_onnx_export(onnx_path, model, (4,))
print(result["passed"])  # True/False
```

### 2. 三级量化 + RL特有精度评估

支持FP16、INT8动态、INT8静态三种量化方式。关键是：**评估的是action-level的精度**，而不是分类准确率。

```python
from rl_deploy_bench import evaluate_quantization

result = evaluate_quantization(fp32_path, int8_path, observation_shape=(4,))
print(result["verdict"])  # "pass" / "caution" / "fail"
print(result["recommendation"])  # 自动给出建议
```

评估指标包括：Action MSE、MAE、Max Error、Cosine Similarity、Relative Error、Per-dimension MSE——这些都是RL部署真正关心的。

### 3. 环境校准数据生成（核心创新点）

这是我认为最有价值的功能：**从Gymnasium环境中采集真实观测数据用于静态量化校准**。

通用工具用随机正态分布数据做校准，效果很差。RL-Deploy-Bench可以用你训练的策略在环境中跑，采集真实的观测分布，这样量化出来的模型精度高得多。

```bash
# 从Pendulum-v1采集500个校准样本
rl-deploy-bench calibrate Pendulum-v1 --output calib.npz --num-samples 500

# 用校准数据做静态INT8量化
rl-deploy-bench quantize policy.onnx --mode static --calibration-file calib.npz
```

### 4. 专业延迟基准

不只是测mean latency，而是完整的百分位统计：P50/P90/P95/P99，加上吞吐量，以及实时系统指标采集（GPU利用率、功耗、显存、温度、CPU、内存）。

```
Model      Mean(ms)  P95(ms)  P99(ms)  FPS      GPU Util  GPU Power
FP32       0.073     0.071     0.120     13699.8  45%       65W
INT8       0.346     0.188     0.410     2892.6   32%       45W
```

### 5. 跨平台支持

自动检测平台，选择对应的监控后端：
- **x86 NVIDIA GPU** → pynvml（GPU利用率/功耗/显存/温度）
- **NVIDIA Jetson** → jetson-stats（Jetson特有指标）
- **CPU-only** → psutil（CPU/内存）

推理后端支持ONNX Runtime（自动选provider）和TensorRT（框架已搭好，带优雅降级）。

### 6. 丰富报告

Markdown报告 + 交互式HTML报告（带6种Plotly图表：延迟分布直方图、百分位柱状图、吞吐量对比、系统指标、Action MSE、Cosine Similarity）。

## 技术选型思考

做这个项目时，我刻意做了几个选择：

### 为什么不直接用MLPerf？

MLPerf是行业标准，但它是为大模型和数据中心设计的，配置复杂，不支持RL特有的action-level指标，也没有环境校准功能。对于RL研究者来说太重了。

### 为什么不集成到Stable Baselines3？

SB3有export功能，但只支持ONNX导出，没有量化、基准、对比、报告。而且不是所有RL框架都是SB3，RL-Deploy-Bench是框架无关的。

## 项目现状

v1.0已经发布，包含：
- 23个Python源文件，9个核心模块
- 3种导出格式（ONNX/TorchScript/SB3）
- 3级量化（FP16/INT8动态/INT8静态）
- 2种推理后端（ONNX Runtime/TensorRT框架）
- 3个完整示例
- 22个单元测试，全部通过
- 6个CLI命令
- GitHub Actions CI

## 下一步计划

- v1.1：发布到PyPI，完善CLI体验
- v1.2：集成ManiSkill/Gibson等具身智能仿真环境
- v2.0：TensorRT完整集成，分布式基准测试

## 结语

RL部署是一个被忽视的领域。大家都在关心怎么训练出更好的策略，但很少有人关心怎么把策略高效、可靠地部署到真实设备上。

如果你也在做RL部署，或者遇到过类似的痛点，欢迎试用、提issue、贡献代码。

**项目地址**：https://github.com/dafahaha/rl-deploy-bench

如果对你有帮助，欢迎点个Star ⭐

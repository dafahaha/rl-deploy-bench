"""RL-Deploy-Bench: Cross-platform RL model deployment and benchmarking toolkit."""

__version__ = "1.1.0"

# Re-export the public Python API documented in README.md / API.md so that
# ``from rl_deploy_bench import export_to_onnx, ...`` works as advertised.
# torch is a core dependency of this package, so importing it at the top level
# is acceptable.
from .benchmark.latency import BenchmarkResult, LatencyStats, benchmark_latency
from .exporter.onnx_export import (
    ExportConfig,
    OnnxablePolicy,
    export_to_onnx,
    verify_onnx_export,
)
from .quantizer.fp16 import FP16Config, convert_onnx_to_fp16, evaluate_fp16_impact
from .quantizer.int8 import (
    CalibrationDataReader,
    QuantizationConfig,
    compare_model_sizes,
    dynamic_quantize,
    evaluate_quantization,
    quantize_and_evaluate,
    static_quantize,
    static_quantize_with_dataset,
)
from .reporter.html import generate_html_report
from .reporter.markdown import generate_markdown_report
from .runtime.onnx_runtime import InferenceResult, OnnxRuntimeInference
from .utils.platform import PlatformInfo, detect_platform

__all__ = [
    "BenchmarkResult",
    "CalibrationDataReader",
    "ExportConfig",
    "FP16Config",
    "InferenceResult",
    "LatencyStats",
    "OnnxRuntimeInference",
    "OnnxablePolicy",
    "PlatformInfo",
    "QuantizationConfig",
    "benchmark_latency",
    "compare_model_sizes",
    "convert_onnx_to_fp16",
    "detect_platform",
    "dynamic_quantize",
    "evaluate_fp16_impact",
    "evaluate_quantization",
    "export_to_onnx",
    "generate_html_report",
    "generate_markdown_report",
    "quantize_and_evaluate",
    "static_quantize",
    "static_quantize_with_dataset",
    "verify_onnx_export",
]

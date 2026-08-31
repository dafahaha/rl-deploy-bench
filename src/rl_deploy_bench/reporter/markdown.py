"""Markdown report generator for deployment benchmark results.

Generates comprehensive Markdown reports with latency tables,
accuracy comparisons, quantization impact analysis, and system metrics.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..benchmark.accuracy import AccuracyComparisonResult
from ..benchmark.latency import BenchmarkResult
from ..quantizer.int8 import compare_model_sizes
from ..utils.platform import PlatformInfo


def generate_markdown_report(
    output_path: str,
    benchmark_results: List[BenchmarkResult],
    model_names: Optional[List[str]] = None,
    accuracy_results: Optional[List[AccuracyComparisonResult]] = None,
    model_paths: Optional[List[str]] = None,
    platform_info: Optional[PlatformInfo] = None,
    title: str = "RL Model Deployment Benchmark Report",
) -> str:
    """Generate a comprehensive Markdown benchmark report.

    Args:
        output_path: Path to save the Markdown report.
        benchmark_results: List of benchmark results (one per model/configuration).
        model_names: Optional names for each benchmark result.
        accuracy_results: Optional accuracy comparison results.
        model_paths: Optional paths to model files (for size comparison).
        platform_info: Platform information.
        title: Report title.

    Returns:
        Absolute path to the generated report.
    """
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(len(benchmark_results))]

    lines = []

    # Header
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Platform info
    if platform_info is not None:
        lines.append("## Platform Information")
        lines.append("")
        lines.append(f"| Property | Value |")
        lines.append(f"|----------|-------|")
        lines.append(f"| OS | {platform_info.os} |")
        lines.append(f"| Architecture | {platform_info.arch} |")
        lines.append(f"| Python | {platform_info.python_version} |")
        lines.append(f"| CPU Cores | {platform_info.cpu_count} |")
        lines.append(f"| Total Memory | {platform_info.total_memory_gb} GB |")
        if platform_info.has_nvidia_gpu:
            lines.append(f"| GPU | {platform_info.gpu_name} (x{platform_info.gpu_count}) |")
        if platform_info.is_jetson:
            lines.append(f"| Platform | NVIDIA Jetson |")
        lines.append("")

    # Latency comparison table
    lines.append("## Latency and Throughput Comparison")
    lines.append("")
    lines.append("| Model | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | Throughput (FPS) |")
    lines.append("|-------|-----------|----------|----------|----------|----------|----------|------------------|")

    for name, result in zip(model_names, benchmark_results):
        lat = result.latency
        lines.append(
            f"| {name} | {lat.mean_ms:.3f} | {lat.p50_ms:.3f} | {lat.p95_ms:.3f} | "
            f"{lat.p99_ms:.3f} | {lat.min_ms:.3f} | {lat.max_ms:.3f} | {lat.throughput_fps:.1f} |"
        )
    lines.append("")

    # System metrics
    has_gpu_metrics = any(r.avg_gpu_utilization is not None for r in benchmark_results)
    if has_gpu_metrics:
        lines.append("## System Metrics During Inference")
        lines.append("")
        lines.append("| Model | GPU Util (%) | GPU Power (W) | GPU Memory (MB) | CPU Util (%) |")
        lines.append("|-------|-------------|---------------|-----------------|-------------|")
        for name, result in zip(model_names, benchmark_results):
            gpu_util = f"{result.avg_gpu_utilization:.1f}" if result.avg_gpu_utilization is not None else "N/A"
            gpu_power = f"{result.avg_gpu_power_w:.2f}" if result.avg_gpu_power_w is not None else "N/A"
            gpu_mem = f"{result.avg_gpu_memory_mb:.1f}" if result.avg_gpu_memory_mb is not None else "N/A"
            cpu_util = f"{result.avg_cpu_utilization:.1f}" if result.avg_cpu_utilization is not None else "N/A"
            lines.append(f"| {name} | {gpu_util} | {gpu_power} | {gpu_mem} | {cpu_util} |")
        lines.append("")

    # Accuracy comparison (filter out None values)
    valid_accuracy = []
    valid_accuracy_names = []
    if accuracy_results is not None:
        for name, acc in zip(model_names, accuracy_results):
            if acc is not None:
                valid_accuracy.append(acc)
                valid_accuracy_names.append(name)

    if len(valid_accuracy) > 0:
        lines.append("## Action Accuracy Comparison")
        lines.append("")
        lines.append("| Model | Action MSE | Action MAE | Max Error | Cosine Similarity | Relative Error |")
        lines.append("|-------|-----------|-----------|-----------|------------------|---------------|")
        for name, acc in zip(valid_accuracy_names, valid_accuracy):
            lines.append(
                f"| {name} | {acc.action_mse:.6f} | {acc.action_mae:.6f} | {acc.action_max_error:.6f} | "
                f"{acc.action_cosine_similarity:.6f} | {acc.action_relative_error:.4f} |"
            )
        lines.append("")

        # Per-dimension MSE
        if len(valid_accuracy) > 0 and len(valid_accuracy[0].per_dimension_mse) > 1:
            lines.append("### Per-Dimension Action MSE")
            lines.append("")
            dim_count = len(valid_accuracy[0].per_dimension_mse)
            header = "| Model | " + " | ".join(f"Dim {i}" for i in range(dim_count)) + " |"
            sep = "|-------|" + "|".join(["------"] * dim_count) + "|"
            lines.append(header)
            lines.append(sep)
            for name, acc in zip(valid_accuracy_names, valid_accuracy):
                row = f"| {name} | " + " | ".join(f"{m:.6f}" for m in acc.per_dimension_mse) + " |"
                lines.append(row)
            lines.append("")

    # Model size comparison
    if model_paths is not None and len(model_paths) >= 2:
        lines.append("## Model Size Comparison")
        lines.append("")
        lines.append("| Model | Size (MB) |")
        lines.append("|-------|-----------|")
        for name, path in zip(model_names, model_paths):
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                lines.append(f"| {name} | {size_mb:.2f} |")
        lines.append("")

        # Size reduction if first is original
        if len(model_paths) >= 2 and os.path.exists(model_paths[0]) and os.path.exists(model_paths[1]):
            size_info = compare_model_sizes(model_paths[0], model_paths[1])
            lines.append(f"**Size reduction:** {size_info['size_reduction_mb']:.2f} MB "
                        f"({size_info['size_reduction_pct']:.1f}%), "
                        f"compression ratio: {size_info['compression_ratio']:.2f}x")
            lines.append("")

    # Inference provider info
    lines.append("## Inference Configuration")
    lines.append("")
    for name, result in zip(model_names, benchmark_results):
        info = result.model_info
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Providers:** {', '.join(info.get('active_providers', []))}")
        lines.append(f"- **Input shape:** {info.get('input_shape', 'N/A')}")
        lines.append(f"- **Batch size:** {result.batch_size}")
        lines.append(f"- **Number of runs:** {result.latency.num_runs}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by rl-deploy-bench*")

    # Write report
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return os.path.abspath(output_path)


def generate_latency_distribution_data(benchmark_results: List[BenchmarkResult], model_names: List[str]) -> dict:
    """Prepare latency distribution data for plotting.

    Args:
        benchmark_results: List of benchmark results.
        model_names: Names for each result.

    Returns:
        Dictionary with data for Plotly charts.
    """
    data = {}
    for name, result in zip(model_names, benchmark_results):
        data[name] = {
            "latencies": result.latency.latencies_ms,
            "p50": result.latency.p50_ms,
            "p95": result.latency.p95_ms,
            "p99": result.latency.p99_ms,
            "mean": result.latency.mean_ms,
        }
    return data

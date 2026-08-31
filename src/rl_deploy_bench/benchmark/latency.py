"""Latency and throughput benchmarking for RL inference.

Measures P50/P95/P99 latency, throughput, and collects system metrics
during inference using the monitor module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..monitor.base import BaseMonitor, SystemMetrics
from ..runtime.onnx_runtime import OnnxRuntimeInference


@dataclass
class LatencyStats:
    """Latency statistics from benchmark run."""

    num_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_fps: float
    latencies_ms: List[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Complete benchmark result including system metrics."""

    latency: LatencyStats
    system_metrics: List[SystemMetrics] = field(default_factory=list)
    avg_gpu_utilization: Optional[float] = None
    avg_gpu_power_w: Optional[float] = None
    avg_gpu_memory_mb: Optional[float] = None
    avg_cpu_utilization: Optional[float] = None
    model_info: dict = field(default_factory=dict)
    batch_size: int = 1


def benchmark_latency(
    inference: OnnxRuntimeInference,
    observation_shape: Sequence[int],
    num_warmup: int = 50,
    num_runs: int = 500,
    batch_size: int = 1,
    monitor: Optional[BaseMonitor] = None,
    monitor_interval_ms: float = 100.0,
) -> BenchmarkResult:
    """Run latency and throughput benchmark.

    Args:
        inference: Inference engine instance.
        observation_shape: Shape of a single observation (without batch dim).
        num_warmup: Number of warmup runs.
        num_runs: Number of benchmark runs.
        batch_size: Batch size for inference.
        monitor: Optional system monitor to collect metrics during benchmark.
        monitor_interval_ms: Interval between monitor snapshots in milliseconds.

    Returns:
        BenchmarkResult with latency stats and system metrics.
    """
    # Warmup
    inference.warmup(num_runs=num_warmup, observation_shape=observation_shape)

    # Generate benchmark data
    observations = np.random.randn(num_runs, batch_size, *observation_shape).astype(np.float32)

    # Start monitor if provided
    if monitor is not None:
        monitor.start()
        system_metrics: List[SystemMetrics] = []
        last_monitor_time = time.time()

    # Run benchmark
    latencies = []
    for i in range(num_runs):
        result = inference.infer(observations[i])
        latencies.append(result.latency_ms)

        # Collect monitor metrics at intervals
        if monitor is not None:
            current_time = time.time()
            if (current_time - last_monitor_time) * 1000 >= monitor_interval_ms:
                system_metrics.append(monitor.snapshot())
                last_monitor_time = current_time

    # Stop monitor
    if monitor is not None:
        # Final snapshot
        try:
            system_metrics.append(monitor.snapshot())
        except Exception:
            pass
        monitor.stop()

    # Compute statistics
    latencies_arr = np.array(latencies)
    total_time_s = np.sum(latencies) / 1000
    throughput = num_runs * batch_size / total_time_s if total_time_s > 0 else 0

    latency_stats = LatencyStats(
        num_runs=num_runs,
        mean_ms=float(np.mean(latencies_arr)),
        std_ms=float(np.std(latencies_arr)),
        min_ms=float(np.min(latencies_arr)),
        max_ms=float(np.max(latencies_arr)),
        p50_ms=float(np.percentile(latencies_arr, 50)),
        p90_ms=float(np.percentile(latencies_arr, 90)),
        p95_ms=float(np.percentile(latencies_arr, 95)),
        p99_ms=float(np.percentile(latencies_arr, 99)),
        throughput_fps=float(throughput),
        latencies_ms=latencies,
    )

    # Aggregate system metrics
    result = BenchmarkResult(
        latency=latency_stats,
        batch_size=batch_size,
        model_info=inference.get_provider_info(),
    )

    if monitor is not None and system_metrics:
        result.system_metrics = system_metrics

        gpu_utils = [m.gpu_utilization for m in system_metrics if m.gpu_utilization is not None]
        gpu_powers = [m.gpu_power_w for m in system_metrics if m.gpu_power_w is not None]
        gpu_mems = [m.gpu_memory_used_mb for m in system_metrics if m.gpu_memory_used_mb is not None]
        cpu_utils = [m.cpu_utilization for m in system_metrics if m.cpu_utilization is not None]

        if gpu_utils:
            result.avg_gpu_utilization = float(np.mean(gpu_utils))
        if gpu_powers:
            result.avg_gpu_power_w = float(np.mean(gpu_powers))
        if gpu_mems:
            result.avg_gpu_memory_mb = float(np.mean(gpu_mems))
        if cpu_utils:
            result.avg_cpu_utilization = float(np.mean(cpu_utils))

    return result

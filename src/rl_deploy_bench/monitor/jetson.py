"""Jetson monitor using jetson-stats (jtop) library."""

from __future__ import annotations

import time
from typing import Optional

from .base import BaseMonitor, SystemMetrics


class JetsonMonitor(BaseMonitor):
    """Monitor Jetson metrics using jetson-stats (jtop).

    Reuses the jetson-stats library which is the standard monitoring
    tool for NVIDIA Jetson platforms, providing CPU/GPU/power/temp
    metrics via Python API.
    """

    def __init__(self):
        self._jtop = None
        self._initialized = False

    def start(self) -> None:
        """Initialize jtop."""
        try:
            from jtop import jtop

            self._jtop = jtop()
            self._jtop.start()
            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize jetson-stats: {e}. "
                "Install with: pip install jetson-stats (requires Jetson hardware)"
            ) from e

    def stop(self) -> None:
        """Stop jtop."""
        if self._jtop is not None:
            try:
                self._jtop.close()
            except Exception:
                pass
        self._initialized = False
        self._jtop = None

    def snapshot(self) -> SystemMetrics:
        """Take a snapshot of Jetson metrics."""
        if not self._initialized or self._jtop is None:
            raise RuntimeError("Monitor not started. Call start() first.")

        metrics = SystemMetrics(timestamp=time.time())

        try:
            stats = self._jtop.stats

            # GPU utilization
            if "GPU" in stats:
                gpu = stats["GPU"]
                if isinstance(gpu, dict):
                    metrics.gpu_utilization = float(gpu.get("status", {}).get("load", 0))
                elif isinstance(gpu, (int, float)):
                    metrics.gpu_utilization = float(gpu)

            # GPU memory
            if "RAM" in stats:
                ram = stats["RAM"]
                if isinstance(ram, dict):
                    metrics.gpu_memory_used_mb = round(ram.get("use", 0) / 1024, 2)
                    metrics.gpu_memory_total_mb = round(ram.get("tot", 0) / 1024, 2)

            # Power
            if "Power" in stats:
                power = stats["Power"]
                if isinstance(power, dict):
                    total_power = power.get("tot", power.get("total", 0))
                    metrics.gpu_power_w = round(total_power / 1000.0, 2)

            # Temperature
            if "Temp" in stats:
                temp = stats["Temp"]
                if isinstance(temp, dict):
                    for key in ["GPU", "gpu", "Tboard", "CPU"]:
                        if key in temp:
                            metrics.gpu_temperature_c = float(temp[key])
                            break

            # CPU
            if "CPU" in stats:
                cpu = stats["CPU"]
                if isinstance(cpu, list) and cpu:
                    total_util = sum(c.get("status", {}).get("load", 0) for c in cpu if isinstance(c, dict))
                    metrics.cpu_utilization = round(total_util / len(cpu), 2)

        except Exception:
            pass

        return metrics

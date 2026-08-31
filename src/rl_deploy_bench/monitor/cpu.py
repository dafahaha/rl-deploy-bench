"""CPU-only monitor using psutil for systems without GPU."""

from __future__ import annotations

import time

from .base import BaseMonitor, SystemMetrics


class CPUMonitor(BaseMonitor):
    """Monitor CPU and memory metrics using psutil.

    Used as fallback when no NVIDIA GPU is detected or on CPU-only
    inference backends.
    """

    def __init__(self):
        self._running = False

    def start(self) -> None:
        """Start monitoring (initialize psutil CPU sampling)."""
        import psutil

        psutil.cpu_percent(interval=None)  # initialize sampling
        self._running = True

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

    def snapshot(self) -> SystemMetrics:
        """Take a snapshot of CPU and memory metrics."""
        if not self._running:
            raise RuntimeError("Monitor not started. Call start() first.")

        import psutil

        metrics = SystemMetrics(timestamp=time.time())

        try:
            metrics.cpu_utilization = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            metrics.cpu_memory_used_mb = round(vm.used / (1024**2), 2)
            metrics.cpu_memory_total_mb = round(vm.total / (1024**2), 2)
        except Exception:
            pass

        return metrics

"""NVIDIA GPU monitor using pynvml (reuses nvidia-ml-py library)."""

from __future__ import annotations

import time
from typing import Optional

from .base import BaseMonitor, SystemMetrics


class NvidiaGPUMonitor(BaseMonitor):
    """Monitor NVIDIA GPU metrics using pynvml.

    Reuses the nvidia-ml-py (pynvml) library which is the official
    Python binding for NVIDIA Management Library (NVML), the same
    backend used by nvidia-smi.
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._handle = None
        self._initialized = False

    def start(self) -> None:
        """Initialize pynvml and get GPU handle."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._initialized = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize pynvml for GPU {self.gpu_index}: {e}. "
                "Install with: pip install nvidia-ml-py"
            ) from e

    def stop(self) -> None:
        """Shutdown pynvml."""
        if self._initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False
            self._handle = None

    def snapshot(self) -> SystemMetrics:
        """Take a snapshot of GPU and system metrics."""
        if not self._initialized or self._handle is None:
            raise RuntimeError("Monitor not started. Call start() first.")

        import pynvml
        import psutil

        metrics = SystemMetrics(timestamp=time.time())

        # GPU utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            metrics.gpu_utilization = float(util.gpu)
        except Exception:
            pass

        # GPU memory
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            metrics.gpu_memory_used_mb = round(mem.used / (1024**2), 2)
            metrics.gpu_memory_total_mb = round(mem.total / (1024**2), 2)
        except Exception:
            pass

        # GPU power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            metrics.gpu_power_w = round(power / 1000.0, 2)  # mW to W
        except Exception:
            pass

        # GPU temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            metrics.gpu_temperature_c = float(temp)
        except Exception:
            pass

        # CPU metrics (via psutil)
        try:
            metrics.cpu_utilization = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            metrics.cpu_memory_used_mb = round(vm.used / (1024**2), 2)
            metrics.cpu_memory_total_mb = round(vm.total / (1024**2), 2)
        except Exception:
            pass

        return metrics

    def get_gpu_name(self) -> Optional[str]:
        """Get GPU name."""
        if not self._initialized or self._handle is None:
            return None
        try:
            import pynvml

            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return name
        except Exception:
            return None

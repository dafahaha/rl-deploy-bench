"""Monitor factory and system metrics collection."""

from __future__ import annotations

from typing import Optional

from ..utils.platform import PlatformInfo, detect_platform, get_monitor_backend
from .base import BaseMonitor, SystemMetrics
from .cpu import CPUMonitor
from .nvidia_gpu import NvidiaGPUMonitor


def create_monitor(
    backend: Optional[str] = None,
    gpu_index: int = 0,
    platform_info: Optional[PlatformInfo] = None,
) -> BaseMonitor:
    """Create the appropriate monitor for the current platform.

    Args:
        backend: Force a specific backend ('nvidia', 'jetson', 'cpu').
            If None, auto-detect.
        gpu_index: GPU index for NVIDIA backend.
        platform_info: Pre-detected platform info. If None, will detect.

    Returns:
        A monitor instance.
    """
    if backend is None:
        if platform_info is None:
            platform_info = detect_platform()
        backend = get_monitor_backend(platform_info)

    if backend == "nvidia":
        return NvidiaGPUMonitor(gpu_index=gpu_index)
    elif backend == "jetson":
        try:
            from .jetson import JetsonMonitor

            return JetsonMonitor()
        except ImportError:
            return CPUMonitor()
    else:
        return CPUMonitor()


__all__ = [
    "BaseMonitor",
    "SystemMetrics",
    "CPUMonitor",
    "NvidiaGPUMonitor",
    "create_monitor",
]

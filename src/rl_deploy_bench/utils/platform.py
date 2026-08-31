"""Platform detection and hardware info utilities."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlatformInfo:
    """Detected platform and hardware information."""

    os: str
    arch: str
    python_version: str
    has_nvidia_gpu: bool
    is_jetson: bool
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    cpu_count: int = 0
    total_memory_gb: float = 0.0


def detect_platform() -> PlatformInfo:
    """Detect current platform and hardware capabilities."""
    info = PlatformInfo(
        os=platform.system(),
        arch=platform.machine(),
        python_version=platform.python_version(),
        has_nvidia_gpu=False,
        is_jetson=False,
        cpu_count=_get_cpu_count(),
        total_memory_gb=_get_total_memory_gb(),
    )

    # Check for NVIDIA GPU
    info.has_nvidia_gpu, info.gpu_count, info.gpu_name = _detect_nvidia_gpu()

    # Check if running on Jetson
    info.is_jetson = _detect_jetson()

    return info


def _get_cpu_count() -> int:
    import psutil

    return psutil.cpu_count(logical=True) or 0


def _get_total_memory_gb() -> float:
    import psutil

    return round(psutil.virtual_memory().total / (1024**3), 2)


def _detect_nvidia_gpu() -> tuple[bool, int, Optional[str]]:
    """Detect NVIDIA GPU using pynvml if available, fallback to nvidia-smi."""
    # Try pynvml first
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            pynvml.nvmlShutdown()
            return True, count, name
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Fallback to nvidia-smi command
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
            return True, len(names), names[0]
    except Exception:
        pass

    return False, 0, None


def _detect_jetson() -> bool:
    """Detect if running on NVIDIA Jetson platform."""
    # Check for Jetson-specific files
    jetson_indicators = [
        "/etc/nv_tegra_release",
        "/proc/device-tree/compatible",
        "/sys/class/gpio/export",
    ]
    import os

    for indicator in jetson_indicators:
        if os.path.exists(indicator):
            try:
                if "nv_tegra" in indicator:
                    return True
                with open(indicator, "r") as f:
                    content = f.read().lower()
                    if "jetson" in content or "tegra" in content:
                        return True
            except Exception:
                continue

    # Check hostname or environment
    hostname = platform.node().lower()
    if "jetson" in hostname or "tegra" in hostname:
        return True

    return False


def get_monitor_backend(info: Optional[PlatformInfo] = None) -> str:
    """Determine the appropriate monitor backend for the current platform."""
    if info is None:
        info = detect_platform()

    if info.is_jetson:
        return "jetson"
    if info.has_nvidia_gpu:
        return "nvidia"
    return "cpu"

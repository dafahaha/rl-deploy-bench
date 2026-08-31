"""Base monitor interface for system metrics collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemMetrics:
    """Snapshot of system metrics during inference."""

    timestamp: float
    gpu_utilization: Optional[float] = None  # percentage 0-100
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_power_w: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    cpu_utilization: Optional[float] = None  # percentage 0-100
    cpu_memory_used_mb: Optional[float] = None
    cpu_memory_total_mb: Optional[float] = None
    extra: dict = field(default_factory=dict)


class BaseMonitor(ABC):
    """Abstract base class for system monitors."""

    @abstractmethod
    def start(self) -> None:
        """Start monitoring."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop monitoring."""
        ...

    @abstractmethod
    def snapshot(self) -> SystemMetrics:
        """Take a single snapshot of current metrics."""
        ...

    def __enter__(self) -> "BaseMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

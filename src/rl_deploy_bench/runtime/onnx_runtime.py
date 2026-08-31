"""ONNX Runtime inference backend.

Reuses onnxruntime library for cross-platform inference (CPU, CUDA, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class InferenceResult:
    """Result of a single inference call."""

    actions: np.ndarray
    latency_ms: float


class OnnxRuntimeInference:
    """ONNX Runtime inference engine for RL policies.

    Supports multiple execution providers (CPU, CUDA, etc.) and
    provides timing for benchmarking.
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[List[dict]] = None,
    ):
        import onnxruntime as ort

        self.model_path = model_path

        # Auto-select providers if not specified
        if providers is None:
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        self.providers = list(providers)
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            provider_options=provider_options,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, observation: np.ndarray) -> InferenceResult:
        """Run a single inference.

        Args:
            observation: Input observation array (batch_size, *obs_shape).

        Returns:
            InferenceResult with actions and latency.
        """
        if observation.ndim == len(self.input_shape) - 1:
            observation = observation[np.newaxis]

        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: observation})
        latency_ms = (time.perf_counter() - start) * 1000

        return InferenceResult(actions=outputs[0], latency_ms=latency_ms)

    def infer_batch(self, observations: np.ndarray) -> InferenceResult:
        """Run batch inference.

        Args:
            observations: Batch of observations (batch_size, *obs_shape).

        Returns:
            InferenceResult with actions and total latency.
        """
        return self.infer(observations)

    def warmup(self, num_runs: int = 10, observation_shape: Optional[Sequence[int]] = None) -> None:
        """Warm up the inference session.

        Args:
            num_runs: Number of warmup runs.
            observation_shape: Shape of a single observation. If None, inferred from model.
        """
        if observation_shape is None:
            # Infer from input shape, replacing dynamic dims with 1
            observation_shape = [1 if isinstance(d, str) or d is None else d for d in self.input_shape[1:]]

        dummy = np.random.randn(1, *observation_shape).astype(np.float32)
        for _ in range(num_runs):
            self.session.run(self.output_names, {self.input_name: dummy})

    def get_provider_info(self) -> dict:
        """Get information about active execution providers."""
        return {
            "active_providers": self.session.get_providers(),
            "model_path": self.model_path,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_names": self.output_names,
        }

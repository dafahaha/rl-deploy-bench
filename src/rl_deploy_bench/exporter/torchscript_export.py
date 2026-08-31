"""TorchScript model export and inference.

TorchScript provides a way to serialize PyTorch models for deployment
in environments without Python dependencies (e.g., LibTorch C++ runtime,
mobile devices, embedded systems).

This is particularly valuable for edge deployment where Python overhead
is unacceptable or where only the C++ LibTorch runtime is available.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TorchScriptConfig:
    """Configuration for TorchScript export."""

    method: str = "trace"  # 'trace' or 'script'
    optimize: bool = True
    strict: bool = True


def export_to_torchscript(
    model: nn.Module,
    observation_shape: Sequence[int],
    output_path: str,
    config: Optional[TorchScriptConfig] = None,
    example_observation: Optional[torch.Tensor] = None,
) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        model: PyTorch model to export.
        observation_shape: Shape of a single observation (without batch dim).
        output_path: Path to save the TorchScript model (.pt).
        config: Export configuration.
        example_observation: Optional example input for tracing.
            If None, a random tensor is used.

    Returns:
        Absolute path to the saved TorchScript model.
    """
    if config is None:
        config = TorchScriptConfig()

    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Prepare example input
    if example_observation is None:
        example_observation = torch.randn(1, *observation_shape, dtype=torch.float32)
    elif example_observation.dim() == len(observation_shape):
        example_observation = example_observation.unsqueeze(0)

    with torch.no_grad():
        if config.method == "trace":
            scripted = torch.jit.trace(
                model, example_observation, strict=config.strict
            )
        elif config.method == "script":
            scripted = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown export method: {config.method}. Use 'trace' or 'script'.")

        # Save before optimization (optimized models may not be serializable)
        scripted.save(output_path)

        # Load and optimize for inference if requested
        if config.optimize:
            try:
                loaded = torch.jit.load(output_path)
                loaded = torch.jit.freeze(loaded)
                optimized = torch.jit.optimize_for_inference(loaded)
                optimized.save(output_path)
            except Exception:
                # If optimization fails, keep the unoptimized version
                pass
    return os.path.abspath(output_path)


def _safe_load_torchscript(model_path: str):
    """Load TorchScript model with support for non-ASCII paths.

    PyTorch's torch.jit.load has issues with non-ASCII paths on Windows.
    This helper copies the file to a temp directory with an ASCII name if needed.
    """
    try:
        return torch.jit.load(model_path)
    except RuntimeError as e:
        if "No such file or directory" in str(e) and any(ord(c) > 127 for c in model_path):
            # Path contains non-ASCII characters, copy to temp dir
            import shutil
            import tempfile

            tmp_path = os.path.join(tempfile.gettempdir(), "rl_deploy_bench_ts_model.pt")
            shutil.copy2(model_path, tmp_path)
            return torch.jit.load(tmp_path)
        raise


def verify_torchscript_export(
    torchscript_path: str,
    model: nn.Module,
    observation_shape: Sequence[int],
    num_samples: int = 100,
    atol: float = 1e-5,
) -> dict:
    """Verify that TorchScript export matches PyTorch output.

    Args:
        torchscript_path: Path to the exported TorchScript model.
        model: Original PyTorch model.
        observation_shape: Shape of a single observation.
        num_samples: Number of test samples.
        atol: Absolute tolerance for comparison.

    Returns:
        Dictionary with verification results.
    """
    model.eval()
    loaded = _safe_load_torchscript(torchscript_path)
    loaded.eval()

    max_diff = 0.0
    mean_diff = 0.0

    with torch.no_grad():
        for _ in range(num_samples):
            obs = torch.randn(1, *observation_shape, dtype=torch.float32)
            torch_output = model(obs).numpy()
            ts_output = loaded(obs).numpy()
            diff = np.abs(torch_output - ts_output).max()
            max_diff = max(max_diff, diff)
            mean_diff += diff

    mean_diff /= num_samples
    passed = max_diff < atol

    return {
        "passed": passed,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "num_samples": num_samples,
        "tolerance": atol,
    }


class TorchScriptInference:
    """TorchScript inference runtime.

    Provides a unified interface for running inference with TorchScript
    models, compatible with the benchmarking and accuracy comparison APIs.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """Load a TorchScript model.

        Args:
            model_path: Path to the .pt TorchScript model.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = _safe_load_torchscript(model_path)
        self.model.eval()

    def infer(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run single inference.

        Args:
            observation: Input observation (batch_size, *obs_shape) or
                single observation (*obs_shape).

        Returns:
            Tuple of (output_actions, latency_ms).
        """
        if observation.ndim == 1:
            observation = observation[np.newaxis]

        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)

        # Warmup on first call
        if not hasattr(self, "_warmed_up"):
            with torch.no_grad():
                _ = self.model(obs_tensor)
            self._warmed_up = True

        import time

        start = time.perf_counter()
        with torch.no_grad():
            output = self.model(obs_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        return output.cpu().numpy(), latency_ms

    def warmup(self, num_runs: int = 10, observation_shape: Optional[Sequence[int]] = None):
        """Warm up the model for consistent benchmarking.

        Args:
            num_runs: Number of warmup runs.
            observation_shape: Shape of observation for warmup.
        """
        if observation_shape is None:
            observation_shape = (4,)

        obs = torch.randn(1, *observation_shape, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._warmed_up = True


def compare_onnx_torchscript(
    onnx_path: str,
    torchscript_path: str,
    observation_shape: Sequence[int],
    num_samples: int = 100,
) -> dict:
    """Compare outputs of ONNX and TorchScript models.

    Useful for verifying that both export paths produce identical results.

    Args:
        onnx_path: Path to ONNX model.
        torchscript_path: Path to TorchScript model.
        observation_shape: Shape of a single observation.
        num_samples: Number of test samples.

    Returns:
        Dictionary with comparison results.
    """
    from ..runtime.onnx_runtime import OnnxRuntimeInference

    onnx_inf = OnnxRuntimeInference(onnx_path)
    ts_inf = TorchScriptInference(torchscript_path)

    max_diff = 0.0
    mean_diff = 0.0

    for _ in range(num_samples):
        obs = np.random.randn(1, *observation_shape).astype(np.float32)
        onnx_output = onnx_inf.infer(obs).actions
        ts_output, _ = ts_inf.infer(obs)
        diff = np.abs(onnx_output - ts_output).max()
        max_diff = max(max_diff, diff)
        mean_diff += diff

    mean_diff /= num_samples

    return {
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "num_samples": num_samples,
        "outputs_match": max_diff < 1e-4,
    }

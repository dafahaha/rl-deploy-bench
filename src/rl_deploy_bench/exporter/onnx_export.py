"""ONNX model export for RL policies.

Reuses PyTorch's native torch.onnx.export API and follows the
pattern from Stable Baselines3 official export guide.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExportConfig:
    """Configuration for ONNX export."""

    opset_version: int = 17
    do_constant_folding: bool = True
    export_params: bool = True
    dynamic_batch: bool = True
    input_names: Tuple[str, ...] = ("observation",)
    output_names: Tuple[str, ...] = ("action",)


class OnnxablePolicy(nn.Module):
    """Wrapper to make an RL policy exportable to ONNX.

    Adapted from Stable Baselines3 official export guide:
    https://stable-baselines3.readthedocs.io/en/master/guide/export.html

    This wrapper handles generic nn.Module policies that return actions
    directly. For SB3 policies, use SB3OnnxablePolicy from sb3.py.
    """

    def __init__(self, policy: nn.Module, action_dim: Optional[int] = None):
        super().__init__()
        self.policy = policy
        self.action_dim = action_dim
        self._normalize = False
        self.register_buffer("_action_low", torch.zeros(1))
        self.register_buffer("_action_high", torch.ones(1))

    def set_action_bounds(self, low: np.ndarray, high: np.ndarray) -> None:
        """Set action space bounds for unscaling continuous actions."""
        self._action_low = torch.tensor(low, dtype=torch.float32).reshape(1, -1)
        self._action_high = torch.tensor(high, dtype=torch.float32).reshape(1, -1)
        self._normalize = True

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference.

        Returns actions directly (not distributions), suitable for deployment.
        Assumes self.policy is a standard nn.Module that returns a tensor
        of actions (or a tuple with actions as first element).
        """
        action = self.policy(observation)

        # Handle tuple outputs (common in RL: (action, value, log_prob))
        if isinstance(action, tuple):
            action = action[0]

        # Unscale actions if bounds are set (tanh output in [-1, 1] -> [low, high])
        if self._normalize:
            low = self._action_low
            high = self._action_high
            # Expand bounds to match batch size
            if low.shape[0] == 1 and action.shape[0] > 1:
                low = low.expand(action.shape[0], -1)
                high = high.expand(action.shape[0], -1)
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = torch.clamp(action, low, high)

        return action


def export_to_onnx(
    policy: nn.Module,
    observation_shape: Sequence[int],
    output_path: str,
    config: Optional[ExportConfig] = None,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
) -> str:
    """Export an RL policy to ONNX format.

    Args:
        policy: The policy network (SB3 policy, CleanRL actor, or generic nn.Module).
        observation_shape: Shape of a single observation (without batch dim).
        output_path: Path to save the ONNX model.
        config: Export configuration.
        action_low: Lower bounds of action space (for continuous actions).
        action_high: Upper bounds of action space.

    Returns:
        Absolute path to the exported ONNX file.
    """
    if config is None:
        config = ExportConfig()

    # Wrap policy
    onnxable = OnnxablePolicy(policy)
    if action_low is not None and action_high is not None:
        onnxable.set_action_bounds(action_low, action_high)
    onnxable.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *observation_shape, dtype=torch.float32)

    # Dynamic axes
    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in config.input_names}
        dynamic_axes.update({name: {0: "batch_size"} for name in config.output_names})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Export
    torch.onnx.export(
        onnxable,
        dummy_input,
        output_path,
        opset_version=config.opset_version,
        do_constant_folding=config.do_constant_folding,
        export_params=config.export_params,
        input_names=list(config.input_names),
        output_names=list(config.output_names),
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use TorchScript-based export for broader compatibility
    )

    return os.path.abspath(output_path)


def verify_onnx_export(
    onnx_path: str,
    policy: nn.Module,
    observation_shape: Sequence[int],
    atol: float = 1e-4,
) -> dict:
    """Verify that ONNX export matches PyTorch output.

    Args:
        onnx_path: Path to exported ONNX model.
        policy: Original PyTorch policy.
        observation_shape: Shape of a single observation.
        atol: Absolute tolerance for comparison.

    Returns:
        Dictionary with verification results.
    """
    import onnxruntime as ort

    # PyTorch inference
    onnxable = OnnxablePolicy(policy)
    onnxable.eval()
    dummy_input = torch.randn(1, *observation_shape, dtype=torch.float32)

    with torch.no_grad():
        torch_output = onnxable(dummy_input).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(torch_output - onnx_output))
    mean_diff = np.mean(np.abs(torch_output - onnx_output))
    passed = max_diff < atol

    return {
        "passed": passed,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "tolerance": atol,
        "torch_output_shape": list(torch_output.shape),
        "onnx_output_shape": list(onnx_output.shape),
    }

"""Stable Baselines3 (SB3) model export utilities.

Reuses SB3's policy structure and follows the official export guide:
https://stable-baselines3.readthedocs.io/en/master/guide/export.html
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from .onnx_export import ExportConfig, OnnxablePolicy, export_to_onnx, verify_onnx_export


class SB3OnnxablePolicy(OnnxablePolicy):
    """ONNX-exportable wrapper for SB3 policies.

    Handles SB3-specific policy structures including observation
    normalization and action scaling.
    """

    def __init__(self, sb3_model):
        # Extract the policy network from SB3 model
        policy = sb3_model.policy
        super().__init__(policy)

        # Store action space bounds
        if hasattr(sb3_model, "action_space"):
            action_space = sb3_model.action_space
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                self.set_action_bounds(action_space.low, action_space.high)

        # Store observation normalization if present
        self._obs_normalize = False
        if hasattr(policy, "obs_rms") and policy.obs_rms is not None:
            self._obs_normalize = True

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass using SB3's _predict method for deterministic inference."""
        # Use SB3's internal _predict which handles normalization properly
        if hasattr(self.policy, "_predict"):
            action = self.policy._predict(observation, deterministic=True)
        else:
            action = super().forward(observation)

        # Unscale actions
        if self._normalize and self._action_low is not None and self._action_high is not None:
            low = self._action_low.to(action.device)
            high = self._action_high.to(action.device)
            action = low + (action + 1.0) * 0.5 * (high - low)
            action = torch.clamp(action, low, high)

        return action


def export_sb3_model(
    sb3_model,
    output_path: str,
    config: Optional[ExportConfig] = None,
) -> str:
    """Export a Stable Baselines3 model to ONNX.

    Args:
        sb3_model: Trained SB3 model (PPO, SAC, DQN, TD3, etc.).
        output_path: Path to save the ONNX model.
        config: Export configuration.

    Returns:
        Absolute path to the exported ONNX file.
    """
    # Get observation shape
    obs_space = sb3_model.observation_space
    if hasattr(obs_space, "shape"):
        obs_shape = obs_space.shape
    else:
        raise ValueError(f"Unsupported observation space: {type(obs_space)}")

    # Wrap and export
    wrapper = SB3OnnxablePolicy(sb3_model)
    wrapper.eval()

    if config is None:
        config = ExportConfig()

    dummy_input = torch.randn(1, *obs_shape, dtype=torch.float32)

    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in config.input_names}
        dynamic_axes.update({name: {0: "batch_size"} for name in config.output_names})

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        wrapper,
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


def verify_sb3_export(
    onnx_path: str,
    sb3_model,
    num_samples: int = 100,
    atol: float = 1e-4,
) -> dict:
    """Verify SB3 ONNX export against original model predictions.

    Args:
        onnx_path: Path to exported ONNX model.
        sb3_model: Original SB3 model.
        num_samples: Number of random observations to test.
        atol: Absolute tolerance.

    Returns:
        Verification results dictionary.
    """
    import onnxruntime as ort

    obs_space = sb3_model.observation_space
    obs_shape = obs_space.shape

    # Generate random observations
    observations = [np.random.randn(*obs_shape).astype(np.float32) for _ in range(num_samples)]

    # SB3 predictions
    sb3_actions = []
    for obs in observations:
        action, _ = sb3_model.predict(obs, deterministic=True)
        sb3_actions.append(action)
    sb3_actions = np.array(sb3_actions)

    # ONNX predictions
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_actions = []
    for obs in observations:
        action = session.run(None, {input_name: obs[np.newaxis]})[0][0]
        onnx_actions.append(action)
    onnx_actions = np.array(onnx_actions)

    # Compare
    max_diff = np.max(np.abs(sb3_actions - onnx_actions))
    mean_diff = np.mean(np.abs(sb3_actions - onnx_actions))
    passed = max_diff < atol

    return {
        "passed": passed,
        "max_abs_diff": float(max_diff),
        "mean_abs_diff": float(mean_diff),
        "tolerance": atol,
        "num_samples": num_samples,
        "sb3_action_shape": list(sb3_actions.shape),
        "onnx_action_shape": list(onnx_actions.shape),
    }


def load_sb3_model(model_path: str, algo: Optional[str] = None, env=None):
    """Load an SB3 model from file.

    Args:
        model_path: Path to the SB3 model zip file.
        algo: Algorithm name ('PPO', 'SAC', 'DQN', 'TD3', etc.).
            If None, tries to infer from model.
        env: Gymnasium environment (optional, needed for some algorithms).

    Returns:
        Loaded SB3 model.
    """
    from stable_baselines3 import PPO, SAC, DQN, TD3, A2C, DDPG

    algo_map = {
        "PPO": PPO,
        "SAC": SAC,
        "DQN": DQN,
        "TD3": TD3,
        "A2C": A2C,
        "DDPG": DDPG,
    }

    if algo is not None:
        algo = algo.upper()
        if algo not in algo_map:
            raise ValueError(f"Unsupported algorithm: {algo}. Supported: {list(algo_map.keys())}")
        model_class = algo_map[algo]
        return model_class.load(model_path, env=env)

    # Try to infer algorithm
    for name, model_class in algo_map.items():
        try:
            model = model_class.load(model_path, env=env)
            return model
        except Exception:
            continue

    raise ValueError(
        f"Could not load model from {model_path}. "
        "Please specify the algorithm explicitly (e.g., algo='PPO')."
    )

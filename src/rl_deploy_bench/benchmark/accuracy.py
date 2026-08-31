"""RL-specific accuracy comparison between original and deployed models.

This is a key differentiator from generic ML benchmark tools:
instead of top-1 accuracy, we measure RL-specific metrics like
action MSE, value function error, and policy divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class AccuracyComparisonResult:
    """Result of comparing original vs deployed model actions."""

    num_samples: int
    action_mse: float
    action_mae: float
    action_max_error: float
    action_cosine_similarity: float
    action_relative_error: float
    per_dimension_mse: List[float]
    actions_original: Optional[np.ndarray] = None
    actions_deployed: Optional[np.ndarray] = None
    observations: Optional[np.ndarray] = None


def compare_actions(
    actions_original: np.ndarray,
    actions_deployed: np.ndarray,
    observations: Optional[np.ndarray] = None,
) -> AccuracyComparisonResult:
    """Compare actions from original model vs deployed model.

    Args:
        actions_original: Actions from original PyTorch/SB3 model.
            Shape: (num_samples, action_dim) or (num_samples,).
        actions_deployed: Actions from deployed ONNX/TensorRT model.
            Same shape as actions_original.
        observations: Optional observations used for reference.

    Returns:
        AccuracyComparisonResult with RL-specific metrics.
    """
    # Ensure 2D
    if actions_original.ndim == 1:
        actions_original = actions_original[:, np.newaxis]
    if actions_deployed.ndim == 1:
        actions_deployed = actions_deployed[:, np.newaxis]

    num_samples = len(actions_original)
    action_dim = actions_original.shape[1] if actions_original.ndim > 1 else 1

    # Basic errors
    diff = actions_original - actions_deployed
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_error = float(np.max(np.abs(diff)))

    # Per-dimension MSE
    per_dim_mse = [float(np.mean(diff[:, d] ** 2)) for d in range(action_dim)]

    # Cosine similarity (average across samples)
    cos_sims = []
    for i in range(num_samples):
        a = actions_original[i]
        b = actions_deployed[i]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            cos_sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
    avg_cos_sim = float(np.mean(cos_sims)) if cos_sims else 0.0

    # Relative error (MSE / variance of original actions)
    original_var = float(np.var(actions_original))
    relative_error = mse / original_var if original_var > 0 else float("inf")

    return AccuracyComparisonResult(
        num_samples=num_samples,
        action_mse=mse,
        action_mae=mae,
        action_max_error=max_error,
        action_cosine_similarity=avg_cos_sim,
        action_relative_error=relative_error,
        per_dimension_mse=per_dim_mse,
        actions_original=actions_original,
        actions_deployed=actions_deployed,
        observations=observations,
    )


def generate_test_observations(
    observation_shape: Sequence[int],
    num_samples: int = 1000,
    distribution: str = "normal",
    low: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate test observations for accuracy comparison.

    Args:
        observation_shape: Shape of a single observation.
        num_samples: Number of samples to generate.
        distribution: 'normal', 'uniform', or 'real' (from environment).
        low: Lower bounds for uniform distribution.
        high: Upper bounds for uniform distribution.

    Returns:
        Array of observations with shape (num_samples, *observation_shape).
    """
    if distribution == "normal":
        return np.random.randn(num_samples, *observation_shape).astype(np.float32)
    elif distribution == "uniform":
        if low is None or high is None:
            return np.random.uniform(-1, 1, (num_samples, *observation_shape)).astype(np.float32)
        return np.random.uniform(low, high, (num_samples, *observation_shape)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def evaluate_quantization_impact(
    original_result: AccuracyComparisonResult,
    quantized_result: AccuracyComparisonResult,
    mse_threshold: float = 0.01,
) -> dict:
    """Evaluate the impact of quantization on action accuracy.

    Args:
        original_result: Comparison result for original (FP32) model.
        quantized_result: Comparison result for quantized model.
        mse_threshold: Acceptable MSE threshold.

    Returns:
        Dictionary with quantization impact assessment.
    """
    mse_increase = quantized_result.action_mse - original_result.action_mse
    mse_ratio = quantized_result.action_mse / original_result.action_mse if original_result.action_mse > 0 else float("inf")

    return {
        "original_mse": original_result.action_mse,
        "quantized_mse": quantized_result.action_mse,
        "mse_increase": mse_increase,
        "mse_ratio": mse_ratio,
        "cosine_similarity_original": original_result.action_cosine_similarity,
        "cosine_similarity_quantized": quantized_result.action_cosine_similarity,
        "within_threshold": quantized_result.action_mse < mse_threshold,
        "threshold": mse_threshold,
        "recommendation": (
            "Quantization accuracy is acceptable"
            if quantized_result.action_mse < mse_threshold
            else "Quantization may cause significant action deviation, consider FP16"
        ),
    }

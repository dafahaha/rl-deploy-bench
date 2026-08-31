"""INT8 quantization for RL policies using ONNX Runtime.

Reuses onnxruntime.quantization API for post-training quantization (PTQ).
Supports both dynamic quantization (no calibration needed) and static
quantization (with calibration data from RL environments).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    quant_format: str = "QDQ"  # 'QDQ' or 'QOperator'
    activation_type: str = "int8"  # 'int8' or 'uint8'
    weight_type: str = "int8"  # 'int8' or 'uint8'
    per_channel: bool = True
    reduce_range: bool = False


def dynamic_quantize(
    onnx_model_path: str,
    output_path: Optional[str] = None,
    config: Optional[QuantizationConfig] = None,
) -> str:
    """Apply dynamic INT8 quantization to an ONNX model.

    Dynamic quantization quantizes weights statically but activations
    are quantized on-the-fly during inference. No calibration data needed.

    Args:
        onnx_model_path: Path to input ONNX model.
        output_path: Path to save quantized model. If None, appends '_int8_dynamic'.
        config: Quantization configuration.

    Returns:
        Absolute path to quantized model.
    """
    import tempfile

    from onnxruntime.quantization import QuantType, quantize_dynamic

    if config is None:
        config = QuantizationConfig()

    if output_path is None:
        base, ext = os.path.splitext(onnx_model_path)
        output_path = f"{base}_int8_dynamic{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Pre-run shape inference to avoid onnxruntime internal path issues
    import onnx
    from onnx import shape_inference

    model = onnx.load(onnx_model_path)
    inferred_model = shape_inference.infer_shapes(model)

    # Save to a temp file for quantization input
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    onnx.save(inferred_model, tmp_path)

    try:
        weight_type = QuantType.QInt8 if config.weight_type == "int8" else QuantType.QUInt8

        quantize_dynamic(
            model_input=tmp_path,
            model_output=output_path,
            weight_type=weight_type,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return os.path.abspath(output_path)


class CalibrationDataReader:
    """Data reader for static quantization calibration.

    Generates calibration data from random observations or from
    an RL environment. This is RL-specific: calibration data should
    be realistic observations, not random noise.
    """

    def __init__(
        self,
        observation_shape: Sequence[int],
        num_samples: int = 100,
        observations: Optional[np.ndarray] = None,
        input_name: str = "observation",
    ):
        self.input_name = input_name
        self.current = 0

        if observations is not None:
            self.observations = observations
        else:
            self.observations = np.random.randn(num_samples, *observation_shape).astype(np.float32)

    def get_next(self):
        if self.current >= len(self.observations):
            return None
        obs = self.observations[self.current]
        self.current += 1
        return {self.input_name: obs}

    def rewind(self):
        self.current = 0


def static_quantize(
    onnx_model_path: str,
    observation_shape: Sequence[int],
    output_path: Optional[str] = None,
    config: Optional[QuantizationConfig] = None,
    calibration_samples: int = 100,
    calibration_observations: Optional[np.ndarray] = None,
) -> str:
    """Apply static INT8 quantization with calibration.

    Static quantization quantizes both weights and activations using
    calibration data. For RL models, calibration data should be
    realistic observations from the environment.

    Args:
        onnx_model_path: Path to input ONNX model.
        observation_shape: Shape of a single observation.
        output_path: Path to save quantized model.
        config: Quantization configuration.
        calibration_samples: Number of calibration samples.
        calibration_observations: Pre-generated calibration observations.
            If None, random normal observations are used.

    Returns:
        Absolute path to quantized model.
    """
    from onnxruntime.quantization import QuantType, quantize_static

    if config is None:
        config = QuantizationConfig()

    if output_path is None:
        base, ext = os.path.splitext(onnx_model_path)
        output_path = f"{base}_int8_static{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Create calibration data reader
    reader = CalibrationDataReader(
        observation_shape=observation_shape,
        num_samples=calibration_samples,
        observations=calibration_observations,
    )

    activation_type = QuantType.QInt8 if config.activation_type == "int8" else QuantType.QUInt8
    weight_type = QuantType.QInt8 if config.weight_type == "int8" else QuantType.QUInt8

    quantize_static(
        model_input=onnx_model_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=getattr(__import__("onnxruntime.quantization", fromlist=["QuantFormat"]), "QuantFormat").QDQ
        if config.quant_format == "QDQ"
        else getattr(__import__("onnxruntime.quantization", fromlist=["QuantFormat"]), "QuantFormat").QOperator,
        weight_type=weight_type,
        activation_type=activation_type,
        per_channel=config.per_channel,
        reduce_range=config.reduce_range,
    )

    return os.path.abspath(output_path)


def get_model_size_mb(model_path: str) -> float:
    """Get model file size in MB."""
    size_bytes = os.path.getsize(model_path)
    return round(size_bytes / (1024 * 1024), 2)


def compare_model_sizes(original_path: str, quantized_path: str) -> dict:
    """Compare file sizes of original and quantized models.

    Args:
        original_path: Path to original FP32 model.
        quantized_path: Path to quantized model.

    Returns:
        Dictionary with size comparison.
    """
    original_size = get_model_size_mb(original_path)
    quantized_size = get_model_size_mb(quantized_path)
    reduction = original_size - quantized_size
    reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0
    ratio = original_size / quantized_size if quantized_size > 0 else float("inf")

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "size_reduction_mb": round(reduction, 2),
        "size_reduction_pct": round(reduction_pct, 2),
        "compression_ratio": round(ratio, 2),
    }


def static_quantize_with_dataset(
    onnx_model_path: str,
    calibration_dataset,
    output_path: Optional[str] = None,
    config: Optional[QuantizationConfig] = None,
    input_name: str = "observation",
) -> str:
    """Apply static INT8 quantization using a CalibrationDataset.

    This is the recommended quantization method for RL models because it
    uses realistic observations from the simulation environment for
    calibration, rather than random noise.

    Args:
        onnx_model_path: Path to input ONNX model.
        calibration_dataset: CalibrationDataset instance (from benchmark.calibration).
        output_path: Path to save quantized model.
        config: Quantization configuration.
        input_name: Name of the model input.

    Returns:
        Absolute path to quantized model.
    """
    import tempfile

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    if config is None:
        config = QuantizationConfig()

    if output_path is None:
        base, ext = os.path.splitext(onnx_model_path)
        output_path = f"{base}_int8_static{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Pre-run shape inference
    import onnx
    from onnx import shape_inference

    model = onnx.load(onnx_model_path)
    inferred_model = shape_inference.infer_shapes(model)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    onnx.save(inferred_model, tmp_path)

    # Create calibration data reader from dataset
    class _DatasetReader:
        def __init__(self, dataset, name):
            self.dataset = dataset
            self.input_name = name
            self.current = 0

        def get_next(self):
            if self.current >= len(self.dataset):
                return None
            obs = self.dataset[self.current]
            self.current += 1
            return {self.input_name: obs[np.newaxis].astype(np.float32)}

        def rewind(self):
            self.current = 0

    reader = _DatasetReader(calibration_dataset, input_name)

    activation_type = QuantType.QInt8 if config.activation_type == "int8" else QuantType.QUInt8
    weight_type = QuantType.QInt8 if config.weight_type == "int8" else QuantType.QUInt8
    quant_format = QuantFormat.QDQ if config.quant_format == "QDQ" else QuantFormat.QOperator

    try:
        quantize_static(
            model_input=tmp_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=quant_format,
            weight_type=weight_type,
            activation_type=activation_type,
            per_channel=config.per_channel,
            reduce_range=config.reduce_range,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return os.path.abspath(output_path)


def evaluate_quantization(
    original_model_path: str,
    quantized_model_path: str,
    observation_shape: Sequence[int],
    num_samples: int = 500,
    mse_threshold: float = 0.01,
    cosine_threshold: float = 0.99,
) -> dict:
    """Evaluate the impact of quantization on action accuracy.

    Runs both models on the same test observations and computes RL-specific
    accuracy metrics, then provides an automated recommendation.

    Args:
        original_model_path: Path to original FP32 ONNX model.
        quantized_model_path: Path to quantized ONNX model.
        observation_shape: Shape of a single observation.
        num_samples: Number of test samples.
        mse_threshold: Maximum acceptable action MSE.
        cosine_threshold: Minimum acceptable cosine similarity.

    Returns:
        Dictionary with evaluation results and recommendation.
    """
    from ..benchmark.accuracy import compare_actions, generate_test_observations
    from ..runtime.onnx_runtime import OnnxRuntimeInference

    # Load models
    orig_inf = OnnxRuntimeInference(original_model_path)
    quant_inf = OnnxRuntimeInference(quantized_model_path)

    # Generate test observations
    observations = generate_test_observations(observation_shape, num_samples)

    # Run inference
    orig_actions = []
    quant_actions = []
    for obs in observations:
        orig_actions.append(orig_inf.infer(obs).actions[0])
        quant_actions.append(quant_inf.infer(obs).actions[0])
    orig_actions = np.array(orig_actions)
    quant_actions = np.array(quant_actions)

    # Compare
    acc = compare_actions(orig_actions, quant_actions, observations)

    # Size comparison
    size_info = compare_model_sizes(original_model_path, quantized_model_path)

    # Determine recommendation
    mse_ok = acc.action_mse < mse_threshold
    cosine_ok = acc.action_cosine_similarity > cosine_threshold

    if mse_ok and cosine_ok:
        recommendation = "QUANTIZATION ACCEPTABLE: Action deviation is within thresholds. INT8 deployment is safe."
        verdict = "pass"
    elif mse_ok and not cosine_ok:
        recommendation = "QUANTIZATION CAUTION: MSE is acceptable but cosine similarity is low. Action direction may deviate. Consider FP16 or more calibration data."
        verdict = "caution"
    elif not mse_ok and cosine_ok:
        recommendation = "QUANTIZATION CAUTION: Cosine similarity is good but MSE exceeds threshold. Action magnitude may deviate. Consider static quantization with environment calibration data."
        verdict = "caution"
    else:
        recommendation = "QUANTIZATION NOT RECOMMENDED: Significant action deviation detected. Use FP16 or FP32 instead. Try collecting more calibration data from the actual environment."
        verdict = "fail"

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "action_mse": acc.action_mse,
        "action_mae": acc.action_mae,
        "action_max_error": acc.action_max_error,
        "cosine_similarity": acc.action_cosine_similarity,
        "relative_error": acc.action_relative_error,
        "per_dimension_mse": acc.per_dimension_mse,
        "mse_threshold": mse_threshold,
        "cosine_threshold": cosine_threshold,
        "mse_within_threshold": mse_ok,
        "cosine_within_threshold": cosine_ok,
        "size_comparison": size_info,
        "num_test_samples": num_samples,
    }


def quantize_and_evaluate(
    onnx_model_path: str,
    observation_shape: Sequence[int],
    output_dir: Optional[str] = None,
    mode: str = "dynamic",
    calibration_dataset=None,
    num_eval_samples: int = 500,
    mse_threshold: float = 0.01,
) -> dict:
    """One-click quantization and evaluation.

    Quantizes the model, evaluates the accuracy impact, and returns
    a complete report with recommendation.

    Args:
        onnx_model_path: Path to input ONNX model.
        observation_shape: Shape of a single observation.
        output_dir: Directory to save quantized model. If None, same dir as input.
        mode: Quantization mode ('dynamic' or 'static').
        calibration_dataset: CalibrationDataset for static mode.
        num_eval_samples: Number of samples for evaluation.
        mse_threshold: MSE threshold for evaluation.

    Returns:
        Dictionary with quantization path, evaluation results, and recommendation.
    """
    base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(onnx_model_path))

    if mode == "dynamic":
        quantized_path = os.path.join(output_dir, f"{base_name}_int8_dynamic.onnx")
        quantized_path = dynamic_quantize(onnx_model_path, quantized_path)
    elif mode == "static":
        if calibration_dataset is None:
            raise ValueError("calibration_dataset is required for static quantization")
        quantized_path = os.path.join(output_dir, f"{base_name}_int8_static.onnx")
        quantized_path = static_quantize_with_dataset(
            onnx_model_path, calibration_dataset, quantized_path
        )
    else:
        raise ValueError(f"Unknown quantization mode: {mode}. Use 'dynamic' or 'static'.")

    # Evaluate
    eval_result = evaluate_quantization(
        onnx_model_path,
        quantized_path,
        observation_shape,
        num_samples=num_eval_samples,
        mse_threshold=mse_threshold,
    )

    return {
        "original_model": onnx_model_path,
        "quantized_model": quantized_path,
        "mode": mode,
        "evaluation": eval_result,
    }

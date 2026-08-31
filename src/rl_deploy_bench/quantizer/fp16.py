"""FP16 quantization and inference for RL models.

FP16 (half-precision) provides significant speedups on GPUs with native
FP16 support (NVIDIA Jetson Xavier/Orin, RTX 20-series+, AMD RDNA2+).
For RL policies (typically MLPs), FP16 usually maintains near-FP32 accuracy
while reducing memory bandwidth and increasing throughput.

This module provides:
- ONNX to FP16 conversion with automatic Cast node insertion
- FP16 inference via ONNX Runtime
- FP16 accuracy impact evaluation
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class FP16Config:
    """Configuration for FP16 conversion."""

    keep_io_in_fp32: bool = True  # Keep model inputs/outputs in FP32
    convert_weights: bool = True  # Convert initializer weights to FP16
    op_blocklist: tuple = ()  # Ops to keep in FP32 (e.g., 'Softmax', 'Exp')


def convert_onnx_to_fp16(
    onnx_model_path: str,
    output_path: Optional[str] = None,
    config: Optional[FP16Config] = None,
) -> str:
    """Convert an ONNX model to FP16 precision.

    Inserts Cast nodes at inputs/outputs and converts weights and
    computation to FP16. The resulting model can be run with ONNX Runtime
    on any provider, with actual FP16 acceleration on supported GPUs.

    Args:
        onnx_model_path: Path to input FP32 ONNX model.
        output_path: Path to save FP16 model. If None, appends '_fp16'.
        config: FP16 conversion configuration.

    Returns:
        Absolute path to the FP16 model.
    """
    import onnx
    from onnx import helper, TensorProto

    if config is None:
        config = FP16Config()

    if output_path is None:
        base, ext = os.path.splitext(onnx_model_path)
        output_path = f"{base}_fp16{ext}"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    model = onnx.load(onnx_model_path)

    # Method 1: Try onnxruntime's float16 converter first (best quality)
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        fp16_model = convert_float_to_float16(
            model,
            keep_io_types=config.keep_io_in_fp32,
            op_blocklist=list(config.op_blocklist),
        )
        onnx.save(fp16_model, output_path)
        return os.path.abspath(output_path)
    except (ImportError, Exception):
        pass

    # Method 2: Manual conversion using onnx helper
    # Convert weights to FP16
    graph = model.graph

    if config.convert_weights:
        for initializer in graph.initializer:
            if initializer.data_type == TensorProto.FLOAT:
                # Convert float32 data to float16
                float_data = np.array(
                    list(initializer.float_data), dtype=np.float32
                ) if initializer.float_data else onnx.numpy_helper.to_array(initializer)
                fp16_data = float_data.astype(np.float16)
                new_initializer = onnx.numpy_helper.from_array(fp16_data, name=initializer.name)
                # Copy over raw_data
                initializer.CopyFrom(new_initializer)

    # Add Cast nodes for inputs (FP32 -> FP16)
    if config.keep_io_in_fp32:
        new_nodes = []
        input_names = {inp.name for inp in graph.input}
        output_names = {out.name for out in graph.output}

        # For each input that is used by nodes, add a Cast node
        cast_count = 0
        for inp in graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                cast_name = f"fp16_cast_input_{cast_count}"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp.name],
                    outputs=[cast_name],
                    to=TensorProto.FLOAT16,
                    name=f"cast_input_{cast_count}",
                )
                new_nodes.append(cast_node)
                # Replace references to this input in nodes
                for node in graph.node:
                    for i, inp_name in enumerate(node.input):
                        if inp_name == inp.name:
                            node.input[i] = cast_name
                cast_count += 1

        # Add Cast nodes for outputs (FP16 -> FP32)
        for out in graph.output:
            if out.type.tensor_type.elem_type == TensorProto.FLOAT:
                # Find the node that produces this output
                fp16_output_name = f"fp16_output_{out.name}"
                for node in graph.node:
                    for i, out_name in enumerate(node.output):
                        if out_name == out.name:
                            node.output[i] = fp16_output_name

                cast_node = helper.make_node(
                    "Cast",
                    inputs=[fp16_output_name],
                    outputs=[out.name],
                    to=TensorProto.FLOAT,
                    name=f"cast_output_{out.name}",
                )
                new_nodes.append(cast_node)

        # Insert cast nodes at the beginning
        for node in reversed(new_nodes):
            graph.node.insert(0, node)

    # Update value info types
    for value_info in graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Save model
    onnx.save(model, output_path)
    return os.path.abspath(output_path)


def evaluate_fp16_impact(
    original_model_path: str,
    fp16_model_path: str,
    observation_shape: Sequence[int],
    num_samples: int = 500,
    mse_threshold: float = 0.01,
) -> dict:
    """Evaluate the impact of FP16 conversion on action accuracy.

    Args:
        original_model_path: Path to original FP32 ONNX model.
        fp16_model_path: Path to FP16 ONNX model.
        observation_shape: Shape of a single observation.
        num_samples: Number of test samples.
        mse_threshold: Maximum acceptable action MSE.

    Returns:
        Dictionary with evaluation results and recommendation.
    """
    from ..benchmark.accuracy import compare_actions, generate_test_observations
    from ..runtime.onnx_runtime import OnnxRuntimeInference
    from .int8 import compare_model_sizes

    orig_inf = OnnxRuntimeInference(original_model_path)
    fp16_inf = OnnxRuntimeInference(fp16_model_path)

    observations = generate_test_observations(observation_shape, num_samples)

    orig_actions = []
    fp16_actions = []
    for obs in observations:
        orig_actions.append(orig_inf.infer(obs).actions[0])
        fp16_actions.append(fp16_inf.infer(obs).actions[0])
    orig_actions = np.array(orig_actions)
    fp16_actions = np.array(fp16_actions)

    acc = compare_actions(orig_actions, fp16_actions, observations)
    size_info = compare_model_sizes(original_model_path, fp16_model_path)

    mse_ok = acc.action_mse < mse_threshold
    cosine_ok = acc.action_cosine_similarity > 0.99

    if mse_ok and cosine_ok:
        verdict = "pass"
        recommendation = "FP16 ACCEPTABLE: Action deviation is within thresholds. FP16 deployment is safe and recommended for GPU acceleration."
    elif mse_ok:
        verdict = "caution"
        recommendation = "FP16 CAUTION: MSE acceptable but cosine similarity low. Action direction may deviate. Consider selective FP16 (keep critical layers in FP32)."
    elif cosine_ok:
        verdict = "caution"
        recommendation = "FP16 CAUTION: Cosine similarity good but MSE exceeds threshold. Action magnitude may deviate. Verify on actual environment rollouts."
    else:
        verdict = "fail"
        recommendation = "FP16 NOT RECOMMENDED: Significant action deviation. Use FP32 or selective FP16 with op_blocklist."

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
        "size_comparison": size_info,
        "num_test_samples": num_samples,
    }


def get_fp16_supported_gpus() -> list:
    """Return list of GPU architectures with native FP16 support.

    Useful for determining whether FP16 will provide actual speedup.
    """
    return [
        # NVIDIA
        "NVIDIA Ampere (A100, A30, RTX 30-series)",
        "NVIDIA Ada Lovelace (RTX 40-series)",
        "NVIDIA Hopper (H100)",
        "NVIDIA Turing (RTX 20-series, T4) - limited FP16",
        "NVIDIA Jetson Xavier (AGX Xavier, Xavier NX)",
        "NVIDIA Jetson Orin (AGX Orin, Orin NX, Orin Nano)",
        # AMD
        "AMD RDNA2 (RX 6000-series)",
        "AMD RDNA3 (RX 7000-series)",
        "AMD CDNA (Instinct MI100/MI200)",
    ]

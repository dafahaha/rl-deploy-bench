"""TensorRT inference backend for high-performance deployment.

Provides ONNX to TensorRT Engine conversion, FP16/INT8 optimization,
and high-throughput inference. Falls back gracefully with clear
installation instructions when TensorRT is not available.

TensorRT delivers significant speedups over ONNX Runtime on NVIDIA GPUs
by optimizing the computation graph for the specific GPU architecture.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

TENSORRT_AVAILABLE = False
TENSORRT_IMPORT_ERROR = None

try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_IMPORT_ERROR = str(e)


TENSORRT_INSTALL_GUIDE = """
TensorRT is not installed. To use the TensorRT backend:

1. Install CUDA Toolkit (11.8+ or 12.x):
   https://developer.nvidia.com/cuda-toolkit

2. Install TensorRT:
   - Windows/Linux: Download from https://developer.nvidia.com/tensorrt
   - pip install tensorrt (may require CUDA to be installed first)

3. Verify installation:
   python -c "import tensorrt; print(tensorrt.__version__)"

For Jetson devices, TensorRT comes pre-installed with JetPack.
"""


def is_tensorrt_available() -> bool:
    """Check if TensorRT is available."""
    return TENSORRT_AVAILABLE


def require_tensorrt():
    """Raise a clear error if TensorRT is not available."""
    if not TENSORRT_AVAILABLE:
        raise ImportError(
            f"TensorRT is not available: {TENSORRT_IMPORT_ERROR}\n{TENSORRT_INSTALL_GUIDE}"
        )


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT engine building."""

    precision: str = "fp16"  # 'fp32', 'fp16', 'int8'
    max_workspace_size: int = 1 << 30  # 1 GB
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 32
    use_dla: bool = False  # Jetson DLA (Deep Learning Accelerator)
    dla_core: int = 0
    builder_optimization_level: int = 3  # 0-5, higher = more optimization


class TensorRTEngine:
    """TensorRT inference engine wrapper.

    Handles ONNX to TensorRT Engine conversion and high-performance
    inference on NVIDIA GPUs.
    """

    def __init__(self, engine_path: Optional[str] = None):
        """Initialize TensorRT engine.

        Args:
            engine_path: Path to a pre-built TensorRT engine file.
                If None, use build_from_onnx() to build one.
        """
        require_tensorrt()
        self.engine = None
        self.context = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self._logger = trt.Logger(trt.Logger.WARNING)
        # Cached device buffers. Allocated lazily on first infer() based on the
        # actual binding shape, then reused across calls to avoid the very large
        # per-call cuda.mem_alloc/free overhead that otherwise dominates latency
        # benchmarking. Freed in close()/__del__.
        self._d_input = None
        self._d_output = None
        self._d_input_nbytes = 0
        self._d_output_nbytes = 0

        if engine_path and os.path.exists(engine_path):
            self.load_engine(engine_path)

    def build_from_onnx(
        self,
        onnx_path: str,
        output_path: Optional[str] = None,
        config: Optional[TensorRTConfig] = None,
        observation_shape: Optional[Sequence[int]] = None,
    ) -> str:
        """Build a TensorRT engine from an ONNX model.

        Args:
            onnx_path: Path to input ONNX model.
            output_path: Path to save the TensorRT engine.
            config: TensorRT build configuration.
            observation_shape: Shape of a single observation (without batch dim).
                Required for dynamic batch size models.

        Returns:
            Path to the saved TensorRT engine.
        """
        require_tensorrt()

        if config is None:
            config = TensorRTConfig()

        builder = trt.Builder(self._logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self._logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = []
                for i in range(parser.num_errors):
                    errors.append(parser.get_error(i))
                raise RuntimeError(f"Failed to parse ONNX model: {errors}")

        # Configure builder
        builder_config = builder.create_builder_config()
        builder_config.max_workspace_size = config.max_workspace_size
        builder_config.builder_optimization_level = config.builder_optimization_level

        # Set precision
        if config.precision == "fp16" and builder.platform_has_fast_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        elif config.precision == "int8" and builder.platform_has_fast_int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
            # Note: For proper INT8, calibration data is needed.
            # This uses default calibration which may not be optimal for RL.
            # Use static_quantize_with_dataset first for better INT8 results.

        # DLA support (Jetson only)
        if config.use_dla and builder.num_DLA_cores > 0:
            builder_config.default_device_type = trt.DeviceType.DLA
            builder_config.DLA_core = min(config.dla_core, builder.num_DLA_cores - 1)

        # Set optimization profile for dynamic batch
        if observation_shape is not None:
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            min_shape = (config.min_batch_size, *observation_shape)
            opt_shape = (config.opt_batch_size, *observation_shape)
            max_shape = (config.max_batch_size, *observation_shape)
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            builder_config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, builder_config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        if output_path is None:
            base, _ = os.path.splitext(onnx_path)
            output_path = f"{base}_{config.precision}.engine"

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(serialized_engine)

        # Load engine for inference
        self.load_engine(output_path)

        return os.path.abspath(output_path)

    def load_engine(self, engine_path: str):
        """Load a pre-built TensorRT engine from file.

        Args:
            engine_path: Path to the .engine file.
        """
        require_tensorrt()

        runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output info
        self.input_name = self.engine.get_tensor_name(0)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        # Find first output
        for i in range(1, self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_name = name
                break

    def infer(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference with TensorRT.

        Args:
            observation: Input observation array (batch_size, *obs_shape).

        Returns:
            Tuple of (output_actions, latency_ms).
        """
        require_tensorrt()

        if self.context is None:
            raise RuntimeError("Engine not loaded. Call build_from_onnx() or load_engine() first.")

        import pycuda.autoinit  # noqa: F401  (side-effect: initializes CUDA context)
        import pycuda.driver as cuda

        # Ensure batch dimension
        if observation.ndim == len(self.input_shape) - 1:
            observation = observation[np.newaxis]

        # Set input shape
        self.context.set_input_shape(self.input_name, observation.shape)

        # Allocate host output buffer sized for the post-set_input_shape output.
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, dtype=np.float32)

        # (Re)allocate device buffers only when size changes; otherwise reuse.
        # This avoids cuda.mem_alloc/free on every inference, which previously
        # dominated measured latency and made benchmark numbers meaningless.
        in_nbytes = observation.nbytes
        out_nbytes = output.nbytes
        if self._d_input is None or in_nbytes != self._d_input_nbytes:
            if self._d_input is not None:
                self._d_input.free()
            self._d_input = cuda.mem_alloc(in_nbytes)
            self._d_input_nbytes = in_nbytes
        if self._d_output is None or out_nbytes != self._d_output_nbytes:
            if self._d_output is not None:
                self._d_output.free()
            self._d_output = cuda.mem_alloc(out_nbytes)
            self._d_output_nbytes = out_nbytes

        # Copy input to device
        cuda.memcpy_htod(self._d_input, observation)

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self._d_input))
        self.context.set_tensor_address(self.output_name, int(self._d_output))

        # Run inference
        start = time.perf_counter()
        self.context.execute_async_v3(stream_handle=0)
        cuda.Context.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        # Copy output to host
        cuda.memcpy_dtoh(output, self._d_output)

        return output, latency_ms

    def close(self) -> None:
        """Release cached device buffers. Safe to call multiple times."""
        if self._d_input is not None:
            try:
                self._d_input.free()
            except Exception:
                pass
            self._d_input = None
            self._d_input_nbytes = 0
        if self._d_output is not None:
            try:
                self._d_output.free()
            except Exception:
                pass
            self._d_output = None
            self._d_output_nbytes = 0

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def get_engine_info(self) -> dict:
        """Get information about the loaded engine."""
        if self.engine is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "output_name": self.output_name,
            "num_layers": self.engine.num_layers,
            "engine_capability": str(self.engine.engine_capability),
        }


def convert_onnx_to_tensorrt(
    onnx_path: str,
    output_path: Optional[str] = None,
    precision: str = "fp16",
    observation_shape: Optional[Sequence[int]] = None,
) -> str:
    """Convenience function to convert ONNX to TensorRT engine.

    Args:
        onnx_path: Path to input ONNX model.
        output_path: Path to save TensorRT engine.
        precision: Precision mode ('fp32', 'fp16', 'int8').
        observation_shape: Shape of a single observation.

    Returns:
        Path to the saved TensorRT engine.
    """
    config = TensorRTConfig(precision=precision)
    engine = TensorRTEngine()
    return engine.build_from_onnx(
        onnx_path, output_path, config=config, observation_shape=observation_shape
    )

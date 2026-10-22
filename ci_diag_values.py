"""Temporary CI diagnostic: emit CPU info and the first DynamicQuantizeLinear
outputs (quantized/scale/zp) to localize the machine-specific runtime error."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

import numpy as np
import torch

from test_core import SimplePolicy
from rl_deploy_bench.exporter.onnx_export import export_to_onnx
from rl_deploy_bench.quantizer.int8 import dynamic_quantize, evaluate_quantization

import onnx
import onnxruntime as ort

# CPU model name
try:
    cpu = [l for l in open('/proc/cpuinfo').read().splitlines() if 'model name' in l][0]
except Exception as exc:
    cpu = f"cpuinfo unavailable: {exc}"
print(f"::warning title=cpu::{cpu.strip()}")

torch.manual_seed(0)
policy = SimplePolicy()
policy.eval()

d = tempfile.mkdtemp()
onnx_path = os.path.join(d, "test.onnx")
export_to_onnx(policy, (4,), onnx_path)
q = dynamic_quantize(onnx_path)

# Run the first DQL node in isolation on the first 3 observations.
m = onnx.load(q)
first = next(n for n in m.graph.node if n.op_type == 'DynamicQuantizeLinear')
for out in first.output:
    m.graph.output.append(onnx.helper.make_tensor_value_info(out, 1, None))
m.graph.output[-3].type.tensor_type.elem_type = onnx.TensorProto.UINT8
m.graph.output[-1].type.tensor_type.elem_type = onnx.TensorProto.UINT8
sess = ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])

for idx in range(3):
    obs = np.random.RandomState(0).randn(3, 4).astype(np.float32)[idx:idx+1]
    qy, sc, zp = sess.run(None, {'observation': obs})
    print(f"::warning title=dql obs{idx}::q={qy[0].tolist()} scale={float(sc):.6f} zp={int(zp)}")

r = evaluate_quantization(onnx_path, q, (4,), num_samples=20, seed=0)
print(f"::warning title=verdict::mse={r['action_mse']:.6e} cos={r['cosine_similarity']:.6f} {r['verdict']}")

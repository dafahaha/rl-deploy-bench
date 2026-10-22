"""Pytest configuration: make the test environment deterministic.

The CI runner is a shared, CPU-limited machine (2 cores). Without limiting
torch's thread pool, concurrent tests / a loaded runner can vary the order
and timing of operations and produce intermittent failures. Seeding the
global RNGs and pinning torch to a single thread removes that source of
flakiness; individual tests that need a specific seed still set their own.
"""

from __future__ import annotations

import os

# Pin thread pools before importing numpy/torch.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

np.random.seed(0)
torch.manual_seed(0)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Interop threads can only be set once and before any interop work;
    # ignore if the runtime has already configured them.
    pass

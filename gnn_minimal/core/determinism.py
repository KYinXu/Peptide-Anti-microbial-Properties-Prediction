"""Reproducible GPU inference: scatter/cuBLAS kernels otherwise vary run to run."""

from __future__ import annotations

import os


def enable_deterministic_inference() -> None:
    """Make inference bit-identical across runs. Call before the first CUDA op."""
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

"""Subprocess environment tweaks (e.g. MKL vs libgomp on Linux/WSL)."""

from __future__ import annotations

import os


def subprocess_env() -> dict[str, str]:
    """
    Child processes (ESMFold, torch, numpy): MKL_THREADING_LAYER=INTEL often conflicts
    with libgomp; GNU is compatible with mixed OpenMP stacks.
    """
    env = os.environ.copy()
    cur = (env.get("MKL_THREADING_LAYER") or "").strip().upper()
    if cur in ("", "INTEL"):
        env["MKL_THREADING_LAYER"] = "GNU"
    return env

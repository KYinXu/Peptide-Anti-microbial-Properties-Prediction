#!/usr/bin/env python3
"""
Backward-compatible CLI for sequence normalization.

Implementation lives in peptide_pipeline.steps.normalize.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from peptide_pipeline.steps.normalize import main, normalize_to_canonical

__all__ = ["normalize_to_canonical", "main"]

if __name__ == "__main__":
    main()

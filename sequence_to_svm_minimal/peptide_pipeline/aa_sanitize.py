"""Single-letter sequences safe for ESM-2 / ESMFold (invalid chars → X)."""

from __future__ import annotations

# fair-esm protein standard_toks for ESM-2 checkpoints (single-letter residues)
ESM2_SINGLE_LETTER = frozenset("ARNDCQEGHILKMFPSTWYVXBZUO")


def sanitize_for_esm2(seq: str) -> str:
    s = seq.upper()
    return "".join(c if c in ESM2_SINGLE_LETTER else "X" for c in s)

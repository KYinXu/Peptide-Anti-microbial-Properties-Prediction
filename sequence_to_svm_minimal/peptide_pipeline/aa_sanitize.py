"""Single-letter sequences: standard-20 filtering vs ESM-2 extended alphabet."""

from __future__ import annotations

# fair-esm protein standard_toks for ESM-2 checkpoints (single-letter residues)
ESM2_SINGLE_LETTER = frozenset("ARNDCQEGHILKMFPSTWYVXBZUO")

# Standard proteinogenic one-letter codes (pipeline graphs / training use this set only)
STANDARD_AA_20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def canonical_standard_aa_sequence(seq: str) -> str | None:
    """
    Uppercase, strip spaces/tabs. Return the string if non-empty and every letter
    is a standard amino acid; otherwise None (invalid letters, X, U, digits, etc.).
    """
    s = seq.replace(" ", "").replace("\t", "").upper()
    if not s:
        return None
    for c in s:
        if c not in STANDARD_AA_20:
            return None
    return s


def sanitize_for_esm2(seq: str) -> str:
    s = seq.upper()
    return "".join(c if c in ESM2_SINGLE_LETTER else "X" for c in s)

"""Substitution schemes: physicochemical-preserving, BLOSUM62-softmax, uniform."""

from __future__ import annotations

import math
import random

from Bio.Align import substitution_matrices


STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Physicochemical-preserving substitutions (Figure 5, rules 1-6). A residue absent
# from this table is never substituted: P and C are protected (rule 6), and G has no
# allowed mutation since rule 5 lists A -> G only.
PHYSICOCHEMICAL = {
    "K": "RH", "R": "KH", "H": "KR",
    "D": "E", "E": "D",
    "F": "YW", "Y": "FW", "W": "FY",
    "S": "TNQ", "T": "SNQ", "N": "QST", "Q": "NST",
    "L": "IVM", "I": "LVM", "V": "LIM", "M": "LIV",
    "A": "G",
}

SCHEMES = ("physicochemical", "blosum62", "uniform")

DEFAULT_BLOSUM_TEMPERATURE = 1.0

_BLOSUM62 = substitution_matrices.load("BLOSUM62")


def _blosum_weights(temperature: float) -> dict[str, tuple[str, list[float]]]:
    """Softmax over each BLOSUM62 row, parent residue excluded and renormalized."""
    table = {}
    for origin in STANDARD_AA:
        targets = [aa for aa in STANDARD_AA if aa != origin]
        scores = [float(_BLOSUM62[origin, aa]) / temperature for aa in targets]
        top = max(scores)
        weights = [math.exp(score - top) for score in scores]
        table[origin] = ("".join(targets), weights)
    return table


def substitutions_for(scheme: str, residue: str, blosum_table) -> tuple[str, list[float] | None]:
    """Allowed replacements for one residue, with sampling weights when weighted."""
    if scheme == "physicochemical":
        return PHYSICOCHEMICAL.get(residue, ""), None
    if scheme == "uniform":
        return "".join(aa for aa in STANDARD_AA if aa != residue), None
    if scheme == "blosum62":
        return blosum_table[residue]
    raise ValueError(f"Unknown mutation scheme: {scheme}")


def eligible_positions(sequence: str, scheme: str, blosum_table) -> list[int]:
    return [
        index
        for index, residue in enumerate(sequence)
        if residue in STANDARD_AA and substitutions_for(scheme, residue, blosum_table)[0]
    ]


def substitution_count(p: float, length: int) -> int:
    """k = round(p * L), half-up (Python's round() is half-to-even)."""
    return int(math.floor(p * length + 0.5))


def sample_replacement(
    rng: random.Random, scheme: str, residue: str, blosum_table
) -> str:
    choices, weights = substitutions_for(scheme, residue, blosum_table)
    if weights is None:
        return rng.choice(choices)
    return rng.choices(choices, weights=weights, k=1)[0]


def build_blosum_table(temperature: float = DEFAULT_BLOSUM_TEMPERATURE):
    if temperature <= 0:
        raise ValueError("--blosum-temperature must be > 0.")
    return _blosum_weights(temperature)

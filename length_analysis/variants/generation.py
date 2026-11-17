"""Enumerate N/C flank extensions for parent peptides."""

from __future__ import annotations

import math
from typing import Iterable, Iterator

from .common import ParentRecord, VariantRecord
from .deletion import TERMINI


def generate_variants(
    parents: Iterable[ParentRecord],
    *,
    max_flank_fraction: float = 0.5,
    stride: int = 1,
    termini: Iterable[str] = TERMINI,
    separate_termini: bool = False,
) -> Iterator[VariantRecord]:
    if max_flank_fraction < 0:
        raise ValueError("--max-flank-fraction must be >= 0.")
    if stride < 1:
        raise ValueError("--stride must be at least 1.")
    termini = tuple(termini)
    for terminus in termini:
        if terminus not in TERMINI:
            raise ValueError(f"Unknown terminus: {terminus!r} (expected 'N' or 'C').")
    if not termini:
        raise ValueError("--termini must list at least one terminus.")

    for parent in parents:
        yield from generate_variants_for_parent(
            parent,
            max_flank_fraction=max_flank_fraction,
            stride=stride,
            termini=termini,
            separate_termini=separate_termini,
        )


def generate_variants_for_parent(
    parent: ParentRecord,
    *,
    max_flank_fraction: float = 0.5,
    stride: int = 1,
    termini: Iterable[str] = TERMINI,
    separate_termini: bool = False,
) -> Iterator[VariantRecord]:
    core_length = len(parent.sequence)
    max_ext = math.floor(max_flank_fraction * core_length)
    left_limit = min(max_ext, len(parent.left_flank)) if "N" in termini else 0
    right_limit = min(max_ext, len(parent.right_flank)) if "C" in termini else 0

    if separate_termini:
        pairs = [(0, 0)]
        pairs += [(left, 0) for left in range(stride, left_limit + 1, stride)]
        pairs += [(0, right) for right in range(stride, right_limit + 1, stride)]
    else:
        pairs = [
            (left, right)
            for left in range(0, left_limit + 1, stride)
            for right in range(0, right_limit + 1, stride)
        ]

    for left, right in pairs:
        sequence = (
            (parent.left_flank[-left:] if left else "")
            + parent.sequence
            + parent.right_flank[:right]
        )
        yield VariantRecord(
            variant_id=f"{parent.id}_L{left}_R{right}",
            parent_id=parent.id,
            sequence=sequence,
            left_added=left,
            right_added=right,
            core_length=core_length,
            source_protein_id=parent.source_protein_id,
            start=parent.start,
            end=parent.end,
            extras=dict(parent.extras),
        )

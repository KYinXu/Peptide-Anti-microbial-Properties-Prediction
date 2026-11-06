"""Enumerate N/C terminal truncations for parent peptides.

Deletion is deterministic: one variant per (parent, terminus, p), so there is no
replicate axis. k = round(p * L) uses the same half-up rule as the mutation sweep
(Python's round() is half-to-even and would give k=2 for L=10, p=0.25).
"""

from __future__ import annotations

import math
from typing import Iterable, Iterator

from .common import DeletionRecord, SubsetRecord


DEFAULT_P_LEVELS = (0.05, 0.10, 0.20)
TERMINI = ("N", "C")


def deletion_count(p: float, length: int) -> int:
    return int(math.floor(p * length + 0.5))


def generate_deletions(
    parents: Iterable[SubsetRecord],
    *,
    p_levels: Iterable[float] = DEFAULT_P_LEVELS,
    termini: Iterable[str] = TERMINI,
) -> Iterator[DeletionRecord]:
    p_levels = list(p_levels)
    termini = list(termini)
    for terminus in termini:
        if terminus not in TERMINI:
            raise ValueError(f"Unknown terminus: {terminus!r} (expected 'N' or 'C').")

    for parent in parents:
        yield _parent_record(parent)
        length = len(parent.sequence)
        for terminus in termini:
            for p in p_levels:
                k = deletion_count(p, length)
                # k == 0 reproduces the parent exactly; k >= length leaves nothing.
                if k == 0 or k >= length:
                    continue
                yield _truncate(parent, terminus=terminus, p=p, k=k)


def _truncate(parent: SubsetRecord, *, terminus: str, p: float, k: int) -> DeletionRecord:
    sequence = parent.sequence[k:] if terminus == "N" else parent.sequence[:-k]
    return DeletionRecord(
        variant_id=f"{parent.id}_{terminus}del_p{round(p * 100):02d}",
        parent_id=parent.id,
        parent_sequence=parent.sequence,
        sequence=sequence,
        terminus=terminus,
        p=p,
        k=k,
        extras=dict(parent.extras),
    )


def _parent_record(parent: SubsetRecord) -> DeletionRecord:
    return DeletionRecord(
        variant_id=f"{parent.id}_parent",
        parent_id=parent.id,
        parent_sequence=parent.sequence,
        sequence=parent.sequence,
        terminus="parent",
        p=0.0,
        k=0,
        extras=dict(parent.extras),
    )

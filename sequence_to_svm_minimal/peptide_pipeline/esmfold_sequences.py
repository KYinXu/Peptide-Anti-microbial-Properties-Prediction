"""Stream ESMFold input sequences without holding the full corpus in RAM."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence

_PLAIN_POS_INT = re.compile(r"^[1-9]\d*$")

EsmfoldRecord = tuple[str, str, str, int]


@dataclass
class ParseStats:
    n_skipped_invalid: int = 0
    n_valid: int = 0


@dataclass(frozen=True)
class EsmfoldWorkSummary:
    n_valid: int
    n_remaining: int
    n_foldable: int


def esmfold_unique_id(idx: str, prefix: str) -> str:
    if prefix == "SEQ":
        return f"{prefix}_{idx}" if _PLAIN_POS_INT.fullmatch(idx) else idx
    return f"{prefix}_{idx}"


def iter_esmfold_sequence_file(
    input_file: str | Path,
    label: int,
    prefix: str,
    stats: ParseStats | None = None,
) -> Iterator[EsmfoldRecord]:
    """
    Yield ``(unique_id, original_idx, sequence, label)`` one record at a time.

    Same TXT rules as the previous list-based parser: blank / ``#`` lines skipped;
    ``id seq`` or bare ``seq``; non-standard letters dropped.
    """
    n_valid = 0
    with open(input_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(None, 1)
            if len(parts) == 2:
                idx, seq = parts[0].strip(), parts[1].strip()
            elif len(parts) == 1:
                seq = parts[0].strip()
                idx = str(n_valid + 1)
            else:
                continue

            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                if stats is not None:
                    stats.n_skipped_invalid += 1
                continue

            n_valid += 1
            if stats is not None:
                stats.n_valid += 1
            yield esmfold_unique_id(idx, prefix), idx, canon, label


def iter_esmfold_inputs(
    *,
    unlabeled: bool,
    amp_file: str | Path | None,
    decoy_file: str | Path | None = None,
    amp_only: bool = False,
    decoy_only: bool = False,
    stats: ParseStats | None = None,
) -> Iterator[EsmfoldRecord]:
    if unlabeled:
        if amp_file is None:
            return
        yield from iter_esmfold_sequence_file(amp_file, 0, "SEQ", stats)
        return
    if not decoy_only and amp_file is not None:
        yield from iter_esmfold_sequence_file(amp_file, 1, "AMP", stats)
    if not amp_only and decoy_file is not None:
        yield from iter_esmfold_sequence_file(decoy_file, -1, "DECOY", stats)


def summarize_esmfold_work(
    records: Iterable[EsmfoldRecord],
    *,
    completed_ids: set[str],
    max_length: int,
) -> EsmfoldWorkSummary:
    n_valid = 0
    n_remaining = 0
    n_foldable = 0
    for unique_id, _, seq, _ in records:
        n_valid += 1
        if unique_id in completed_ids:
            continue
        n_remaining += 1
        if len(seq) <= max_length:
            n_foldable += 1
    return EsmfoldWorkSummary(
        n_valid=n_valid,
        n_remaining=n_remaining,
        n_foldable=n_foldable,
    )

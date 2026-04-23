"""Sliding-window expansion for peptide sequences (range of lengths + stride)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Window:
    seq_index: int
    parent_id: str
    start: int
    length: int
    window_seq: str

    @property
    def window_id(self) -> str:
        return f"{self.parent_id}__s{self.start}__l{self.length}"

    @property
    def peptide_id(self) -> str:
        """Matches ESMFold unlabeled naming for numeric line ids: ``SEQ_<n>``."""
        return f"SEQ_{self.seq_index}"


def iter_window_slices(
    seq: str,
    *,
    min_len: int,
    max_len: int,
    stride: int,
) -> Iterator[tuple[int, int, str]]:
    """
    Yield ``(start, length, subsequence)`` for each window, in deterministic order:
    for each length from ``min_len`` to ``max_len``, all starts left-to-right by ``stride``.
    """
    if min_len <= 0 or max_len <= 0 or stride <= 0:
        raise ValueError("min_len, max_len, and stride must be positive")
    if min_len > max_len:
        raise ValueError("min_len cannot be greater than max_len")
    n = len(seq)
    for win_len in range(min_len, max_len + 1):
        if win_len > n:
            continue
        for start in range(0, n - win_len + 1, stride):
            yield start, win_len, seq[start : start + win_len]


def expand_records_to_windows(
    records: list[tuple[str, str]],
    *,
    min_len: int,
    max_len: int,
    stride: int,
) -> tuple[list[tuple[str, str]], list[Window]]:
    """
    Expand ``(parent_id, sequence)`` into windowed canonical records and metadata.

    Canonical lines use **integer** indices (``1``, ``2``, …) as required by QSAR descriptor
    generation and ESMFold ``SEQ_<n>`` naming for unlabeled numeric ids.

    Parents shorter than ``min_len`` yield no windows.
    """
    out: list[tuple[str, str]] = []
    meta: list[Window] = []
    seq_index = 1
    for pid, seq in records:
        if not seq:
            continue
        for start, win_len, wseq in iter_window_slices(
            seq, min_len=min_len, max_len=max_len, stride=stride
        ):
            w = Window(
                seq_index=seq_index,
                parent_id=pid,
                start=start,
                length=win_len,
                window_seq=wseq,
            )
            out.append((str(seq_index), wseq))
            meta.append(w)
            seq_index += 1
    return out, meta

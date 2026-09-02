"""Single-sequence sliding-window profile generation."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .analysis import build_output_row, finalize_positions, update_profiles
from .common import BestWindow, ProfileConfig, SequenceRecord, WindowRecord, WindowScorer


def profile_sequence(record: SequenceRecord, scorer: WindowScorer, config: ProfileConfig) -> dict[str, Any]:
    length = len(record.sequence)
    p_amp_sum = np.zeros(length, dtype=np.float64)
    distance_sum = np.zeros(length, dtype=np.float64)
    coverage = np.zeros(length, dtype=np.int32)
    p_amp_profile = np.full(length, np.nan, dtype=np.float64)
    distance_profile = np.full(length, np.nan, dtype=np.float64)
    best = BestWindow()
    window_count = 0
    finalized_through = -1

    for batch_end_start, windows in iter_window_batches(record.sequence, config):
        scores = scorer.score(windows)
        update_profiles(windows, scores.p_amp, scores.hyperplane_distance, p_amp_sum, distance_sum, coverage, best)
        window_count += len(windows)
        finalize_positions(
            finalized_through + 1,
            batch_end_start,
            p_amp_sum,
            distance_sum,
            coverage,
            p_amp_profile,
            distance_profile,
        )
        finalized_through = batch_end_start

    finalize_positions(
        finalized_through + 1,
        length - 1,
        p_amp_sum,
        distance_sum,
        coverage,
        p_amp_profile,
        distance_profile,
    )
    return build_output_row(record, p_amp_profile, distance_profile, coverage, best, window_count, config)


def iter_window_batches(sequence: str, config: ProfileConfig) -> Iterator[tuple[int, list[WindowRecord]]]:
    length = len(sequence)
    if length < config.min_len:
        return
    starts = range(0, length - config.min_len + 1, config.stride)
    pending: list[WindowRecord] = []
    batch_end_start = -1
    for start_index, start in enumerate(starts, start=1):
        pending.extend(windows_at_start(sequence, start, config.min_len, config.max_len))
        batch_end_start = start
        if start_index % config.batch_starts == 0:
            yield batch_end_start, pending
            pending = []
    if pending:
        yield batch_end_start, pending


def windows_at_start(sequence: str, start: int, min_len: int, max_len: int) -> list[WindowRecord]:
    windows = []
    for window_len in range(min_len, max_len + 1):
        end = start + window_len
        if end <= len(sequence):
            windows.append(WindowRecord(start=start, end=end, length=window_len, sequence=sequence[start:end]))
    return windows

"""Single-sequence sliding-window profile generation."""

from __future__ import annotations

from typing import Any, Iterator

from .aggregation import ResidueProfileAccumulator
from .analysis import build_output_row, update_best_window
from .common import BestWindow, ProfileConfig, SequenceRecord, WindowConfig, WindowRecord, WindowScorer


def profile_sequence(record: SequenceRecord, scorer: WindowScorer, config: ProfileConfig) -> dict[str, Any]:
    length = len(record.sequence)
    accumulator = ResidueProfileAccumulator.create(
        length,
        ("p_amp", "hyperplane_distance"),
        config.aggregation,
    )
    best = BestWindow()
    window_count = 0

    for _, windows in iter_window_batches(record.sequence, config):
        scores = scorer.score(windows)
        accumulator.update(
            windows,
            {
                "p_amp": scores.p_amp,
                "hyperplane_distance": scores.hyperplane_distance,
            },
        )
        update_best_window(windows, scores.p_amp, scores.hyperplane_distance, best)
        window_count += len(windows)

    profiles = accumulator.profiles()
    return build_output_row(
        record,
        profiles["p_amp"],
        profiles["hyperplane_distance"],
        best,
        window_count,
        config,
    )


def iter_window_batches(sequence: str, config: WindowConfig) -> Iterator[tuple[int, list[WindowRecord]]]:
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

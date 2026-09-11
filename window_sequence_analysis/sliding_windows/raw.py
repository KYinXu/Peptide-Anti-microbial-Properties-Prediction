"""Single-sequence raw per-window prediction generation."""

from __future__ import annotations

from typing import Any, Iterator

from .common import SequenceRecord, WindowConfig, WindowRecord, WindowScorer, WindowScores
from .sequence import iter_window_batches


def iter_raw_window_rows(
    record: SequenceRecord,
    scorer: WindowScorer,
    config: WindowConfig,
) -> Iterator[dict[str, Any]]:
    for _, windows in iter_window_batches(record.sequence, config):
        scores = scorer.score(windows)
        yield from build_raw_rows(record, windows, scores)


def build_raw_rows(
    record: SequenceRecord,
    windows: list[WindowRecord],
    scores: WindowScores,
) -> list[dict[str, Any]]:
    if len(windows) != len(scores.prediction):
        raise ValueError("Window and prediction counts do not match.")
    if len(windows) != len(scores.sigma) or len(windows) != len(scores.p_amp):
        raise ValueError("Window and score counts do not match.")
    rows = []
    for window, prediction, sigma, p_amp in zip(
        windows,
        scores.prediction,
        scores.sigma,
        scores.p_amp,
        strict=True,
    ):
        rows.append(
            {
                "id": record.id,
                "window_start_0based": window.start,
                "window_end_0based_exclusive": window.end,
                "prediction": int(prediction),
                "sigma": float(sigma),
                "p_amp": float(p_amp),
            }
        )
    return rows

"""Orchestration helpers used by model-specific runners."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .common import ProfileConfig, SequenceRecord, WindowScorer
from .progress import NullProgressReporter, ProgressReporter
from .results_io import ProfileCsvWriter
from .sequence import profile_sequence


def run_window_profile_analysis(
    records: Iterable[SequenceRecord],
    scorer: WindowScorer,
    config: ProfileConfig,
    output: Path,
    *,
    label_columns: Iterable[str] = (),
    progress: ProgressReporter | None = None,
) -> int:
    reporter = NullProgressReporter() if progress is None else progress
    count = 0
    try:
        with ProfileCsvWriter(output, label_columns) as writer:
            for record in records:
                writer.write_row(profile_sequence(record, scorer, config))
                count += 1
                reporter.update(count, record.id)
    finally:
        reporter.close()
    return count

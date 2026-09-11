"""Orchestration helpers used by model-specific runners."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator

from .common import ProfileConfig, SequenceRecord, WindowConfig, WindowScorer
from .parallel import iter_parallel_profile_rows
from .progress import NullProgressReporter, ProgressReporter
from .raw import iter_raw_window_rows
from .raw_results_io import RawWindowCsvWriter
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
    workers: int = 1,
    scorer_factory: Callable[[], WindowScorer] | None = None,
    run_id: str | None = None,
) -> int:
    reporter = NullProgressReporter() if progress is None else progress
    count = 0
    try:
        with ProfileCsvWriter(output, label_columns, run_id) as writer:
            for row in iter_profile_rows(
                records,
                scorer,
                config,
                workers=workers,
                scorer_factory=scorer_factory,
            ):
                writer.write_row(row)
                count += 1
                reporter.update(count, str(row.get("id", "")))
    finally:
        reporter.close()
    return count


def run_raw_window_analysis(
    records: Iterable[SequenceRecord],
    scorer: WindowScorer,
    config: WindowConfig,
    output: Path,
    *,
    progress: ProgressReporter | None = None,
    run_id: str | None = None,
) -> int:
    reporter = NullProgressReporter() if progress is None else progress
    count = 0
    sequences_done = 0
    try:
        with RawWindowCsvWriter(output, run_id) as writer:
            for record in records:
                for row in iter_raw_window_rows(record, scorer, config):
                    writer.write_row(row)
                    count += 1
                sequences_done += 1
                reporter.update(sequences_done, record.id)
    finally:
        reporter.close()
    return count


def iter_profile_rows(
    records: Iterable[SequenceRecord],
    scorer: WindowScorer,
    config: ProfileConfig,
    *,
    workers: int = 1,
    scorer_factory: Callable[[], WindowScorer] | None = None,
) -> Iterator[dict]:
    if workers <= 1:
        for record in records:
            yield profile_sequence(record, scorer, config)
        return
    if scorer_factory is None:
        raise ValueError("scorer_factory is required when workers > 1.")
    yield from iter_parallel_profile_rows(records, scorer_factory, config, workers)

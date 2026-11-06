"""Orchestrate N/C truncation generation, GNN scoring, and CSV writing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..models.gnn import GnnScorer
from ..variants.common import DeletionRecord, SubsetRecord
from ..variants.deletion import generate_deletions
from .aggregate import summarize_parent
from .results_io import DeletionCurveCsvWriter, DeletionRobustnessCsvWriter
from .runner import NullProgressReporter, ProgressReporter


def run_deletion_scoring(
    parents: Iterable[SubsetRecord],
    scorer: GnnScorer,
    *,
    p_levels: Iterable[float],
    termini: Iterable[str],
    output: Path,
    curves_output: Path,
    workspace: Path,
    extra_columns: Iterable[str] = (),
    progress: ProgressReporter | None = None,
    run_id: str | None = None,
) -> int:
    reporter = NullProgressReporter() if progress is None else progress
    records = list(generate_deletions(parents, p_levels=p_levels, termini=termini))
    reporter.set_labels([record.parent_id for record in records])
    with reporter.quieted():
        predictions = scorer.score_variants(records, workspace, progress=reporter.stage)

    by_parent: dict[str, list[DeletionRecord]] = defaultdict(list)
    for record in records:
        by_parent[record.parent_id].append(record)

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    for parent_records in by_parent.values():
        reference = next(
            (predictions[record.variant_id] for record in parent_records if record.terminus == "parent"),
            None,
        )
        scored = [predictions[record.variant_id] for record in parent_records]
        summary, curves = summarize_parent(parent_records, scored, reference)
        summary_rows.extend(summary)
        curve_rows.extend(curves)

    count = 0
    try:
        with DeletionCurveCsvWriter(
            curves_output, extra_columns=extra_columns, run_id=run_id
        ) as curve_writer:
            for row in curve_rows:
                curve_writer.write_row(row)
        with DeletionRobustnessCsvWriter(
            output, extra_columns=extra_columns, run_id=run_id
        ) as summary_writer:
            for row in summary_rows:
                summary_writer.write_row(row)
                count += 1
                reporter.update(count, row["parent_id"])
    finally:
        reporter.close()
    return count

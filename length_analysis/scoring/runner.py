"""Orchestrate variant generation, GNN scoring, and CSV writing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

from ..scoring.results_io import LengthPredictionCsvWriter
from ..variants.common import ParentRecord, VariantRecord
from ..variants.generation import generate_variants
from ..models.gnn import GnnPrediction, GnnScorer


class ProgressReporter(Protocol):
    def update(self, count: int, label: str = "") -> None: ...

    def close(self) -> None: ...


class NullProgressReporter:
    def update(self, count: int, label: str = "") -> None:
        return None

    def close(self) -> None:
        return None


def run_length_scoring(
    parents: Iterable[ParentRecord],
    scorer: GnnScorer,
    *,
    max_flank_fraction: float,
    stride: int,
    output: Path,
    workspace: Path,
    extra_columns: Iterable[str] = (),
    progress: ProgressReporter | None = None,
    run_id: str | None = None,
) -> int:
    reporter = NullProgressReporter() if progress is None else progress
    parent_list = list(parents)
    variants = list(
        generate_variants(
            parent_list,
            max_flank_fraction=max_flank_fraction,
            stride=stride,
        )
    )
    predictions = scorer.score_variants(variants, workspace)
    count = 0
    try:
        with LengthPredictionCsvWriter(
            output,
            extra_columns=extra_columns,
            run_id=run_id,
        ) as writer:
            for variant in variants:
                prediction = predictions[variant.variant_id]
                writer.write_row(_row_from_variant(variant, prediction))
                count += 1
                reporter.update(count, variant.variant_id)
    finally:
        reporter.close()
    return count


def _row_from_variant(variant: VariantRecord, prediction: GnnPrediction) -> dict:
    row = {
        "variant_id": variant.variant_id,
        "parent_id": variant.parent_id,
        "source_protein_id": variant.source_protein_id,
        "start": variant.start,
        "end": variant.end,
        "left_added": variant.left_added,
        "right_added": variant.right_added,
        "core_length": variant.core_length,
        "variant_length": variant.variant_length,
        "sequence": variant.sequence,
        "pred": prediction.pred,
        "prob_AMP": prediction.prob_AMP,
        "confidence": prediction.confidence,
        "logit_AMP": prediction.logit_AMP,
        "logit_nonAMP": prediction.logit_nonAMP,
        "logit_margin": prediction.logit_margin,
        "score_z": prediction.score_z,
    }
    row.update(variant.extras)
    return row

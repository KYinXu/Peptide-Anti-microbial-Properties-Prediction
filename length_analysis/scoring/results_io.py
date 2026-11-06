"""CSV output for length-extension predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from analysis_outputs import RunCsvWriter


BASE_COLUMNS = [
    "variant_id",
    "parent_id",
    "source_protein_id",
    "start",
    "end",
    "left_added",
    "right_added",
    "core_length",
    "variant_length",
    "sequence",
    "pred",
    "prob_AMP",
    "confidence",
    "logit_AMP",
    "logit_nonAMP",
    "logit_margin",
    "score_z",
]


DELETION_ROBUSTNESS_COLUMNS = [
    "parent_id",
    "terminus",
    "sequence",
    "length",
    "parent_pred",
    "parent_prob_AMP",
    "parent_logit_margin",
    "n_variants",
    "n_p_levels",
    "mean_k",
    "retention",
    "retention_first_p",
    "retention_last_p",
    "mean_prob_AMP",
    "mean_delta_prob_AMP",
    "margin_auc",
    "margin_slope",
]

DELETION_CURVE_COLUMNS = [
    "parent_id",
    "terminus",
    "p",
    "k",
    "parent_length",
    "variant_length",
    "sequence",
    "pred",
    "prob_AMP",
    "logit_margin",
    "delta_prob_AMP",
    "delta_logit_margin",
]


class DeletionRobustnessCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        *,
        extra_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        extras = [column for column in extra_columns if column not in DELETION_ROBUSTNESS_COLUMNS]
        super().__init__(path, [*DELETION_ROBUSTNESS_COLUMNS, *extras], run_id)


class DeletionCurveCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        *,
        extra_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        extras = [column for column in extra_columns if column not in DELETION_CURVE_COLUMNS]
        super().__init__(path, [*DELETION_CURVE_COLUMNS, *extras], run_id)


class LengthPredictionCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        *,
        extra_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        extras = [column for column in extra_columns if column not in BASE_COLUMNS]
        super().__init__(path, [*BASE_COLUMNS, *extras], run_id)

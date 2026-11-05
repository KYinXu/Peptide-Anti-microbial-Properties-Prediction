"""CSV output for mutation-variant predictions and per-peptide robustness scores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from analysis_outputs import RunCsvWriter


BASE_COLUMNS = [
    "mutant_id",
    "parent_id",
    "scheme",
    "p",
    "k_requested",
    "k_applied",
    "replicate",
    "mutations",
    "sequence",
    "length",
    "backbone",
    "pred",
    "prob_AMP",
    "logit_AMP",
    "logit_nonAMP",
    "logit_margin",
    "delta_prob_AMP",
    "delta_logit_margin",
]


ROBUSTNESS_COLUMNS = [
    "parent_id",
    "scheme",
    "backbone",
    "sequence",
    "length",
    "parent_pred",
    "parent_prob_AMP",
    "parent_logit_margin",
    "n_variants",
    "n_p_levels",
    "mean_k_requested",
    "mean_k_applied",
    "retention_auc",
    "retention_first_p",
    "retention_last_p",
    "p50_retention",
    "mean_prob_AMP",
    "mean_delta_prob_AMP",
    "margin_auc",
    "margin_slope",
]

CURVE_COLUMNS = [
    "parent_id",
    "scheme",
    "backbone",
    "p",
    "n_variants",
    "k_requested",
    "k_applied",
    "retention",
    "mean_prob_AMP",
    "sd_prob_AMP",
    "mean_delta_prob_AMP",
    "mean_logit_margin",
    "sd_logit_margin",
]


class MutationPredictionCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        *,
        extra_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        extras = [column for column in extra_columns if column not in BASE_COLUMNS]
        super().__init__(path, [*BASE_COLUMNS, *extras], run_id)


class MutationRobustnessCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        *,
        extra_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        extras = [column for column in extra_columns if column not in ROBUSTNESS_COLUMNS]
        super().__init__(path, [*ROBUSTNESS_COLUMNS, *extras], run_id)


class MutationCurveCsvWriter(RunCsvWriter):
    def __init__(self, path: Path, *, run_id: str | None = None) -> None:
        super().__init__(path, CURVE_COLUMNS, run_id)

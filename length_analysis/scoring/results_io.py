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

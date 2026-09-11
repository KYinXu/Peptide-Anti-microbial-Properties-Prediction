"""Compact profile CSV output for sliding-window analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from analysis_outputs import RunCsvWriter


OUTPUT_PREFIX_COLUMNS = [
    "id",
]
OUTPUT_METRIC_COLUMNS = [
    "sequence_length",
    "window_min_len",
    "window_max_len",
    "stride",
    "window_count",
    "profile_aggregation",
    "profile_index_base",
    "profile_delimiter",
    "p_amp_mean",
    "p_amp_max_residue_mean",
    "p_amp_max_residue_1based",
    "hyperplane_distance_mean",
    "hyperplane_distance_max_residue_mean",
    "hyperplane_distance_max_residue_1based",
    "best_window_p_amp",
    "best_window_hyperplane_distance",
    "best_window_start_0based",
    "best_window_end_0based_exclusive",
    "best_window_start_1based",
    "best_window_end_1based_inclusive",
    "best_window_length",
    "best_window_sequence",
    "p_amp_profile",
    "hyperplane_distance_profile",
    "p_amp_mean_profile",
    "hyperplane_distance_mean_profile",
    "p_amp_max_profile",
    "hyperplane_distance_max_profile",
]


class ProfileCsvWriter(RunCsvWriter):
    def __init__(
        self,
        path: Path,
        label_columns: Iterable[str] = (),
        run_id: str | None = None,
    ) -> None:
        super().__init__(path, profile_columns(label_columns), run_id)


def profile_columns(label_columns: Iterable[str] = ()) -> list[str]:
    known = set(OUTPUT_PREFIX_COLUMNS + OUTPUT_METRIC_COLUMNS)
    safe_label_columns = [name if name not in known else f"label_{name}" for name in label_columns]
    return OUTPUT_PREFIX_COLUMNS + safe_label_columns + OUTPUT_METRIC_COLUMNS

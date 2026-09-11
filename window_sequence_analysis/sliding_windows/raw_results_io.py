"""Per-window CSV output for raw sliding-window predictions."""

from __future__ import annotations

from pathlib import Path

from analysis_outputs import RunCsvWriter


OUTPUT_COLUMNS = [
    "id",
    "window_start_0based",
    "window_end_0based_exclusive",
    "prediction",
    "sigma",
    "p_amp",
]


class RawWindowCsvWriter(RunCsvWriter):
    def __init__(self, path: Path, run_id: str | None = None) -> None:
        super().__init__(path, OUTPUT_COLUMNS, run_id)

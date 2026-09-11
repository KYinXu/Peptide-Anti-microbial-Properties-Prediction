"""CSV writer that binds every row to its run manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable


RUN_ID_COLUMN = "run_id"


class RunCsvWriter:
    def __init__(self, path: Path, columns: Iterable[str], run_id: str | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.columns = _columns_with_run_id(columns, run_id)
        self.handle: Any = None
        self.writer: csv.DictWriter | None = None

    def __enter__(self) -> "RunCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.columns, extrasaction="ignore")
        self.writer.writeheader()
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self.handle is not None:
            self.handle.close()

    def write_row(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            raise RuntimeError("RunCsvWriter must be opened before writing rows.")
        output_row = row if self.run_id is None else {**row, RUN_ID_COLUMN: self.run_id}
        self.writer.writerow(output_row)

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.write_row(row)


def _columns_with_run_id(columns: Iterable[str], run_id: str | None) -> list[str]:
    remaining = [column for column in columns if column != RUN_ID_COLUMN]
    return remaining if run_id is None else [RUN_ID_COLUMN, *remaining]

"""Load normalized sequence CSV datasets into streamable Python objects."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from data_normalizer.shared.records import normalize_sequence
from window_sequence_analysis.sliding_windows.common import SequenceRecord


REQUIRED_COLUMNS = {"id", "sequence"}


@dataclass(frozen=True)
class NormalizedSequenceDataset:
    path: Path
    fieldnames: list[str]
    label_columns: list[str]

    @classmethod
    def from_csv(cls, path: Path) -> "NormalizedSequenceDataset":
        if not path.is_file():
            raise FileNotFoundError(f"Normalized sequence CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("Window sequence analysis accepts normalized CSV input only.")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"Normalized sequence CSV is missing required column(s): {missing}")
        label_columns = [name for name in fieldnames if name not in REQUIRED_COLUMNS]
        return cls(path=path, fieldnames=fieldnames, label_columns=label_columns)

    def records(self) -> Iterator[SequenceRecord]:
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                record_id = (row.get("id") or "").strip()
                if not record_id:
                    raise ValueError(f"{self.path}: missing id on CSV row {row_number}.")
                sequence = normalize_sequence(row.get("sequence", ""), record_id)
                yield SequenceRecord(
                    id=record_id,
                    sequence=sequence,
                    extras={name: row.get(name, "") for name in self.label_columns},
                )

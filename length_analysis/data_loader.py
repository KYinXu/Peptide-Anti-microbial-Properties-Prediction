"""Load normalized subset CSVs for length analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from data_normalizer.shared.records import normalize_sequence

from .variants.common import SubsetRecord


REQUIRED_COLUMNS = {"id", "sequence"}


class SubsetSequenceDataset:
    def __init__(self, path: Path, fieldnames: list[str], label_columns: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.label_columns = label_columns

    @classmethod
    def from_csv(cls, path: Path) -> "SubsetSequenceDataset":
        if not path.is_file():
            raise FileNotFoundError(f"Subset sequence CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("Length analysis accepts normalized CSV input only.")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"Subset CSV is missing required column(s): {missing}")
        label_columns = [name for name in fieldnames if name not in REQUIRED_COLUMNS]
        return cls(path=path, fieldnames=fieldnames, label_columns=label_columns)

    def count_records(self) -> int:
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return sum(1 for _ in reader)

    def records(self) -> Iterator[SubsetRecord]:
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                record_id = (row.get("id") or "").strip()
                if not record_id:
                    raise ValueError(f"{self.path}: missing id on CSV row {row_number}.")
                sequence = normalize_sequence(row.get("sequence", ""), record_id)
                yield SubsetRecord(
                    id=record_id,
                    sequence=sequence,
                    extras={name: row.get(name, "") or "" for name in self.label_columns},
                )

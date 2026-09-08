"""Load normalized sequence CSV datasets for sequence-level analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REQUIRED_COLUMNS = {"id", "sequence"}


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    extras: dict[str, str]


@dataclass(frozen=True)
class NormalizedSequenceDataset:
    path: Path
    fieldnames: list[str]
    extra_columns: list[str]

    @classmethod
    def from_csv(cls, path: Path) -> "NormalizedSequenceDataset":
        if not path.is_file():
            raise FileNotFoundError(f"Normalized sequence CSV not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError("Sequence analysis accepts normalized CSV input only.")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"Normalized sequence CSV is missing required column(s): {missing}")
        return cls(
            path=path,
            fieldnames=fieldnames,
            extra_columns=[name for name in fieldnames if name not in REQUIRED_COLUMNS],
        )

    def records(self) -> Iterator[SequenceRecord]:
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                record_id = (row.get("id") or "").strip()
                if not record_id:
                    raise ValueError(f"{self.path}: missing id on CSV row {row_number}.")
                yield SequenceRecord(
                    id=record_id,
                    sequence=normalize_sequence(row.get("sequence", ""), record_id),
                    extras={name: row.get(name, "") for name in self.extra_columns},
                )


def normalize_sequence(sequence: str, record_id: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"{record_id}: empty sequence.")
    return normalized

"""Shared helpers for prediction CSV analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: float
    mean: float
    maximum: float


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV contains no rows: {path}")
    return rows


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def numeric_values(rows: Iterable[dict[str, str]], column: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(column, "")
        if raw in {"", "nan", "NaN", "None", "null"}:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def summarize_numeric(values: list[float]) -> NumericSummary | None:
    if not values:
        return None
    return NumericSummary(
        count=len(values),
        minimum=min(values),
        mean=mean(values),
        maximum=max(values),
    )


def sequence_length(row: dict[str, str], sequence_column: str | None) -> int | None:
    if not sequence_column:
        return None
    sequence = row.get(sequence_column)
    return len(sequence) if sequence else None

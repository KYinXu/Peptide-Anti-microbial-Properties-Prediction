#!/usr/bin/env python3
"""Normalize sequence CSV input into the shared id/sequence dataset format."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_normalizer.shared.records import NormalizedSequenceRecord, normalize_sequence
from data_normalizer.shared.writers import write_normalized_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "normalized_sequences.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a CSV with id and sequence columns into the standard sequence dataset CSV."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CSV containing id and sequence columns.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def normalize_csv_to_csv(input_csv: Path, output_csv: Path) -> int:
    return write_normalized_csv(output_csv, iter_normalized_csv(input_csv))


def iter_normalized_csv(path: Path) -> Iterator[NormalizedSequenceRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("normalize_csv accepts CSV input only.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted({"id", "sequence"} - set(fieldnames))
        if missing:
            raise ValueError(f"Input CSV is missing required column(s): {missing}")
        extra_columns = [name for name in fieldnames if name not in {"id", "sequence"}]
        for row_number, row in enumerate(reader, start=2):
            record_id = (row.get("id") or "").strip()
            if not record_id:
                raise ValueError(f"{path}: missing id on CSV row {row_number}.")
            yield NormalizedSequenceRecord(
                id=record_id,
                sequence=normalize_sequence(row.get("sequence", ""), record_id),
                extras={name: row.get(name, "") for name in extra_columns},
            )


def main() -> int:
    args = parse_args()
    try:
        count = normalize_csv_to_csv(args.input, args.output)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {count} normalized sequence row(s) to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

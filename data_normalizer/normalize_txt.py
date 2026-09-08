#!/usr/bin/env python3
"""Normalize TXT sequence input into the shared id/sequence CSV format."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Normalize TXT rows into the standard id/sequence CSV format.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input TXT with `id sequence` rows.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def normalize_txt_to_csv(input_txt: Path, output_csv: Path) -> int:
    return write_normalized_csv(output_csv, iter_normalized_txt(input_txt))


def iter_normalized_txt(path: Path) -> Iterator[NormalizedSequenceRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Input TXT not found: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path}: line {line_number} must contain `id sequence`.")
        record_id, sequence = parts[0].strip(), parts[1].strip()
        if not record_id:
            raise ValueError(f"{path}: line {line_number} is missing an id.")
        yield NormalizedSequenceRecord(
            id=record_id,
            sequence=normalize_sequence(sequence, record_id),
        )


def main() -> int:
    args = parse_args()
    try:
        count = normalize_txt_to_csv(args.input, args.output)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {count} normalized CSV sequence row(s) to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


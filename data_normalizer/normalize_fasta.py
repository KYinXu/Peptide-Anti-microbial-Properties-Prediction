#!/usr/bin/env python3
"""Normalize FASTA records into the CSV dataset format used by prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_normalizer.shared.label_parser import parse_fasta_label
from data_normalizer.shared.records import NormalizedSequenceRecord, normalize_sequence
from data_normalizer.shared.writers import write_normalized_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "normalized_sequences.csv"
FASTA_EXTENSIONS = {".fa", ".fasta", ".faa", ".fna"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize FASTA headers and sequence lines into one CSV row per sequence."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input FASTA file (.fa, .fasta, .faa, .fna).")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    return parser.parse_args()


def normalize_fasta_to_csv(input_fasta: Path, output_csv: Path) -> int:
    return write_normalized_csv(output_csv, iter_normalized_fasta(input_fasta))


def iter_normalized_fasta(path: Path) -> Iterator[NormalizedSequenceRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {path}")
    if path.suffix.lower() not in FASTA_EXTENSIONS:
        raise ValueError("normalize_fasta accepts FASTA input only (.fa, .fasta, .faa, .fna).")

    current_header: str | None = None
    chunks: list[str] = []

    def flush() -> NormalizedSequenceRecord | None:
        if current_header is None:
            return None
        label = parse_fasta_label(current_header)
        return NormalizedSequenceRecord(
            id=label.id,
            sequence=normalize_sequence("".join(chunks), label.id),
            extras={
                **({"name": label.name} if label.name else {}),
                "rawtext": label.rawtext,
                **label.fields,
            },
        )

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            row = flush()
            if row is not None:
                yield row
            current_header = line[1:].strip()
            if not current_header:
                raise ValueError(f"{path}: FASTA header on line {line_number} is missing a sequence ID.")
            chunks = []
        else:
            if current_header is None:
                raise ValueError(f"{path}: sequence data before first FASTA header on line {line_number}.")
            chunks.append(line)

    row = flush()
    if row is not None:
        yield row


def main() -> int:
    args = parse_args()
    try:
        count = normalize_fasta_to_csv(args.input, args.output)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {count} normalized sequence row(s) to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

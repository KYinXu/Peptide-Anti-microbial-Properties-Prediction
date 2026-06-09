"""Txt / FASTA / CSV -> canonical SVM-style lines (ESMFold / ESM-2 compatible)."""

from __future__ import annotations

import argparse
from pathlib import Path

from peptide_pipeline.sequence_io import read_sequence_records, write_canonical


def _filter_len(seq: str, min_len: int | None, max_len: int | None) -> bool:
    n = len(seq)
    if min_len is not None and n < min_len:
        return False
    if max_len is not None and n > max_len:
        return False
    return True


def normalize_to_canonical(
    input_path: Path,
    output_path: Path,
    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    input_format = "fasta" if suffix in (".fa", ".fasta", ".faa") else "csv" if suffix == ".csv" else "txt"
    stats = {
        "n_written": 0,
        "n_skipped_len": 0,
        "n_skipped_empty": 0,
        "n_skipped_invalid": 0,
        "format": input_format,
    }

    inv: dict = {}
    records = read_sequence_records(input_path, invalid_stats=inv)
    stats["n_skipped_invalid"] = inv.get("n_skipped_invalid", 0)

    if stats["format"] == "txt":
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() == "":
                    stats["n_skipped_empty"] += 1

    lines_out: list[tuple[str, str]] = []
    for idx, seq in records:
        if not seq:
            stats["n_skipped_empty"] += 1
            continue
        if not _filter_len(seq, min_len, max_len):
            stats["n_skipped_len"] += 1
            continue
        lines_out.append((idx, seq))

    stats["n_written"] = len(lines_out)
    write_canonical(output_path, lines_out)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize sequence file to canonical SVM-style format.")
    ap.add_argument("--input", "-i", type=str, required=True)
    ap.add_argument("--output", "-o", type=str, required=True)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    args = ap.parse_args()
    st = normalize_to_canonical(
        Path(args.input),
        Path(args.output),
        min_len=args.min_len,
        max_len=args.max_len,
    )
    print(st)


if __name__ == "__main__":
    main()

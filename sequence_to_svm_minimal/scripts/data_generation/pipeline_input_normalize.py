#!/usr/bin/env python3
"""
Normalize txt / FASTA sequence input into SVM-style lines for ESMFold and ESM-2.

Output format (matches models/run_esmfold_peptides.parse_sequence_file):
  - Two fields per line: index sequence (whitespace-separated; index is first token)
  - One sequence per line → auto index 1, 2, 3, ...
  - Empty lines and # comments skipped
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _iter_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Return list of (record_id, sequence) from FASTA."""
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                header = line[1:].strip()
                current_id = header.split()[0] if header else str(len(records) + 1)
                chunks = []
            else:
                chunks.append(line.strip())
        if current_id is not None:
            records.append((current_id, "".join(chunks)))
    return records


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
    """
    Read input_path (plain txt or .fasta/.fa), write canonical two-field lines.

    Returns stats dict: n_written, n_skipped_empty, n_skipped_len, format (txt|fasta).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = input_path.suffix.lower()
    stats = {
        "n_written": 0,
        "n_skipped_len": 0,
        "n_skipped_empty": 0,
        "format": "fasta" if suffix in (".fa", ".fasta", ".faa") else "txt",
    }

    lines_out: list[str] = []

    if stats["format"] == "fasta":
        for rid, seq in _iter_fasta_records(input_path):
            seq = seq.upper().replace(" ", "")
            if not seq:
                stats["n_skipped_empty"] += 1
                continue
            if not _filter_len(seq, min_len, max_len):
                stats["n_skipped_len"] += 1
                continue
            lines_out.append(f"{rid} {seq}")
    else:
        auto_i = 0
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    if raw == "":
                        stats["n_skipped_empty"] += 1
                    continue
                parts = raw.split(None, 1)
                if len(parts) == 2:
                    idx, seq = parts[0].strip(), parts[1].strip()
                elif len(parts) == 1:
                    auto_i += 1
                    idx, seq = str(auto_i), parts[0].strip()
                else:
                    continue
                seq = seq.upper().replace(" ", "")
                if not seq:
                    stats["n_skipped_empty"] += 1
                    continue
                if not _filter_len(seq, min_len, max_len):
                    stats["n_skipped_len"] += 1
                    continue
                lines_out.append(f"{idx} {seq}")

    stats["n_written"] = len(lines_out)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines_out))
        if lines_out:
            out.write("\n")

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

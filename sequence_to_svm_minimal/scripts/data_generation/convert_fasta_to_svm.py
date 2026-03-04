#!/usr/bin/env python3
"""
Convert FASTA files to SVM format for ESMFold and GNN pipeline.

Reads amps.fasta and decoys.fasta, optionally subsamples decoys, filters by length,
and writes seqs_AMP.txt and seqs_decoy.txt in the format expected by run_esmfold_peptides.
"""

import argparse
import random
import sys
from pathlib import Path

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")


def _parse_same_line(header_line: str) -> tuple[str, str] | None:
    """
    Parse FASTA line where header and sequence are on same line: >IDSEQUENCE.
    Returns (id, sequence) or None if unparseable.
    """
    content = header_line[1:].strip()
    if not content:
        return None
    for i in range(1, len(content) + 1):
        candidate_seq = content[i:].upper()
        if len(candidate_seq) >= 2 and all(c in VALID_AA for c in candidate_seq):
            seq_id = content[:i].strip()
            if seq_id:
                return seq_id, candidate_seq
    return None


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """
    Parse FASTA file. Handles standard format and same-line header+sequence.
    Returns list of (id, sequence) tuples.
    """
    records = []
    current_id = None
    current_seq_parts = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq_parts).replace(" ", "").upper()
                    if seq:
                        records.append((current_id, seq))
                same_line = _parse_same_line(line)
                if same_line:
                    records.append(same_line)
                    current_id = None
                    current_seq_parts = []
                else:
                    current_id = line[1:].split()[0].strip()
                    current_seq_parts = []
            else:
                if current_id is not None:
                    current_seq_parts.append(line)
                elif records and current_seq_parts == []:
                    pass

        if current_id is not None:
            seq = "".join(current_seq_parts).replace(" ", "").upper()
            if seq:
                records.append((current_id, seq))

    return records


def _is_valid_id(rid: str, max_id_len: int = 50) -> bool:
    if not rid or len(rid) > max_id_len:
        return False
    if " " in rid or "\t" in rid:
        return False
    return True


def filter_by_length(
    records: list[tuple[str, str]],
    min_len: int,
    max_len: int,
) -> list[tuple[str, str]]:
    return [
        (rid, seq)
        for rid, seq in records
        if min_len <= len(seq) <= max_len
    ]


def filter_valid_ids(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(rid, seq) for rid, seq in records if _is_valid_id(rid)]


def subsample(
    records: list[tuple[str, str]],
    n: int,
    seed: int | None,
) -> list[tuple[str, str]]:
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def write_svm(records: list[tuple[str, str]], path: Path) -> int:
    with open(path, "w") as f:
        for rid, seq in records:
            f.write(f"{rid}\t{seq}\n")
    return len(records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FASTA to SVM format for ESMFold pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_fasta_to_svm.py
  python scripts/convert_fasta_to_svm.py --max-decoys 300 --output-dir data/gnn_training_dataset
  python scripts/convert_fasta_to_svm.py --amp-fasta amps.fasta --decoy-fasta decoys.fasta
        """,
    )
    base = Path(__file__).parent.parent / "data" / "gnn_training_dataset"
    parser.add_argument(
        "--amp-fasta",
        type=Path,
        default=base / "amps.fasta",
        help="AMP FASTA file",
    )
    parser.add_argument(
        "--decoy-fasta",
        type=Path,
        default=base / "decoys.fasta",
        help="Decoy FASTA file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as amp-fasta parent)",
    )
    parser.add_argument(
        "--max-decoys",
        type=int,
        default=10000,
        help="Subsample decoys to N (default: 10,000)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=200,
        help="Skip sequences longer than N aa (default: 200)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=5,
        help="Skip sequences shorter than N aa (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for decoy subsampling (default: 42)",
    )

    args = parser.parse_args()
    out_dir = args.output_dir or args.amp_fasta.parent

    if not args.amp_fasta.exists():
        print(f"Error: AMP FASTA not found: {args.amp_fasta}", file=sys.stderr)
        sys.exit(1)
    if not args.decoy_fasta.exists():
        print(f"Error: Decoy FASTA not found: {args.decoy_fasta}", file=sys.stderr)
        sys.exit(1)

    amp_records = parse_fasta(args.amp_fasta)
    decoy_records = parse_fasta(args.decoy_fasta)

    amp_records = filter_valid_ids(amp_records)
    decoy_records = filter_valid_ids(decoy_records)

    amp_filtered = filter_by_length(
        amp_records, args.min_length, args.max_length
    )
    decoy_filtered = filter_by_length(
        decoy_records, args.min_length, args.max_length
    )

    amp_skipped = len(amp_records) - len(amp_filtered)
    decoy_skipped = len(decoy_records) - len(decoy_filtered)
    if amp_skipped or decoy_skipped:
        print(f"Skipped (length): AMP {amp_skipped}, decoy {decoy_skipped}")

    decoy_subsampled = subsample(decoy_filtered, args.max_decoys, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    amp_path = out_dir / "seqs_AMP.txt"
    decoy_path = out_dir / "seqs_decoy.txt"

    n_amp = write_svm(amp_filtered, amp_path)
    n_decoy = write_svm(decoy_subsampled, decoy_path)

    print(f"Wrote {n_amp} AMP sequences to {amp_path}")
    print(f"Wrote {n_decoy} decoy sequences to {decoy_path}")


if __name__ == "__main__":
    main()

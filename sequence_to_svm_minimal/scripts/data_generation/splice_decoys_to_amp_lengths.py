#!/usr/bin/env python3
"""
Splice decoy sequences into multiple AMP-length windows (10-60 AA).

Target lengths are sampled from the empirical AMP length distribution (skewed,
not hardcoded). Windows are packed into the central region of each source
sequence, avoiding both ends (controlled by --end-margin). Run once per source
file, then combine with clean_fasta_file.py; use --max-records to balance sources.
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

RecordList = list[tuple[str, str]]


def parse_fasta(path: Path) -> RecordList:
    records = []
    current_id = None
    current_seq: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if current_id is not None:
                    seq = "".join(current_seq).replace(" ", "").upper()
                    if seq:
                        records.append((current_id, seq))
                current_id = line[1:].split()[0].strip() or None
                current_seq = []
            elif current_id is not None:
                current_seq.append(line.replace(" ", "").upper())
    if current_id is not None:
        seq = "".join(current_seq).replace(" ", "").upper()
        if seq:
            records.append((current_id, seq))
    return records


def write_fasta(records: RecordList, path: Path | None) -> None:
    if path is None:
        for rid, seq in records:
            print(f">{rid}\n{seq}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rid, seq in records:
                f.write(f">{rid}\n{seq}\n")


def build_length_pmf(sequences: list[str], min_len: int, max_len: int) -> np.ndarray:
    """Normalized histogram of AMP lengths over [min_len, max_len]."""
    counts = np.zeros(max_len - min_len + 1)
    for s in sequences:
        length = len(s)
        if min_len <= length <= max_len:
            counts[length - min_len] += 1
    total = counts.sum()
    if total == 0:
        raise ValueError(f"No AMP sequences found in length range [{min_len}, {max_len}].")
    return counts / total


def sample_length(pmf: np.ndarray, min_len: int, rng: np.random.Generator,
                  max_allowed: int | None = None) -> int:
    """Sample a length from the AMP PMF, optionally capped at max_allowed."""
    if max_allowed is None or max_allowed >= min_len + len(pmf) - 1:
        return int(rng.choice(len(pmf), p=pmf)) + min_len
    cutoff = max_allowed - min_len + 1
    sub = pmf[:cutoff].copy()
    sub /= sub.sum()
    return int(rng.choice(len(sub), p=sub)) + min_len


def multi_splice(seq: str, rid: str, pmf: np.ndarray, min_len: int, max_len: int,
                 rng: np.random.Generator, end_margin_fraction: float) -> RecordList:
    """Pack non-overlapping windows into the central region of seq."""
    margin = int(len(seq) * end_margin_fraction)
    usable = seq[margin: len(seq) - margin]

    segments: RecordList = []
    pos = 0
    while len(usable) - pos >= min_len:
        target_len = sample_length(pmf, min_len, rng, max_allowed=min(max_len, len(usable) - pos))
        label = f"{rid}_seg{len(segments) + 1}"
        segments.append((label, usable[pos: pos + target_len]))
        pos += target_len
    return segments


def splice_all_records(decoy_records: RecordList, pmf: np.ndarray, min_len: int,
                       max_len: int, rng: np.random.Generator,
                       end_margin_fraction: float) -> tuple[RecordList, dict]:
    out: RecordList = []
    stats = {"kept": 0, "spliced_sources": 0, "spliced_segments": 0, "discarded": 0}

    for rid, seq in decoy_records:
        if len(seq) < min_len:
            stats["discarded"] += 1
        elif len(seq) <= max_len:
            out.append((rid, seq))
            stats["kept"] += 1
        else:
            segs = multi_splice(seq, rid, pmf, min_len, max_len, rng, end_margin_fraction)
            out.extend(segs)
            stats["spliced_sources"] += 1
            stats["spliced_segments"] += len(segs)

    return out, stats


def subsample(records: RecordList, n: int, rng: np.random.Generator) -> RecordList:
    indices = rng.choice(len(records), size=n, replace=False)
    return [records[i] for i in sorted(indices)]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Splice decoy sequences to match AMP length distribution."
    )
    p.add_argument("--decoys", type=Path, required=True, help="Input decoy FASTA")
    p.add_argument("--amps", type=Path, required=True, help="Reference AMP FASTA (sets length distribution)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output FASTA (default: results/spliced_decoys_<timestamp>.fasta)")
    p.add_argument("--min-len", type=int, default=10, metavar="N")
    p.add_argument("--max-len", type=int, default=60, metavar="N")
    p.add_argument("--end-margin", type=float, default=0.15, metavar="F",
                   help="Fraction of each source sequence to exclude at each end (default: 0.15)")
    p.add_argument("--max-records", type=int, default=None, metavar="N",
                   help="Subsample output to at most N records (for balancing sources)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for path, name in [(args.decoys, "--decoys"), (args.amps, "--amps")]:
        if not path.exists():
            p.error(f"{name}: {path} not found")

    rng = np.random.default_rng(args.seed)

    amp_records = parse_fasta(args.amps)
    pmf = build_length_pmf([s for _, s in amp_records], args.min_len, args.max_len)
    print(f"Built AMP PMF from {len(amp_records)} sequences.", file=sys.stderr)

    decoy_records = parse_fasta(args.decoys)
    print(f"Loaded {len(decoy_records)} decoy sequences.", file=sys.stderr)

    out, stats = splice_all_records(
        decoy_records, pmf, args.min_len, args.max_len, rng, args.end_margin
    )

    print(f"  Kept as-is: {stats['kept']}", file=sys.stderr)
    print(f"  Spliced: {stats['spliced_sources']} sources -> {stats['spliced_segments']} segments", file=sys.stderr)
    print(f"  Discarded (<{args.min_len} AA): {stats['discarded']}", file=sys.stderr)
    print(f"  Total before subsampling: {len(out)}", file=sys.stderr)

    if args.max_records is not None and len(out) > args.max_records:
        out = subsample(out, args.max_records, rng)
        print(f"  Subsampled to {args.max_records} records.", file=sys.stderr)

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"spliced_decoys_{timestamp}.fasta"

    write_fasta(out, args.output)
    print(f"Wrote {len(out)} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

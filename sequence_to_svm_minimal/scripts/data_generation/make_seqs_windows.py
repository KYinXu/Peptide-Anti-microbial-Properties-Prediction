#!/usr/bin/env python3
"""
Make seqs.txt by windowing a raw sequence text file.

- Input file may contain spaces/newlines/etc.; we keep only A..Z and uppercase.
- Outputs two-column "seqs.txt" format: "index sequence" per line.
- Generates all sliding windows for lengths [--min-len .. --max-len] with given --stride.

Usage:
  python scripts/make_seqs_windows.py \
    --in /path/to/raw.txt \
    --out /path/to/predictionsParameters/seqs.txt \
    --min-len 10 --max-len 35 --stride 1
"""
import argparse
import os
import re

def main():
    ap = argparse.ArgumentParser(description="Window raw AA sequence file into seqs.txt format")
    ap.add_argument("--in", dest="inp", required=True, help="Input raw sequence text file")
    ap.add_argument("--out", dest="out", required=True, help="Output seqs.txt path")
    ap.add_argument("--min-len", dest="min_len", type=int, required=True, help="Minimum window length")
    ap.add_argument("--max-len", dest="max_len", type=int, required=True, help="Maximum window length")
    ap.add_argument("--stride", dest="stride", type=int, default=1, help="Stride for sliding windows (default: 1)")
    args = ap.parse_args()

    if args.min_len <= 0 or args.max_len <= 0 or args.stride <= 0:
        raise SystemExit("min-len, max-len, and stride must be positive integers")
    if args.min_len > args.max_len:
        raise SystemExit("min-len cannot be greater than max-len")

    # Read and clean
    with open(args.inp, "r") as f:
        raw = f.read()
    seq = re.sub(r"[^A-Za-z]", "", raw).upper()
    if not seq:
        raise SystemExit("No amino-acid letters found in input file")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    lines = []
    idx = 0
    for W in range(args.min_len, args.max_len + 1):
        if W > len(seq):
            continue
        for start in range(0, len(seq) - W + 1, args.stride):
            idx += 1
            lines.append(f"{idx} {seq[start:start+W]}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Wrote {idx} windows (len {args.min_len}..{args.max_len}, stride {args.stride}) -> {args.out}")

if __name__ == "__main__":
    main()

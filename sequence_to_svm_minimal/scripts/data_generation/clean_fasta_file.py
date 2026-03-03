#!/usr/bin/env python3
"""
FASTA cleaner: remove duplicates and apply optional filters.
Deduplication and each filter are toggleable via flags.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

RecordList = list[tuple[str, str]]


@dataclass
class FilterOpts:
    min_len: int | None = None
    max_len: int | None = None
    drop_empty_id: bool = False

    def active_filters(self) -> list[Callable[[RecordList], RecordList]]:
        """Ordered list of filter fns to apply (only those that are enabled)."""
        out: list[Callable[[RecordList], RecordList]] = []
        if self.min_len:
            n = self.min_len
            out.append(lambda r, n=n: filter_min_len(r, n))
        if self.max_len:
            n = self.max_len
            out.append(lambda r, n=n: filter_max_len(r, n))
        if self.drop_empty_id:
            out.append(filter_drop_empty_id)
        return out


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse FASTA; return list of (id, sequence). Sequence uppercased, whitespace removed."""
    records = []
    current_id = None
    current_seq = []

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


def write_fasta(records: list[tuple[str, str]], path: Path | None) -> None:
    if path is None:
        for rid, seq in records:
            print(f">{rid}\n{seq}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for rid, seq in records:
                f.write(f">{rid}\n{seq}\n")


def dedupe_by_seq(records: list[tuple[str, str]], keep: str = "first") -> list[tuple[str, str]]:
    seen = set()
    out = []
    for r in (reversed(records) if keep == "last" else records):
        key = r[1]
        if key not in seen:
            seen.add(key)
            out.append(r)
    if keep == "last":
        out.reverse()
    return out


def dedupe_by_id(records: list[tuple[str, str]], keep: str = "first") -> list[tuple[str, str]]:
    seen = set()
    out = []
    for r in (reversed(records) if keep == "last" else records):
        key = r[0]
        if key not in seen:
            seen.add(key)
            out.append(r)
    if keep == "last":
        out.reverse()
    return out


def filter_min_len(records: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    return [(i, s) for i, s in records if len(s) >= n]


def filter_max_len(records: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
    return [(i, s) for i, s in records if len(s) <= n]


def filter_drop_empty_id(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(i, s) for i, s in records if (i or "").strip()]


def apply_filters(records: RecordList, opts: FilterOpts) -> RecordList:
    for fn in opts.active_filters():
        records = fn(records)
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="FASTA: remove duplicates and apply optional filters.")
    p.add_argument("--input", type=Path, help="Input FASTA")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output FASTA (default: stdout)")
    p.add_argument("--keep", choices=("first", "last"), default="first",
                   help="When deduping by sequence, keep first or last occurrence (default: first)")

    g = p.add_argument_group("Filters (optional)")
    g.add_argument("--min-len", type=int, default=None, metavar="N", help="Drop sequences shorter than N")
    g.add_argument("--max-len", type=int, default=None, metavar="N", help="Drop sequences longer than N")
    g.add_argument("--drop-empty-id", action="store_true", help="Drop records with empty id")

    args = p.parse_args()

    if not args.input.exists():
        p.error(f"Input not found: {args.input}")

    opts = FilterOpts(
        min_len=args.min_len,
        max_len=args.max_len,
        drop_empty_id=args.drop_empty_id,
    )

    records = parse_fasta(args.input)
    n_in = len(records)
    
    records = apply_filters(
        records,
        opts=opts,
    )

    records = dedupe_by_seq(records, keep=args.keep)

    write_fasta(records, args.output)
    if args.output and n_in != len(records):
        print(f"Wrote {len(records)} records (from {n_in}) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

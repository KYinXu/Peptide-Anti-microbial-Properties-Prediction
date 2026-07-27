#!/usr/bin/env python3
"""Deduplicate a CSV by the sequence column, writing unique rows to a new file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_sequence_column(columns: pd.Index, name: str) -> str:
    exact = {c: c for c in columns}
    if name in exact:
        return name
    lower = {str(c).lower(): c for c in columns}
    key = name.lower()
    if key in lower:
        return lower[key]
    raise SystemExit(
        f"Column {name!r} not found. Available: {list(columns)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Drop duplicate rows by sequence column; write remaining rows to a new CSV."
    )
    ap.add_argument("input", type=Path, help="Input CSV path")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>_dedup.csv next to input)",
    )
    ap.add_argument(
        "--column",
        default="sequence",
        help="Column to deduplicate on (case-insensitive match; default: sequence)",
    )
    ap.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="Which duplicate row to keep (default: first)",
    )
    args = ap.parse_args()

    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.output
    if out is None:
        out = inp.with_name(f"{inp.stem}_dedup{inp.suffix}")

    df = pd.read_csv(inp)
    col = resolve_sequence_column(df.columns, args.column)
    before = len(df)
    deduped = df.drop_duplicates(subset=[col], keep=args.keep)
    after = len(deduped)
    deduped.to_csv(out, index=False)
    print(f"Wrote {out} ({after} rows; removed {before - after} duplicates of {before})")


if __name__ == "__main__":
    main()

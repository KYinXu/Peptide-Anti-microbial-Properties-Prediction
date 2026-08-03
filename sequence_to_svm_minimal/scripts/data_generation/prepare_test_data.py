#!/usr/bin/env python3
"""
Convert a peptide CSV into ``id sequence`` text for ``run_data_pipeline``.

Resolves sequence / id columns from built-in alias lists (or CLI overrides).
If no id column is found, assigns ``SEQ_1``, ``SEQ_2``, …

Examples:
  python scripts/data_generation/prepare_test_data.py -i peptides.csv -o seqs.txt
  python scripts/data_generation/prepare_test_data.py -i mapp.csv -o seqs.txt \\
      --sequence-col Sequence --map-csv mapp_id_map.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence

# Prefer earlier aliases when several columns match (case-insensitive).
DEFAULT_SEQUENCE_ALIASES = (
    "Sequence",
    "sequence",
    "seq",
    "SEQ",
)

DEFAULT_ID_ALIASES = (
    "peptide_id",
    "Peptide_ID",
    "Peptide id",
    "id",
)


def clean_name(name: object, *, fallback: str = "SEQ") -> str:
    """Make a whitespace-free id safe for pipeline / filesystem use."""
    s = str(name).strip()
    if not s or s.lower() in {"nan", "none"}:
        return fallback
    s = s.replace(" ", "_").replace("?", "")
    s = re.sub(r"[^a-zA-Z0-9_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or fallback


def _normalize_key(s: str) -> str:
    return re.sub(r"[\s_]+", "", str(s).strip().lower())


def resolve_column(
    columns: pd.Index,
    candidates: list[str],
    *,
    required: bool,
    kind: str,
) -> str | None:
    """Pick the first candidate that matches a column (exact, then normalized)."""
    exact = {str(c): str(c) for c in columns}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
    norm_map = {_normalize_key(c): str(c) for c in columns}
    for cand in candidates:
        key = _normalize_key(cand)
        if key in norm_map:
            return norm_map[key]
    if required:
        raise ValueError(
            f"No {kind} column found. Tried {candidates!r}. Available: {list(columns)}"
        )
    return None


def _parse_alias_list(raw: str | None, defaults: tuple[str, ...]) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(defaults)
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return parts or list(defaults)


def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode CSV: {path}")


def csv_to_pipeline_txt(
    input_path: Path,
    output_path: Path,
    *,
    sequence_aliases: list[str],
    id_aliases: list[str],
    sequence_col: str | None = None,
    id_col: str | None = None,
    id_prefix: str = "SEQ",
    map_csv: Path | None = None,
    dedup_sequence: bool = False,
    keep_invalid: bool = False,
    legacy_two_col: bool = False,
    force_auto_id: bool = False,
) -> dict:
    df = _read_csv(input_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {input_path}")

    if legacy_two_col:
        if df.shape[1] < 2:
            raise ValueError("--legacy-two-col requires at least two CSV columns (name, seq)")
        id_name = None if force_auto_id else str(df.columns[0])
        seq_name = str(df.columns[1])
    else:
        seq_name = sequence_col or resolve_column(
            df.columns, sequence_aliases, required=True, kind="sequence"
        )
        assert seq_name is not None
        if force_auto_id:
            id_name = None
        elif id_col:
            if id_col not in df.columns:
                id_name = resolve_column(df.columns, [id_col], required=True, kind="id")
            else:
                id_name = id_col
        else:
            id_name = resolve_column(
                df.columns, id_aliases, required=False, kind="id"
            )

    records: list[dict] = []
    n_skipped_empty = 0
    n_skipped_invalid = 0
    n_skipped_dup = 0
    seen_seq: set[str] = set()
    used_ids: set[str] = set()
    auto_i = 0

    for src_row, row in df.iterrows():
        raw_seq = row[seq_name]
        if pd.isna(raw_seq) or not str(raw_seq).strip():
            n_skipped_empty += 1
            continue
        canon = canonical_standard_aa_sequence(str(raw_seq))
        if canon is None:
            n_skipped_invalid += 1
            if not keep_invalid:
                continue
            canon = re.sub(r"\s+", "", str(raw_seq).upper())

        if dedup_sequence and canon in seen_seq:
            n_skipped_dup += 1
            continue
        seen_seq.add(canon)

        if id_name is not None and not pd.isna(row[id_name]) and str(row[id_name]).strip():
            base = clean_name(row[id_name], fallback=f"{id_prefix}_0")
        else:
            auto_i += 1
            base = f"{id_prefix}_{auto_i}"

        pid = base
        suffix = 2
        while pid in used_ids:
            pid = f"{base}_{suffix}"
            suffix += 1
        used_ids.add(pid)

        records.append(
            {
                "peptide_id": pid,
                "sequence": canon,
                "source_row": int(src_row) if isinstance(src_row, (int,)) else src_row,
                "source_id_column": id_name,
                "source_id_value": (
                    None if id_name is None or pd.isna(row[id_name]) else str(row[id_name])
                ),
                "source_sequence_column": seq_name,
            }
        )

    if not records:
        raise ValueError(
            f"No sequences written from {input_path} "
            f"(empty={n_skipped_empty}, invalid_aa={n_skipped_invalid}, dup={n_skipped_dup})"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r['peptide_id']} {r['sequence']}\n")

    if map_csv is not None:
        map_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(map_csv, index=False)

    return {
        "n_written": len(records),
        "n_skipped_empty": n_skipped_empty,
        "n_skipped_invalid": n_skipped_invalid,
        "n_skipped_dup": n_skipped_dup,
        "sequence_column": seq_name,
        "id_column": id_name,
        "output": str(output_path.resolve()),
        "map_csv": str(map_csv.resolve()) if map_csv is not None else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Convert a peptide CSV to id+sequence text for run_data_pipeline. "
            "Auto-detects sequence/id columns from alias lists; missing id → SEQ_n."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/data_generation/prepare_test_data.py -i data.csv -o seqs.txt\n"
            "  python scripts/data_generation/prepare_test_data.py -i mapp.csv -o seqs.txt \\\n"
            "      --sequence-aliases Sequence,seq --id-prefix SEQ --map-csv mapp_map.csv\n"
            "\n"
            "Then:\n"
            "  python scripts/run_data_pipeline.py --mode blind --input seqs.txt --work-dir .../generated\n"
        ),
    )
    ap.add_argument("--input", "-i", type=Path, required=True, help="Input CSV path")
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output pipeline txt (peptide_id sequence per line)",
    )
    ap.add_argument(
        "--sequence-col",
        type=str,
        default=None,
        help="Exact sequence column name (skips alias search)",
    )
    ap.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Exact id column name (skips alias search). If omitted and no alias matches, use SEQ_n.",
    )
    ap.add_argument(
        "--sequence-aliases",
        type=str,
        default=None,
        help=(
            "Comma-separated sequence column candidates "
            f"(default: {','.join(DEFAULT_SEQUENCE_ALIASES[:6])},...)"
        ),
    )
    ap.add_argument(
        "--id-aliases",
        type=str,
        default=None,
        help=(
            "Comma-separated id column candidates "
            f"(default: {','.join(DEFAULT_ID_ALIASES[:6])},...)"
        ),
    )
    ap.add_argument(
        "--id-prefix",
        type=str,
        default="SEQ",
        help="Prefix for auto ids when no id column is used (default: SEQ → SEQ_1,...)",
    )
    ap.add_argument(
        "--auto-id",
        action="store_true",
        help="Always assign SEQ_n (or --id-prefix_n); ignore id columns / --id-col",
    )
    ap.add_argument(
        "--map-csv",
        type=Path,
        default=None,
        help="Optional sidecar CSV mapping peptide_id ↔ source row / original id",
    )
    ap.add_argument(
        "--dedup-sequence",
        action="store_true",
        help="Keep first occurrence of each unique sequence",
    )
    ap.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep sequences with non-standard AA letters (default: drop them)",
    )
    ap.add_argument(
        "--legacy-two-col",
        action="store_true",
        help="Old behavior: column 0 = name, column 1 = sequence (ignore headers/aliases)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    try:
        stats = csv_to_pipeline_txt(
            args.input,
            args.output,
            sequence_aliases=_parse_alias_list(args.sequence_aliases, DEFAULT_SEQUENCE_ALIASES),
            id_aliases=_parse_alias_list(args.id_aliases, DEFAULT_ID_ALIASES),
            sequence_col=args.sequence_col,
            id_col=args.id_col,
            id_prefix=args.id_prefix,
            map_csv=args.map_csv,
            dedup_sequence=args.dedup_sequence,
            keep_invalid=args.keep_invalid,
            legacy_two_col=args.legacy_two_col,
            force_auto_id=args.auto_id,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(
        f"Wrote {stats['n_written']} sequences → {stats['output']}\n"
        f"  sequence_column={stats['sequence_column']!r} id_column={stats['id_column']!r}\n"
        f"  skipped empty={stats['n_skipped_empty']} invalid_aa={stats['n_skipped_invalid']} "
        f"dup={stats['n_skipped_dup']}"
    )
    if stats["map_csv"]:
        print(f"  map_csv={stats['map_csv']}")
    print(
        "\nNext:\n"
        f"  python scripts/run_data_pipeline.py --mode blind --input {args.output} "
        f"--work-dir <dir>/generated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

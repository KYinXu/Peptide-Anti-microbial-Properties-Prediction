import argparse
import re
from pathlib import Path

import pandas as pd


def excel_sheet_name(raw: str, used: set[str]) -> str:
    """31 chars max; no []:*?/\\; unique within used."""
    s = re.sub(r'[\[\]:*?/\\]', "_", str(raw).strip()) or "sheet"
    s = s[:31]
    base, n = s, 1
    while s in used:
        suffix = f"_{n}"
        s = (base[: 31 - len(suffix)] + suffix)[:31]
        n += 1
    used.add(s)
    return s


def resolve_window_map(gen_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for c in (gen_dir / "inputs" / "window_map_notebook.csv", gen_dir / "window_map_notebook.csv"):
        if c.is_file():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert model_comparison CSV in a generated folder to Excel in the same folder. "
            "When a window map provides parent_id, each parent sequence is written to its own sheet "
            "(windows sorted by start position)."
        )
    )
    parser.add_argument(
        "generated_dir",
        type=str,
        nargs="?",
        default="sequence_to_svm_minimal/data/test/H2A_homologues_windowed/generated",
        help="Directory containing the model comparison CSV (e.g. .../generated)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="model_comparison_latest.csv",
        help="CSV filename inside generated_dir (default: model_comparison_latest.csv)",
    )
    parser.add_argument(
        "--window-map",
        type=str,
        default=None,
        help="Window map CSV; if omitted, uses <generated_dir>/inputs/window_map_notebook.csv when present",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="ESM+Combined32",
        help="Model prefix (e.g. ESM+Combined32, ESM+Geo20, ESM-only)",
    )
    args = parser.parse_args()

    gen_dir = Path(args.generated_dir).resolve()
    if not gen_dir.is_dir():
        raise SystemExit(f"Not a directory: {gen_dir}")

    input_path = gen_dir / args.csv
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_path = input_path.with_suffix(".xlsx")

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    prefix = args.model_prefix
    if f"{prefix}_pred" not in df.columns:
        for p in ("ESM+Combined32", "ESM+Geo20", "ESM-only"):
            if f"{p}_pred" in df.columns:
                print(f"Prefix {prefix!r} not found; using {p!r}.")
                prefix = p
                break

    required = ("peptide_id", f"{prefix}_pred", f"{prefix}_logit_margin", f"{prefix}_prob_AMP")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns for prefix {prefix!r}: {missing}\nHave: {list(df.columns)}")

    print(f"Using model {prefix!r} for predictions and logits.")

    out_df = pd.DataFrame()
    out_df["seqIndex"] = df["peptide_id"]
    if "sequence" in df.columns:
        out_df["sequence"] = df["sequence"]

    window_map_path = resolve_window_map(gen_dir, args.window_map)
    start_map: dict[str, int] = {}
    if args.window_map and window_map_path is None:
        print(f"Warning: --window-map path does not exist: {args.window_map}")
    elif window_map_path is not None:
        print(f"Reading window map {window_map_path}...")
        window_df = pd.read_csv(window_map_path)
        if "peptide_id" in window_df.columns and "parent_id" in window_df.columns:
            parent_map = window_df.set_index("peptide_id")["parent_id"].to_dict()
            out_df["parent_id"] = out_df["seqIndex"].map(parent_map)
            if "start" in window_df.columns:
                start_map = (
                    window_df.set_index("peptide_id")["start"].astype(int).to_dict()
                )
        else:
            print("Warning: window map missing peptide_id or parent_id columns.")

    out_df["prediction"] = df[f"{prefix}_pred"].apply(lambda x: 1 if x == 1 else -1)
    out_df["distToMargin"] = df[f"{prefix}_logit_margin"]
    out_df["P(-1)"] = 1.0 - df[f"{prefix}_prob_AMP"]
    out_df["P(+1)"] = df[f"{prefix}_prob_AMP"]

    if "parent_id" in out_df.columns and out_df["parent_id"].notna().any():
        out_df["_sheet_group"] = out_df["parent_id"].fillna("_unmapped")
    else:
        out_df["_sheet_group"] = "_all"

    if start_map:
        out_df["_win_start"] = out_df["seqIndex"].map(start_map)
    else:
        out_df["_win_start"] = pd.NA

    export_cols = [c for c in out_df.columns if c not in ("_sheet_group", "_win_start")]

    print(f"Writing {output_path}...")
    used_names: set[str] = set()
    groups = list(out_df.groupby("_sheet_group", sort=False))
    groups.sort(key=lambda kv: (kv[0] == "_all", kv[0] == "_unmapped", str(kv[0])))

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for group_key, part in groups:
            part = part.sort_values(
                ["_win_start", "seqIndex"],
                na_position="last",
                kind="mergesort",
            )
            part_out = part[export_cols]
            if group_key == "_unmapped":
                label = "unmapped"
            elif group_key == "_all":
                label = "all"
            else:
                label = str(group_key)
            sheet_name = excel_sheet_name(label, used_names)
            part_out.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Done.")


if __name__ == "__main__":
    main()

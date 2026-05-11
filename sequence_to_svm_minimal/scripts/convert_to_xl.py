import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def excel_sheet_name(raw: str, used: set[str]) -> str:
    """31 chars max; no []:*?/\\; unique within used."""
    s = re.sub(r"[\[\]:*?/\\]", "_", str(raw).strip()) or "sheet"
    s = s[:31]
    base, n = s, 1
    while s in used:
        suffix = f"_{n}"
        s = (base[: 31 - len(suffix)] + suffix)[:31]
        n += 1
    used.add(s)
    return s


def load_canonical_by_id(gen_dir: Path) -> dict[str, str]:
    """Parse inputs/canonical_seqs.txt -> {id: sequence}."""
    p = gen_dir / "inputs" / "canonical_seqs.txt"
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pid = parts[0].strip()
        seq = "".join(parts[1:]).strip().upper()
        if pid and seq:
            out[pid] = seq
    return out


def discover_pred_prefixes(df: pd.DataFrame) -> list[str]:
    """Model stems with *_pred columns (used to pick --model-prefix; sheets stay slim)."""
    prefs: list[str] = []
    for c in df.columns:
        if c.endswith("_pred") and not c.startswith("_"):
            prefs.append(c[: -len("_pred")])
    prefs.sort(key=lambda x: (len(x), x))
    return prefs


def resolve_window_map(gen_dir: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for c in (
        gen_dir / "inputs" / "window_map_notebook.csv",
        gen_dir / "window_map_notebook.csv",
        gen_dir / "inputs" / "window_map.csv",
        gen_dir / "window_map.csv",
    ):
        if c.is_file():
            return c
    return None


def is_windowed_comparison_df(df: pd.DataFrame) -> bool:
    """Per-window rows (e.g. model_comparison_windowed_latest.csv)."""
    return (
        "parent_id" in df.columns
        and "start" in df.columns
        and ("length" in df.columns or "window_length" in df.columns)
    )


def _length_col(df: pd.DataFrame) -> str:
    return "length" if "length" in df.columns else "window_length"


def resolve_margin_column(df: pd.DataFrame, prefix: str) -> str | None:
    m = f"{prefix}_logit_margin"
    if m in df.columns:
        return m
    if prefix == "SVM":
        if "SVM_distance" in df.columns:
            return "SVM_distance"
        if "SVM_hyperplane_distance" in df.columns:
            return "SVM_hyperplane_distance"
    return None


def build_windowed_summary(
    df: pd.DataFrame,
    prefix: str,
    canon: dict[str, str],
) -> pd.DataFrame:
    """One row per parent_id with aggregate stats for the chosen model prefix."""
    prob_col = f"{prefix}_prob_AMP"
    pred_col = f"{prefix}_pred"
    margin_col = resolve_margin_column(df, prefix)
    if prob_col not in df.columns:
        raise SystemExit(f"Windowed CSV missing {prob_col!r}")

    L = _length_col(df)
    rows: list[dict] = []
    for parent_id, g in df.groupby("parent_id", sort=True):
        pid = str(parent_id)
        prob = pd.to_numeric(g[prob_col], errors="coerce")
        pred = pd.to_numeric(g[pred_col], errors="coerce") if pred_col in g.columns else pd.Series(dtype=float)
        margin = (
            pd.to_numeric(g[margin_col], errors="coerce")
            if margin_col and margin_col in g.columns
            else pd.Series(dtype=float)
        )

        best = None
        if prob.notna().any():
            try:
                best = g.loc[prob.idxmax()]
            except (KeyError, TypeError, ValueError):
                best = None

        row: dict = {
            "parent_id": pid,
            "full_sequence": canon.get(pid, ""),
            "n_windows": int(len(g)),
            f"{prefix}_mean_prob_AMP": float(prob.mean()) if prob.notna().any() else np.nan,
            f"{prefix}_max_prob_AMP": float(prob.max()) if prob.notna().any() else np.nan,
            f"{prefix}_std_prob_AMP": float(prob.std(ddof=0)) if prob.notna().any() else np.nan,
            f"{prefix}_n_windows_pred_AMP": int((pred == 1).sum()) if len(pred) else 0,
            f"{prefix}_mean_logit_margin": float(margin.mean()) if margin.notna().any() else np.nan,
        }
        if best is not None:
            for k in ("seqIndex", "window_id", "start", L, "sequence", "peptide_id"):
                if k in best.index:
                    row[f"{prefix}_best_{k}"] = best[k]
        rows.append(row)
    return pd.DataFrame(rows)


SHEET_COL_ORDER = ("seqIndex", "sequence", "parent_id", "prediction", "distToMargin", "P(-1)", "P(+1)")


def _sheet_seq_index_col(sub: pd.DataFrame) -> pd.Series:
    """First column matches window exports: prefer peptide_id (e.g. SEQ_1), else seqIndex."""
    if "peptide_id" in sub.columns:
        return sub["peptide_id"].astype(str)
    if "seqIndex" in sub.columns:
        return sub["seqIndex"].astype(str)
    return pd.Series([""] * len(sub), index=sub.index)


def slim_per_sheet_rows(sub: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Seven columns per sheet: seqIndex, sequence, parent_id, prediction, distToMargin, P(-1), P(+1)."""
    sub = sub.copy()
    pred_col = f"{prefix}_pred"
    prob_col = f"{prefix}_prob_AMP"
    margin_col = resolve_margin_column(sub, prefix)

    def _bin_pred(x: object) -> int:
        try:
            return 1 if int(float(x)) == 1 else -1
        except (TypeError, ValueError):
            return -1

    seq_col = (
        sub["sequence"].astype(str)
        if "sequence" in sub.columns
        else pd.Series("", index=sub.index, dtype=str)
    )
    out = pd.DataFrame(
        {
            "seqIndex": _sheet_seq_index_col(sub),
            "sequence": seq_col,
            "parent_id": (
                sub["parent_id"].astype(str)
                if "parent_id" in sub.columns
                else pd.Series("", index=sub.index, dtype=str)
            ),
            "prediction": sub[pred_col].apply(_bin_pred) if pred_col in sub.columns else -1,
            "distToMargin": sub[margin_col].astype(float)
            if margin_col and margin_col in sub.columns
            else np.nan,
            "P(-1)": np.nan,
            "P(+1)": np.nan,
        },
        index=sub.index,
    )
    if prob_col in sub.columns:
        p1 = pd.to_numeric(sub[prob_col], errors="coerce")
        out["P(+1)"] = p1
        out["P(-1)"] = 1.0 - p1
    return out[list(SHEET_COL_ORDER)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert model comparison CSV in a generated folder to Excel. "
            "Windowed exports (model_comparison_windowed_*.csv): sheet 1 = per-parent summary; "
            "further sheets = all windows for each parent (sorted by start). "
            "Parent-level CSV: optional window map joins parent_id; one sheet per parent or 'all'."
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
        default=None,
        help=(
            "CSV filename inside generated_dir. Default: model_comparison_windowed_latest.csv if "
            "present, else model_comparison_latest.csv"
        ),
    )
    parser.add_argument(
        "--window-map",
        type=str,
        default=None,
        help="Window map CSV; if omitted, tries inputs/window_map*.csv under generated_dir",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="ESM+Combined32",
        help="Primary model prefix for summary + distToMargin-style columns (e.g. ESM+Combined32)",
    )
    parser.add_argument(
        "--force-windowed",
        action="store_true",
        help="Treat input as per-window rows even if columns look ambiguous",
    )
    parser.add_argument(
        "--force-legacy",
        action="store_true",
        help="Treat input as parent-level rows (ignore windowed auto-detect)",
    )
    args = parser.parse_args()

    gen_dir = Path(args.generated_dir).resolve()
    if not gen_dir.is_dir():
        raise SystemExit(f"Not a directory: {gen_dir}")

    csv_name = args.csv
    if not csv_name:
        wdef = gen_dir / "model_comparison_windowed_latest.csv"
        pdef = gen_dir / "model_comparison_latest.csv"
        if wdef.is_file():
            csv_name = wdef.name
        elif pdef.is_file():
            csv_name = pdef.name
        else:
            csv_name = "model_comparison_latest.csv"

    input_path = gen_dir / csv_name
    if not input_path.is_file():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_path = input_path.with_suffix(".xlsx")

    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)

    prefixes = discover_pred_prefixes(df)
    if not prefixes:
        raise SystemExit(f"No *_pred columns found. Columns: {list(df.columns)}")

    prefix = args.model_prefix
    if f"{prefix}_pred" not in df.columns:
        for p in ("ESM+Combined32", "ESM+Geo20", "ESM+QSAR12", "ESM-only", "SVM"):
            if f"{p}_pred" in df.columns:
                print(f"Prefix {prefix!r} not found; using {p!r}.")
                prefix = p
                break
        if f"{prefix}_pred" not in df.columns:
            prefix = prefixes[0]
            print(f"Using first discovered prefix {prefix!r}.")

    required = ("peptide_id", f"{prefix}_pred", f"{prefix}_prob_AMP")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns for prefix {prefix!r}: {missing}\nHave: {list(df.columns)}")

    margin_col = resolve_margin_column(df, prefix)
    if margin_col is None and prefix != "SVM":
        print(f"Note: no logit margin column for {prefix!r}; distToMargin column will be omitted.")

    windowed = args.force_windowed or (not args.force_legacy and is_windowed_comparison_df(df))
    print(f"Mode: {'windowed (per-parent sheets + summary)' if windowed else 'legacy (parent / flat)'}.")
    print(f"Primary model prefix: {prefix!r}. All detected: {prefixes!r}")

    used_names: set[str] = set()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if windowed:
            canon = load_canonical_by_id(gen_dir)
            summary = build_windowed_summary(df, prefix, canon)
            summary.to_excel(writer, sheet_name=excel_sheet_name("Summary", used_names), index=False)

            win_start = "start"
            groups = list(df.groupby("parent_id", sort=True))
            for parent_key, part in groups:
                if pd.isna(parent_key):
                    continue
                sort_keys = [win_start] + (["seqIndex"] if "seqIndex" in part.columns else [])
                part2 = part.sort_values(sort_keys, na_position="last", kind="mergesort")
                part_out = slim_per_sheet_rows(part2, prefix)
                sheet_name = excel_sheet_name(str(parent_key), used_names)
                part_out.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            out_df = pd.DataFrame(index=df.index)
            out_df["parent_id"] = ""

            window_map_path = resolve_window_map(gen_dir, args.window_map)
            start_map: dict[str, int] = {}
            if args.window_map and window_map_path is None:
                print(f"Warning: --window-map path does not exist: {args.window_map}")
            elif window_map_path is not None:
                print(f"Reading window map {window_map_path}...")
                window_df = pd.read_csv(window_map_path)
                if "peptide_id" in window_df.columns and "parent_id" in window_df.columns:
                    parent_map = window_df.set_index("peptide_id")["parent_id"].to_dict()
                    pid = df["peptide_id"].astype(str)
                    out_df["parent_id"] = pid.map(parent_map).fillna("").astype(str)
                    if "start" in window_df.columns:
                        start_map = window_df.set_index("peptide_id")["start"].astype(int).to_dict()
                else:
                    print("Warning: window map missing peptide_id or parent_id columns.")

            pred_col = f"{prefix}_pred"
            prob_col = f"{prefix}_prob_AMP"

            def _bin_pred(x: object) -> int:
                try:
                    return 1 if int(float(x)) == 1 else -1
                except (TypeError, ValueError):
                    return -1

            out_df["seqIndex"] = _sheet_seq_index_col(df)
            out_df["sequence"] = (
                df["sequence"].astype(str) if "sequence" in df.columns else pd.Series("", index=df.index)
            )
            out_df["prediction"] = df[pred_col].apply(_bin_pred)
            out_df["distToMargin"] = (
                pd.to_numeric(df[margin_col], errors="coerce")
                if (margin_col and margin_col in df.columns)
                else pd.Series(np.nan, index=df.index)
            )
            p1 = pd.to_numeric(df[prob_col], errors="coerce")
            out_df["P(+1)"] = p1
            out_df["P(-1)"] = 1.0 - p1

            pid_for_group = out_df["parent_id"].replace("", pd.NA)
            if pid_for_group.notna().any():
                out_df["_sheet_group"] = pid_for_group.astype(str).fillna("_unmapped")
            else:
                out_df["_sheet_group"] = "_all"

            if start_map:
                out_df["_win_start"] = df["peptide_id"].astype(str).map(start_map)
            else:
                out_df["_win_start"] = pd.NA

            export_cols = list(SHEET_COL_ORDER)

            groups = list(out_df.groupby("_sheet_group", sort=False))
            groups.sort(key=lambda kv: (kv[0] == "_all", kv[0] == "_unmapped", str(kv[0])))

            summary_rows: list[dict] = []
            for group_key, part in groups:
                if group_key in ("_all", "_unmapped"):
                    continue
                prob = part["P(+1)"] if "P(+1)" in part.columns else pd.Series(dtype=float)
                summary_rows.append(
                    {
                        "parent_id": str(group_key),
                        "n_rows": len(part),
                        f"{prefix}_mean_prob_AMP": float(prob.mean()) if len(prob) else np.nan,
                        f"{prefix}_max_prob_AMP": float(prob.max()) if len(prob) else np.nan,
                        f"{prefix}_n_pred_AMP": int((part["prediction"] == 1).sum()) if "prediction" in part.columns else 0,
                    }
                )
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(
                    writer, sheet_name=excel_sheet_name("Summary", used_names), index=False
                )
            else:
                prob = out_df["P(+1)"] if "P(+1)" in out_df.columns else pd.Series(dtype=float)
                pd.DataFrame(
                    [
                        {
                            "scope": "all_rows",
                            "n_rows": len(out_df),
                            f"{prefix}_mean_prob_AMP": float(prob.mean()) if len(prob) else np.nan,
                            f"{prefix}_max_prob_AMP": float(prob.max()) if len(prob) else np.nan,
                            f"{prefix}_n_pred_AMP": int((out_df["prediction"] == 1).sum())
                            if "prediction" in out_df.columns
                            else 0,
                        }
                    ]
                ).to_excel(writer, sheet_name=excel_sheet_name("Summary", used_names), index=False)

            for group_key, part in groups:
                part = part.sort_values(
                    ["_win_start", "seqIndex"],
                    na_position="last",
                    kind="mergesort",
                )
                part_out = part[export_cols].copy()
                if group_key == "_unmapped":
                    label = "unmapped"
                elif group_key == "_all":
                    label = "all"
                else:
                    label = str(group_key)
                sheet_name = excel_sheet_name(label, used_names)
                part_out.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Wrote {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()

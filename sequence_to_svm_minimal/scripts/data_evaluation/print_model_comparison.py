#!/usr/bin/env python3
"""
Pretty-print the contents of results/test_model_comparison.csv (or a given CSV)
as a readable CLI table, showing every peptide and every model's prediction.

Usage:
    python scripts/data_evaluation/print_model_comparison.py
    python scripts/data_evaluation/print_model_comparison.py --csv path/to/file.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def _format_float(x, ndigits: int = 3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    try:
        return f"{float(x):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(x)


def _summarize_models(df: pd.DataFrame, models: list[str]) -> None:
    print("\nSummary statistics")
    header = f"{'Model':<18}{'N':>8}{'AMP':>8}{'%AMP':>8}{'Mean conf':>12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for m in models:
        pred_col = f"{m}_pred"
        conf_col = f"{m}_confidence" if f"{m}_confidence" in df.columns else f"{m}_prob_AMP"
        if pred_col not in df.columns:
            continue
        pred = df[pred_col].dropna().astype(int)
        n = int(pred.shape[0])
        n_amp = int((pred == 1).sum())
        pct = (100.0 * n_amp / n) if n else 0.0
        conf = df.loc[pred.index, conf_col] if conf_col in df.columns else None
        mean_conf = float(np.nanmean(conf.values.astype(float))) if conf is not None else float("nan")
        print(f"{m:<18}{n:>8d}{n_amp:>8d}{pct:>7.1f}%{mean_conf:>12.4f}")


def _pairwise_agreement(df: pd.DataFrame, models: list[str]) -> None:
    preds = {}
    for m in models:
        col = f"{m}_pred"
        if col in df.columns:
            preds[m] = df[col]
    names = list(preds.keys())
    if len(names) < 2:
        return

    # Use rows where both models have predictions
    agree = np.zeros((len(names), len(names)), dtype=int)
    total = np.zeros((len(names), len(names)), dtype=int)
    for i, mi in enumerate(names):
        for j, mj in enumerate(names):
            pi = preds[mi]
            pj = preds[mj]
            mask = pi.notna() & pj.notna()
            total[i, j] = int(mask.sum())
            if total[i, j] == 0:
                agree[i, j] = 0
            else:
                agree[i, j] = int((pi[mask].astype(int).values == pj[mask].astype(int).values).sum())

    print("\nPairwise agreement (same 0/1 prediction)")
    col_w = max(10, max(len(n) for n in names))
    header = " " * (col_w + 2) + "  ".join(f"{n[:col_w]:>{col_w}}" for n in names)
    print(header)
    for i, mi in enumerate(names):
        row = [f"{mi[:col_w]:<{col_w}}"]
        for j in range(len(names)):
            if total[i, j] == 0:
                cell = "0/0"
            else:
                cell = f"{agree[i, j]}/{total[i, j]}"
            row.append(f"{cell:>{col_w}}")
        print("  ".join(row))


def main():
    base_dir = Path(__file__).resolve().parents[2]
    default_csv = base_dir / "results" / "test_model_comparison.csv"

    ap = argparse.ArgumentParser(description="Pretty-print model comparison CSV as a CLI table.")
    ap.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help=f"Path to model comparison CSV (default: {default_csv})",
    )
    ap.add_argument("--show", type=int, default=10, help="Number of rows to display (default: 10)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"\nNo rows in {csv_path}")
        return

    # Sort by peptide_id for stable display if the column exists
    if "peptide_id" in df.columns:
        df = df.sort_values("peptide_id")

    print("\n" + "=" * 100)
    print(f"MODEL COMPARISON TABLE  –  {csv_path}")
    print("=" * 100 + "\n")

    # Determine models from *_pred columns and build compact display:
    # one column per model, with a checkmark and confidence.
    model_pred_cols = [c for c in df.columns if c.endswith("_pred") and c != "peptide_id"]
    models = []
    for c in model_pred_cols:
        base = c[:-5]  # strip '_pred'
        models.append(base)

    # Stable, user-friendly ordering and labels
    display_order = ["SVM", "Graph-only", "Graph+Geo20", "Graph+Combined32"]
    label_map = {
        "SVM": "SVM",
        "Graph-only": "GNN Graph-only",
        "Graph+Geo20": "GNN Geo",
        "Graph+Combined32": "GNN QSAR+Geo",
    }
    ordered_models = [m for m in display_order if m in models]
    # Include any other models at the end
    for m in models:
        if m not in ordered_models:
            ordered_models.append(m)

    # Column widths for compact table
    id_col = "peptide_id" if "peptide_id" in df.columns else df.columns[0]
    id_width = max(12, min(24, max(len(str(id_col)), *(len(str(v)) for v in df[id_col].head(50)))))

    model_widths = {}
    for m in ordered_models:
        label = label_map.get(m, m)
        model_widths[m] = max(len(label), 10)

    # Header
    header_parts = [f"{id_col:<{id_width}}"]
    for m in ordered_models:
        label = label_map.get(m, m)
        header_parts.append(f"{label:^{model_widths[m]}}")
    header_line = "  ".join(header_parts)

    print(header_line)
    print("-" * len(header_line))

    # Rows: show ✓ or · with confidence (e.g., ✓0.95)
    n_show = max(0, int(args.show))
    df_show = df.head(n_show) if n_show else df.iloc[0:0]
    for _, row in df_show.iterrows():
        parts = []
        pid_val = str(row[id_col]) if pd.notna(row[id_col]) else ""
        parts.append(f"{pid_val:<{id_width}}")

        for m in ordered_models:
            pred_col = f"{m}_pred"
            conf_col = f"{m}_confidence" if f"{m}_confidence" in df.columns else f"{m}_prob_AMP"

            pred = row[pred_col] if pred_col in df.columns else None
            conf = row[conf_col] if conf_col in df.columns else None

            if pd.isna(pred):
                cell = ""
            else:
                mark = "✓" if int(pred) == 1 else "·"
                conf_str = _format_float(conf, 2) if conf is not None and pd.notna(conf) else ""
                # Confidence first, checkmark/dot to the right
                if conf_str:
                    cell = f"{conf_str}{mark}"
                else:
                    cell = mark

            parts.append(f"{cell:^{model_widths[m]}}")

        print("  ".join(parts))

    print("\n" + "=" * len(header_line))
    if n_show and len(df) > n_show:
        print(f"Showing first {n_show} of {len(df)} peptides")
    else:
        print(f"Total peptides shown: {len(df)}")
    print("=" * len(header_line))

    _summarize_models(df, ordered_models)
    _pairwise_agreement(df, ordered_models)


if __name__ == "__main__":
    main()


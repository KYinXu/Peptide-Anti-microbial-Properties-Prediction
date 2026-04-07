#!/usr/bin/env python3
"""
Pretty-print the contents of results/comparisons/model_comparison_latest.csv (or a given CSV)
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


def _metric_col_for_model(df: pd.DataFrame, model: str, metric_mode: str) -> str | None:
    if metric_mode == "prob":
        col = f"{model}_prob_AMP"
        if col in df.columns:
            return col
        fallback = f"{model}_confidence"
        return fallback if fallback in df.columns else None

    if metric_mode == "confidence":
        col = f"{model}_confidence"
        if col in df.columns:
            return col
        fallback = f"{model}_prob_AMP"
        return fallback if fallback in df.columns else None

    if metric_mode == "logit_margin":
        col = f"{model}_logit_margin"
        return col if col in df.columns else None

    if metric_mode == "svm_distance":
        if model == "SVM":
            for col in ("SVM_hyperplane_distance", "SVM_distance"):
                if col in df.columns:
                    return col
            return None
        return None

    if metric_mode == "distance_like":
        if model == "SVM":
            for col in ("SVM_hyperplane_distance", "SVM_distance"):
                if col in df.columns:
                    return col
            return None
        col = f"{model}_logit_margin"
        return col if col in df.columns else None

    if metric_mode == "score_z":
        col = f"{model}_score_z"
        return col if col in df.columns else None

    return None


def _metric_label(metric_mode: str) -> str:
    if metric_mode == "prob":
        return "P(AMP)"
    if metric_mode == "confidence":
        return "Confidence"
    if metric_mode == "logit_margin":
        return "Logit margin"
    if metric_mode == "svm_distance":
        return "SVM hyperplane distance"
    if metric_mode == "distance_like":
        return "Distance-like (SVM distance / GNN logit margin)"
    if metric_mode == "score_z":
        return "Z-score (per model, this CSV)"
    return "Metric"


def _print_metric_legend(models: list[str], df: pd.DataFrame, metric_mode: str) -> None:
    print("\nLegend")
    print("- ✓ = predicted AMP (1), · = predicted non-AMP (0)")
    print(f"- Displayed numeric value = {_metric_label(metric_mode)}")
    if metric_mode == "distance_like":
        print("- Side-by-side mapping: SVM uses decision_function distance; GNN models use logit_margin")
    if metric_mode == "score_z":
        print("- Z-score columns from compare_model_predictions.py; comparable scale across models on this run")
    for m in models:
        col = _metric_col_for_model(df, m, metric_mode)
        label = col if col is not None else "N/A for this model"
        print(f"- {m}: {label}")


def _summarize_models(df: pd.DataFrame, models: list[str], metric_mode: str) -> None:
    print("\nSummary statistics")
    header = f"{'Model':<18}{'N':>8}{'AMP':>8}{'%AMP':>8}{('Mean ' + _metric_label(metric_mode)):>18}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for m in models:
        pred_col = f"{m}_pred"
        metric_col = _metric_col_for_model(df, m, metric_mode)
        if pred_col not in df.columns:
            continue
        pred = df[pred_col].dropna().astype(int)
        n = int(pred.shape[0])
        n_amp = int((pred == 1).sum())
        pct = (100.0 * n_amp / n) if n else 0.0
        metric_vals = df.loc[pred.index, metric_col] if metric_col in df.columns else None
        mean_metric = float(np.nanmean(metric_vals.values.astype(float))) if metric_vals is not None else float("nan")
        print(f"{m:<18}{n:>8d}{n_amp:>8d}{pct:>7.1f}%{mean_metric:>18.4f}")


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
    default_csv = base_dir / "results" / "comparisons" / "model_comparison_latest.csv"

    ap = argparse.ArgumentParser(description="Pretty-print model comparison CSV as a CLI table.")
    ap.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help=f"Path to model comparison CSV (default: {default_csv})",
    )
    ap.add_argument("--show", type=int, default=10, help="Number of rows to display (default: 10)")
    ap.add_argument(
        "--metric",
        type=str,
        default="prob",
        choices=["prob", "confidence", "logit_margin", "svm_distance", "distance_like", "score_z"],
        help="Numeric metric to display in model cells",
    )
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
    display_order = ["SVM", "ESM-only", "ESM+Geo20", "ESM+QSAR12", "ESM+Combined32"]
    label_map = {
        "SVM": "SVM",
        "ESM-only": "GNN ESM-only",
        "ESM+Geo20": "GNN ESM+Geo",
        "ESM+QSAR12": "GNN ESM+QSAR",
        "ESM+Combined32": "GNN ESM+QSAR+Geo",
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
            metric_col = _metric_col_for_model(df, m, args.metric)

            pred = row[pred_col] if pred_col in df.columns else None
            metric_val = row[metric_col] if metric_col in df.columns else None

            if pd.isna(pred):
                cell = ""
            else:
                mark = "✓" if int(pred) == 1 else "·"
                conf_str = _format_float(metric_val, 2) if metric_val is not None and pd.notna(metric_val) else ""
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

    _print_metric_legend(ordered_models, df, args.metric)
    _summarize_models(df, ordered_models, args.metric)
    _pairwise_agreement(df, ordered_models)


if __name__ == "__main__":
    main()


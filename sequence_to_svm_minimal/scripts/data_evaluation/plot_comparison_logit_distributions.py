#!/usr/bin/env python3
"""
Single multi-panel figure: signed AMP vs non-AMP score per model from a comparison CSV.

Bar heights are counts per bin; the vertical axis is labeled Density. GNN panels share one
x-axis range and bin edges so shapes are comparable. SVM keeps its own scale (different units).

Uses the same scalar as distance_like in print_model_comparison.py — negative → non-AMP,
positive → AMP (SVM: decision_function; GNN: class margin = logit_AMP − logit_nonAMP).

Run from sequence_to_svm_minimal:
  python scripts/data_evaluation/plot_comparison_logit_distributions.py
  python scripts/data_evaluation/plot_comparison_logit_distributions.py --csv results/comparisons/file.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DISPLAY_ORDER = ["SVM", "ESM-only", "ESM+Geo20", "ESM+QSAR12", "ESM+Combined32"]


def _models_from_df(df: pd.DataFrame) -> list[str]:
    preds = [c for c in df.columns if c.endswith("_pred")]
    models = [c[:-5] for c in preds]
    ordered = [m for m in DISPLAY_ORDER if m in models]
    for m in models:
        if m not in ordered:
            ordered.append(m)
    return ordered


def _signed_score_column(df: pd.DataFrame, model: str) -> str | None:
    if model == "SVM":
        for col in ("SVM_hyperplane_distance", "SVM_distance"):
            if col in df.columns:
                return col
        return None
    col = f"{model}_logit_margin"
    return col if col in df.columns else None


def _gnn_shared_x_range(df: pd.DataFrame, models: list[str], pad_frac: float = 0.05) -> tuple[float, float] | None:
    chunks: list[np.ndarray] = []
    for m in models:
        if m == "SVM":
            continue
        col = _signed_score_column(df, m)
        if col is None:
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        if x.size:
            chunks.append(x)
    if not chunks:
        return None
    all_x = np.concatenate(chunks)
    lo = float(np.nanmin(all_x))
    hi = float(np.nanmax(all_x))
    span = hi - lo
    pad = pad_frac * span if span > 0 else 1.0
    return (lo - pad, hi + pad)


def _plot_combined(
    df: pd.DataFrame,
    models: list[str],
    bins: int,
    layout: str,
    out_path: Path,
) -> None:
    n = len(models)
    if n == 0:
        raise ValueError("No models to plot.")

    gnn_xlim = _gnn_shared_x_range(df, models)
    gnn_bin_edges = (
        np.linspace(gnn_xlim[0], gnn_xlim[1], bins + 1) if gnn_xlim is not None else None
    )

    if layout == "cols":
        nrows, ncols = 1, n
        figsize = (3.4 * n, 3.2)
    else:
        nrows, ncols = n, 1
        figsize = (7.5, 2.4 * n)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    ax_flat = axes.ravel()
    cmap = plt.cm.tab10(np.linspace(0, 1, min(n, 10)))

    for i, model in enumerate(models):
        ax = ax_flat[i]
        col = _signed_score_column(df, model)
        if col is None:
            ax.set_visible(False)
            continue
        x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
        color = cmap[i % len(cmap)]
        if model == "SVM":
            ax.hist(
                x,
                bins=bins,
                density=False,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
            )
            subtitle = "SVM decision function"
        else:
            hbins = gnn_bin_edges if gnn_bin_edges is not None else bins
            ax.hist(
                x,
                bins=hbins,
                density=False,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
            )
            if gnn_xlim is not None:
                ax.set_xlim(gnn_xlim)
            subtitle = "Raw logit outputs"
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.75)
        ax.set_ylabel("Density")
        ax.set_title(f"{model}\n({subtitle})")

    ax_flat[n - 1].set_xlabel("Class Margin (logit_AMP - logit_nonAMP)")
    fig.suptitle("AMP vs non-AMP score distribution", y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default_csv = ROOT / "results" / "comparisons" / "model_comparison_latest.csv"
    default_out = ROOT / "results" / "comparison_plots" / "signed_score_distributions.png"

    ap = argparse.ArgumentParser(
        description="One figure: histogram of signed score per model (comparison CSV).",
    )
    ap.add_argument("--csv", type=str, default=str(default_csv), help="Model comparison CSV")
    ap.add_argument(
        "--output",
        type=str,
        default=str(default_out),
        help="Output PNG path",
    )
    ap.add_argument(
        "--layout",
        type=str,
        choices=["rows", "cols"],
        default="rows",
        help="Stack panels top-to-bottom (rows) or left-to-right (cols)",
    )
    ap.add_argument("--bins", type=int, default=50, help="Histogram bin count")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows in {csv_path}")

    models = _models_from_df(df)
    if not models:
        raise ValueError("No *_pred columns found; is this a comparison CSV?")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_combined(df, models, args.bins, args.layout, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single multi-panel figure: overlapping signed score distributions per model from TWO comparison CSVs.

Bar heights are normalized (Density); the vertical axis is labeled Density.
The distributions for Dataset 1 and Dataset 2 are plotted overlapping for each model.

Run from sequence_to_svm_minimal:
  python scripts/data_evaluation/plot_comparison_overlapping_logit_distributions.py \
    --csv1 results/comparisons/file1.csv \
    --csv2 results/comparisons/file2.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DISPLAY_ORDER = ["SVM", "ESM-only", "ESM+Geo20", "ESM+QSAR12", "ESM+Combined32"]


def _models_from_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
    preds1 = [c for c in df1.columns if c.endswith("_pred")]
    preds2 = [c for c in df2.columns if c.endswith("_pred")]
    models1 = {c[:-5] for c in preds1}
    models2 = {c[:-5] for c in preds2}
    models = models1.intersection(models2)
    
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


def _gnn_shared_x_range(df1: pd.DataFrame, df2: pd.DataFrame, models: list[str], drop_negative: bool = False, pad_frac: float = 0.05) -> tuple[float, float] | None:
    chunks: list[np.ndarray] = []
    for df in (df1, df2):
        for m in models:
            if m == "SVM":
                continue
            col = _signed_score_column(df, m)
            if col is None:
                continue
            x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if drop_negative:
                x = x[x >= 0]
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


def _dataset_name_from_csv_path(csv_path: Path) -> str:
    parts_l = [p.lower() for p in csv_path.parts]
    if "generated" in parts_l:
        i = parts_l.index("generated")
        if i - 1 >= 0:
            return csv_path.parts[i - 1]
    return csv_path.stem


def _sanitize_for_filename(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s).strip("_")


def _write_plot_metadata_json(
    *,
    json_path: Path,
    csv1_path: Path,
    csv2_path: Path,
    out_path: Path,
    dataset_name1: str,
    dataset_name2: str,
    models: list[str],
    bins: int,
    layout: str,
) -> None:
    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "dataset_name1": dataset_name1,
        "dataset_name2": dataset_name2,
        "csv1_path": str(csv1_path.resolve()),
        "csv2_path": str(csv2_path.resolve()),
        "output_png": str(out_path.resolve()),
        "models": models,
        "bins": int(bins),
        "layout": layout,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def _plot_combined(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    models: list[str],
    bins: int,
    layout: str,
    out_path: Path,
    dataset_name1: str,
    dataset_name2: str,
    drop_negative: bool = False,
) -> None:
    n = len(models)
    if n == 0:
        raise ValueError("No models to plot.")

    gnn_xlim = _gnn_shared_x_range(df1, df2, models, drop_negative=drop_negative)
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
    
    color1 = "tab:blue"
    color2 = "tab:orange"

    for i, model in enumerate(models):
        ax = ax_flat[i]
        col1 = _signed_score_column(df1, model)
        col2 = _signed_score_column(df2, model)
        
        if col1 is None or col2 is None:
            ax.set_visible(False)
            continue
            
        score1 = pd.to_numeric(df1[col1], errors="coerce").to_numpy()
        score2 = pd.to_numeric(df2[col2], errors="coerce").to_numpy()
        
        score1 = score1[np.isfinite(score1)]
        score2 = score2[np.isfinite(score2)]

        if drop_negative:
            score1 = score1[score1 >= 0]
            score2 = score2[score2 >= 0]

        def get_stat_text(x, name):
            if drop_negative:
                # If already dropped, we calculate stats on the remaining values
                pos_x = x
            else:
                pos_x = x[x > 0]
            if pos_x.size:
                mu = float(np.mean(pos_x))
                var = float(np.var(pos_x))
                return f"{name}: μ={mu:.2f} var={var:.2f} (n={pos_x.size})"
            return f"{name}: No positive scores"

        if drop_negative:
            stat1 = get_stat_text(score1, dataset_name1)
            stat2 = get_stat_text(score2, dataset_name2)
        else:
            stat1 = get_stat_text(score1, f"{dataset_name1} (val > 0)")
            stat2 = get_stat_text(score2, f"{dataset_name2} (val > 0)")

        hbins = gnn_bin_edges if (model != "SVM" and gnn_bin_edges is not None) else bins

        if score1.size > 0:
            ax.hist(
                score1,
                bins=hbins,
                density=True,
                color=color1,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.6,
                label=stat1,
            )
        if score2.size > 0:
            ax.hist(
                score2,
                bins=hbins,
                density=True,
                color=color2,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.6,
                label=stat2,
            )

        if model != "SVM" and gnn_xlim is not None:
            ax.set_xlim(gnn_xlim)

        subtitle = "SVM decision function" if model == "SVM" else "Raw logit outputs"
        
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.75)
        ax.set_ylabel("Density")
        ax.set_title(f"{model}\n({subtitle})")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    ax_flat[n - 1].set_xlabel("Class Margin (logit_AMP - logit_nonAMP)")
    fig.suptitle(
        f"Overlapping score distribution — {dataset_name1} vs {dataset_name2}",
        y=1.01,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One figure: histogram of overlapping signed scores for two datasets per model.",
    )
    ap.add_argument("--csv1", type=str, required=True, help="First model comparison CSV")
    ap.add_argument("--csv2", type=str, required=True, help="Second model comparison CSV")
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path. If omitted, writes a timestamped PNG under results/comparison_plots/.",
    )
    ap.add_argument(
        "--dataset-name1",
        "--name1",
        type=str,
        default=None,
        dest="dataset_name1",
        help="Optional label for the first dataset.",
    )
    ap.add_argument(
        "--dataset-name2",
        "--name2",
        type=str,
        default=None,
        dest="dataset_name2",
        help="Optional label for the second dataset.",
    )
    ap.add_argument(
        "--layout",
        type=str,
        choices=["rows", "cols"],
        default="rows",
        help="Stack panels top-to-bottom (rows) or left-to-right (cols)",
    )
    ap.add_argument("--bins", type=int, default=50, help="Histogram bin count")
    ap.add_argument(
        "--drop-negative",
        action="store_true",
        help="If set, drops negative scores from the plot and analysis entirely.",
    )
    args = ap.parse_args()

    csv1_path = Path(args.csv1)
    if not csv1_path.is_file():
        raise FileNotFoundError(f"CSV1 not found: {csv1_path}")
        
    csv2_path = Path(args.csv2)
    if not csv2_path.is_file():
        raise FileNotFoundError(f"CSV2 not found: {csv2_path}")

    df1 = pd.read_csv(csv1_path)
    if df1.empty:
        raise ValueError(f"No rows in {csv1_path}")
        
    df2 = pd.read_csv(csv2_path)
    if df2.empty:
        raise ValueError(f"No rows in {csv2_path}")

    models = _models_from_dfs(df1, df2)
    if not models:
        raise ValueError("No matching *_pred columns found in both CSVs.")

    dataset_name1 = args.dataset_name1 or _dataset_name_from_csv_path(csv1_path)
    dataset_name2 = args.dataset_name2 or _dataset_name_from_csv_path(csv2_path)
    
    slug1 = _sanitize_for_filename(dataset_name1)
    slug2 = _sanitize_for_filename(dataset_name2)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = ROOT / "results" / "comparison_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"overlapping_score_distributions_{slug1}_vs_{slug2}_{ts}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_combined(df1, df2, models, args.bins, args.layout, out_path, dataset_name1, dataset_name2, drop_negative=args.drop_negative)
    _write_plot_metadata_json(
        json_path=out_path.with_suffix("").with_name(out_path.stem + "_meta.json"),
        csv1_path=csv1_path,
        csv2_path=csv2_path,
        out_path=out_path,
        dataset_name1=dataset_name1,
        dataset_name2=dataset_name2,
        models=models,
        bins=args.bins,
        layout=args.layout,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

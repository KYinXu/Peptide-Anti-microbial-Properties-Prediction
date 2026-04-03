#!/usr/bin/env python3
"""
Plot normalized histograms of classifier margins from compare_model_predictions.py CSV output.

Typical x-axis: GNN logit margin (logit_AMP − logit_nonAMP), comparable to "HDC margin" style figures.
Categories (Natural AMP, Synthetic AMP, Decoy) come from a small mapping CSV or id-list files.

Run from sequence_to_svm_minimal:

  python scripts/data_evaluation/plot_comparison_margins.py \\
    --comparison results/comparisons/test_model_comparison.csv \\
    --categories-csv path/to/peptide_categories.csv \\
    --model "ESM+Combined32" \\
    --out results/comparisons/margin_histograms.png

categories CSV columns: peptide_id, category
  category values (case-insensitive): natural / natural_amp, synthetic / synthetic_amp, decoy

Alternatively use --natural-ids, --synthetic-ids, --decoy-ids (one peptide_id per line).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# (internal_key, subplot_title_prefix, bar facecolor)
PANEL_SPECS = [
    ("natural_amp", "Natural AMP", "#7B9EB8"),
    ("synthetic_amp", "Synthetic AMP", "#5BA39A"),
    ("decoy", "Decoy (Test nonAMP)", "#D98880"),
]


def _normalize_category(raw: str) -> str | None:
    s = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("natural", "nat", "natural_amp"):
        return "natural_amp"
    if s in ("synthetic", "syn", "synthetic_amp", "artificial"):
        return "synthetic_amp"
    if s in ("decoy", "negative", "nonamp", "non_amp", "test_nonamp"):
        return "decoy"
    return None


def _load_id_set(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")}


def _build_category_series(
    peptide_ids: pd.Series,
    categories_csv: Path | None,
    natural_ids: Path | None,
    synthetic_ids: Path | None,
    decoy_ids: Path | None,
) -> pd.Series:
    """Return a Series aligned by index with values natural_amp | synthetic_amp | decoy | pd.NA."""
    n = len(peptide_ids)
    out = pd.Series(pd.NA, index=peptide_ids.index, dtype="object")

    if categories_csv is not None and categories_csv.is_file():
        m = pd.read_csv(categories_csv)
        if "peptide_id" not in m.columns:
            raise ValueError("categories CSV must contain column 'peptide_id'")
        cat_col = "category" if "category" in m.columns else m.columns[m.columns != "peptide_id"][0]
        m = m[["peptide_id", cat_col]].copy()
        m.columns = ["peptide_id", "category"]
        m["peptide_id"] = m["peptide_id"].astype(str)
        m["_k"] = m["category"].map(_normalize_category)
        if m["_k"].isna().any():
            bad = m.loc[m["_k"].isna(), "category"].unique()[:10]
            raise ValueError(f"Unknown category value(s) in CSV: {list(bad)}")
        lookup = dict(zip(m["peptide_id"], m["_k"], strict=False))
        for i, pid in peptide_ids.astype(str).items():
            if pid in lookup:
                out.loc[i] = lookup[pid]
        return out

    nat = _load_id_set(natural_ids)
    syn = _load_id_set(synthetic_ids)
    dec = _load_id_set(decoy_ids)
    if not nat and not syn and not dec:
        raise ValueError(
            "Provide --categories-csv, or all of --natural-ids, --synthetic-ids, --decoy-ids "
            "(non-empty id lists)."
        )
    for i, pid in peptide_ids.astype(str).items():
        if pid in nat:
            out.loc[i] = "natural_amp"
        elif pid in syn:
            out.loc[i] = "synthetic_amp"
        elif pid in dec:
            out.loc[i] = "decoy"
    return out


def _pick_default_margin_col(df: pd.DataFrame) -> str | None:
    suffix = "_logit_margin"
    cands = [c for c in df.columns if c.endswith(suffix)]
    if not cands:
        return None
    for preferred in ("ESM+Combined32_logit_margin", "ESM+QSAR12_logit_margin", "ESM+Geo20_logit_margin"):
        if preferred in cands:
            return preferred
    return sorted(cands)[-1]


def plot_margin_histograms(
    df: pd.DataFrame,
    margin_col: str,
    category: pd.Series,
    *,
    out_path: Path,
    x_label: str = "Logit margin (AMP − nonAMP)",
    suptitle: str | None = "Full dataset — normalized histograms",
    bins: int = 40,
    share_x: bool = True,
    share_y: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if margin_col not in df.columns:
        raise ValueError(f"Column not in comparison CSV: {margin_col}")

    values = pd.to_numeric(df[margin_col], errors="coerce")
    valid = values.notna() & category.notna()
    df_plot = df.loc[valid].copy()
    df_plot["_margin"] = values[valid]
    df_plot["_cat"] = category[valid]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=share_x, sharey=share_y, constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)

    x_min = float(df_plot["_margin"].min())
    x_max = float(df_plot["_margin"].max())
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    x_range = (x_min - pad, x_max + pad)

    for ax, (key, title_base, color) in zip(axes, PANEL_SPECS, strict=True):
        sub = df_plot.loc[df_plot["_cat"] == key, "_margin"]
        n = len(sub)
        ax.set_title(f"{title_base} (n={n})", fontsize=10)
        if n > 0:
            ax.hist(
                sub.values,
                bins=bins,
                range=x_range,
                density=True,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                alpha=0.92,
            )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel(x_label, fontsize=9)

    axes[0].set_ylabel("Density", fontsize=9)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Histograms of margins from model comparison CSV")
    ap.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="CSV from compare_model_predictions.py",
    )
    ap.add_argument(
        "--categories-csv",
        type=Path,
        default=None,
        help="peptide_id + category (natural / synthetic / decoy)",
    )
    ap.add_argument("--natural-ids", type=Path, default=None, help="Text file: one peptide_id per line")
    ap.add_argument("--synthetic-ids", type=Path, default=None, help="Text file: one peptide_id per line")
    ap.add_argument("--decoy-ids", type=Path, default=None, help="Text file: one peptide_id per line")
    ap.add_argument(
        "--margin-col",
        type=str,
        default=None,
        help="Column to histogram (e.g. ESM+Combined32_logit_margin). Default: first Combined32/QSAR12/Geo logit margin.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Shorthand: use {model}_logit_margin (e.g. 'ESM+Combined32')",
    )
    ap.add_argument("--out", "-o", type=Path, default=ROOT / "results/comparisons/margin_histograms.png")
    ap.add_argument(
        "--x-label",
        type=str,
        default="Logit margin (AMP − nonAMP)",
        help='X-axis label (e.g. "HDC margin (AMP - nonAMP)" for publication)',
    )
    ap.add_argument("--title", type=str, default="Full dataset — normalized histograms", help="Figure suptitle")
    ap.add_argument("--no-title", action="store_true", help="Omit suptitle")
    ap.add_argument("--bins", type=int, default=40)
    args = ap.parse_args()

    comp = Path(args.comparison)
    if not comp.is_file():
        print(f"Not found: {comp}", file=sys.stderr)
        return 1

    df = pd.read_csv(comp)
    id_col = "peptide_id" if "peptide_id" in df.columns else df.columns[0]

    margin_col = args.margin_col
    if args.model:
        margin_col = f"{args.model}_logit_margin"
    if margin_col is None:
        margin_col = _pick_default_margin_col(df)
    if not margin_col or margin_col not in df.columns:
        print(
            "Could not resolve margin column. Pass --margin-col or --model. "
            f"Columns ending with _logit_margin: {[c for c in df.columns if c.endswith('_logit_margin')]}",
            file=sys.stderr,
        )
        return 1

    cat = _build_category_series(
        df[id_col],
        args.categories_csv,
        args.natural_ids,
        args.synthetic_ids,
        args.decoy_ids,
    )
    n_assigned = cat.notna().sum()
    if n_assigned == 0:
        print("No peptides matched categories. Check peptide_id spelling vs mapping files.", file=sys.stderr)
        return 1

    n_total = len(df)
    if n_assigned < n_total:
        print(f"Note: {n_assigned}/{n_total} rows have a category; others are omitted from histograms.")

    plot_margin_histograms(
        df,
        margin_col,
        cat,
        out_path=args.out,
        x_label=args.x_label,
        suptitle=None if args.no_title else args.title,
        bins=args.bins,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

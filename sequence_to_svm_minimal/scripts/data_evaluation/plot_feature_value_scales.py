#!/usr/bin/env python3
"""
Compare raw numeric scales of geometric (Geo20), QSAR12, and ESM2 embedding columns
after the same merge as GNN training. Produces matplotlib figures and a short stdout summary.

Run from `sequence_to_svm_minimal`:
  python scripts/data_evaluation/plot_feature_value_scales.py
  python scripts/data_evaluation/plot_feature_value_scales.py --out_dir results/feature_scale_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project root: sequence_to_svm_minimal/
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GEO_CSV = "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/geometric_features_clustered.csv"
DEFAULT_QSAR_CSV = "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/qsar12_descriptors.csv"
DEFAULT_ESM2_AMP = "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/esm2_amp.csv"
DEFAULT_ESM2_DECOY = "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/esm2_decoy.csv"

GEO_COLS = [
    "radius_gyration",
    "end_to_end_distance",
    "max_pairwise_distance",
    "centroid_distance_mean",
    "centroid_distance_std",
    "fraction_helix",
    "fraction_sheet",
    "fraction_coil",
    "total_sasa",
    "hydrophobic_sasa",
    "fraction_hydrophobic_sasa",
    "length",
    "net_charge",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "torsion_mean",
    "torsion_std",
]

QSAR_COLS = [
    "netCharge",
    "FC",
    "LW",
    "DP",
    "NK",
    "AE",
    "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
]


def _normalize_core_id(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"^(AMP_|DECOY_)", "", regex=True)
    return s


def _load_esm2_table(
    esm2_csv: str | None,
    esm2_amp_csv: str | None,
    esm2_decoy_csv: str | None,
    esm_prefix: str,
) -> tuple[pd.DataFrame | None, list[str]]:
    if esm2_csv:
        df = pd.read_csv(esm2_csv)
        if "peptide_id" in df.columns:
            id_col = "peptide_id"
        elif "seqIndex" in df.columns:
            id_col = "seqIndex"
        else:
            raise ValueError("ESM2 CSV must contain 'peptide_id' or 'seqIndex'")
        emb_cols = [c for c in df.columns if c.startswith(esm_prefix)]
        if not emb_cols:
            raise ValueError(f"No embedding columns with prefix {esm_prefix!r}")
        out = df[[id_col] + emb_cols].copy()
        out["core_id"] = _normalize_core_id(out[id_col])
        return out[["core_id"] + emb_cols], emb_cols

    if esm2_amp_csv and esm2_decoy_csv:
        amp_df = pd.read_csv(esm2_amp_csv)
        decoy_df = pd.read_csv(esm2_decoy_csv)
        for d in (amp_df, decoy_df):
            if "seqIndex" not in d.columns:
                raise ValueError("ESM2 AMP/DECOY CSVs must contain 'seqIndex'")
        emb_cols = [c for c in amp_df.columns if c.startswith(esm_prefix)]
        if not emb_cols:
            raise ValueError(f"No embedding columns with prefix {esm_prefix!r}")
        amp = amp_df[["seqIndex"] + emb_cols].copy()
        decoy = decoy_df[["seqIndex"] + emb_cols].copy()
        amp["core_id"] = _normalize_core_id(amp["seqIndex"])
        decoy["core_id"] = _normalize_core_id(decoy["seqIndex"])
        merged = pd.concat(
            [amp[["core_id"] + emb_cols], decoy[["core_id"] + emb_cols]],
            axis=0,
            ignore_index=True,
        )
        return merged.drop_duplicates(subset=["core_id"]), emb_cols

    return None, []


def load_merged_features(
    csv_path: Path,
    qsar_csv: Path,
    esm2_csv: str | None,
    esm2_amp_csv: str | None,
    esm2_decoy_csv: str | None,
    esm_prefix: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    geo_df = pd.read_csv(csv_path)
    qsar_df = pd.read_csv(qsar_csv)
    merged_df = geo_df.merge(
        qsar_df[["peptide_id"] + QSAR_COLS], on="peptide_id", how="left"
    )
    esm2_df, esm2_cols = _load_esm2_table(
        esm2_csv, esm2_amp_csv, esm2_decoy_csv, esm_prefix
    )
    if esm2_df is not None:
        merged_df["core_id"] = _normalize_core_id(merged_df["peptide_id"])
        merged_df = merged_df.merge(esm2_df, on="core_id", how="left")
        merged_df = merged_df.drop(columns=["core_id"])
    return merged_df, QSAR_COLS, esm2_cols


def _block_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    X = df[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def subsample(arr: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if arr.size <= max_n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.size, size=max_n, replace=False)
    return arr.flat[idx]


def summarize_block(name: str, X: np.ndarray) -> dict:
    flat = X.ravel()
    abs_flat = np.abs(flat)
    per_col_std = X.std(axis=0)
    per_col_mad = np.median(np.abs(X - np.median(X, axis=0, keepdims=True)), axis=0)
    per_col_range = X.max(axis=0) - X.min(axis=0)
    return {
        "name": name,
        "n_rows": X.shape[0],
        "n_cols": X.shape[1],
        "flat_min": float(flat.min()),
        "flat_max": float(flat.max()),
        "flat_mean": float(flat.mean()),
        "flat_std": float(flat.std()),
        "abs_mean": float(abs_flat.mean()),
        "abs_median": float(np.median(abs_flat)),
        "per_col_std_median": float(np.median(per_col_std)),
        "per_col_std_mean": float(per_col_std.mean()),
        "per_col_range_median": float(np.median(per_col_range)),
        "per_col_mad_median": float(np.median(per_col_mad)),
        "_flat": flat,
        "_per_col_std": per_col_std,
        "_per_col_range": per_col_range,
        "_per_col_abs_mean": np.abs(X).mean(axis=0),
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== Pooled value scale (all cells, after nan_to_num) ===")
    keys = [
        "name",
        "n_rows",
        "n_cols",
        "flat_min",
        "flat_max",
        "flat_mean",
        "flat_std",
        "abs_mean",
        "abs_median",
    ]
    for r in rows:
        line = " | ".join(f"{k}={r[k]:.6g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys)
        print(line)
    print("\n=== Per-column spread across peptides (median over columns) ===")
    for r in rows:
        print(
            f"{r['name']}: median std(col)={r['per_col_std_median']:.6g}, "
            f"median range(col)={r['per_col_range_median']:.6g}, "
            f"median MAD(col)={r['per_col_mad_median']:.6g}"
        )


def plot_figure(
    summaries: list[dict],
    out_path: Path,
    subsample_n: int,
    seed: int,
    dpi: int,
) -> None:
    labels = [s["name"] for s in summaries]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Raw feature value scales: Geo vs QSAR vs ESM2 (training merge)", fontsize=12)

    # (0,0) Overlaid histograms of subsampled pooled values
    ax = axes[0, 0]
    for s, c, lab in zip(summaries, colors, labels):
        sub = subsample(s["_flat"], subsample_n, seed)
        ax.hist(
            sub,
            bins=80,
            density=True,
            alpha=0.45,
            color=c,
            label=f"{lab} (n≈{min(subsample_n, s['_flat'].size):,})",
        )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Pooled values (subsampled)")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    # (0,1) Same on log|x|+eps for heavy tails
    ax = axes[0, 1]
    eps = 1e-12
    for s, c, lab in zip(summaries, colors, labels):
        sub = subsample(np.log10(np.abs(s["_flat"]) + eps), subsample_n, seed + 1)
        ax.hist(sub, bins=60, density=True, alpha=0.45, color=c, label=lab)
    ax.set_xlabel("log10(|value| + 1e-12)")
    ax.set_ylabel("Density")
    ax.set_title("|Value| on log10 scale (subsampled)")
    ax.legend(fontsize=8)

    # (1,0) Boxplot: per-column std across peptides (one value per feature column)
    ax = axes[1, 0]
    data_std = [s["_per_col_std"] for s in summaries]
    bp = ax.boxplot(data_std, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("Std across peptides")
    ax.set_title("Per-feature variation (std over rows)")
    ax.set_yscale("symlog", linthresh=1e-6)

    # (1,1) Boxplot: per-column value range (max - min) across peptides
    ax = axes[1, 1]
    data_rng = [s["_per_col_range"] for s in summaries]
    bp = ax.boxplot(data_rng, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("max(col) - min(col)")
    ax.set_title("Per-feature span in dataset")
    ax.set_yscale("symlog", linthresh=1e-6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_per_feature_abs_mean(summaries: list[dict], out_path: Path, dpi: int) -> None:
    """Strip plot alternative: distribution of mean |x| per column."""
    labels = [s["name"] for s in summaries]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [s["_per_col_abs_mean"] for s in summaries]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("Mean |value| within column (across peptides)")
    ax.set_title("Typical magnitude per feature dimension")
    ax.set_yscale("log")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Geo / QSAR / ESM2 raw value scale comparison.")
    ap.add_argument("--geo_csv", type=str, default=DEFAULT_GEO_CSV)
    ap.add_argument("--qsar_csv", type=str, default=DEFAULT_QSAR_CSV)
    ap.add_argument("--esm2_csv", type=str, default=None, help="Single merged ESM2 table (optional)")
    ap.add_argument("--esm2_amp_csv", type=str, default=DEFAULT_ESM2_AMP)
    ap.add_argument("--esm2_decoy_csv", type=str, default=DEFAULT_ESM2_DECOY)
    ap.add_argument("--no_esm2", action="store_true", help="Skip ESM2 merge (Geo + QSAR only)")
    ap.add_argument(
        "--esm_prefix",
        type=str,
        default="esm2_dim_",
        help="Column prefix for sequence embeddings (e.g. esm2_dim_ or esmfold_dim_)",
    )
    ap.add_argument("--out_dir", type=str, default="results/feature_scale_plots")
    ap.add_argument("--subsample", type=int, default=120_000, help="Max pooled values per group for histograms")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    geo_path = (ROOT / args.geo_csv).resolve()
    qsar_path = (ROOT / args.qsar_csv).resolve()
    esm2_csv = args.esm2_csv
    esm_amp = None if args.no_esm2 else str((ROOT / args.esm2_amp_csv).resolve())
    esm_decoy = None if args.no_esm2 else str((ROOT / args.esm2_decoy_csv).resolve())
    if args.esm2_csv:
        esm2_csv = str((ROOT / args.esm2_csv).resolve())
        esm_amp = esm_decoy = None

    merged, qsar_cols, esm2_cols = load_merged_features(
        geo_path,
        qsar_path,
        esm2_csv,
        esm_amp,
        esm_decoy,
        args.esm_prefix,
    )

    summaries = []
    Xg = _block_matrix(merged, GEO_COLS)
    summaries.append(summarize_block("Geo (20)", Xg))
    Xq = _block_matrix(merged, qsar_cols)
    summaries.append(summarize_block("QSAR (12)", Xq))
    if esm2_cols:
        Xe = _block_matrix(merged, esm2_cols)
        summaries.append(summarize_block(f"ESM2 ({len(esm2_cols)} dims)", Xe))
    else:
        print("Warning: no ESM2 columns loaded; only Geo and QSAR plots.", file=sys.stderr)

    print_summary(summaries)

    out_dir = (ROOT / args.out_dir).resolve()
    plot_figure(summaries, out_dir / "feature_scale_overview.png", args.subsample, args.seed, args.dpi)
    plot_per_feature_abs_mean(summaries, out_dir / "feature_scale_per_dim_abs_mean.png", args.dpi)
    print(f"\nWrote:\n  {out_dir / 'feature_scale_overview.png'}\n  {out_dir / 'feature_scale_per_dim_abs_mean.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

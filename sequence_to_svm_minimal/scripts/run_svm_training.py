#!/usr/bin/env python3
"""
Train a ready-to-run SVM for use with compare_model_predictions.py.

This script:
  - Loads the same training geometric + QSAR-12 features used in GNN/SVM baselines
  - Trains an SVM with RBF kernel on labeled data (AMP vs DECOY)
  - Saves:
      (1) a .pkl file with the trained SVM
      (2) a Z-score descriptor file (names, means, stds) compatible with
          _load_svm_predictions() in compare_model_predictions.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
import joblib


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute standard binary classification metrics."""
    y_pred = (y_prob >= 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def _normalize_core_id(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"^(AMP_|DECOY_)", "", regex=True)
    return s


def _load_esm2_features(esm2_csv: str | None, esm2_amp_csv: str | None, esm2_decoy_csv: str | None):
    """Load ESM2 embeddings from a single CSV or AMP/DECOY CSV pair."""
    if esm2_csv:
        df = pd.read_csv(esm2_csv)
        if "peptide_id" in df.columns:
            id_col = "peptide_id"
        elif "seqIndex" in df.columns:
            id_col = "seqIndex"
        else:
            raise ValueError("ESM2 CSV must contain 'peptide_id' or 'seqIndex'")
        esm2_cols = [c for c in df.columns if c.startswith("esm2_dim_")]
        if not esm2_cols:
            raise ValueError("No ESM2 columns found in ESM2 CSV (expected prefix 'esm2_dim_').")
        out = df[[id_col] + esm2_cols].copy()
        out["core_id"] = _normalize_core_id(out[id_col])
        return out[["core_id"] + esm2_cols], esm2_cols

    if esm2_amp_csv and esm2_decoy_csv:
        amp_df = pd.read_csv(esm2_amp_csv)
        decoy_df = pd.read_csv(esm2_decoy_csv)
        for d in (amp_df, decoy_df):
            if "seqIndex" not in d.columns:
                raise ValueError("ESM2 AMP/DECOY CSV must contain 'seqIndex'.")
        esm2_cols = [c for c in amp_df.columns if c.startswith("esm2_dim_")]
        if not esm2_cols:
            raise ValueError("No ESM2 columns found in AMP/DECOY CSVs (expected prefix 'esm2_dim_').")

        amp = amp_df[["seqIndex"] + esm2_cols].copy()
        decoy = decoy_df[["seqIndex"] + esm2_cols].copy()
        amp["core_id"] = _normalize_core_id(amp["seqIndex"])
        decoy["core_id"] = _normalize_core_id(decoy["seqIndex"])

        merged = pd.concat([amp[["core_id"] + esm2_cols], decoy[["core_id"] + esm2_cols]], axis=0, ignore_index=True)
        merged = merged.drop_duplicates(subset=["core_id"])
        return merged, esm2_cols

    return None, []


def main():
    base_dir = Path(__file__).resolve().parents[1]

    default_geo = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "geometric_features_clustered.csv"
    default_qsar = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "qsar12_descriptors.csv"
    default_esm2_amp = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "esm2_amp.csv"
    default_esm2_decoy = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "esm2_decoy.csv"
    default_out_dir = base_dir / "results" / "svm"

    ap = argparse.ArgumentParser(description="Train SVM for use with compare_model_predictions.py")
    ap.add_argument("--geo_csv", type=str, default=str(default_geo), help="Geometric features CSV used for training")
    ap.add_argument("--qsar_csv", type=str, default=str(default_qsar), help="QSAR-12 descriptors CSV aligned with geo_csv")
    ap.add_argument("--esm2_csv", type=str, default=None,
                    help="Optional single ESM2 CSV with peptide_id/seqIndex + esm2_dim_*")
    ap.add_argument("--esm2_amp_csv", type=str, default=str(default_esm2_amp),
                    help="AMP ESM2 CSV (used if --esm2_csv not provided)")
    ap.add_argument("--esm2_decoy_csv", type=str, default=str(default_esm2_decoy),
                    help="DECOY ESM2 CSV (used if --esm2_csv not provided)")
    ap.add_argument("--out_dir", type=str, default=str(default_out_dir), help="Output directory for SVM .pkl and Z-score file")
    ap.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="SVM kernel")
    args = ap.parse_args()

    geo_csv = Path(args.geo_csv)
    qsar_csv = Path(args.qsar_csv)
    esm2_csv = Path(args.esm2_csv) if args.esm2_csv else None
    esm2_amp_csv = Path(args.esm2_amp_csv) if args.esm2_amp_csv else None
    esm2_decoy_csv = Path(args.esm2_decoy_csv) if args.esm2_decoy_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training SVM for compare_model_predictions.py ===")
    print(f"Geo CSV : {geo_csv}")
    print(f"QSAR CSV: {qsar_csv}")
    print(f"ESM2 CSV: {esm2_csv if esm2_csv else f'{esm2_amp_csv} + {esm2_decoy_csv}'}")
    print(f"Out dir : {out_dir}")

    geo_df = pd.read_csv(geo_csv)
    qsar_df = pd.read_csv(qsar_csv)

    # Ensure alignment by peptide_id
    assert list(geo_df["peptide_id"]) == list(qsar_df["peptide_id"]), "Peptide ID mismatch between geo and QSAR CSVs"

    # Labels: convert -1/1 to 0/1
    raw_labels = geo_df["label"].values
    y = np.where(raw_labels == 1, 1, 0).astype(np.int64)

    # Feature columns: use QSAR-12 only (to mirror SVM baselines)
    qsar_cols = [
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

    for c in qsar_cols:
        if c not in qsar_df.columns:
            raise ValueError(f"QSAR column missing from {qsar_csv}: {c}")

    # Merge ESM2 embeddings via normalized core IDs (strip AMP_/DECOY_ prefixes)
    esm2_df, esm2_cols = _load_esm2_features(
        str(esm2_csv) if esm2_csv else None,
        str(esm2_amp_csv) if esm2_amp_csv else None,
        str(esm2_decoy_csv) if esm2_decoy_csv else None,
    )
    if esm2_df is None or not esm2_cols:
        raise ValueError("ESM2 embeddings are required but were not loaded.")

    train_df = geo_df[["peptide_id", "label"]].merge(qsar_df[["peptide_id"] + qsar_cols], on="peptide_id", how="left")
    train_df["core_id"] = _normalize_core_id(train_df["peptide_id"])
    train_df = train_df.merge(esm2_df, on="core_id", how="left")

    feature_cols = qsar_cols + esm2_cols
    missing_after_merge = train_df[feature_cols].isna().any(axis=1).sum()
    if missing_after_merge > 0:
        raise ValueError(f"Found {missing_after_merge} rows with missing QSAR/ESM2 features after merge.")

    X_raw = train_df[feature_cols].values.astype(np.float64)

    # Z-score normalization (full dataset) and save stats for compare_model_predictions.py
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    # Avoid zeros
    stds_safe = np.where(stds > 0, stds, 1.0)
    X = (X_raw - means) / stds_safe

    # Simple train/val split (stratified 80/20) for reporting metrics
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    svm = SVC(kernel=args.kernel, probability=True, C=1.0, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)
    y_prob_val = svm.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_prob_val)

    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    # Retrain on full normalized dataset
    svm.fit(X, y)

    # Save SVM model
    svm_path = out_dir / "svm_qsar12_esm2_model.pkl"
    joblib.dump(svm, svm_path)
    print(f"\nSaved SVM model: {svm_path}")

    # Save Z-score descriptor file expected by _load_svm_predictions:
    # line 1: comma-separated descriptor names in order
    # line 2: comma-separated means
    # line 3: comma-separated stds
    z_path = out_dir / "svm_qsar12_esm2_zscores.txt"
    with z_path.open("w") as f:
        f.write(",".join(feature_cols) + "\n")
        f.write(",".join(f"{m:.10f}" for m in means) + "\n")
        f.write(",".join(f"{s:.10f}" for s in stds_safe) + "\n")
    print(f"Saved Z-score file: {z_path}")

    print("\nDone. To use this SVM in compare_model_predictions.py, run it with:")
    print("  --svm_descriptor_csv <descriptor CSV containing all features listed in z_file line 1>")
    print(f"  --svm_z_file {z_path}")
    print(f"  --svm_pkl {svm_path}")


if __name__ == "__main__":
    main()


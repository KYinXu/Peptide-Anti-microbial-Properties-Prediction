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


def main():
    base_dir = Path(__file__).resolve().parents[1]

    default_geo = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "geometric_features_clustered.csv"
    default_qsar = base_dir / "data" / "gnn_training_dataset" / "alpha_and_beta_combined" / "generated" / "spliced" / "qsar12_descriptors.csv"
    default_out_dir = base_dir / "results" / "svm"

    ap = argparse.ArgumentParser(description="Train SVM for use with compare_model_predictions.py")
    ap.add_argument("--geo_csv", type=str, default=str(default_geo), help="Geometric features CSV used for training")
    ap.add_argument("--qsar_csv", type=str, default=str(default_qsar), help="QSAR-12 descriptors CSV aligned with geo_csv")
    ap.add_argument("--out_dir", type=str, default=str(default_out_dir), help="Output directory for SVM .pkl and Z-score file")
    ap.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="SVM kernel")
    args = ap.parse_args()

    geo_csv = Path(args.geo_csv)
    qsar_csv = Path(args.qsar_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training SVM for compare_model_predictions.py ===")
    print(f"Geo CSV : {geo_csv}")
    print(f"QSAR CSV: {qsar_csv}")
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

    X_raw = qsar_df[qsar_cols].values.astype(np.float64)

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
    svm_path = out_dir / "svm_qsar12_model.pkl"
    joblib.dump(svm, svm_path)
    print(f"\nSaved SVM model: {svm_path}")

    # Save Z-score descriptor file expected by _load_svm_predictions:
    # line 1: comma-separated descriptor names in order
    # line 2: comma-separated means
    # line 3: comma-separated stds
    z_path = out_dir / "svm_qsar12_zscores.txt"
    with z_path.open("w") as f:
        f.write(",".join(qsar_cols) + "\n")
        f.write(",".join(f"{m:.10f}" for m in means) + "\n")
        f.write(",".join(f"{s:.10f}" for s in stds_safe) + "\n")
    print(f"Saved Z-score file: {z_path}")

    print("\nDone. To use this SVM in compare_model_predictions.py, run it with:")
    print(f"  --svm_descriptor_csv {qsar_csv}")
    print(f"  --svm_z_file {z_path}")
    print(f"  --svm_pkl {svm_path}")


if __name__ == "__main__":
    main()


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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

# Ensure `sequence_to_svm_minimal/` is on sys.path even when the script is run
# from outside that directory (common when launching from WSL or a different cwd).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    # Optional dependency: only needed when using the pipeline workspace mode.
    from peptide_pipeline.manifest_paths import load_pipeline_manifest, resolve_generated_workspace
except Exception:  # pragma: no cover
    load_pipeline_manifest = None
    resolve_generated_workspace = None


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


def _resolve_pipeline_workspace(generated: str | None, input_dir: str | None) -> Path | None:
    """
    Resolve a pipeline workspace (directory containing pipeline_manifest.json).

    Accepts either the generated/ folder itself, or a parent containing generated/.
    Mirrors compare_model_predictions.py behavior.
    """
    if generated and input_dir:
        gp = Path(generated).expanduser().resolve()
        ip = Path(input_dir).expanduser().resolve()
        if gp != ip:
            raise SystemExit("Use only one of GENERATED (positional) or --input-dir / --pipeline-work-dir.")
    chosen = generated or input_dir
    if not chosen:
        return None

    if resolve_generated_workspace is None:
        raise SystemExit(
            "Pipeline workspace mode requested, but `peptide_pipeline` could not be imported. "
            "Run from the `sequence_to_svm_minimal/` directory, or ensure that directory is on PYTHONPATH."
        )
    return resolve_generated_workspace(chosen)


def _normalize_manifest_path(p: str) -> Path:
    """
    Normalize manifest paths across Windows / WSL.

    - On Windows: convert WSL-style `/mnt/<drive>/...` to `X:\...`
    - On WSL/Linux: convert Windows-style `X:\...` to `/mnt/x/...`
    - Otherwise: keep path unchanged
    """
    s = str(p)
    if sys.platform.startswith("win"):
        # /mnt/c/Users/...  ->  C:\Users\...
        if s.startswith("/mnt/") and len(s) >= 7 and s[5].isalpha() and s[6:7] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
        return Path(s)

    # WSL/Linux: convert C:\Users\... -> /mnt/c/Users/...
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def _load_esm2_features(esm2_csv: str | None):
    """Load ESM2 embeddings from a single merged CSV (peptide_id or seqIndex + esm2_dim_*)."""
    if not esm2_csv:
        return None, []
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


def main():
    base_dir = Path(__file__).resolve().parents[1]

    # Legacy defaults (pre-pipeline) retained for backwards compatibility.
    default_geo = (
        base_dir
        / "data"
        / "gnn_training_dataset"
        / "alpha_and_beta_combined"
        / "generated"
        / "spliced"
        / "geometric_features_clustered.csv"
    )
    default_qsar = (
        base_dir
        / "data"
        / "gnn_training_dataset"
        / "alpha_and_beta_combined"
        / "generated"
        / "spliced"
        / "qsar12_descriptors.csv"
    )
    default_esm2_merged = default_geo.parent / "esm2_embeddings.csv"
    default_out_dir = base_dir / "results" / "svm"

    ap = argparse.ArgumentParser(description="Train SVM for use with compare_model_predictions.py")
    ap.add_argument(
        "generated",
        nargs="?",
        default=None,
        metavar="GENERATED",
        help=(
            "Pipeline generated/ directory or parent containing generated/. "
            "When provided, geo/qsar/esm2 paths default to pipeline_manifest.json "
            "(overridden by explicit --geo_csv/--qsar_csv/--esm2_csv flags)."
        ),
    )
    ap.add_argument(
        "--input-dir",
        "--pipeline-work-dir",
        type=str,
        default=None,
        dest="input_dir",
        help="Same as positional GENERATED (pipeline workspace containing pipeline_manifest.json).",
    )
    ap.add_argument("--geo_csv", type=str, default=str(default_geo), help="Geometric features CSV used for training")
    ap.add_argument("--qsar_csv", type=str, default=str(default_qsar), help="QSAR-12 descriptors CSV aligned with geo_csv")
    ap.add_argument(
        "--svm_feature_set",
        type=str,
        default="qsar12",
        choices=["qsar12", "qsar12+esm2"],
        help=(
            "Which features to train on. "
            "'qsar12' trains a QSAR-only SVM (no ESM2 required). "
            "'qsar12+esm2' trains the newer fused SVM (default)."
        ),
    )
    ap.add_argument(
        "--esm2_csv",
        type=str,
        default=str(default_esm2_merged),
        help="Merged ESM2 CSV with peptide_id or seqIndex + esm2_dim_* (required for qsar12+esm2 unless using pipeline manifest defaults).",
    )
    ap.add_argument("--out_dir", type=str, default=str(default_out_dir), help="Output directory for SVM .pkl and Z-score file")
    ap.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="SVM kernel")
    args = ap.parse_args()

    # If a pipeline workspace is provided, use manifest paths as defaults unless user explicitly overrides.
    ws = _resolve_pipeline_workspace(getattr(args, "generated", None), getattr(args, "input_dir", None))
    if ws is not None:
        if load_pipeline_manifest is None:
            raise SystemExit(
                "Pipeline workspace mode requested, but `peptide_pipeline` could not be imported. "
                "Run from the `sequence_to_svm_minimal/` directory, or ensure that directory is on PYTHONPATH."
            )
        m = load_pipeline_manifest(ws)
        # These keys are written by run_data_pipeline when QSAR/ESM2 are enabled.
        required = ["geometric_features", "qsar12_descriptors"]
        if args.svm_feature_set == "qsar12+esm2":
            required.append("esm2_embeddings")
        for k in required:
            if not m.get(k):
                raise SystemExit(
                    f"Manifest missing {k!r} in {ws}. "
                    "Run the pipeline without --skip-qsar (and without --skip-esm2 for qsar12+esm2), "
                    "or pass explicit --geo_csv/--qsar_csv/(--esm2_csv)."
                )

        # Treat manifest paths as defaults: only apply if user did not set the corresponding CLI flags.
        if args.geo_csv == str(default_geo):
            args.geo_csv = str(_normalize_manifest_path(m["geometric_features"]).resolve())
        if args.qsar_csv == str(default_qsar):
            args.qsar_csv = str(_normalize_manifest_path(m["qsar12_descriptors"]).resolve())
        if args.svm_feature_set == "qsar12+esm2":
            if args.esm2_csv == str(default_esm2_merged):
                args.esm2_csv = str(_normalize_manifest_path(m["esm2_embeddings"]).resolve())

    geo_csv = Path(args.geo_csv)
    qsar_csv = Path(args.qsar_csv)
    esm2_csv = Path(args.esm2_csv) if args.esm2_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training SVM for compare_model_predictions.py ===")
    if ws is not None:
        print(f"Pipeline workspace: {ws}")
    print(f"Geo CSV : {geo_csv}")
    print(f"QSAR CSV: {qsar_csv}")
    if args.svm_feature_set == "qsar12+esm2":
        print(f"ESM2 CSV: {esm2_csv}")
    else:
        print("ESM2 CSV: (not used; --svm_feature_set=qsar12)")
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

    train_df = geo_df[["peptide_id", "label"]].merge(qsar_df[["peptide_id"] + qsar_cols], on="peptide_id", how="left")
    feature_cols = list(qsar_cols)
    if args.svm_feature_set == "qsar12+esm2":
        # Merge ESM2 embeddings via normalized core IDs (strip AMP_/DECOY_ prefixes)
        esm2_df, esm2_cols = _load_esm2_features(str(esm2_csv) if esm2_csv else None)
        if esm2_df is None or not esm2_cols:
            raise ValueError(
                "ESM2 embeddings are required for --svm_feature_set=qsar12+esm2 but were not loaded. "
                "Provide --esm2_csv pointing to a merged embeddings table, or switch to --svm_feature_set=qsar12."
            )
        train_df["core_id"] = _normalize_core_id(train_df["peptide_id"])
        train_df = train_df.merge(esm2_df, on="core_id", how="left")
        feature_cols = feature_cols + list(esm2_cols)
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
    svm_path = out_dir / "svm_qsar12_model.pkl"
    joblib.dump(svm, svm_path)
    print(f"\nSaved SVM model: {svm_path}")
    if args.svm_feature_set == "qsar12+esm2":
        svm_path_legacy = out_dir / "svm_qsar12_esm2_model.pkl"
        joblib.dump(svm, svm_path_legacy)
        print(f"Saved (legacy name): {svm_path_legacy}")

    # Save Z-score descriptor file expected by _load_svm_predictions:
    # line 1: comma-separated descriptor names in order
    # line 2: comma-separated means
    # line 3: comma-separated stds
    z_path = out_dir / "svm_qsar12_zscores.txt"
    with z_path.open("w") as f:
        f.write(",".join(feature_cols) + "\n")
        f.write(",".join(f"{m:.10f}" for m in means) + "\n")
        f.write(",".join(f"{s:.10f}" for s in stds_safe) + "\n")
    print(f"Saved Z-score file: {z_path}")
    if args.svm_feature_set == "qsar12+esm2":
        z_path_legacy = out_dir / "svm_qsar12_esm2_zscores.txt"
        z_path_legacy.write_text(z_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Saved (legacy name): {z_path_legacy}")

    print("\nDone. To use this SVM in compare_model_predictions.py, run it with:")
    print("  --svm_descriptor_csv <descriptor CSV containing all features listed in z_file line 1>")
    print(f"  --svm_z_file {z_path}")
    print(f"  --svm_pkl {svm_path}")


if __name__ == "__main__":
    main()


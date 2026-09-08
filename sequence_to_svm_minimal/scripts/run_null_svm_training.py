#!/usr/bin/env python3
"""Train a QSAR-12 SVM with the last four sequence-order descriptors nulled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent))

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence
from peptide_pipeline.sequence_io import read_sequence_records
from sequence_analysis.utils.descriptor_ablation import null_descriptor_values, parse_null_descriptor_names


QSAR_COLUMNS = (
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
)
SEQUENCE_ORDER_QSAR_COLUMNS = (
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
)
DEFAULT_NULL_QSAR_COLUMNS = SEQUENCE_ORDER_QSAR_COLUMNS

CHARGE = {
    "A": 0,
    "C": 0,
    "D": -1,
    "E": -1,
    "F": 0,
    "G": 0,
    "H": 1,
    "I": 0,
    "K": 1,
    "L": 0,
    "M": 0,
    "N": 0,
    "P": 0,
    "Q": 0,
    "R": 1,
    "S": 0,
    "T": 0,
    "V": 0,
    "W": 0,
    "Y": 0,
}


def parse_args() -> argparse.Namespace:
    default_out_dir = BASE_DIR / "results" / "null_svm"
    parser = argparse.ArgumentParser(
        description=(
            "Train a QSAR-12 SVM from peptide sequences while forcing the last "
            "four GRAR740104 sequence-order descriptors to zero."
        )
    )
    parser.add_argument(
        "--amp_sequences",
        "--positive_sequences",
        type=Path,
        required=True,
        help="Positive AMP sequence file: FASTA, CSV with sequence column, or text id/sequence lines.",
    )
    parser.add_argument(
        "--decoy_sequences",
        type=Path,
        required=True,
        help="Negative decoy sequence file: FASTA, CSV with sequence column, or text id/sequence lines.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=default_out_dir,
        help="Output directory for SVM .pkl, Z-score file, and training summaries.",
    )
    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear"], help="SVM kernel.")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter.")
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Downsample to equal AMP/decoy counts before fitting. By default, all examples are used.",
    )
    parser.add_argument(
        "--no_class_weight",
        action="store_true",
        help="Disable SVC class_weight='balanced'. By default, class imbalance is automatically weighted.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splits and optional downsampling.")
    parser.add_argument(
        "--no_validation_split",
        action="store_true",
        help="Skip the 80/20 train/validation split and train directly on all sequences.",
    )
    parser.add_argument(
        "--write_descriptor_csv",
        type=Path,
        default=None,
        help="Path to write the nulled QSAR-12 training descriptor table.",
    )
    parser.add_argument(
        "--null-descriptors",
        "--null_descriptors",
        action="append",
        default=[],
        help=(
            "Descriptor names to force to 0.0 before fitting. Accepts comma-separated "
            "names and may be repeated. Defaults to the four GRAR740104 sequence-order descriptors."
        ),
    )
    return parser.parse_args()


def compute_metrics(y_true: np.ndarray, sigma: np.ndarray) -> dict:
    y_pred = np.where(sigma >= 0, 1, -1)
    has_both_classes = len(np.unique(y_true)) == 2
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == -1) & (y_pred == -1)).sum())
    fp = int(((y_true == -1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == -1)).sum())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "auc_roc_sigma": roc_auc_score(y_true, sigma) if has_both_classes else float("nan"),
        "auc_pr_sigma": average_precision_score((y_true == 1).astype(int), sigma) if has_both_classes else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def prefixed_id(prefix: str, peptide_id: str) -> str:
    peptide_id = str(peptide_id).strip() or "seq"
    return peptide_id if peptide_id.startswith(prefix) else f"{prefix}{peptide_id}"


def read_csv_sequence_records(path: Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError(f"{path} must contain a 'sequence' column.")
    id_col = next((c for c in ("peptide_id", "sequence_id", "seqIndex", "id", "name") if c in df.columns), None)
    records = []
    for index, row in df.iterrows():
        raw_id = row[id_col] if id_col else f"seq_{index + 1}"
        sequence = canonical_standard_aa_sequence(str(row["sequence"]))
        if sequence is None:
            raise ValueError(f"{path}: invalid amino acid sequence for record {raw_id!r}.")
        records.append((str(raw_id), sequence))
    return records


def read_sequence_file(path: Path, prefix: str) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    records = read_csv_sequence_records(path) if path.suffix.lower() == ".csv" else read_sequence_records(path)
    if not records:
        raise ValueError(f"No valid sequences found in {path}.")
    return [(prefixed_id(prefix, peptide_id), sequence) for peptide_id, sequence in records]


def null_descriptors_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.null_descriptors:
        return parse_null_descriptor_names(args.null_descriptors, QSAR_COLUMNS)
    return DEFAULT_NULL_QSAR_COLUMNS


def needs_sequence_order(null_descriptors: tuple[str, ...]) -> bool:
    return bool(set(SEQUENCE_ORDER_QSAR_COLUMNS) - set(null_descriptors))


def load_grar740104_matrix():
    try:
        from propy import AAIndex
    except Exception as error:
        raise RuntimeError(
            "Could not import ProPy. Install the project requirements, including propy3, "
            "before training the QSAR-12 SVM."
        ) from error
    aaindex_dir = BASE_DIR.parent / "descriptors" / "aaindex"
    try:
        return AAIndex.GetAAIndex23("GRAR740104", path=str(aaindex_dir))
    except Exception as error:
        raise RuntimeError(f"Could not load GRAR740104 from {aaindex_dir}.") from error


def compute_sequence_order_descriptors(descriptor, sequence: str, grar740104_matrix) -> dict[str, float]:
    try:
        socn = descriptor.GetSOCNp(maxlag=30, distancematrix=grar740104_matrix)
        qso = descriptor.GetQSOp(maxlag=30, weight=0.05, distancematrix=grar740104_matrix)
    except Exception as error:
        raise RuntimeError(
            f"Could not compute GRAR740104 descriptors for sequence {sequence!r} "
            f"(length {len(sequence)})."
        ) from error

    required = {"tau2": socn, "tau4": socn, "QSO50": qso, "QSO29": qso}
    missing = [name for name, values in required.items() if name not in values]
    if missing:
        raise RuntimeError(
            f"ProPy omitted required descriptors for sequence {sequence!r}: {', '.join(missing)}."
        )

    length = len(sequence)
    return {
        "tau2_GRAR740104": float(socn["tau2"] / (length - 2)) if length > 2 else 0.0,
        "tau4_GRAR740104": float(socn["tau4"] / (length - 4)) if length > 4 else 0.0,
        "QSO50_GRAR740104": float(qso["QSO50"]),
        "QSO29_GRAR740104": float(qso["QSO29"]),
    }


def require_finite_descriptors(features: dict[str, float], sequence: str) -> None:
    invalid = [name for name, value in features.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(
            f"Non-finite descriptors for sequence {sequence!r}: {', '.join(invalid)}."
        )


def compute_ablatable_qsar12(
    sequence: str,
    null_descriptors: tuple[str, ...],
    grar740104_matrix,
) -> dict[str, float]:
    try:
        from propy import ProCheck
        from propy.PyPro import GetProDes
    except Exception as error:
        raise RuntimeError("Could not import ProPy descriptor APIs.") from error

    if ProCheck.ProteinCheck(sequence) == 0:
        raise ValueError(f"ProPy rejected sequence: {sequence}")

    descriptor = GetProDes(sequence)
    dpc = descriptor.GetDPComp()
    ctd = descriptor.GetCTD()
    sequence_order = (
        compute_sequence_order_descriptors(descriptor, sequence, grar740104_matrix)
        if grar740104_matrix is not None
        else {name: 0.0 for name in SEQUENCE_ORDER_QSAR_COLUMNS}
    )
    methionine = sequence.count("M")
    lysine = sequence.count("K")
    features = {
        "netCharge": float(sum(CHARGE[residue] for residue in sequence)),
        **{name: float(f"{dpc[name]:.2f}") for name in ("FC", "LW", "DP", "NK", "AE")},
        "pcMK": 0.0 if methionine == 0 else methionine / (methionine + lysine),
        "_SolventAccessibilityD1025": float(ctd["_SolventAccessibilityD1025"]),
        **sequence_order,
    }
    null_descriptor_values(features, null_descriptors)
    require_finite_descriptors(features, sequence)
    return features


def build_null_descriptor_table(
    amp_records: list[tuple[str, str]],
    decoy_records: list[tuple[str, str]],
    null_descriptors: tuple[str, ...],
) -> pd.DataFrame:
    grar740104_matrix = load_grar740104_matrix() if needs_sequence_order(null_descriptors) else None
    labeled_records = [(peptide_id, sequence, 1) for peptide_id, sequence in amp_records]
    labeled_records += [(peptide_id, sequence, -1) for peptide_id, sequence in decoy_records]
    rows = []
    for index, (peptide_id, sequence, label) in enumerate(labeled_records, start=1):
        if index == 1 or index % 100 == 0:
            print(f"   Computed nulled descriptors for {index}/{len(labeled_records)} sequences...")
        features = compute_ablatable_qsar12(sequence, null_descriptors, grar740104_matrix)
        rows.append({"peptide_id": peptide_id, "sequence": sequence, "label": label, **features})
    return pd.DataFrame(rows)


def validate_training_table(df: pd.DataFrame, require_validation_split: bool) -> None:
    duplicates = sorted(df.loc[df["peptide_id"].duplicated(), "peptide_id"].unique())
    if duplicates:
        raise ValueError(f"peptide_id values must be unique; duplicates include: {', '.join(duplicates[:5])}")
    counts = df["label"].value_counts()
    if counts.get(1, 0) < 1 or counts.get(-1, 0) < 1:
        raise ValueError("Training requires at least one AMP and one decoy sequence.")
    if require_validation_split and (counts.get(1, 0) < 2 or counts.get(-1, 0) < 2):
        raise ValueError("Training requires at least two AMP and two decoy sequences for the validation split.")
    values = df[list(QSAR_COLUMNS)].to_numpy(dtype=np.float64)
    invalid_rows = (~np.isfinite(values)).any(axis=1).sum()
    if invalid_rows:
        raise ValueError(f"Found {invalid_rows} rows with non-finite QSAR-12 descriptor values.")


def downsample_balanced_training_table(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    amp_df = df[df["label"] == 1]
    decoy_df = df[df["label"] == -1]
    n_per_class = min(len(amp_df), len(decoy_df))
    balanced = pd.concat(
        [
            amp_df.sample(n=n_per_class, random_state=random_state),
            decoy_df.sample(n=n_per_class, random_state=random_state),
        ],
        ignore_index=True,
    )
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def p_amp_probabilities(svm: SVC, x: np.ndarray) -> np.ndarray:
    probabilities = svm.predict_proba(x)
    pos_idx = int(np.where(svm.classes_ == 1)[0][0])
    return probabilities[:, pos_idx]


def write_score_outputs(path: Path, df: pd.DataFrame, svm: SVC, x: np.ndarray) -> None:
    sigma = np.asarray(svm.decision_function(x)).ravel()
    out = df[["peptide_id", "sequence", "label"]].copy()
    out["prediction"] = np.where(sigma >= 0, 1, -1)
    out["sigma"] = sigma
    out["P(AMP)"] = p_amp_probabilities(svm, x)
    out = out.sort_values("sigma", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def normalized_feature_matrix(fit_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_raw = fit_df[feature_cols].values.astype(np.float64)
    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0)
    constant_features = [feature_cols[index] for index in np.flatnonzero(stds == 0)]
    if constant_features:
        print(
            "WARNING: These descriptors are constant in the fitting data and cannot influence the SVM: "
            + ", ".join(constant_features)
        )
    stds_safe = np.where(stds > 0, stds, 1.0)
    return (x_raw - means) / stds_safe, means, stds_safe


def fit_validation_report(args: argparse.Namespace, svm: SVC, x: np.ndarray, y: np.ndarray) -> None:
    if args.no_validation_split:
        print("\nValidation split: disabled (--no_validation_split)")
        return

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
    train_idx, val_idx = next(splitter.split(x, y))
    svm.fit(x[train_idx], y[train_idx])
    sigma_val = np.asarray(svm.decision_function(x[val_idx])).ravel()
    metrics = compute_metrics(y[val_idx], sigma_val)

    print("\nValidation metrics using sigma >= 0:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")


def write_zscores(path: Path, feature_cols: list[str], means: np.ndarray, stds: np.ndarray) -> None:
    with path.open("w") as handle:
        handle.write(",".join(feature_cols) + "\n")
        handle.write(",".join(f"{m:.10f}" for m in means) + "\n")
        handle.write(",".join(f"{s:.10f}" for s in stds) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Training QSAR-12 SVM with nulled sequence-order descriptors ===")
    print(f"AMP sequences  : {args.amp_sequences}")
    print(f"Decoy sequences: {args.decoy_sequences}")
    print(f"Out dir        : {out_dir}")
    null_descriptors = null_descriptors_from_args(args)
    print("Nulled columns : " + ", ".join(null_descriptors))

    amp_records = read_sequence_file(args.amp_sequences, "AMP_")
    decoy_records = read_sequence_file(args.decoy_sequences, "DECOY_")
    print(f"Loaded {len(amp_records)} AMP and {len(decoy_records)} decoy sequences.")

    all_df = build_null_descriptor_table(amp_records, decoy_records, null_descriptors)
    validate_training_table(all_df, require_validation_split=not args.no_validation_split)
    print(f"Computed descriptors for {len(all_df)} total sequences.")

    fit_df = downsample_balanced_training_table(all_df, args.random_state) if args.balance_classes else all_df
    if args.balance_classes:
        fit_counts = fit_df["label"].value_counts()
        print(f"Class downsampling: {fit_counts.get(1, 0)} AMP and {fit_counts.get(-1, 0)} decoy sequences")
    else:
        print("Class downsampling: disabled; fitting with all examples")

    descriptor_csv = args.write_descriptor_csv or (out_dir / "svm_qsar12_training_descriptors.csv")
    descriptor_csv.parent.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(descriptor_csv, index=False)
    print(f"Saved nulled training descriptor CSV: {descriptor_csv}")

    y = fit_df["label"].values.astype(np.int64)
    feature_cols = list(QSAR_COLUMNS)
    x, means, stds = normalized_feature_matrix(fit_df, feature_cols)

    class_weight = None if args.no_class_weight else "balanced"
    print(f"Class weights: {class_weight or 'disabled'}")
    print("Probability output: P(AMP) from Platt-calibrated SVM probabilities")
    svm = SVC(
        kernel=args.kernel,
        probability=True,
        C=args.C,
        gamma="scale",
        class_weight=class_weight,
        random_state=args.random_state,
    )
    fit_validation_report(args, svm, x, y)

    svm.fit(x, y)

    svm_path = out_dir / "svm_qsar12_model.pkl"
    joblib.dump(svm, svm_path)
    print(f"\nSaved SVM model: {svm_path}")

    train_scores_path = out_dir / "svm_qsar12_training_scores.csv"
    write_score_outputs(train_scores_path, fit_df, svm, x)
    print(f"Saved training score output: {train_scores_path}")

    z_path = out_dir / "svm_qsar12_zscores.txt"
    write_zscores(z_path, feature_cols, means, stds)
    print(f"Saved Z-score file: {z_path}")

    print("\nDone. To use this null-tail SVM in compare_model_predictions.py, run it with:")
    print(f"  --svm_descriptor_csv {descriptor_csv}")
    print(f"  --svm_z_file {z_path}")
    print(f"  --svm_pkl {svm_path}")
    print(f"  training scores written to {train_scores_path}")


if __name__ == "__main__":
    main()

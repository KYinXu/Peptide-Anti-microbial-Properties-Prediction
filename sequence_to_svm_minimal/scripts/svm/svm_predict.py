#!/usr/bin/env python3
"""
Run QSAR-12 SVM inference from peptide sequences.

This script is self-contained apart from its Python dependencies and one model
file. The model must be a joblib containing a fitted sklearn Pipeline with its
StandardScaler and SVC, such as the output of
convert_svm_pkl_zscores_to_pipeline.py.

Input CSV files require a `sequence` column and may include `peptide_id`.
Text input accepts either one sequence per line or `peptide_id<TAB>sequence`.

Example:
    python svm_predict.py --input peptides.csv --model amp_svm.joblib \
        --output predictions.csv

Dependencies:
    numpy, scikit-learn, joblib, propy3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

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

CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1, "I": 0,
    "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict AMP probability from peptide sequences with a QSAR-12 SVM."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="CSV or text sequence file")
    parser.add_argument("--model", "-m", type=Path, required=True, help="Pipeline joblib file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Prediction CSV path")
    parser.add_argument(
        "--write-features",
        type=Path,
        help="Optional QSAR-12 CSV for inspection or reuse",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[tuple[str, str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    records = read_csv(path) if path.suffix.lower() == ".csv" else read_text(path)
    if not records:
        raise ValueError("Input contains no sequences.")
    ids = [record[0] for record in records]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"peptide_id values must be unique; duplicates include: {', '.join(duplicates[:5])}")
    return [
        (peptide_id, normalize_sequence(sequence, peptide_id), name)
        for peptide_id, sequence, name in records
    ]


def read_csv(path: Path) -> list[tuple[str, str, str | None]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0]:
        raise ValueError("CSV input must have a header row.")
    columns = set(rows[0])
    if "sequence" not in columns:
        raise ValueError("CSV input requires a 'sequence' column.")
    id_column = next(
        (name for name in ("peptide_id", "sequence_id", "id", "name") if name in columns),
        None,
    )
    return [
        (
            row[id_column].strip() if id_column and row[id_column] else f"seq_{index}",
            row["sequence"],
            row.get("name") or None,
        )
        for index, row in enumerate(rows, start=1)
    ]


def read_text(path: Path) -> list[tuple[str, str, str | None]]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        peptide_id, sequence = line.split("\t", 1) if "\t" in line else (f"seq_{index}", line)
        records.append((peptide_id.strip(), sequence.strip(), None))
    return records


def normalize_sequence(sequence: str, peptide_id: str) -> str:
    normalized = sequence.strip().upper()
    invalid = sorted(set(normalized) - set(CHARGE))
    if not normalized or invalid:
        detail = f"invalid residues {''.join(invalid)!r}" if invalid else "an empty sequence"
        raise ValueError(f"{peptide_id}: {detail}")
    return normalized


def qsar12(sequence: str) -> dict[str, float]:
    from propy import ProCheck
    from propy.PyPro import GetProDes

    if ProCheck.ProteinCheck(sequence) == 0:
        raise ValueError("ProPy rejected the sequence.")
    descriptor = GetProDes(sequence)
    dpc = descriptor.GetDPComp()
    ctd = descriptor.GetCTD()
    try:
        socn = descriptor.GetSOCN(maxlag=30)
    except Exception:
        socn = {}
    try:
        qso = descriptor.GetQSO(maxlag=30, weight=0.05)
    except Exception:
        qso = {}
    length = len(sequence)
    methionine = sequence.count("M")
    lysine = sequence.count("K")
    return {
        "netCharge": float(sum(CHARGE[residue] for residue in sequence)),
        **{name: round(dpc.get(name, 0), 2) for name in ("FC", "LW", "DP", "NK", "AE")},
        "pcMK": 0.0 if methionine == 0 else methionine / (methionine + lysine),
        "_SolventAccessibilityD1025": float(ctd.get("_SolventAccessibilityD1025", 0)),
        "tau2_GRAR740104": float(socn.get("tau2", 0) / (length - 2) if length > 2 else 0),
        "tau4_GRAR740104": float(socn.get("tau4", 0) / (length - 4) if length > 4 else 0),
        "QSO50_GRAR740104": float(qso.get("QSO50", 0)),
        "QSO29_GRAR740104": float(qso.get("QSO29", 0)),
    }


def build_features(records: list[tuple[str, str, str | None]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = []
    values = []
    for peptide_id, sequence, name in records:
        try:
            features = qsar12(sequence)
        except Exception as error:
            raise ValueError(f"{peptide_id}: could not compute QSAR-12 descriptors: {error}") from error
        rows.append(
            {
                "peptide_id": peptide_id,
                "sequence": sequence,
                **({"name": name} if name else {}),
                **features,
            }
        )
        values.append([features[name] for name in QSAR_COLUMNS])
    return rows, np.asarray(values, dtype=np.float64)


def load_model(path: Path) -> tuple[Any, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    import joblib

    model = joblib.load(path)
    if not hasattr(model, "named_steps") or not hasattr(model, "predict"):
        raise TypeError(
            "Model must be a fitted sklearn Pipeline containing both scaling and SVC. "
            "Convert a legacy SVC + z-score file before using this script."
        )
    scaler = next(
        (step for step in model.named_steps.values() if hasattr(step, "mean_") and hasattr(step, "scale_")),
        None,
    )
    if scaler is None:
        raise TypeError("Pipeline has no fitted scaler; raw QSAR values cannot be used safely.")
    names = tuple(str(name) for name in getattr(scaler, "feature_names_in_", QSAR_COLUMNS))
    if set(names) != set(QSAR_COLUMNS):
        raise ValueError("Model feature names do not match the supported QSAR-12 descriptor contract.")
    return model, names


def predict(model: Any, feature_rows: list[dict[str, Any]], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray([[row[name] for name in names] for row in feature_rows], dtype=np.float64)
    raw_predictions = np.asarray(model.predict(matrix)).ravel()
    estimator = model.named_steps[list(model.named_steps)[-1]]
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    positive_class = 1 if 1 in classes else classes[-1]
    probabilities = np.asarray(model.predict_proba(matrix))[:, int(np.where(classes == positive_class)[0][0])]
    distance = np.asarray(model.decision_function(matrix)).ravel()
    return (raw_predictions == positive_class).astype(int), probabilities, distance


def write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        records = read_records(args.input)
        feature_rows, _ = build_features(records)
        model, feature_order = load_model(args.model)
        predictions, probabilities, distance = predict(model, feature_rows, feature_order)
        output_rows = [
            {
                "peptide_id": row["peptide_id"],
                "sequence": row["sequence"],
                **({"name": row["name"]} if "name" in row else {}),
                "SVM_pred": int(prediction),
                "SVM_prob_AMP": float(probability),
                "SVM_hyperplane_distance": float(score),
            }
            for row, prediction, probability, score in zip(feature_rows, predictions, probabilities, distance)
        ]
        output_columns = ("peptide_id", "sequence")
        if any("name" in row for row in output_rows):
            output_columns += ("name",)
        output_columns += ("SVM_pred", "SVM_prob_AMP", "SVM_hyperplane_distance")
        write_csv(args.output, output_rows, output_columns)
        if args.write_features:
            feature_columns = ("peptide_id", "sequence")
            if any("name" in row for row in feature_rows):
                feature_columns += ("name",)
            write_csv(args.write_features, feature_rows, (*feature_columns, *QSAR_COLUMNS))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(records)} predictions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

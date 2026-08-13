#!/usr/bin/env python3
"""
Run QSAR-12 SVM inference on fixed-length sequence windows.

Example:
    python svm_predict_windowed.py --input peptides.csv --model amp_svm.joblib \
        --window-size 20 --stride 5 --output window_predictions.csv \
        --summary-output window_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from svm_predict import QSAR_COLUMNS, build_features, load_model, predict, read_records, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score fixed-length peptide windows with a QSAR-12 SVM."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="CSV or text sequence file")
    parser.add_argument("--model", "-m", type=Path, required=True, help="Pipeline joblib file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Per-window prediction CSV")
    parser.add_argument("--window-size", type=int, help="Fixed window length in residues (mutually exclusive with min-len/max-len)")
    parser.add_argument("--window-min-len", type=int, help="Minimum window length in residues (if not using fixed --window-size)")
    parser.add_argument("--window-max-len", type=int, help="Maximum window length in residues (if not using fixed --window-size)")
    parser.add_argument("--stride", type=int, default=1, help="Residues between window starts (default: 1)")
    parser.add_argument(
        "--include-terminal-window",
        action="store_true",
        help="Add a final full-length window ending at the sequence C-terminus when stride misses it.",
    )
    parser.add_argument("--summary-output", type=Path, help="Optional per-parent summary CSV")
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used in summary hit counts (default: 0.5)",
    )
    parser.add_argument("--write-features", type=Path, help="Optional per-window QSAR-12 CSV")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size is None and (args.window_min_len is None or args.window_max_len is None):
        raise ValueError("Must provide either --window-size OR both --window-min-len and --window-max-len.")
    if args.window_size is not None and (args.window_min_len is not None or args.window_max_len is not None):
        raise ValueError("--window-size is mutually exclusive with --window-min-len and --window-max-len.")
    
    if args.window_size is not None and args.window_size < 1:
        raise ValueError("--window-size must be at least 1.")
    if args.window_min_len is not None and (args.window_min_len < 1 or args.window_max_len < args.window_min_len):
        raise ValueError("Invalid min/max window lengths.")
        
    if args.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if not 0 <= args.hit_threshold <= 1:
        raise ValueError("--hit-threshold must be between 0 and 1.")


def window_starts(length: int, window_size: int, stride: int, include_terminal: bool) -> list[int]:
    if length < window_size:
        return []
    starts = list(range(0, length - window_size + 1, stride))
    terminal_start = length - window_size
    if include_terminal and terminal_start not in starts:
        starts.append(terminal_start)
    return sorted(starts)


def build_windows(
    records: list[tuple[str, str, str | None]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    windows = []
    
    min_len = args.window_size if args.window_size is not None else args.window_min_len
    max_len = args.window_size if args.window_size is not None else args.window_max_len
    
    for parent_id, sequence, name in records:
        for win_len in range(min_len, max_len + 1):
            for ordinal, start in enumerate(
                window_starts(len(sequence), win_len, args.stride, args.include_terminal_window)
            ):
                end = start + win_len
                windows.append(
                    {
                        "parent_id": parent_id,
                        "parent_sequence_length": len(sequence),
                        "window_id": f"{parent_id}_l{win_len}_s{start:04d}",
                        "window_start": start,
                        "window_end": end,
                        "sequence": sequence[start:end],
                        **({"name": name} if name else {}),
                    }
                )
    return windows


def score_windows(windows: list[dict[str, Any]], model: Any, feature_order: tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows, _ = build_features(
        [(window["window_id"], window["sequence"], window.get("name")) for window in windows]
    )
    predictions, probabilities, distances = predict(model, feature_rows, feature_order)
    output = []
    for window, prediction, probability, distance, features in zip(
        windows, predictions, probabilities, distances, feature_rows
    ):
        output.append(
            {
                **window,
                "SVM_pred": int(prediction),
                "SVM_prob_AMP": float(probability),
                "SVM_hyperplane_distance": float(distance),
            }
        )
        features.update(
            {
                "parent_id": window["parent_id"],
                "window_id": window["window_id"],
                "window_start": window["window_start"],
                "window_end": window["window_end"],
            }
        )
    return output, feature_rows


def summarize(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(row)
    summaries = []
    for parent_id, windows in grouped.items():
        strongest = max(windows, key=lambda row: row["SVM_prob_AMP"])
        hits = [row for row in windows if row["SVM_prob_AMP"] >= threshold]
        summaries.append(
            {
                "parent_id": parent_id,
                "parent_sequence_length": windows[0]["parent_sequence_length"],
                **({"name": windows[0]["name"]} if "name" in windows[0] else {}),
                "window_count": len(windows),
                "hit_threshold": threshold,
                "hit_window_count": len(hits),
                "hit_window_fraction": len(hits) / len(windows),
                "max_SVM_prob_AMP": strongest["SVM_prob_AMP"],
                "max_probability_window_start": strongest["window_start"],
                "max_probability_window_end": strongest["window_end"],
                "max_probability_window_sequence": strongest["sequence"],
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        records = read_records(args.input)
        windows = build_windows(
            records,
            args,
        )
        if not windows:
            raise ValueError(
                f"No input sequences are long enough for the requested window sizes."
            )
        model, feature_order = load_model(args.model)
        output_rows, feature_rows = score_windows(windows, model, feature_order)
        columns = (
            "parent_id",
            "parent_sequence_length",
            "window_id",
            "window_start",
            "window_end",
            "sequence",
        )
        if any("name" in row for row in output_rows):
            columns += ("name",)
        columns += ("SVM_pred", "SVM_prob_AMP", "SVM_hyperplane_distance")
        write_csv(args.output, output_rows, columns)
        if args.write_features:
            feature_columns = ("parent_id", "window_id", "window_start", "window_end", "peptide_id", "sequence")
            if any("name" in row for row in feature_rows):
                feature_columns += ("name",)
            write_csv(args.write_features, feature_rows, (*feature_columns, *QSAR_COLUMNS))
        if args.summary_output:
            summary_rows = summarize(output_rows, args.hit_threshold)
            summary_columns = ("parent_id", "parent_sequence_length")
            if any("name" in row for row in summary_rows):
                summary_columns += ("name",)
            summary_columns += (
                "window_count",
                "hit_threshold",
                "hit_window_count",
                "hit_window_fraction",
                "max_SVM_prob_AMP",
                "max_probability_window_start",
                "max_probability_window_end",
                "max_probability_window_sequence",
            )
            write_csv(args.summary_output, summary_rows, summary_columns)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(output_rows)} window predictions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

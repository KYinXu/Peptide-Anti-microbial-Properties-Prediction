#!/usr/bin/env python3
"""Run one-time sequence inference and print prediction summaries."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "sequence_analysis"

from .models import (
    DEFAULT_GNN_ROOT,
    GnnSequencePrediction,
    GnnSequenceScorer,
    SvmSequencePrediction,
    SvmSequenceScorer,
    resolve_gnn_model_dir,
)
from .utils import add_svm_descriptor_ablation_arguments
from .utils.data_loader import NormalizedSequenceDataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "original_svm"
SVM_PICKLE_SUFFIXES = {".pkl"}
ZSCORE_SUFFIXES = {".csv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequence-level model inference and print a CLI summary.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Normalized CSV with id and sequence columns.")
    parser.add_argument("--model", choices=["svm", "gnn"], default="svm", help="Model adapter to use.")
    parser.add_argument(
        "--gnn-model",
        type=Path,
        help=f"GNN model directory or gnn_model.pt. Defaults to {DEFAULT_GNN_ROOT}.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="GNN inference batch size.")
    parser.add_argument("--device", type=str, default=None, help="Torch device for GNN inference and folding.")
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Regenerate GNN structures and QSAR features before scoring.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "SVM: directory with one pickle and one z-score file "
            f"(default: {DEFAULT_CHECKPOINT_DIR}). "
            "GNN: directory with gnn_model.pt, or a parent of one "
            f"(default: {DEFAULT_GNN_ROOT})."
        ),
    )
    parser.add_argument("--svm-pkl", type=Path, help="SVM pickle path. Defaults to the .pkl file in --checkpoint-dir.")
    parser.add_argument(
        "--zscores",
        type=Path,
        help="Z-score CSV/TXT file path. Defaults to the .csv or .txt file in --checkpoint-dir.",
    )
    parser.add_argument("--top", type=int, default=None, help="Optional limit on rows printed in alphabetical order.")
    add_svm_descriptor_ablation_arguments(parser)
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    svm_pkl = args.svm_pkl or find_single_checkpoint_file(checkpoint_dir, SVM_PICKLE_SUFFIXES, "SVM pickle")
    zscores = args.zscores or find_single_checkpoint_file(checkpoint_dir, ZSCORE_SUFFIXES, "z-score")
    return svm_pkl, zscores


def find_single_checkpoint_file(directory: Path, suffixes: set[str], description: str) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {directory}")
    candidates = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    if not candidates:
        suffix_list = ", ".join(sorted(suffixes))
        raise FileNotFoundError(f"No {description} file ({suffix_list}) found in {directory}")
    if len(candidates) > 1:
        candidate_list = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple {description} files found in {directory}: {candidate_list}")
    return candidates[0]


def display_prediction(value: int) -> str:
    return "AMP" if value == 1 else "."


def print_score_summary(
    title: str,
    details: list[str],
    predictions: list[SvmSequencePrediction | GnnSequencePrediction],
    top_n: int | None,
) -> None:
    if not predictions:
        print("No predictions.")
        return
    sigmas = np.asarray([prediction.sigma for prediction in predictions], dtype=np.float64)
    p_amp = np.asarray([prediction.p_amp for prediction in predictions], dtype=np.float64)
    counts = Counter(prediction.prediction for prediction in predictions)
    rows_to_print = predictions if top_n is None else predictions[:top_n]

    print(title)
    for line in details:
        print(line)
    print(f"Rows scored: {len(predictions)}")
    print(f"Predictions: +1={counts.get(1, 0)}, -1={counts.get(-1, 0)}")
    print(f"sigma: min={sigmas.min():.2f}, mean={sigmas.mean():.2f}, max={sigmas.max():.2f}")
    print(f"P(AMP): min={p_amp.min():.2f}, mean={p_amp.mean():.2f}, max={p_amp.max():.2f}")
    print()
    print("Predictions By Input Sequence Order")
    id_width = max(len("id"), *(len(prediction.id) for prediction in rows_to_print))
    header = f"{'#':>4}  {'id':<{id_width}}  {'pred':>5}  {'sigma':>12}  {'P(AMP)':>10}"
    print(header)
    print("-" * len(header))
    for index, prediction in enumerate(rows_to_print, start=1):
        pred = display_prediction(prediction.prediction)
        print(f"{index:>4}  {prediction.id:<{id_width}}  {pred:>5}  {prediction.sigma:>12.2f}  {prediction.p_amp:>10.2f}")


def print_svm_summary(
    predictions: list[SvmSequencePrediction],
    svm_pkl: Path,
    zscores: Path,
    top_n: int | None,
    null_descriptors: tuple[str, ...] = (),
) -> None:
    nulled = ", ".join(null_descriptors) if null_descriptors else "(none)"
    print_score_summary(
        "SVM Sequence Inference",
        [f"Model: {svm_pkl}", f"Z-scores: {zscores}", f"Nulled descriptors: {nulled}"],
        predictions,
        top_n,
    )


def main() -> int:
    args = parse_args()
    try:
        if args.top is not None and args.top < 1:
            raise ValueError("--top must be at least 1 when provided.")
        dataset = NormalizedSequenceDataset.from_csv(args.input)
        records = list(dataset.records())
        if args.model == "svm":
            svm_pkl, zscores = resolve_checkpoint_paths(args)
            scorer = SvmSequenceScorer.from_paths(
                svm_pkl,
                zscores,
                null_descriptors=args.null_descriptors,
            )
            print_svm_summary(
                scorer.score(records),
                svm_pkl,
                zscores,
                args.top,
                scorer.null_descriptors,
            )
        elif args.model == "gnn":
            if args.null_descriptors:
                raise ValueError("GNN inference does not null QSAR columns. Omit --null-descriptors.")
            model_dir = resolve_gnn_model_dir(args.gnn_model, args.checkpoint_dir)
            gnn = GnnSequenceScorer.from_paths(
                model_dir,
                batch_size=args.batch_size,
                device=args.device,
                force_process=args.force_process,
            )
            print_score_summary(
                "GNN Sequence Inference",
                [
                    f"Model: {model_dir}",
                    "sigma: logit margin (AMP - non-AMP); prediction is +1 when P(AMP) >= 0.5",
                    f"Tabular columns: {', '.join(gnn.feature_cols) if gnn.feature_cols else '(none)'}",
                ],
                gnn.score(records, args.input),
                args.top,
            )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


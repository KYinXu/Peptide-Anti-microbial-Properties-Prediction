#!/usr/bin/env python3
"""Run sequence SVM inference after nulling selected QSAR descriptors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "sequence_analysis"

from .models import SvmSequenceScorer
from .print_predictions import (
    DEFAULT_CHECKPOINT_DIR,
    SVM_PICKLE_SUFFIXES,
    ZSCORE_SUFFIXES,
    find_single_checkpoint_file,
    print_svm_summary,
)
from .utils.data_loader import NormalizedSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVM inference after forcing selected QSAR descriptors to 0.0."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Normalized CSV with id and sequence columns.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing one SVM pickle and one z-score CSV/TXT file (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument("--svm-pkl", type=Path, help="SVM pickle path. Defaults to the .pkl file in --checkpoint-dir.")
    parser.add_argument(
        "--zscores",
        type=Path,
        help="Z-score CSV/TXT file path. Defaults to the .csv or .txt file in --checkpoint-dir.",
    )
    parser.add_argument(
        "--null-descriptors",
        "--null_descriptors",
        action="append",
        default=[],
        help=(
            "Descriptor names to force to 0.0 before z-score scaling. Accepts comma-separated "
            "names and may be repeated."
        ),
    )
    parser.add_argument("--top", type=int, default=None, help="Optional limit on rows printed in input order.")
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    svm_pkl = args.svm_pkl or find_single_checkpoint_file(args.checkpoint_dir, SVM_PICKLE_SUFFIXES, "SVM pickle")
    zscores = args.zscores or find_single_checkpoint_file(args.checkpoint_dir, ZSCORE_SUFFIXES, "z-score")
    return svm_pkl, zscores


def main() -> int:
    args = parse_args()
    try:
        if args.top is not None and args.top < 1:
            raise ValueError("--top must be at least 1 when provided.")
        dataset = NormalizedSequenceDataset.from_csv(args.input)
        svm_pkl, zscores = resolve_checkpoint_paths(args)
        scorer = SvmSequenceScorer.from_paths(svm_pkl, zscores, null_descriptors=args.null_descriptors)
        nulled = ", ".join(scorer.null_descriptors) if scorer.null_descriptors else "(none)"
        print(f"Nulled descriptors: {nulled}")
        print_svm_summary(scorer.score(list(dataset.records())), svm_pkl, zscores, args.top)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

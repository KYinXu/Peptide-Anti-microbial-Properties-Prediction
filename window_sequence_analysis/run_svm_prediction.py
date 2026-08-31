#!/usr/bin/env python3
"""Run SVM prediction for compact per-residue window sequence profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from window_sequence_analysis.data_loader import NormalizedSequenceDataset
from window_sequence_analysis.models.svm import SvmWindowScorer
from window_sequence_analysis.sliding_windows.common import ProfileConfig
from window_sequence_analysis.sliding_windows.progress import build_progress_reporter
from window_sequence_analysis.sliding_windows.runner import run_window_profile_analysis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "svm"
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "window_sequence_profiles.csv"
SVM_PICKLE_SUFFIXES = {".pkl"}
ZSCORE_SUFFIXES = {".csv", ".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score all sequence windows with an SVM and save compact per-residue "
            "mean P(AMP) and hyperplane-distance profiles."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Normalized CSV with required id and sequence columns; extra columns are preserved.",
    )
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
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--window-min-len", type=int, default=10, help="Minimum window length (default: 10).")
    parser.add_argument("--window-max-len", type=int, default=35, help="Maximum window length (default: 35).")
    parser.add_argument("--stride", type=int, default=1, help="Window start stride (default: 1).")
    parser.add_argument(
        "--batch-starts",
        type=int,
        default=64,
        help="Number of start positions to score per bounded in-memory batch (default: 64).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Significant digits for serialized float profiles (default: 6).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress to stderr every N completed sequences (default: 1).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress reporting.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    return ProfileConfig(
        min_len=args.window_min_len,
        max_len=args.window_max_len,
        stride=args.stride,
        batch_starts=args.batch_starts,
        precision=args.precision,
    )


def validate_config(config: ProfileConfig) -> None:
    if config.min_len < 1:
        raise ValueError("--window-min-len must be at least 1.")
    if config.max_len < config.min_len:
        raise ValueError("--window-max-len must be greater than or equal to --window-min-len.")
    if config.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if config.batch_starts < 1:
        raise ValueError("--batch-starts must be at least 1.")
    if config.precision < 1:
        raise ValueError("--precision must be at least 1.")


def validate_args(args: argparse.Namespace) -> None:
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1.")


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    svm_pkl = args.svm_pkl or find_single_checkpoint_file(args.checkpoint_dir, SVM_PICKLE_SUFFIXES, "SVM pickle")
    zscores = args.zscores or find_single_checkpoint_file(args.checkpoint_dir, ZSCORE_SUFFIXES, "z-score")
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


def main() -> int:
    args = parse_args()
    try:
        config = config_from_args(args)
        validate_config(config)
        validate_args(args)
        svm_pkl, zscores = resolve_checkpoint_paths(args)
        dataset = NormalizedSequenceDataset.from_csv(args.input)
        scorer = SvmWindowScorer.from_paths(svm_pkl, zscores)
        progress = build_progress_reporter(quiet=args.quiet, every=args.progress_every)
        count = run_window_profile_analysis(
            dataset.records(),
            scorer,
            config,
            args.output,
            label_columns=dataset.label_columns,
            progress=progress,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {count} sequence profile row(s) to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

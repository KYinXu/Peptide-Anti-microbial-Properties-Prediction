#!/usr/bin/env python3
"""Run raw per-window predictions for normalized sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "window_sequence_analysis"

from analysis_outputs import RunConfiguration, RunOutput, RunSpec, execute_csv_run

from .data_loader import NormalizedSequenceDataset
from .models import SvmScorerFactory
from .run_options import add_window_arguments, validate_window_config, window_config_from_args
from .sliding_windows import WindowConfig, build_progress_reporter, run_raw_window_analysis


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "svm_no_class_weight"
OUTPUT_FILENAME = "raw_window_predictions.csv"
SVM_PICKLE_SUFFIXES = {".pkl"}
ZSCORE_SUFFIXES = {".csv", ".txt"}
SUPPORTED_MODELS = ("svm",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score all sequence windows with a selected model and save one CSV row per "
            "window with prediction, sigma, and P(AMP)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Normalized CSV with required id and sequence columns.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="svm",
        help="Model adapter to use (default: svm).",
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
    parser.add_argument(
        "--null-descriptors",
        "--null_descriptors",
        action="append",
        default=[],
        help=(
            "Descriptor names to force to 0.0 before z-score scaling. Accepts comma-separated "
            "names and may be repeated; use the same values passed to run_null_svm_training."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=f"Output CSV name and root (default: <input directory>/results/{OUTPUT_FILENAME}).",
    )
    add_window_arguments(parser)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> WindowConfig:
    return window_config_from_args(args)


def output_path_from_args(args: argparse.Namespace) -> Path:
    return args.output or args.input.parent / "results" / OUTPUT_FILENAME


def validate_config(config: WindowConfig) -> None:
    validate_window_config(config)


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


def run_svm_raw_prediction(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    validate_config(config)
    svm_pkl, zscores = resolve_checkpoint_paths(args)
    dataset = NormalizedSequenceDataset.from_csv(args.input)
    scorer = SvmScorerFactory(svm_pkl, zscores, tuple(args.null_descriptors))()
    progress = build_progress_reporter(
        quiet=args.quiet,
        total=None if args.quiet else dataset.count_records(),
    )
    output_target = output_path_from_args(args)
    spec = RunSpec(
        output_root=output_target.parent,
        output_filename=output_target.name,
        runner="window_sequence_analysis.raw",
        model=args.model,
        config=RunConfiguration(
            output_mode="raw",
            arguments=vars(args),
            effective={
                "window": config,
                "null_descriptors": scorer.null_descriptors,
            },
        ),
        inputs={"sequences": args.input},
        model_files={"svm_pickle": svm_pkl, "zscores": zscores},
        repository_root=ROOT,
    )

    def write_csv(run: RunOutput) -> int:
        return run_raw_window_analysis(
            dataset.records(),
            scorer,
            config,
            run.csv_path,
            progress=progress,
            run_id=run.run_id,
        )

    run = execute_csv_run(spec, write_csv)
    row_count = run.manifest["output"]["row_count"]
    print(f"Saved {row_count} window prediction row(s) to {run.csv_path}")
    print(f"Run manifest: {run.manifest_path}")
    return row_count


def main() -> int:
    args = parse_args()
    try:
        if args.model == "svm":
            run_svm_raw_prediction(args)
        else:
            raise ValueError(f"Unsupported model: {args.model}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

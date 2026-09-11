#!/usr/bin/env python3
"""Run sequence SVM inference and save predictions to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "sequence_analysis"

from analysis_outputs import RunConfiguration, RunCsvWriter, RunOutput, RunSpec, execute_csv_run

from .models import SvmSequencePrediction, SvmSequenceScorer
from .print_predictions import (
    DEFAULT_CHECKPOINT_DIR,
    SVM_PICKLE_SUFFIXES,
    ZSCORE_SUFFIXES,
    find_single_checkpoint_file,
)
from .utils import add_svm_descriptor_ablation_arguments
from .utils.data_loader import NormalizedSequenceDataset, SequenceRecord


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILENAME = "sequence_predictions.csv"
PREDICTION_COLUMNS = ("prediction", "sigma", "p_amp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequence SVM inference and save predictions to CSV.")
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
        "--output",
        "-o",
        type=Path,
        help=f"Output CSV name and root (default: <input directory>/results/{OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--include-descriptors",
        "--save-descriptors",
        action="store_true",
        help="Include the raw descriptor values used for SVM inference in the prediction CSV.",
    )
    add_svm_descriptor_ablation_arguments(parser)
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    svm_pkl = args.svm_pkl or find_single_checkpoint_file(args.checkpoint_dir, SVM_PICKLE_SUFFIXES, "SVM pickle")
    zscores = args.zscores or find_single_checkpoint_file(args.checkpoint_dir, ZSCORE_SUFFIXES, "z-score")
    return svm_pkl, zscores


def output_path_from_args(args: argparse.Namespace) -> Path:
    return args.output or args.input.parent / "results" / OUTPUT_FILENAME


def prediction_row(
    record: SequenceRecord,
    prediction: SvmSequencePrediction,
    descriptors: dict[str, float],
) -> dict[str, object]:
    return {
        "id": record.id,
        "sequence": record.sequence,
        **record.extras,
        **descriptors,
        "prediction": prediction.prediction,
        "sigma": prediction.sigma,
        "p_amp": prediction.p_amp,
    }


def write_predictions(
    output: Path,
    extra_columns: list[str],
    records: list[SequenceRecord],
    predictions: list[SvmSequencePrediction],
    descriptor_names: tuple[str, ...] = (),
    descriptor_values: np.ndarray | None = None,
    run_id: str | None = None,
) -> None:
    if len(records) != len(predictions):
        raise ValueError("Input record and prediction counts do not match.")
    if descriptor_values is not None and descriptor_values.shape != (len(records), len(descriptor_names)):
        raise ValueError("Descriptor matrix dimensions do not match records and descriptor names.")
    excluded_columns = {"id", "sequence", *PREDICTION_COLUMNS, *descriptor_names}
    preserved_columns = [name for name in extra_columns if name not in excluded_columns]
    fieldnames = ["id", "sequence", *preserved_columns, *descriptor_names, *PREDICTION_COLUMNS]
    with RunCsvWriter(output, fieldnames, run_id) as writer:
        for index, (record, prediction) in enumerate(zip(records, predictions)):
            descriptors = (
                dict(zip(descriptor_names, descriptor_values[index], strict=True))
                if descriptor_values is not None
                else {}
            )
            writer.write_row(prediction_row(record, prediction, descriptors))


def execute_prediction_run(
    args: argparse.Namespace,
    dataset: NormalizedSequenceDataset,
    records: list[SequenceRecord],
    predictions: list[SvmSequencePrediction],
    descriptor_names: tuple[str, ...],
    descriptor_values: np.ndarray | None,
    scorer: SvmSequenceScorer,
    svm_pkl: Path,
    zscores: Path,
) -> RunOutput:
    output_target = output_path_from_args(args)
    spec = RunSpec(
        output_root=output_target.parent,
        output_filename=output_target.name,
        runner="sequence_analysis.predictions",
        model="svm",
        config=RunConfiguration(
            output_mode="sequence",
            arguments=vars(args),
            effective={
                "include_descriptors": args.include_descriptors,
                "null_descriptors": scorer.null_descriptors,
            },
        ),
        inputs={"sequences": args.input},
        model_files={"svm_pickle": svm_pkl, "zscores": zscores},
        repository_root=ROOT,
    )

    def write_csv(run: RunOutput) -> int:
        write_predictions(
            run.csv_path,
            dataset.extra_columns,
            records,
            predictions,
            descriptor_names,
            descriptor_values,
            run.run_id,
        )
        return len(predictions)

    return execute_csv_run(spec, write_csv)


def main() -> int:
    args = parse_args()
    try:
        dataset = NormalizedSequenceDataset.from_csv(args.input)
        records = list(dataset.records())
        svm_pkl, zscores = resolve_checkpoint_paths(args)
        scorer = SvmSequenceScorer.from_paths(
            svm_pkl,
            zscores,
            null_descriptors=args.null_descriptors,
        )
        predictions, descriptor_values = scorer.score_with_descriptors(records)
        descriptor_names = tuple(scorer.descriptor_names) if args.include_descriptors else ()
        saved_descriptors = descriptor_values if args.include_descriptors else None
        run = execute_prediction_run(
            args,
            dataset,
            records,
            predictions,
            descriptor_names,
            saved_descriptors,
            scorer,
            svm_pkl,
            zscores,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(predictions)} prediction row(s) to {run.csv_path}")
    print(f"Run manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run sequence model inference and save predictions to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "sequence_analysis"

from analysis_outputs import RunConfiguration, RunCsvWriter, RunOutput, RunSpec, execute_csv_run

from .models import (
    DEFAULT_GNN_ROOT,
    GnnSequencePrediction,
    GnnSequenceScorer,
    SvmSequencePrediction,
    SvmSequenceScorer,
    resolve_gnn_model_dir,
)
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
    parser = argparse.ArgumentParser(description="Run sequence model inference and save predictions to CSV.")
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
    parser.add_argument("--input", "-i", type=Path, required=True, help="Normalized CSV with id and sequence columns.")
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
        help="Include the raw descriptor values used for inference in the prediction CSV.",
    )
    add_svm_descriptor_ablation_arguments(parser)
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_dir = args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR
    svm_pkl = args.svm_pkl or find_single_checkpoint_file(checkpoint_dir, SVM_PICKLE_SUFFIXES, "SVM pickle")
    zscores = args.zscores or find_single_checkpoint_file(checkpoint_dir, ZSCORE_SUFFIXES, "z-score")
    return svm_pkl, zscores


def output_path_from_args(args: argparse.Namespace) -> Path:
    return args.output or args.input.parent / "results" / OUTPUT_FILENAME


def prediction_row(
    record: SequenceRecord,
    prediction: SvmSequencePrediction | GnnSequencePrediction,
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
    predictions: list[SvmSequencePrediction | GnnSequencePrediction],
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
    predictions: list[SvmSequencePrediction | GnnSequencePrediction],
    descriptor_names: tuple[str, ...],
    descriptor_values: np.ndarray | None,
    *,
    model_name: str,
    model_files: dict[str, Path],
    effective: dict[str, object],
) -> RunOutput:
    output_target = output_path_from_args(args)
    spec = RunSpec(
        output_root=output_target.parent,
        output_filename=output_target.name,
        runner="sequence_analysis.predictions",
        model=model_name,
        config=RunConfiguration(
            output_mode="sequence",
            arguments=vars(args),
            effective=effective,
        ),
        inputs={"sequences": args.input},
        model_files=model_files,
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


def score_svm(args: argparse.Namespace, records: list[SequenceRecord]):
    svm_pkl, zscores = resolve_checkpoint_paths(args)
    scorer = SvmSequenceScorer.from_paths(svm_pkl, zscores, null_descriptors=args.null_descriptors)
    predictions, descriptor_values = scorer.score_with_descriptors(records)
    return predictions, descriptor_values, scorer, {"svm_pickle": svm_pkl, "zscores": zscores}


def score_gnn(args: argparse.Namespace, records: list[SequenceRecord]):
    if args.null_descriptors:
        raise ValueError("GNN inference does not null QSAR columns. Omit --null-descriptors.")
    model_dir = resolve_gnn_model_dir(args.gnn_model, args.checkpoint_dir)
    scorer = GnnSequenceScorer.from_paths(
        model_dir,
        batch_size=args.batch_size,
        device=args.device,
        force_process=args.force_process,
    )
    predictions, descriptor_values = scorer.score_with_descriptors(records, args.input)
    return predictions, descriptor_values, scorer, {"gnn_model": model_dir}


def main() -> int:
    args = parse_args()
    try:
        dataset = NormalizedSequenceDataset.from_csv(args.input)
        records = list(dataset.records())
        if args.model == "svm":
            predictions, descriptor_values, scorer, model_files = score_svm(args, records)
            effective = {
                "include_descriptors": args.include_descriptors,
                "null_descriptors": scorer.null_descriptors,
            }
            names = scorer.descriptor_names
        else:
            predictions, descriptor_values, scorer, model_files = score_gnn(args, records)
            effective = {
                "include_descriptors": args.include_descriptors,
                "tabular_columns": scorer.feature_cols,
            }
            names = scorer.feature_cols
        descriptor_names = tuple(names) if args.include_descriptors else ()
        saved_descriptors = descriptor_values if args.include_descriptors else None
        run = execute_prediction_run(
            args,
            dataset,
            records,
            predictions,
            descriptor_names,
            saved_descriptors,
            model_name=args.model,
            model_files=model_files,
            effective=effective,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(predictions)} prediction row(s) to {run.csv_path}")
    print(f"Run manifest: {run.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

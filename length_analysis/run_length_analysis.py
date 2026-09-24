#!/usr/bin/env python3
"""Run length-extension analysis for normalized peptide subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "length_analysis"

from analysis_outputs import RunConfiguration, RunOutput, RunSpec, execute_csv_run

from .data_loader import SubsetSequenceDataset
from .models import GnnScorerFactory, resolve_gnn_checkpoint
from .provenance import recover_parents, load_candidate_index
from .run_options import (
    SUPPORTED_MODELS,
    add_length_arguments,
    add_provenance_arguments,
    length_config_from_args,
    validate_length_config,
)
from .scoring.runner import NullProgressReporter, run_length_scoring


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "sequence_to_svm_minimal" / "checkpoints" / "latest"
OUTPUT_FILENAME = "length_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover parent flanks for subset peptides, enumerate N/C length extensions, "
            "and score variants with a GNN checkpoint."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Normalized subset CSV with required id and sequence columns.",
    )
    add_provenance_arguments(parser)
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="gnn",
        help="Model adapter to use (default: gnn).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory containing GNN .pt checkpoint(s) (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        type=Path,
        help="Explicit GNN .pt path. Defaults to gat_ready_QSAR.pt in --checkpoint-dir when present.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=f"Output CSV name and root (default: <input directory>/results/{OUTPUT_FILENAME}).",
    )
    add_length_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=32, help="GNN batch size (default: 32).")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Device for feature processing / inference (default: auto).",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Rebuild feature artifacts even when they already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser.parse_args()


def output_path_from_args(args: argparse.Namespace) -> Path:
    return args.output or args.input.parent / "results" / OUTPUT_FILENAME


def main() -> int:
    args = parse_args()
    try:
        if args.model != "gnn":
            raise ValueError(f"Unsupported model: {args.model}")
        config = length_config_from_args(args)
        validate_length_config(config)
        checkpoint = resolve_gnn_checkpoint(
            checkpoint=args.gnn_checkpoint,
            checkpoint_dir=args.checkpoint_dir,
        )
        dataset = SubsetSequenceDataset.from_csv(args.input)
        candidates = load_candidate_index(args.candidates_table)
        parents = recover_parents(dataset.records(), candidates, args.proteome_fasta)
        scorer = GnnScorerFactory(
            checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            skip_if_exists=not args.force_process,
            force_process=args.force_process,
        )()
        output_target = output_path_from_args(args)
        progress = NullProgressReporter()
        if not args.quiet:
            print(f"Recovered {len(parents)} parent peptide(s).", flush=True)
            print(f"Using GNN checkpoint: {checkpoint}", flush=True)

        spec = RunSpec(
            output_root=output_target.parent,
            output_filename=output_target.name,
            runner="length_analysis",
            model=args.model,
            config=RunConfiguration(
                output_mode="length_variants",
                arguments=vars(args),
                effective={
                    "length": {
                        "max_flank_fraction": config.max_flank_fraction,
                        "stride": config.stride,
                    },
                    "batch_size": args.batch_size,
                    "device": args.device,
                },
            ),
            inputs={
                "sequences": args.input,
                "candidates_table": args.candidates_table,
                "proteome_fasta": args.proteome_fasta,
            },
            model_files={"gnn_checkpoint": checkpoint},
            repository_root=ROOT,
        )

        def write_csv(run: RunOutput) -> int:
            workspace = run.directory / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            return run_length_scoring(
                parents,
                scorer,
                max_flank_fraction=config.max_flank_fraction,
                stride=config.stride,
                output=run.csv_path,
                workspace=workspace,
                extra_columns=dataset.label_columns,
                progress=progress,
                run_id=run.run_id,
            )

        run = execute_csv_run(spec, write_csv)
        row_count = run.manifest["output"]["row_count"]
        print(f"Saved {row_count} length-variant prediction row(s) to {run.csv_path}")
        print(f"Run manifest: {run.manifest_path}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

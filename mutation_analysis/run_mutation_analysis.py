#!/usr/bin/env python3
"""Run mutation-robustness analysis for normalized peptide subsets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mutation_analysis"

from analysis_outputs import RunConfiguration, RunOutput, RunSpec, execute_csv_run

from .data_loader import SubsetSequenceDataset
from .models.gnn import DEFAULT_CHECKPOINT_DIR, MutationScorer, resolve_parent_structures
from .mutants.generation import DEFAULT_P_LEVELS, DEFAULT_REPLICATES, DEFAULT_SEED
from .mutants.rules import DEFAULT_BLOSUM_TEMPERATURE, SCHEMES
from .scoring.aggregate import rank_by_worst_scheme
from .scoring.runner import run_mutation_scoring


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILENAME = "mutation_robustness.csv"
CURVES_FILENAME = "mutation_curves.csv"
VARIANTS_FILENAME = "mutation_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mutate peptides under BLOSUM62-softmax, uniform and physicochemical-preserving "
            "schemes at k=round(p*L) substitutions, and score each variant with a GNN checkpoint."
        )
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Normalized subset CSV with required id and sequence columns.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
        help=f"GNN checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help=f"Output CSV name and root (default: <input directory>/results/{OUTPUT_FILENAME}).",
    )
    parser.add_argument(
        "--p-levels", type=float, nargs="+", default=list(DEFAULT_P_LEVELS),
        help="Mutation fractions; k = round(p * L) (default: 0.10 ... 0.50).",
    )
    parser.add_argument(
        "--replicates", type=int, default=DEFAULT_REPLICATES,
        help=f"Draws per (peptide, p, scheme) (default: {DEFAULT_REPLICATES}).",
    )
    parser.add_argument(
        "--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES),
        help="Mutation schemes to run (default: all three).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base seed; draws are reproducible per parent/scheme/p/replicate (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--blosum-temperature", type=float, default=DEFAULT_BLOSUM_TEMPERATURE,
        help="Softmax temperature over BLOSUM62 rows (default: 1.0).",
    )
    parser.add_argument(
        "--refold-sample", type=int, default=0,
        help=(
            "Additionally ESMFold this many mutants per (parent, scheme, p) cell and score them "
            "against their own structure, as a control on the fixed-backbone assumption "
            "(default: 0 = off). Costs ~0.65 s per extra fold."
        ),
    )
    parser.add_argument(
        "--parent-structures", type=Path, default=None,
        help=(
            "Directory of <parent id>.pdb structures to reuse instead of folding. "
            "Default: <input dir>/generated/structures when every parent matches it, "
            "otherwise fold into the run workspace."
        ),
    )
    parser.add_argument(
        "--fitting", "-f", choices=["temperature", "platt", "softmax"], default=None,
        help=(
            "P(AMP) mapping. Default: temperature when gnn_temperature_scaling exists, "
            "else platt, else softmax. Only platt shifts the 0.5 crossing off margin=0, "
            "so this changes pred and every retention metric."
        ),
    )
    parser.add_argument(
        "--write-variants", action="store_true",
        help=(
            f"Also write every scored variant to {VARIANTS_FILENAME} (~13.5k rows per peptide); "
            "by default only the per-peptide and per-cell collapses are kept."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Parent peptides to score in parallel processes (default: 1). Each worker "
            "loads its own checkpoint (~2 GB resident), so size this against RAM, not "
            "cores. Output is identical to a serial run. Cannot be used with "
            "--refold-sample."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32, help="GNN batch size (default: 32).")
    parser.add_argument(
        "--device", choices=("cuda", "cpu"), default=None,
        help="Device for folding and inference (default: auto).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser.parse_args()


def output_path_from_args(args: argparse.Namespace) -> Path:
    return args.output or args.input.parent / "results" / OUTPUT_FILENAME


def print_ranking(summaries: list[dict], top: int = 10) -> None:
    """Print retention AUC and applied k for every scheme, ranked by the weakest scheme."""
    ranked = rank_by_worst_scheme(summaries)
    if not ranked:
        print("No AMP-called parents to rank.")
        return
    shown = ranked[:top]
    schemes = sorted({scheme for record in ranked for scheme in record["by_scheme"]})
    id_width = max(len(record["parent_id"]) for record in shown)
    cell_width = max(14, *(len(scheme) for scheme in schemes))

    print(f"\nMost mutation-resistant of {len(ranked)} AMP-called parent(s), "
          "retention AUC per scheme (k = mean applied substitutions):")
    header = f"      {'parent':<{id_width}}  {'L':>3} {'k_req':>5}  "
    header += "  ".join(f"{scheme:<{cell_width}}" for scheme in schemes)
    print(header + "  worst")
    capped = False
    for position, record in enumerate(shown, start=1):
        line = (
            f"  {position:2d}. {record['parent_id']:<{id_width}}  "
            f"{record['length']:>3} {record['mean_k_requested']:>5.2f}  "
        )
        cells = []
        for scheme in schemes:
            entry = record["by_scheme"].get(scheme)
            if entry is None:
                cells.append(f"{'-':<{cell_width}}")
                continue
            marker = "*" if entry["mean_k_applied"] < record["mean_k_requested"] - 1e-9 else " "
            capped = capped or marker == "*"
            cells.append(
                f"{entry['retention_auc']:.3f} k={entry['mean_k_applied']:.2f}{marker}".ljust(cell_width)
            )
        print(line + "  ".join(cells) + f"  {record['worst_scheme']}")
    if capped:
        print("  * k capped below k_req by eligible-position count; "
              "that scheme's score is inflated.")


def main() -> int:
    args = parse_args()
    try:
        for p in args.p_levels:
            if not 0 < p <= 1:
                raise ValueError(f"--p-levels entries must be in (0, 1]; got {p}.")
        if args.replicates < 1:
            raise ValueError("--replicates must be at least 1.")
        if args.workers < 1:
            raise ValueError("--workers must be at least 1.")

        dataset = SubsetSequenceDataset.from_csv(args.input)
        parents = list(dataset.records())
        scorer = MutationScorer(
            args.checkpoint_dir,
            batch_size=args.batch_size,
            device=args.device,
            fitting=args.fitting,
        )
        output_target = output_path_from_args(args)
        print(f"Calibration: {scorer.calibration_description}", flush=True)
        if not args.quiet:
            per_parent = len(args.schemes) * len(args.p_levels) * args.replicates + 1
            print(f"Loaded {len(parents)} parent peptide(s).", flush=True)
            print(f"Up to {per_parent} variant(s) per parent.", flush=True)

        spec = RunSpec(
            output_root=output_target.parent,
            output_filename=output_target.name,
            runner="mutation_analysis",
            model="gnn",
            config=RunConfiguration(
                output_mode="mutation_variants",
                arguments=vars(args),
                effective={
                    "mutation": {
                        "p_levels": args.p_levels,
                        "replicates": args.replicates,
                        "schemes": args.schemes,
                        "seed": args.seed,
                        "blosum_temperature": args.blosum_temperature,
                        "backbone": "fixed",
                        "refold_sample": args.refold_sample,
                    },
                    "write_variants": args.write_variants,
                    "parent_structures": args.parent_structures,
                    "fitting": scorer.fitting,
                    "workers": args.workers,
                    "batch_size": args.batch_size,
                    "device": args.device,
                },
            ),
            inputs={"sequences": args.input},
            model_files={"checkpoint_dir": args.checkpoint_dir},
            repository_root=ROOT,
        )

        summaries: list[dict] = []

        def write_csv(run: RunOutput) -> int:
            workspace = run.directory / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            structures = resolve_parent_structures(
                parents,
                workspace=workspace,
                device=args.device,
                explicit=args.parent_structures,
                input_dir=args.input.parent,
            )
            run.manifest["config"]["effective"]["parent_structures_used"] = str(structures)
            summaries.extend(run_mutation_scoring(
                parents,
                scorer,
                p_levels=args.p_levels,
                replicates=args.replicates,
                schemes=args.schemes,
                seed=args.seed,
                blosum_temperature=args.blosum_temperature,
                output=run.csv_path,
                curves_output=run.directory / CURVES_FILENAME,
                workspace=workspace,
                structures=structures,
                checkpoint_dir=args.checkpoint_dir,
                batch_size=args.batch_size,
                workers=args.workers,
                variants_output=(
                    run.directory / VARIANTS_FILENAME if args.write_variants else None
                ),
                extra_columns=dataset.label_columns,
                run_id=run.run_id,
                device=args.device,
                quiet=args.quiet,
                refold_sample=args.refold_sample,
            ))
            return len(summaries)

        run = execute_csv_run(spec, write_csv)
        row_count = run.manifest["output"]["row_count"]
        print(f"Saved {row_count} per-peptide robustness row(s) to {run.csv_path}")
        print(f"Degradation curves: {run.directory / CURVES_FILENAME}")
        if args.write_variants:
            print(f"Scored variants: {run.directory / VARIANTS_FILENAME}")
        print_ranking(summaries)
        print(f"Run manifest: {run.manifest_path}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

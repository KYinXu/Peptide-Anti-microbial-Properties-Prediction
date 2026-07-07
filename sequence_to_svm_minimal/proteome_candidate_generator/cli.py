"""Command-line orchestration for proteome candidate generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from proteome_candidate_generator.candidates import (
    generate_candidates,
    generate_paper_candidates,
    resolve_candidates_table_path,
    write_candidates_table,
    write_pipeline_txt,
)
from proteome_candidate_generator.cleavage import (
    inspect_pepsickle_schema,
    load_union_from_outputs,
    write_sites_jsonl,
)
from proteome_candidate_generator.fasta import BatchFile, ProteinRecord, preprocess_fasta, read_valid_proteins
from proteome_candidate_generator.pepsickle_runner import build_tasks, run_tasks
from proteome_candidate_generator.pddp_scoring import (
    PaneScorer,
    compute_nonzero_mean_threshold,
    is_mapp_database,
    load_known_amp_sequences,
    load_mapp_reference_scorer,
    load_score_matrix,
)

DEFAULT_INPUT = Path("data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta")
DEFAULT_OUTPUT = Path("data/proteomes")
MODELS = ("constitutive", "immunoproteasome")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AMP-like peptides from proteome pepsickle cleavage predictions.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "preprocess", "run-pepsickle", "build-candidates", "validate"],
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pepsickle-bin", default="pepsickle")
    parser.add_argument("--protocol", choices=["current", "paper_pddp", "mapp_database"], default="current")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-proteins", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--min-charge", type=int, default=None)
    parser.add_argument("--min-hydrophobicity", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--amp-score-matrix", type=Path, default=None)
    parser.add_argument("--mapp-database", type=Path, default=None)
    parser.add_argument("--known-amps", type=Path, nargs="*", default=None)
    parser.add_argument("--amp-score-threshold", type=float, default=None)
    parser.add_argument("--pane-m-exponent", type=float, default=1.0)
    parser.add_argument("--pane-n-exponent", type=float, default=1.0)
    parser.add_argument("--require-cationic-cterm", action="store_true")
    parser.add_argument("--cationic-cterm-residues", default="KRH")
    parser.add_argument(
        "--overlap-policy",
        choices=["top_score", "longest", "keep_all"],
        default="top_score",
        help="How paper mode handles overlapping peptides from the same source protein.",
    )
    parser.add_argument(
        "--no-terminal-boundaries",
        action="store_true",
        help="Do not add protein N/C termini as candidate fragment boundaries.",
    )
    parser.add_argument("--output-format", choices=["auto", "csv", "parquet"], default="auto")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars/status messages for long-running stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys
    from configs.load_config import argv_without_config_flags
    from proteome_candidate_generator.config import parser_defaults

    if argv is None:
        argv = sys.argv[1:]
    
    cfg_path, remaining_argv = argv_without_config_flags(argv)
    parser = build_parser()
    
    if cfg_path:
        defaults = parser_defaults([cfg_path])
        parser.set_defaults(**defaults)
        
    args = parser.parse_args(remaining_argv)
    try:
        _apply_protocol_defaults(args)
        _validate_args(args)
        return _dispatch(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _apply_protocol_defaults(args: argparse.Namespace) -> None:
    if args.protocol in ("paper_pddp", "mapp_database"):
        args.min_len = 10 if args.min_len is None else args.min_len
        args.max_len = 50 if args.max_len is None else args.max_len
        args.min_charge = 0 if args.min_charge is None else args.min_charge
        args.min_hydrophobicity = 0.0 if args.min_hydrophobicity is None else args.min_hydrophobicity
        return
    args.min_len = 8 if args.min_len is None else args.min_len
    args.max_len = 30 if args.max_len is None else args.max_len
    args.min_charge = 2 if args.min_charge is None else args.min_charge
    args.min_hydrophobicity = 0.30 if args.min_hydrophobicity is None else args.min_hydrophobicity
    args.top_n = 400000 if args.top_n is None else args.top_n


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "preprocess":
        _run_preprocess(args)
    elif args.command == "run-pepsickle":
        _run_pepsickle(args)
    elif args.command == "build-candidates":
        _run_build_candidates(args)
    elif args.command == "validate":
        _run_validate(args)
    else:
        _run_preprocess(args)
        _run_pepsickle(args)
        _run_build_candidates(args)
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.threshold < 0 or args.threshold > 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size <= 0 or args.workers <= 0:
        raise ValueError("--batch-size and --workers must be positive")
    if args.min_len <= 0 or args.max_len < args.min_len:
        raise ValueError("--min-len must be positive and <= --max-len")
    if args.min_hydrophobicity < 0 or args.min_hydrophobicity > 1:
        raise ValueError("--min-hydrophobicity must be between 0 and 1")
    if args.top_n is not None and args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.protocol == "mapp_database":
        if args.mapp_database is None:
            raise ValueError("--protocol mapp_database requires --mapp-database")
    if args.protocol == "paper_pddp":
        pass  # Pane scorer doesn't require extra files unless we want to compute threshold from known AMPs
        if args.amp_score_matrix is not None:
            raise ValueError("--protocol paper_pddp uses Pane scorer, do not provide --amp-score-matrix")
        if not args.cationic_cterm_residues:
            raise ValueError("--cationic-cterm-residues cannot be empty")
    if str(args.pepsickle_bin).startswith("/path/to/"):
        raise ValueError(
            "--pepsickle-bin is still a placeholder. Activate the pepsickle "
            "environment or pass the real path to its pepsickle executable."
        )


def _layout(output_dir: Path) -> dict[str, Path]:
    root = output_dir.resolve() / "generated"
    return {
        "root": root,
        "batches": root / "inputs" / "batches",
        "pepsickle": root / "pepsickle",
        "sites": root / "cleavage_sites.jsonl",
        "final_table": root / "final_candidates",
        "final_txt": root / "final_candidates.txt",
        "manifest": root / "run_manifest.json",
    }


def _run_preprocess(args: argparse.Namespace) -> tuple[list[ProteinRecord], list[BatchFile]]:
    paths = _layout(args.output_dir)
    result = preprocess_fasta(
        args.input,
        paths["batches"],
        batch_size=args.batch_size,
        limit=args.limit_proteins,
        show_progress=not args.no_progress,
    )
    _update_manifest(paths["manifest"], {"preprocessing": result.stats, "input": str(args.input)})
    print(f"Wrote {len(result.batches)} batch FASTA files under {paths['batches']}")
    return result.records, result.batches


def _run_pepsickle(args: argparse.Namespace) -> None:
    paths = _layout(args.output_dir)
    batches = _discover_batches(paths["batches"])
    if not batches:
        raise FileNotFoundError(f"No batch FASTA files found under {paths['batches']}")
    tasks = build_tasks(batches, paths["pepsickle"], pepsickle_bin=args.pepsickle_bin, threshold=args.threshold)
    print(f"Running {len(tasks)} pepsickle task(s) with {args.workers} worker(s).", flush=True)
    results = run_tasks(
        tasks,
        workers=args.workers,
        resume=args.resume,
        force=args.force,
        show_progress=not args.no_progress,
    )
    failures = [result for result in results if result.returncode != 0]
    _update_manifest(paths["manifest"], {"pepsickle": [_task_result_dict(result) for result in results]})
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"{len(failures)} pepsickle task(s) failed; first failure was "
            f"{first.status} for batch {first.task.batch.index} "
            f"({first.task.model_name}). See {paths['manifest']} for details."
        )
    print(f"Pepsickle outputs ready: {len(results)} files")


def _run_build_candidates(args: argparse.Namespace) -> None:
    paths = _layout(args.output_dir)
    records, stats = read_valid_proteins(
        args.input,
        limit=args.limit_proteins,
        show_progress=not args.no_progress,
    )
    batches = _discover_batches(paths["batches"])
    if not batches:
        raise FileNotFoundError(
            f"No batch FASTA files found under {paths['batches']}. "
            "Run the `preprocess` step first, or run the full `all` command."
        )
    output_paths = _pepsickle_output_paths(batches, paths["pepsickle"])
    if not output_paths:
        raise FileNotFoundError(
            f"No expected pepsickle outputs could be derived from batches under {paths['batches']}."
        )
    missing = [path for path, _ in output_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing pepsickle outputs under {paths['pepsickle']}; first missing file: {missing[0]}. "
            "Run `run-pepsickle`, or run the full `all` command."
        )
    lengths = {record.protein_id: len(record.sequence) for record in records}
    
    # Check if we can reuse the existing cleavage_sites.jsonl
    if paths["sites"].exists():
        print(f"Loading existing cleavage sites from {paths['sites']}")
        import json
        from proteome_candidate_generator.cleavage import ProteinCleavageSites
        sites = {}
        with paths["sites"].open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Apply the threshold filter when loading from cache!
                filtered_probs = {
                    int(k): float(v) 
                    for k, v in data["site_probabilities"].items() 
                    if float(v) > args.threshold  # The paper says "threshold of 0.5"
                }
                sites[data["protein_id"]] = ProteinCleavageSites(
                    protein_id=data["protein_id"],
                    length=data["length"],
                    site_probabilities=filtered_probs
                )
    else:
        print(f"Parsing raw Pepsickle TSVs to build cleavage sites...")
        sites = load_union_from_outputs(
            output_paths,
            lengths=lengths,
            threshold=args.threshold,
            show_progress=not args.no_progress,
        )
        write_sites_jsonl(sites, paths["sites"])
    scoring_metadata: dict[str, object] = {"protocol": args.protocol}
    if args.protocol in ("paper_pddp", "mapp_database"):
        scorer, threshold, threshold_source, known_amp_count = _build_paper_scorer(args)
        candidates, candidate_stats = generate_paper_candidates(
            records,
            sites,
            min_len=args.min_len,
            max_len=args.max_len,
            scorer=scorer,
            score_threshold=threshold,
            require_cationic_cterm=args.require_cationic_cterm,
            cationic_cterm_residues=args.cationic_cterm_residues,
            overlap_policy=args.overlap_policy,
            include_terminal_boundaries=not args.no_terminal_boundaries,
            show_progress=not args.no_progress,
            finalize_outputs=(paths["final_table"], paths["final_txt"], args.output_format),
        )
        scoring_metadata.update(
            {
                "amp_score_matrix": str(args.amp_score_matrix) if args.amp_score_matrix else None,
                "mapp_database": str(args.mapp_database) if args.mapp_database else None,
                "pane_m_exponent": args.pane_m_exponent,
                "pane_n_exponent": args.pane_n_exponent,
                "known_amps": [str(path) for path in (args.known_amps or [])],
                "known_amp_count": known_amp_count,
                "amp_score_threshold": threshold,
                "amp_score_threshold_source": threshold_source,
                "require_cationic_cterm": args.require_cationic_cterm,
                "cationic_cterm_residues": args.cationic_cterm_residues,
                "overlap_policy": args.overlap_policy,
            }
        )
    else:
        candidates, candidate_stats = generate_candidates(
            records,
            sites,
            min_len=args.min_len,
            max_len=args.max_len,
            min_charge=args.min_charge,
            min_hydrophobicity=args.min_hydrophobicity,
            top_n=args.top_n,
            include_terminal_boundaries=not args.no_terminal_boundaries,
            show_progress=not args.no_progress,
        )
    if candidates:
        table_path = write_candidates_table(candidates, paths["final_table"], output_format=args.output_format)
        write_pipeline_txt(candidates, paths["final_txt"])
    else:
        table_path = resolve_candidates_table_path(paths["final_table"], args.output_format)
    _update_manifest(
        paths["manifest"],
        {
            "input_stats": stats,
            "candidate_generation": asdict(candidate_stats),
            "candidate_protocol": scoring_metadata,
            "outputs": {
                "cleavage_sites": str(paths["sites"]),
                "final_table": str(table_path),
                "final_pipeline_txt": str(paths["final_txt"]),
            },
        },
    )
    print(f"Wrote {candidate_stats.retained} final candidates to {table_path}")


def _build_paper_scorer(args: argparse.Namespace):
    if args.protocol == "mapp_database":
        scorer = load_mapp_reference_scorer(args.mapp_database)
        threshold = 0.0 if args.amp_score_threshold is None else args.amp_score_threshold
        return scorer, threshold, "mapp_treatment_total", 0

    if args.protocol == "paper_pddp":
        scorer = PaneScorer(m=args.pane_m_exponent, n=args.pane_n_exponent)
        threshold = args.amp_score_threshold
        threshold_source = "cli"
        known_amp_count = 0
        if threshold is None:
            if args.known_amps:
                known_amp_sequences = load_known_amp_sequences(args.known_amps)
                known_amp_count = len(known_amp_sequences)
                threshold = compute_nonzero_mean_threshold(known_amp_sequences, scorer)
                threshold_source = "known_amps_nonzero_mean"
            else:
                threshold = 5.0
                threshold_source = "default"
        return scorer, threshold, threshold_source, known_amp_count

    raise ValueError(f"Unknown protocol: {args.protocol}")


def _run_validate(args: argparse.Namespace) -> None:
    args.limit_proteins = args.limit_proteins or 1
    _run_preprocess(args)
    _run_pepsickle(args)
    paths = _layout(args.output_dir)
    outputs = _pepsickle_output_paths(_discover_batches(paths["batches"]), paths["pepsickle"])
    if not outputs:
        raise RuntimeError("No pepsickle outputs were produced")
    schema = inspect_pepsickle_schema(outputs[0][0])
    _update_manifest(paths["manifest"], {"pepsickle_schema": schema})
    print(f"Validated pepsickle TSV schema: {schema}")


def _discover_batches(batches_dir: Path) -> list[BatchFile]:
    batches: list[BatchFile] = []
    for path in sorted(batches_dir.glob("batch_*.fasta")):
        index = int(path.stem.split("_")[-1])
        n_records = _count_fasta_records(path)
        batches.append(BatchFile(index=index, path=path, n_records=n_records))
    return batches


def _count_fasta_records(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def _pepsickle_output_paths(batches: list[BatchFile], pepsickle_dir: Path) -> list[tuple[Path, str]]:
    return [
        (pepsickle_dir / f"{batch.path.stem}.{model}.tsv", model)
        for batch in batches
        for model in MODELS
    ]


def _task_result_dict(result) -> dict[str, object]:
    return {
        "batch": result.task.batch.index,
        "model": result.task.model_name,
        "output": str(result.task.output_path),
        "status": result.status,
        "returncode": result.returncode,
        "command": result.task.command,
    }


def _update_manifest(path: Path, updates: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

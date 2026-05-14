"""Command-line orchestration for proteome candidate generation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from proteome_candidate_generator.candidates import (
    generate_candidates,
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

DEFAULT_INPUT = Path("data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta")
DEFAULT_OUTPUT = Path("data/proteomes/generated")
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-proteins", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--min-hydrophobicity", type=float, default=0.30)
    parser.add_argument("--top-n", type=int, default=400000)
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
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        return _dispatch(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


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
    if str(args.pepsickle_bin).startswith("/path/to/"):
        raise ValueError(
            "--pepsickle-bin is still a placeholder. Activate the pepsickle "
            "environment or pass the real path to its pepsickle executable."
        )


def _layout(output_dir: Path) -> dict[str, Path]:
    root = output_dir.resolve()
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
    output_paths = _pepsickle_output_paths(_discover_batches(paths["batches"]), paths["pepsickle"])
    missing = [path for path, _ in output_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pepsickle outputs; first missing file: {missing[0]}")
    lengths = {record.protein_id: len(record.sequence) for record in records}
    sites = load_union_from_outputs(
        output_paths,
        lengths=lengths,
        threshold=args.threshold,
        show_progress=not args.no_progress,
    )
    write_sites_jsonl(sites, paths["sites"])
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
    table_path = write_candidates_table(candidates, paths["final_table"], output_format=args.output_format)
    write_pipeline_txt(candidates, paths["final_txt"])
    _update_manifest(
        paths["manifest"],
        {
            "input_stats": stats,
            "candidate_generation": asdict(candidate_stats),
            "outputs": {
                "cleavage_sites": str(paths["sites"]),
                "final_table": str(table_path),
                "final_pipeline_txt": str(paths["final_txt"]),
            },
        },
    )
    print(f"Wrote {len(candidates)} final candidates to {table_path}")


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

"""Pepsickle subprocess command construction and execution."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from proteome_candidate_generator.fasta import BatchFile
from proteome_candidate_generator.progress import progress_iter

PROTEASOME_MODELS = {
    "constitutive": ("in-vitro", "C"),
    "immunoproteasome": ("in-vitro", "I"),
}


@dataclass(frozen=True)
class PepsickleTask:
    batch: BatchFile
    model_name: str
    output_path: Path
    command: list[str]


@dataclass(frozen=True)
class PepsickleTaskResult:
    task: PepsickleTask
    status: str
    returncode: int


def build_task(
    batch: BatchFile,
    output_dir: Path,
    *,
    model_name: str,
    pepsickle_bin: str,
) -> PepsickleTask:
    if model_name not in PROTEASOME_MODELS:
        raise ValueError(f"Unknown pepsickle model: {model_name}")
    model_type, proteasome_type = PROTEASOME_MODELS[model_name]
    output_path = output_dir / f"{batch.path.stem}.{model_name}.tsv"
    command = [
        pepsickle_bin,
        "-f",
        str(batch.path),
        "-m",
        model_type,
        "-p",
        proteasome_type,
        "-o",
        str(output_path),
    ]
    return PepsickleTask(batch=batch, model_name=model_name, output_path=output_path, command=command)


def build_tasks(
    batches: list[BatchFile],
    output_dir: Path,
    *,
    pepsickle_bin: str,
    threshold: float,
    model_names: tuple[str, ...] = ("constitutive", "immunoproteasome"),
) -> list[PepsickleTask]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        build_task(
            batch,
            output_dir,
            model_name=model,
            pepsickle_bin=pepsickle_bin,
        )
        for batch in batches
        for model in model_names
    ]


def run_task(task: PepsickleTask, *, resume: bool, force: bool) -> PepsickleTaskResult:
    if task.output_path.exists() and resume and not force:
        return PepsickleTaskResult(task=task, status="skipped_existing", returncode=0)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(task.command, check=False)
    except FileNotFoundError:
        return PepsickleTaskResult(task=task, status="missing_executable", returncode=127)
    status = "completed" if completed.returncode == 0 else "failed"
    return PepsickleTaskResult(task=task, status=status, returncode=completed.returncode)


def run_tasks(
    tasks: list[PepsickleTask],
    *,
    workers: int,
    resume: bool,
    force: bool,
    show_progress: bool = False,
) -> list[PepsickleTaskResult]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1:
        iterator = progress_iter(tasks, desc="Running pepsickle", total=len(tasks)) if show_progress else tasks
        return [run_task(task, resume=resume, force=force) for task in iterator]
    results: list[PepsickleTaskResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task, resume=resume, force=force) for task in tasks]
        completed = as_completed(futures)
        if show_progress:
            completed = progress_iter(completed, desc="Running pepsickle", total=len(futures))
        for future in completed:
            results.append(future.result())
    return sorted(results, key=lambda r: (r.task.batch.index, r.task.model_name))

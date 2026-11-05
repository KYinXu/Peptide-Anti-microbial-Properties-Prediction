"""Generate mutants per parent, score them against a fixed or refolded backbone.

Parents are independent: a mutant's RNG is seeded from its parent id, scheme, p and
replicate, so a worker handed one parent reproduces exactly the mutants a serial run
would give it. That makes sharding by parent safe, and ``Pool.imap`` preserves input
order, so parallel output is row-for-row identical to serial output.
"""

from __future__ import annotations

import io
import os
import multiprocessing
import random
from collections import defaultdict
from contextlib import ExitStack, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..models.gnn import MutationScorer, fold_parents
from ..mutants.common import MutantRecord, SubsetRecord
from ..mutants.generation import generate_mutants
from .aggregate import summarize_parent
from .progress import build_mutation_progress
from .results_io import (
    MutationCurveCsvWriter,
    MutationPredictionCsvWriter,
    MutationRobustnessCsvWriter,
)


@dataclass(frozen=True)
class ScoringConfig:
    """Per-run settings shipped to each worker; every field must be picklable."""

    p_levels: tuple[float, ...]
    replicates: int
    schemes: tuple[str, ...]
    seed: int
    blosum_temperature: float
    structures: Path
    workspace: Path
    device: str | None
    refold_sample: int
    want_variants: bool
    checkpoint_dir: Path
    batch_size: int
    fitting: str | None


def parent_rows(
    parent: SubsetRecord, scorer: MutationScorer, config: ScoringConfig, progress=None
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Score every mutant of one parent: (summary, curve, variant rows, count)."""
    records = list(
        generate_mutants(
            [parent],
            p_levels=config.p_levels,
            replicates=config.replicates,
            schemes=config.schemes,
            seed=config.seed,
            blosum_temperature=config.blosum_temperature,
        )
    )
    panel = _refold_panel(records, config.refold_sample, config.seed) if config.refold_sample else []
    total_units = len(records) + len(panel)

    def stage(offset: int, label: str = ""):
        if progress is None:
            return None
        return lambda phase, done, _total: progress(f"{label}{phase}", offset + done, total_units)

    backbone = scorer.load_backbone(_parent_pdb(config.structures, parent.id))
    predictions = scorer.score_on_backbone(
        [record.mutant_id for record in records],
        [record.sequence for record in records],
        backbone,
        progress=stage(0),
    )
    reference = next(
        (p for record, p in zip(records, predictions) if record.scheme == "parent"), None
    )

    summary, curves = summarize_parent(records, predictions, reference, "fixed")
    variants = (
        [_row(r, p, reference, "fixed") for r, p in zip(records, predictions)]
        if config.want_variants
        else []
    )
    counted = len(records)

    if panel:
        refold_structures = fold_parents(
            pd.DataFrame(
                {"id": [r.mutant_id for r in panel], "sequence": [r.sequence for r in panel]}
            ),
            config.workspace / "refold" / parent.id,
            config.device,
        )
        refold_predictions = scorer.score(
            [r.mutant_id for r in panel],
            [r.sequence for r in panel],
            refold_structures,
            progress=stage(len(records), "refold "),
        )
        refold_summary, refold_curves = summarize_parent(
            panel, refold_predictions, reference, "refold"
        )
        summary += refold_summary
        curves += refold_curves
        if config.want_variants:
            variants += [
                _row(r, p, reference, "refold") for r, p in zip(panel, refold_predictions)
            ]
        counted += len(panel)

    return summary, curves, variants, counted


_WORKER: dict = {}


def _init_worker(config: ScoringConfig) -> None:
    """Load the checkpoint once per worker process, not once per parent.

    Torch defaults to one CPU thread per core, so N workers would each spawn a full
    thread pool and oversubscribe the machine several times over. Pinning each worker
    to one thread is what makes the process-level speedup real; set before torch is
    first imported, since OpenMP reads the environment at init.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with redirect_stdout(io.StringIO()):
        _WORKER["scorer"] = MutationScorer(
            config.checkpoint_dir,
            batch_size=config.batch_size,
            device=config.device,
            fitting=config.fitting,
        )
    import torch

    torch.set_num_threads(1)
    _WORKER["config"] = config


def _worker_task(parent: SubsetRecord):
    """Worker stdout is dropped: compute_qsar12 prints per 100 rows and would interleave."""
    with redirect_stdout(io.StringIO()):
        return parent_rows(parent, _WORKER["scorer"], _WORKER["config"])


def run_mutation_scoring(
    parents: Iterable[SubsetRecord],
    scorer: MutationScorer,
    *,
    p_levels: Iterable[float],
    replicates: int,
    schemes: Iterable[str],
    seed: int,
    blosum_temperature: float,
    output: Path,
    curves_output: Path,
    workspace: Path,
    structures: Path,
    checkpoint_dir: Path,
    batch_size: int,
    variants_output: Path | None = None,
    extra_columns: Iterable[str] = (),
    run_id: str | None = None,
    device: str | None = None,
    quiet: bool = False,
    refold_sample: int = 0,
    workers: int = 1,
) -> list[dict]:
    parent_list = list(parents)
    if workers > 1 and refold_sample:
        raise ValueError(
            "--workers above 1 cannot be combined with --refold-sample: refolding loads "
            "ESMFold in every worker (~10 GB each). Run the refold panel as its own "
            "single-worker run."
        )

    config = ScoringConfig(
        p_levels=tuple(p_levels),
        replicates=replicates,
        schemes=tuple(schemes),
        seed=seed,
        blosum_temperature=blosum_temperature,
        structures=structures,
        workspace=workspace,
        device=device,
        refold_sample=refold_sample,
        want_variants=variants_output is not None,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        fitting=scorer.fitting,
    )

    summaries: list[dict] = []
    with ExitStack() as stack:
        writer = stack.enter_context(
            MutationRobustnessCsvWriter(output, extra_columns=extra_columns, run_id=run_id)
        )
        curve_writer = stack.enter_context(MutationCurveCsvWriter(curves_output, run_id=run_id))
        variant_writer = (
            stack.enter_context(
                MutationPredictionCsvWriter(
                    variants_output, extra_columns=extra_columns, run_id=run_id
                )
            )
            if variants_output is not None
            else None
        )

        reporter = build_mutation_progress(
            quiet=quiet, total_parents=len(parent_list), per_parent=workers == 1
        )

        def emit(result, parent_id: str) -> None:
            summary, curves, variants, counted = result
            if variant_writer is not None:
                variant_writer.write_rows(variants)
            writer.write_rows(summary)
            curve_writer.write_rows(curves)
            summaries.extend(summary)
            reporter.finish_parent(parent_id, counted)

        try:
            if workers > 1:
                context = multiprocessing.get_context("spawn")
                with context.Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
                    for parent, result in zip(
                        parent_list, pool.imap(_worker_task, parent_list, chunksize=1)
                    ):
                        emit(result, parent.id)
            else:
                for parent in parent_list:
                    report = reporter.start_parent(parent.id)
                    # compute_qsar12 prints per 100 rows and would break up the bars.
                    quieted = redirect_stdout(io.StringIO()) if report else nullcontext()
                    with quieted:
                        result = parent_rows(parent, scorer, config, progress=report)
                    emit(result, parent.id)
        finally:
            reporter.close()

    return summaries


def _refold_panel(batch: list[MutantRecord], per_cell: int, seed: int) -> list[MutantRecord]:
    """Sample up to per_cell mutants from each (scheme, p) cell, excluding the parent."""
    cells: dict[tuple[str, float], list[MutantRecord]] = defaultdict(list)
    for record in batch:
        if record.scheme != "parent":
            cells[(record.scheme, record.p)].append(record)
    panel: list[MutantRecord] = []
    for key in sorted(cells):
        members = cells[key]
        rng = random.Random(f"{seed}|refold|{key[0]}|{round(key[1] * 100)}")
        panel.extend(rng.sample(members, min(per_cell, len(members))))
    return panel


def _parent_pdb(structures: Path, parent_id: str) -> Path:
    path = structures / f"{parent_id}.pdb"
    if not path.is_file():
        raise FileNotFoundError(f"No parent structure for {parent_id!r}: {path}")
    return path


def _row(record: MutantRecord, prediction, reference, backbone: str) -> dict:
    row = {
        "mutant_id": record.mutant_id,
        "parent_id": record.parent_id,
        "scheme": record.scheme,
        "p": record.p,
        "k_requested": record.k_requested,
        "k_applied": record.k_applied,
        "replicate": record.replicate,
        "mutations": record.mutations,
        "sequence": record.sequence,
        "length": record.length,
        "backbone": backbone,
        "pred": prediction.pred,
        "prob_AMP": prediction.prob_AMP,
        "logit_AMP": prediction.logit_AMP,
        "logit_nonAMP": prediction.logit_nonAMP,
        "logit_margin": prediction.logit_margin,
        "delta_prob_AMP": (
            prediction.prob_AMP - reference.prob_AMP if reference is not None else ""
        ),
        "delta_logit_margin": (
            prediction.logit_margin - reference.logit_margin if reference is not None else ""
        ),
    }
    row.update(record.extras)
    return row

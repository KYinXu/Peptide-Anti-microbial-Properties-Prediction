"""Minimal peptide_id + sequence table for compare_model_predictions alignment."""

from __future__ import annotations

import csv
from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.constants import CANONICAL_WINDOWS_SIDECAR
from peptide_pipeline.context import RunContext
from peptide_pipeline.sequence_io import read_sequence_records


def inference_sequences_path(ctx: RunContext, cfg: RunConfig) -> Path:
    """Sequences scored at inference time (windows or canonical parents)."""
    win_side = ctx.inputs_dir / CANONICAL_WINDOWS_SIDECAR
    if cfg.uses_windowing() and not cfg.window_expand_canonical:
        return win_side
    return ctx.canonical


def write_inference_samples_csv(path: Path, records: list[tuple[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["peptide_id", "sequence"])
        writer.writeheader()
        for pid, seq in records:
            writer.writerow({"peptide_id": pid, "sequence": seq})


def build_inference_samples(ctx: RunContext, cfg: RunConfig) -> Path:
    """Write geometric_features.csv with peptide_id + sequence only (no PDB geometry)."""
    seq_path = inference_sequences_path(ctx, cfg)
    records = [(pid, seq) for pid, seq in read_sequence_records(seq_path) if seq]
    if not records:
        raise ValueError(f"No sequences for inference table: {seq_path}")
    write_inference_samples_csv(ctx.geo_csv, records)
    return ctx.geo_csv

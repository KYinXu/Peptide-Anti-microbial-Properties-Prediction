"""Importable data pipeline (config, context, runner). See DATA_PROCESSING.md."""

from __future__ import annotations

from peptide_pipeline.config import RunConfig
from peptide_pipeline.constants import REPO_ROOT
from peptide_pipeline.manifest_paths import (
    gnn_final_training_paths_from_work_dir,
    gnn_legacy_training_paths_from_work_dir,
    load_pipeline_manifest,
    resolve_generated_workspace,
)
from peptide_pipeline.runner import run_pipeline

__all__ = [
    "REPO_ROOT",
    "RunConfig",
    "gnn_final_training_paths_from_work_dir",
    "gnn_legacy_training_paths_from_work_dir",
    "load_pipeline_manifest",
    "resolve_generated_workspace",
    "run_pipeline",
]

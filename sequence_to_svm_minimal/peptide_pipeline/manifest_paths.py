"""Resolve training data paths from a pipeline workspace (pipeline_manifest.json)."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "pipeline_manifest.json"


def load_pipeline_manifest(work_dir: Path) -> dict:
    work_dir = Path(work_dir).resolve()
    path = work_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline manifest not found: {path}. "
            "Use the pipeline workspace directory (same as --work-dir from run_data_pipeline)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def gnn_final_training_paths_from_work_dir(work_dir: Path) -> dict[str, str]:
    m = load_pipeline_manifest(work_dir)
    keys = ("geometric_features", "structures_dir", "qsar12_descriptors", "esm2_embeddings")
    missing = [k for k in keys if not m.get(k)]
    if missing:
        raise KeyError(
            f"Manifest {work_dir / MANIFEST_NAME} missing or empty keys: {missing}. "
            "Run the full pipeline without --skip-qsar / --skip-esm2 for final GNN training."
        )
    return {
        "csv_path": str(Path(m["geometric_features"]).resolve()),
        "pdb_dir": str(Path(m["structures_dir"]).resolve()),
        "qsar_csv": str(Path(m["qsar12_descriptors"]).resolve()),
        "esm2_csv": str(Path(m["esm2_embeddings"]).resolve()),
    }


def gnn_legacy_training_paths_from_work_dir(work_dir: Path) -> dict[str, str]:
    m = load_pipeline_manifest(work_dir)
    keys = ("geometric_features", "structures_dir")
    missing = [k for k in keys if not m.get(k)]
    if missing:
        raise KeyError(
            f"Manifest {work_dir / MANIFEST_NAME} missing or empty keys: {missing}."
        )
    return {
        "csv_path": str(Path(m["geometric_features"]).resolve()),
        "pdb_dir": str(Path(m["structures_dir"]).resolve()),
    }

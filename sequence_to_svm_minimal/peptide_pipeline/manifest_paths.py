"""Resolve training data paths from a pipeline workspace (pipeline_manifest.json)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MANIFEST_NAME = "pipeline_manifest.json"


def normalize_manifest_path(p: str | Path) -> Path:
    """
    Map paths stored in pipeline_manifest.json across WSL ↔ Windows.

    Manifests often record Linux paths like ``/mnt/c/Users/...``. On native Windows,
    ``Path(...).resolve()`` can turn those into invalid ``C:\\mnt\\c\\...``. This
    normalizes first, then callers typically call ``.resolve()``.
    """
    s = str(p).strip()
    if sys.platform.startswith("win"):
        if s.startswith("/mnt/") and len(s) >= 7 and s[5].isalpha() and s[6:7] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
        m = re.match(r"^([A-Za-z]):\\mnt\\([a-z])\\(.*)$", s)
        if m:
            return Path(f"{m.group(1).upper()}:\\{m.group(3)}")
        return Path(s)
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(s)


def _resolve_manifest_path_str(raw: str) -> str:
    return str(normalize_manifest_path(raw).expanduser().resolve())


def resolve_generated_workspace(path: Path | str) -> Path:
    """
    Resolve the pipeline workspace directory that contains ``pipeline_manifest.json``.

    Accepts either the ``generated/`` folder itself, or a parent directory that
    contains ``generated/`` (as produced by ``run_data_pipeline`` defaults).
    """
    p = Path(path).expanduser().resolve()
    if (p / MANIFEST_NAME).is_file():
        return p
    nested = p / "generated"
    if (nested / MANIFEST_NAME).is_file():
        return nested
    raise FileNotFoundError(
        f"Could not find {MANIFEST_NAME} in {p} or {nested}. "
        "Pass the pipeline generated/ directory (or its parent containing generated/)."
    )


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
    esm2_csv = Path(_resolve_manifest_path_str(str(m["esm2_embeddings"])))
    if m.get("esm2_per_residue"):
        esm2_res_dir = _resolve_manifest_path_str(str(m["esm2_per_residue"]))
    else:
        esm2_res_dir = str((esm2_csv.parent / "esm2_per_residue").resolve())
    return {
        "csv_path": _resolve_manifest_path_str(str(m["geometric_features"])),
        "pdb_dir": _resolve_manifest_path_str(str(m["structures_dir"])),
        "qsar_csv": _resolve_manifest_path_str(str(m["qsar12_descriptors"])),
        "esm2_csv": str(esm2_csv),
        "esm2_residue_dir": esm2_res_dir,
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
        "csv_path": _resolve_manifest_path_str(str(m["geometric_features"])),
        "pdb_dir": _resolve_manifest_path_str(str(m["structures_dir"])),
    }

"""Load JSON presets under ``configs/``. Path values are relative to the package root (``sequence_to_svm_minimal/``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def argv_without_config_flags(argv: list[str] | None) -> tuple[str | None, list[str]]:
    """
    Return ``(first_config_path_or_none, argv_without_any_--config_flags)``.
    First ``--config`` / ``-c`` wins; later duplicates are ignored (removed).
    """
    if argv is None:
        argv = []
    cfg: str | None = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--config", "-c"):
            if i + 1 >= len(argv):
                print(f"{argv[i]} requires a path", file=sys.stderr)
                raise SystemExit(2)
            if cfg is None:
                cfg = argv[i + 1]
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return cfg, out


def repo_root() -> Path:
    """Root of the sequence_to_svm_minimal tree (parent of ``configs/``)."""
    return Path(__file__).resolve().parent.parent


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {p}, got {type(data).__name__}")
    return data


def merge_dicts(*parts: dict) -> dict:
    out: dict = {}
    for d in parts:
        out.update(d)
    return out


def normalize_pipeline_arg_keys(raw: dict) -> dict:
    out = dict(raw)
    if "input_path" in out:
        out["input"] = out.pop("input_path")
    return {k: v for k, v in out.items() if not str(k).startswith("_")}


def merge_pipeline_config_paths(paths: list[str | Path] | None) -> dict:
    if not paths:
        return {}
    merged: dict = {}
    for p in paths:
        merged.update(normalize_pipeline_arg_keys(load_json(p)))
    return merged


def resolve_path_str(rel_or_abs: str, root: Path | None = None) -> str:
    root = root or repo_root()
    p = Path(rel_or_abs)
    if p.is_absolute():
        return str(p.resolve())
    return str((root / p).resolve())


def resolve_path_keys(d: dict, keys: frozenset[str]) -> dict:
    out = dict(d)
    root = repo_root()
    for k in keys:
        if k not in out or out[k] is None:
            continue
        v = out[k]
        if isinstance(v, str):
            out[k] = resolve_path_str(v, root)
    return out


GNN_FINAL_PATH_KEYS = frozenset(
    {
        "csv_path",
        "pdb_dir",
        "qsar_csv",
        "esm2_csv",
        "esm2_amp_csv",
        "esm2_decoy_csv",
    }
)

GNN_COMPARISON_PATH_KEYS = frozenset({"csv_path", "pdb_dir", "qsar_csv"})

COMPARE_MODELS_PATH_KEYS = frozenset(
    {
        "geo_csv",
        "pdb_dir",
        "qsar_csv",
        "geometric_qsar_combined_csv",
        "svm_descriptor_csv",
        "svm_z_file",
        "svm_pkl",
        "esm_only_pt",
        "esm_geo_pt",
        "esm_qsar_pt",
        "esm_combined_pt",
        "output_csv",
    }
)


def load_gnn_final_train_bundle(
    config_path: str | None = None,
) -> tuple[dict, dict[str, dict], list[str], dict | None]:
    path = config_path or str(repo_root() / "configs/gnn_final_train.json")
    raw = load_json(path)
    data = dict(raw)
    data.pop("_documentation", None)
    architectures = list(data.pop("architectures", ["gcn", "gat", "egnn"]))
    feature_sets = dict(data.pop("feature_sets", {}))
    node_feature_groups = data.pop("node_feature_groups", None)
    training = resolve_path_keys(data, GNN_FINAL_PATH_KEYS)
    return training, feature_sets, architectures, node_feature_groups


def load_gnn_comparison_bundle(
    config_path: str | None = None,
) -> tuple[dict, dict[str, dict], list[str], dict | None]:
    path = config_path or str(repo_root() / "configs/gnn_comparison.json")
    raw = load_json(path)
    data = dict(raw)
    data.pop("_documentation", None)
    architectures = list(data.pop("architectures", ["gcn", "gat", "egnn"]))
    feature_sets = dict(data.pop("feature_sets", {}))
    node_feature_groups = data.pop("node_feature_groups", None)
    cfg = resolve_path_keys(data, GNN_COMPARISON_PATH_KEYS)
    return cfg, feature_sets, architectures, node_feature_groups


def load_compare_models_config(config_path: str | None = None) -> dict:
    path = config_path or str(repo_root() / "configs/compare_models.json")
    raw = load_json(path)
    data = dict(raw)
    data.pop("_documentation", None)
    return resolve_path_keys(data, COMPARE_MODELS_PATH_KEYS)

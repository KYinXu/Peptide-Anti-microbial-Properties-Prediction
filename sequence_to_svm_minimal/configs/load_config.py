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
    if out.pop("with_svm", False) or out.pop("svm_only", False):
        out["features_only"] = True
    for deprecated in (
        "svm_aaindex",
        "svm_model_pkl",
        "svm_scaler_csv",
        "svm_output_dir",
    ):
        out.pop(deprecated, None)
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


def resolve_path_list(values: list[str]) -> list[str]:
    root = repo_root()
    return [resolve_path_str(item, root) if isinstance(item, str) else item for item in values]


PEPSICKLE_CONFIG_SECTIONS = frozenset(
    {
        "preprocessing",
        "pepsickle",
        "fragment_expansion",
        "filtering",
        "paper_pddp",
        "mapp_database",
        "output",
    }
)


def flatten_pepsickle_config(raw: dict) -> dict:
    out: dict = {}
    for key, value in raw.items():
        if str(key).startswith("_"):
            continue
        if key in PEPSICKLE_CONFIG_SECTIONS and isinstance(value, dict):
            out.update(value)
        else:
            out[key] = value
    return out


def merge_pepsickle_config_paths(paths: list[str | Path] | None) -> dict:
    if not paths:
        return {}
    merged: dict = {}
    for path in paths:
        merged.update(flatten_pepsickle_config(load_json(path)))
    return merged


GNN_FINAL_PATH_KEYS = frozenset(
    {
        "csv_path",
        "pdb_dir",
        "qsar_csv",
        "esm2_csv",
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
) -> tuple[dict, dict[str, dict], list[str], dict | None, list[str] | None]:
    """Load ``gnn_final_train.json``: path defaults, ``post_message_passing_tabular_presets``, optional defaults list."""
    path = config_path or str(repo_root() / "configs/gnn_final_train.json")
    raw = load_json(path)
    data = dict(raw)
    data.pop("_documentation", None)
    architectures = list(data.pop("architectures", ["gcn", "gat", "egnn"]))
    raw_presets = data.pop("post_message_passing_tabular_presets", None)
    if raw_presets is None:
        raw_presets = data.pop("feature_sets", None)
    if raw_presets is None:
        raw_presets = {}
    if not isinstance(raw_presets, dict):
        raise TypeError(f"{path}: post_message_passing_tabular_presets must be a JSON object.")

    post_mp_tabular_presets: dict[str, dict] = {}
    for name, cfg in raw_presets.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: invalid preset name {name!r}")
        if not isinstance(cfg, dict):
            raise TypeError(f"{path}: preset {name!r} must be a JSON object.")
        slim = {k: cfg[k] for k in ("use_geo", "use_qsar") if k in cfg}
        if set(slim.keys()) != {"use_geo", "use_qsar"}:
            raise ValueError(
                f"{path}: preset {name!r} must include boolean use_geo and use_qsar "
                f"(optional 'about' text is ignored by the trainer)."
            )
        if not all(isinstance(slim[k], bool) for k in ("use_geo", "use_qsar")):
            raise TypeError(f"{path}: preset {name!r} use_geo/use_qsar must be booleans.")
        post_mp_tabular_presets[name] = slim

    default_presets = data.pop("feature_sets_default", None)
    if default_presets is None:
        default_presets = data.pop("train_feature_sets", None)
    if default_presets is None:
        default_presets = data.pop("default_post_mp_tabular_presets", None)
    node_feature_groups = data.pop("node_feature_groups", None)
    training = resolve_path_keys(data, GNN_FINAL_PATH_KEYS)

    if default_presets is not None:
        if not isinstance(default_presets, list) or not default_presets:
            raise ValueError(
                f"{path}: feature_sets_default (or train_feature_sets / default_post_mp_tabular_presets) "
                f"must be a non-empty list of names that appear in the preset catalog "
                f"(feature_sets or post_message_passing_tabular_presets)."
            )
        if not all(isinstance(x, str) for x in default_presets):
            raise TypeError(
                f"{path}: feature_sets_default / train_feature_sets / default_post_mp_tabular_presets "
                f"must be a list of strings."
            )
        unknown = [x for x in default_presets if x not in post_mp_tabular_presets]
        if unknown:
            raise ValueError(
                f"{path}: default preset list contains unknown entries {unknown!r}. "
                f"Valid names: {list(post_mp_tabular_presets.keys())}"
            )
        seen: set[str] = set()
        ordered: list[str] = []
        for name in default_presets:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        default_presets = ordered
    else:
        default_presets = None

    return training, post_mp_tabular_presets, architectures, node_feature_groups, default_presets


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


def _compare_models_overlay_dict(path: str | Path) -> dict:
    raw = load_json(path)
    data = {k: v for k, v in raw.items() if not str(k).startswith("_")}
    path_vals = {k: data[k] for k in data if k in COMPARE_MODELS_PATH_KEYS}
    out = resolve_path_keys(path_vals, COMPARE_MODELS_PATH_KEYS)
    for k, v in data.items():
        if k not in COMPARE_MODELS_PATH_KEYS:
            out[k] = v
    return out


def load_compare_models_windowed_config(config_path: str | None = None) -> dict:
    """
    Defaults for ``compare_model_predictions_windowed.py``.

    Starts from ``compare_models.json``, then applies ``compare_models_windowed.json``
    when present, then an optional ``--config`` path (same layering as the pipeline).
    """
    cfg = load_compare_models_config(str(repo_root() / "configs/compare_models.json"))
    overlay = repo_root() / "configs/compare_models_windowed.json"
    if overlay.is_file():
        cfg.update(_compare_models_overlay_dict(overlay))
    if config_path:
        cfg.update(_compare_models_overlay_dict(config_path))
    return cfg

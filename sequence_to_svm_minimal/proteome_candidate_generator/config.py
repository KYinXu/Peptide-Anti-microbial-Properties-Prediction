"""Configuration loading for the proteome candidate generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configs.load_config import (
    flatten_pepsickle_config,
    load_json,
    merge_pepsickle_config_paths,
    resolve_path_keys,
    resolve_path_list,
)

PEPSICKLE_PATH_KEYS = frozenset(
    {
        "input",
        "output_dir",
        "amp_score_matrix",
        "mapp_database",
    }
)


@dataclass(frozen=True)
class ResidueMetrics:
    positive: frozenset[str]
    negative: frozenset[str]
    hydrophobic: frozenset[str]
    hydrophobic_moment_angle_degrees: float

    @classmethod
    def from_args(cls, args) -> ResidueMetrics:
        return cls(
            positive=frozenset(str(args.positive_charge_residues).upper()),
            negative=frozenset(str(args.negative_charge_residues).upper()),
            hydrophobic=frozenset(str(args.hydrophobic_residues).upper()),
            hydrophobic_moment_angle_degrees=float(args.hydrophobic_moment_angle_degrees),
        )


def load_config_dict(paths: list[str | Path] | None) -> dict[str, Any]:
    merged = merge_pepsickle_config_paths(paths)
    if not merged:
        return {}
    flat = flatten_pepsickle_config(merged)
    flat = resolve_path_keys(flat, PEPSICKLE_PATH_KEYS)
    if "known_amps" in flat and flat["known_amps"] is not None:
        flat["known_amps"] = resolve_path_list(flat["known_amps"])
    if "cleavage_models" in flat and flat["cleavage_models"] is not None:
        flat["cleavage_models"] = tuple(flat["cleavage_models"])
    return flat


def parser_defaults(paths: list[str | Path] | None) -> dict[str, Any]:
    defaults = load_config_dict(paths)
    if "output_dir" in defaults and defaults["output_dir"] is not None:
        defaults["output_dir"] = Path(defaults["output_dir"])
    if "input" in defaults and defaults["input"] is not None:
        defaults["input"] = Path(defaults["input"])
    for key in ("amp_score_matrix", "mapp_database"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = Path(defaults[key])
    if "known_amps" in defaults and defaults["known_amps"]:
        defaults["known_amps"] = [Path(path) for path in defaults["known_amps"]]
    return defaults

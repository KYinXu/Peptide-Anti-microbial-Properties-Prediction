"""Sidecar JSON + state-dict helpers so PeptideGNN inference matches training layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .data_utils import (
    NodeFeatureGroups,
    node_feature_groups_for_base_dim,
    node_feature_groups_from_config_value,
    node_input_dim,
)
from .models import esm2_hidden_dim_from_state_dict, esm2_raw_dim_from_state_dict

META_SUFFIX = "_gnn_meta.json"


def sidecar_meta_path(checkpoint_path: str | Path) -> Path:
    p = Path(checkpoint_path)
    return p.with_name(p.stem + META_SUFFIX)


def load_peptide_gnn_meta(checkpoint_path: str | Path) -> dict[str, Any] | None:
    path = sidecar_meta_path(checkpoint_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "node_feature_groups" not in data:
        return None
    return data


def save_peptide_gnn_meta(
    checkpoint_path: str | Path,
    *,
    architecture: str,
    node_feature_groups: NodeFeatureGroups,
    hidden_channels: int,
    num_layers: int,
    dropout: float,
    pooling: str,
    geo_feature_dim: int,
    esm2_raw_dim: int,
    esm2_hidden_dim: int,
) -> Path:
    ng = node_feature_groups
    base = node_input_dim(ng)
    first_msg = base + (esm2_hidden_dim if esm2_raw_dim > 0 else 0)
    payload = {
        "kind": "peptide_gnn",
        "schema": 1,
        "architecture": architecture.lower(),
        "node_feature_groups": {
            "onehot": ng.onehot,
            "pdb_continuous": ng.pdb_continuous,
            "vae_table": ng.vae_table,
            "esm2_residue": ng.esm2_residue,
        },
        "node_input_dim": base,
        "first_message_in_dim": first_msg,
        "hidden_channels": int(hidden_channels),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "pooling": pooling,
        "geo_feature_dim": int(geo_feature_dim),
        "esm2_raw_dim": int(esm2_raw_dim),
        "esm2_hidden_dim": int(esm2_hidden_dim),
    }
    out = sidecar_meta_path(checkpoint_path)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def infer_first_message_in_dim(state_dict: Mapping[str, Any], architecture: str) -> int:
    """Width of the node vector entering the first conv / input_proj (after ESM concat in forward)."""
    arch = architecture.lower().strip()
    keys: list[str] = []
    if arch in ("gcn", "gat"):
        for pre in ("model.", ""):
            keys.extend(
                (
                    f"{pre}convs.0.lin.weight",
                    f"{pre}convs.0.lin_src.weight",
                )
            )
    elif arch == "egnn":
        for pre in ("model.", ""):
            keys.append(f"{pre}input_proj.weight")
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    for k in keys:
        t = state_dict.get(k)
        if t is not None and getattr(t, "ndim", 0) == 2:
            return int(t.shape[1])
    raise ValueError(
        f"Cannot infer first-layer input width for {arch!r}. "
        f"Tried keys (sample): {keys[:6]}"
    )


def resolve_node_layout_for_checkpoint(
    checkpoint_path: str | Path,
    state_dict: Mapping[str, Any],
    architecture: str,
    *,
    user_node_groups: NodeFeatureGroups | None = None,
) -> tuple[NodeFeatureGroups, int, list[str]]:
    """
    Return ``(node_feature_groups, base_node_dim, notes)`` for building
    ``PeptideGraphDataset`` / ``PeptideGNN`` so ``load_state_dict`` succeeds.

    Prefers ``<stem>_gnn_meta.json`` next to the checkpoint; otherwise infers
    base width from the first graph layer and recovers toggles via
    :func:`gnn.data_utils.node_feature_groups_for_base_dim`.
    """
    notes: list[str] = []
    arch = architecture.lower().strip()
    esm2_raw = esm2_raw_dim_from_state_dict(state_dict)
    esm2_h = esm2_hidden_dim_from_state_dict(state_dict)
    first_in = infer_first_message_in_dim(state_dict, arch)
    base_ckpt = first_in - (esm2_h if esm2_raw > 0 else 0)
    if base_ckpt < 0:
        raise ValueError(
            f"Inferred negative base node dim ({base_ckpt}) from first_in={first_in}, "
            f"esm2_raw={esm2_raw}, esm2_hidden={esm2_h}."
        )

    meta = load_peptide_gnn_meta(checkpoint_path)
    if meta is not None:
        ng_meta = node_feature_groups_from_config_value(meta.get("node_feature_groups"))
        ng_eff = ng_meta if ng_meta is not None else NodeFeatureGroups()
        base_meta = node_input_dim(ng_eff)
        meta_first = int(meta.get("first_message_in_dim", base_meta + (esm2_h if esm2_raw > 0 else 0)))
        if meta_first == first_in and base_meta == base_ckpt:
            ng_use = NodeFeatureGroups(
                onehot=ng_eff.onehot,
                pdb_continuous=ng_eff.pdb_continuous,
                vae_table=ng_eff.vae_table,
                esm2_residue=esm2_raw > 0,
            )
            notes.append(f"Using sidecar {sidecar_meta_path(checkpoint_path).name}")
        else:
            notes.append(
                f"Sidecar {sidecar_meta_path(checkpoint_path).name!r} disagrees with weights "
                f"(sidecar base {base_meta}, first_in {meta_first} vs ckpt base {base_ckpt}, "
                f"first_in {first_in}); trusting state_dict."
            )
            ng_use = node_feature_groups_for_base_dim(base_ckpt)
            ng_use.esm2_residue = esm2_raw > 0
    else:
        ng_use = node_feature_groups_for_base_dim(base_ckpt)
        ng_use.esm2_residue = esm2_raw > 0
        notes.append(
            f"No {META_SUFFIX} next to checkpoint; inferred node blocks from weight shapes "
            f"(data.x width {base_ckpt})."
        )

    if user_node_groups is not None:
        u = user_node_groups
        if (
            u.onehot != ng_use.onehot
            or u.pdb_continuous != ng_use.pdb_continuous
            or u.vae_table != ng_use.vae_table
            or bool(u.esm2_residue) != bool(ng_use.esm2_residue)
        ):
            notes.append(
                "Config / CLI node_feature_groups differ from checkpoint layout; using checkpoint."
            )

    return ng_use, base_ckpt, notes

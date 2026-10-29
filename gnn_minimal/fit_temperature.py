#!/usr/bin/env python3
"""Fit temperature scaling for a saved GNN checkpoint.

The GNN weights stay frozen. Temperature is fit on the validation split used
by training: a stratified shuffle of the labeled table, seed 42 and 20% held
out unless overridden. The result is written to ``gnn_temperature_scaling``
in the model directory.

Usage:
  python gnn_minimal/fit_temperature.py --model checkpoints/gnn/gnn_vae_qsar --input labeled.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.artifacts import model_checkpoint_path, resolve_model_dir, temperature_scaling_path
from core.checkpoint_meta import load_peptide_gnn_meta, resolve_node_layout_for_checkpoint
from core.data_utils import PeptideGraphDataset
from core.models import PeptideGNN, esm2_hidden_dim_from_state_dict, esm2_raw_dim_from_state_dict
from core.platt import collect_margins_and_labels
from core.temperature import fit_temperature, save_temperature_json
from inference import resolve_tabular_contract
from process_data import ensure_processed_data, resolve_input_and_generated
from train import binary_labels, load_data_with_features, require_both_classes


def validation_indices(labels: np.ndarray, val_size: float, seed: int) -> np.ndarray:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    _train_idx, val_idx = next(splitter.split(np.arange(len(labels)), labels))
    return val_idx


def labeled_validation_frame(gen_dir: Path, source_csv: Path | None, val_size: float, seed: int):
    frame, _qsar_cols = load_data_with_features(
        str(gen_dir / "geometric_features.csv"),
        str(gen_dir / "qsar12_descriptors.csv"),
        source_csv,
    )
    labels = binary_labels(frame["label"].values)
    require_both_classes(labels, context="Temperature scaling")
    val_idx = validation_indices(labels, val_size, seed)
    return frame.iloc[val_idx].reset_index(drop=True)


def load_frozen_model(model_dir: Path, dataset, state, meta: dict, in_channels: int, esm2_raw: int, device):
    geo_dim = 0
    if len(dataset) > 0 and hasattr(dataset[0], "geo_features"):
        geo_dim = int(dataset[0].geo_features.shape[1])
    model = PeptideGNN(
        architecture=str(meta.get("architecture", "gat")),
        in_channels=in_channels,
        hidden_channels=int(meta.get("hidden_channels", 64)),
        num_layers=int(meta.get("num_layers", 3)),
        num_classes=int(meta.get("num_classes", 2)),
        pooling=str(meta.get("pooling", "mean_max")),
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=esm2_hidden_dim_from_state_dict(state),
    )
    model.load_state_dict(state)
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for a saved GNN checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Model directory or path to gnn_model.pt.")
    parser.add_argument("--input", type=Path, required=True, help="Labeled CSV (id,label,sequence) or its generated/ directory.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed. Match the training run (default: 42).")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation fraction (default: 0.2).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-process", action="store_true", help="Regenerate structures and features before fitting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1.")
    model_dir = resolve_model_dir(args.model)
    checkpoint = model_checkpoint_path(model_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(state)
    meta = load_peptide_gnn_meta(model_dir) or {}
    if int(meta.get("num_classes", 2)) != 2:
        raise ValueError("Temperature scaling supports binary checkpoints only.")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    source_csv, gen_dir = resolve_input_and_generated(args.input)
    gen_dir = ensure_processed_data(
        source_csv,
        gen_dir,
        need_esm2=esm2_raw > 0,
        require_labels=True,
        device=device_name,
        force=args.force_process,
    )
    architecture = str(meta.get("architecture", "gat"))
    _groups, in_channels, _notes = resolve_node_layout_for_checkpoint(model_dir, state, architecture, user_node_groups=None)
    scaler_path, feature_cols = resolve_tabular_contract(model_dir, meta)
    val_frame = labeled_validation_frame(gen_dir, source_csv, args.val_size, args.seed)
    dataset = PeptideGraphDataset(
        csv_path=str(gen_dir / "geometric_features.csv"),
        pdb_dir=str(gen_dir / "structures"),
        use_geometric_features=bool(feature_cols),
        geometric_feature_cols=feature_cols,
        tabular_scaler_path=str(scaler_path) if scaler_path else None,
        esm2_residue_dir=str(gen_dir / "esm2_per_residue") if esm2_raw > 0 else None,
        node_feature_groups=_groups,
        dataframe=val_frame,
    )
    device = torch.device(device_name)
    model = load_frozen_model(model_dir, dataset, state, meta, in_channels, esm2_raw, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    margins, labels = collect_margins_and_labels(model, loader, device)
    payload = fit_temperature(margins, labels)
    if payload is None:
        raise ValueError("Temperature fit needs both classes on the validation split.")
    payload["seed"] = args.seed
    payload["val_size"] = args.val_size
    output = temperature_scaling_path(model_dir)
    save_temperature_json(output, payload)
    print(
        f"Saved temperature {payload['temperature']:.6f} "
        f"({payload['n_calib']} validation peptides) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

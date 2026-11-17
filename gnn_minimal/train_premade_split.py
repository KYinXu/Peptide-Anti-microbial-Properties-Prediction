#!/usr/bin/env python3
"""
Train single GNN models on a premade split: one labeled CSV per split.

Training settings (architecture, feature presets, epochs, lr, node feature
groups, ...) come from the same config as train.py; only the split differs.
train.py derives a stratified train/val split from one CSV and has no test
set, so held-out numbers from this script are not comparable to its val
metrics.

Usage:
  python train_premade_split.py --train train.csv --val val.csv --test test.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.data_utils import (
    NodeFeatureGroups,
    node_input_dim,
    wants_esm2_residue_nodes,
)
from core.models import PeptideGNN
from core.platt import (
    collect_margins_and_labels,
    fit_platt,
    save_platt_json,
)
from core.train import evaluate, run_training
from core.extra_feature_scaler import ExtraFeatureRobustScaler, save_extra_feature_scaler
from core.checkpoint_meta import save_peptide_gnn_meta
from core.artifacts import (
    model_checkpoint_path,
    platt_scaling_path,
    tabular_scaler_path,
)
from process_data import ensure_processed_data, input_has_labels, resolve_input_and_generated
from train import (
    CustomPeptideDataset,
    binary_labels,
    create_feature_cols,
    infer_esm2_raw_dim,
    load_data_with_features,
    require_both_classes,
    set_seed,
)

SPLIT_NAMES = ("train", "val", "test")


def resolve_split_input(path: Path | str) -> tuple[Path | None, Path]:
    """Give each split CSV its own generated/generated_<stem>/ so sibling splits never share artifacts."""
    csv_path, gen_dir = resolve_input_and_generated(path)
    if csv_path is None:
        return None, gen_dir
    return csv_path, csv_path.parent / "generated" / f"generated_{csv_path.stem}"


def prepare_split(
    name: str,
    path: Path,
    *,
    need_esm2: bool,
    device: str,
    force: bool,
) -> dict:
    csv_path, gen_dir = resolve_split_input(path)
    if csv_path is not None and not input_has_labels(csv_path):
        raise ValueError(
            f"{name} CSV must include a 'label' column (id,label,sequence): {csv_path}"
        )
    gen_dir = ensure_processed_data(
        csv_path,
        gen_dir,
        need_esm2=need_esm2,
        require_labels=True,
        device=device,
        force=force,
    )
    df, qsar_cols = load_data_with_features(
        gen_dir / "geometric_features.csv",
        gen_dir / "qsar12_descriptors.csv",
        source_csv=csv_path,
    )
    raw_labels = df["label"].values
    labels = binary_labels(raw_labels)
    require_both_classes(labels, context=f"{name} split")
    df = df.copy()
    df["label"] = labels
    return {
        "name": name,
        "df": df,
        "labels": labels,
        "raw_labels": raw_labels,
        "qsar_cols": qsar_cols,
        "gen_dir": gen_dir,
        "pdb_dir": gen_dir / "structures",
        "esm2_dir": gen_dir / "esm2_per_residue",
    }


def check_disjoint(splits: dict) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = set(splits[left]["df"]["peptide_id"]) & set(splits[right]["df"]["peptide_id"])
        if shared:
            raise ValueError(
                f"{left} and {right} splits share {len(shared)} peptide_id(s), "
                f"e.g. {sorted(shared)[:5]}"
            )


def train_on_splits(
    arch: str,
    feature_name: str,
    feature_cfg: dict,
    splits: dict,
    args,
    device: torch.device,
    out_dir: Path,
    esm2_raw_dim: int,
    node_feature_groups: NodeFeatureGroups | None = None,
):
    qsar_cols = splits["train"]["qsar_cols"]
    feature_cols = create_feature_cols(feature_cfg["use_geo"], feature_cfg["use_qsar"], qsar_cols)
    num_classes = 2

    tabular_scaler = None
    if feature_cols:
        tabular_scaler = ExtraFeatureRobustScaler.fit(
            splits["train"]["df"], feature_cols, balance_blocks=True
        )

    want_esm2_nodes = wants_esm2_residue_nodes(node_feature_groups)

    from torch_geometric.loader import DataLoader

    loaders = {}
    for name in SPLIT_NAMES:
        split = splits[name]
        dataset = CustomPeptideDataset(
            split["df"],
            split["pdb_dir"],
            feature_cols if feature_cols else None,
            args.distance_threshold,
            tabular_scaler=tabular_scaler,
            esm2_residue_dir=str(split["esm2_dir"].resolve()) if want_esm2_nodes else None,
            node_feature_groups=node_feature_groups,
        )
        graphs = [dataset[i] for i in range(len(dataset))]
        loaders[name] = DataLoader(
            graphs, batch_size=args.batch_size, shuffle=(name == "train")
        )

    train_y = splits["train"]["labels"]
    counts = np.bincount(train_y, minlength=num_classes)
    if int(counts.min()) > 0:
        n_tr = int(train_y.shape[0])
        w = np.array([n_tr / (num_classes * int(counts[c])) for c in range(num_classes)], dtype=np.float32)
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        class_weights = None

    geo_dim = len(feature_cols)
    esm2_raw = esm2_raw_dim if want_esm2_nodes else 0
    in_ch = node_input_dim(node_feature_groups)

    model = PeptideGNN(
        architecture=arch,
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
    )

    print(f"\n=== Training {arch.upper()} on {feature_name} ===")
    _, best_metrics = run_training(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights,
        verbose=True,
        label_smoothing=args.label_smoothing,
        logit_penalty=args.logit_penalty,
    )

    test_metrics = evaluate(model, loaders["test"], device)
    print(
        "Held-out test: "
        f"auc_roc={test_metrics['auc_roc']:.4f} "
        f"auc_pr={test_metrics['auc_pr']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"mcc={test_metrics['mcc']:.4f}"
    )

    model_dir = out_dir / f"{arch}_{feature_name.replace('+', '_plus_')}"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_checkpoint_path(model_dir)
    torch.save(model.state_dict(), ckpt_path)

    if tabular_scaler is not None:
        save_extra_feature_scaler(tabular_scaler, str(tabular_scaler_path(model_dir)))

    margins, y_val = collect_margins_and_labels(model, loaders["val"], device)
    platt_payload = fit_platt(margins, y_val)
    if platt_payload is not None:
        save_platt_json(platt_scaling_path(model_dir), platt_payload)
        print(f"Saved Platt calibration: {platt_scaling_path(model_dir)}")

    raw_labels = np.concatenate([splits[name]["raw_labels"] for name in SPLIT_NAMES])
    label_map = {label: 1 if int(label) == 1 else 0 for label in sorted(set(raw_labels))}
    metrics = dict(best_metrics)
    metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    _meta_ng = node_feature_groups if node_feature_groups is not None else NodeFeatureGroups()
    save_peptide_gnn_meta(
        model_dir,
        architecture=arch,
        node_feature_groups=_meta_ng,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
        num_classes=num_classes,
        label_map=label_map,
        tabular_feature_cols=feature_cols,
        feature_set=feature_name,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
    )
    print(f"Saved model directory: {model_dir}")

    return str(model_dir), best_metrics, test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN on a premade train/val/test split (one labeled CSV per split)"
    )
    parser.add_argument("--train", type=Path, required=True, help="Training CSV (id,label,sequence) or its generated/ folder")
    parser.add_argument("--val", type=Path, required=True, help="Validation CSV (early stopping + Platt calibration)")
    parser.add_argument("--test", type=Path, required=True, help="Held-out test CSV (scored once after training)")
    parser.add_argument("--config", type=str, default="configs/gnn_final_train.json", help="Path to config JSON")
    parser.add_argument("--output_dir", type=str, default="results/gnn_models")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--esm2_hidden_dim", type=int, default=64)
    parser.add_argument("--presets", nargs="+", default=None, help="Post-pooling tabular presets to train")
    parser.add_argument("--force-process", action="store_true", help="Regenerate processed features before training")
    parser.add_argument("--device", type=str, default=None, help="Device for on-demand processing (default: auto)")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent / args.config
    with open(config_path) as f:
        config = json.load(f)

    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    node_feature_groups = NodeFeatureGroups(**args.node_feature_groups)
    need_esm2 = wants_esm2_residue_nodes(node_feature_groups)
    process_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    split_paths = {"train": args.train, "val": args.val, "test": args.test}
    try:
        splits = {
            name: prepare_split(
                name,
                path,
                need_esm2=need_esm2,
                device=process_device,
                force=args.force_process,
            )
            for name, path in split_paths.items()
        }
        check_disjoint(splits)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    for name in SPLIT_NAMES:
        labels = splits[name]["labels"]
        print(
            f"{name:5s}: {len(labels)} peptides "
            f"({int(labels.sum())} positive, {int((labels == 0).sum())} negative) "
            f"from {splits[name]['gen_dir']}"
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    esm2_raw_dim = 0
    if need_esm2:
        esm2_raw_dim = infer_esm2_raw_dim(splits["train"]["df"], splits["train"]["esm2_dir"])

    feature_sets = args.feature_sets
    selected_presets = (
        args.presets
        or getattr(args, "feature_sets_default", None)
        or getattr(args, "train_feature_sets", None)
        or list(feature_sets)
    )
    unknown_presets = [name for name in selected_presets if name not in feature_sets]
    if unknown_presets:
        raise ValueError(
            f"Unknown presets {unknown_presets}; available: {list(feature_sets)}"
        )

    for preset_name in selected_presets:
        f_cfg = feature_sets[preset_name]
        for arch in args.architectures:
            model_dir, val_metrics, test_metrics = train_on_splits(
                arch=arch,
                feature_name=preset_name,
                feature_cfg=f_cfg,
                splits=splits,
                args=args,
                device=device,
                out_dir=out_dir,
                esm2_raw_dim=esm2_raw_dim,
                node_feature_groups=node_feature_groups,
            )
            print(
                f"Finished {arch}/{preset_name}: "
                f"val_auc={val_metrics.get('auc_roc', float('nan')):.4f} "
                f"test_auc={test_metrics.get('auc_roc', float('nan')):.4f} -> {model_dir}"
            )

    print(f"\nModels saved under: {out_dir}")


if __name__ == "__main__":
    main()

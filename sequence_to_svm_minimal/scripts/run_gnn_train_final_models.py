#!/usr/bin/env python3
"""
Train single GNN models (no CV) for each architecture/feature config,
and save checkpoints ready for use with `data_evaluation/compare_model_predictions.py`.

Configs mirror `run_gnn_comparison.py`:
- Graph-only
- Graph+Geo20
- Graph+Combined32 (geo + QSAR)
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))

from gnn.models import PeptideGNN
from gnn.train import run_training, evaluate


CONFIG = {
    "csv_path": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/geometric_features_clustered.csv",
    "pdb_dir": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced",
    "qsar_csv": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/qsar12_descriptors.csv",
    "seed": 42,
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 30,
    "hidden_channels": 64,
    "num_layers": 3,
    "dropout": 0.2,
    "distance_threshold": 8.0,
}

FEATURE_CONFIGS = {
    "Graph-only": {"use_geo": False, "use_qsar": False, "geo_dim": 0},
    "Graph+Geo20": {"use_geo": True, "use_qsar": False, "geo_dim": 20},
    "Graph+QSAR12": {"use_geo": False, "use_qsar": True, "geo_dim": 12},
    "Graph+Combined32": {"use_geo": True, "use_qsar": True, "geo_dim": 32},
}

ARCHITECTURES = ["gcn", "gat", "egnn"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_with_features(csv_path: str, qsar_csv: str):
    """Load geometric CSV and merge QSAR if available."""
    geo_df = pd.read_csv(csv_path)
    qsar_df = pd.read_csv(qsar_csv)

    qsar_cols = [
        "netCharge",
        "FC",
        "LW",
        "DP",
        "NK",
        "AE",
        "pcMK",
        "_SolventAccessibilityD1025",
        "tau2_GRAR740104",
        "tau4_GRAR740104",
        "QSO50_GRAR740104",
        "QSO29_GRAR740104",
    ]

    merged_df = geo_df.merge(qsar_df[["peptide_id"] + qsar_cols], on="peptide_id", how="left")
    return merged_df, qsar_cols


def create_feature_cols(use_geo: bool, use_qsar: bool, qsar_cols):
    geo_cols = [
        "radius_gyration",
        "end_to_end_distance",
        "max_pairwise_distance",
        "centroid_distance_mean",
        "centroid_distance_std",
        "fraction_helix",
        "fraction_sheet",
        "fraction_coil",
        "total_sasa",
        "hydrophobic_sasa",
        "fraction_hydrophobic_sasa",
        "length",
        "net_charge",
        "mean_hydrophobicity",
        "hydrophobic_moment",
        "curvature_mean",
        "curvature_std",
        "curvature_max",
        "torsion_mean",
        "torsion_std",
    ]
    cols = []
    if use_geo:
        cols.extend(geo_cols)
    if use_qsar and qsar_cols:
        cols.extend(qsar_cols)
    return cols


class CustomPeptideDataset:
    """Same dataset as in run_gnn_comparison, but without CV."""

    def __init__(self, df, pdb_dir, feature_cols, distance_threshold: float = 8.0):
        self.df = df
        self.pdb_dir = Path(pdb_dir)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold

        from gnn.data_utils import pdb_to_graph, parse_pdb, compute_node_features, compute_edges

        self.pdb_to_graph = pdb_to_graph
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        from gnn.data_utils import parse_pdb, compute_node_features, compute_edges

        row = self.df.iloc[idx]

        pdb_file = row.get("pdb_file", f"{row['peptide_id']}.pdb")
        pdb_path = None
        for subdir in ["structures/AMP", "structures/DECOY", "structures", ""]:
            candidate = self.pdb_dir / subdir / pdb_file
            if candidate.exists():
                pdb_path = candidate
                break

        if pdb_path is None:
            raise FileNotFoundError(f"PDB not found: {pdb_file}")

        aa_sequence, ca_coords, plddt_values = parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)

        x = compute_node_features(aa_sequence, plddt_values, n_residues)
        edge_index, edge_attr = compute_edges(ca_coords, self.distance_threshold)
        pos = torch.tensor(ca_coords, dtype=torch.float32)

        raw_label = int(row["label"])
        label = 1 if raw_label == 1 else 0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_residues,
        )

        if self.feature_cols:
            extra = row[self.feature_cols].values.astype(np.float32)
            extra = np.nan_to_num(extra, nan=0.0)
            data.geo_features = torch.tensor(extra, dtype=torch.float32).unsqueeze(0)

        return data


def train_single_model(arch: str, feature_name: str, feature_cfg: dict, df: pd.DataFrame, qsar_cols, args, device: torch.device, out_dir: Path):
    """Train a single model with a train/val split and save checkpoint."""
    feature_cols = create_feature_cols(feature_cfg["use_geo"], feature_cfg["use_qsar"], qsar_cols)
    dataset = CustomPeptideDataset(df, args.pdb_dir, feature_cols if feature_cols else None, args.distance_threshold)

    labels = np.where(df["label"].values == 1, 1, 0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))

    from torch_geometric.loader import DataLoader

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    class_weights = torch.tensor(
        [n_samples / (n_classes * np.sum(labels == c)) for c in range(n_classes)],
        dtype=torch.float32,
        device=device,
    )

    geo_dim = feature_cfg["geo_dim"]
    model = PeptideGNN(
        architecture=arch,
        in_channels=26,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
    )

    print(f"\n=== Training {arch.upper()} on {feature_name} ===")
    _, best_metrics = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights,
        verbose=True,
    )

    print("Validation metrics:")
    for k, v in best_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    model_name = f"{arch}_ready_{feature_name.replace('+', '_plus_')}.pt"
    ckpt_path = out_dir / model_name
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    return str(ckpt_path), best_metrics


def parse_args():
    ap = argparse.ArgumentParser(description="Train single GNN models (no CV) for test-time inference.")
    ap.add_argument("--csv_path", type=str, default=CONFIG["csv_path"])
    ap.add_argument("--pdb_dir", type=str, default=CONFIG["pdb_dir"])
    ap.add_argument("--qsar_csv", type=str, default=CONFIG["qsar_csv"])
    ap.add_argument("--architectures", type=str, nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    ap.add_argument("--feature_sets", type=str, nargs="+", default=list(FEATURE_CONFIGS.keys()), choices=list(FEATURE_CONFIGS.keys()))
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit model selections as ARCH:FEATURE (e.g. "
            "gat:Graph-only gat:Graph+QSAR12). If set, --architectures and "
            "--feature_sets are ignored."
        ),
    )
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=CONFIG["lr"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=CONFIG["patience"])
    ap.add_argument("--hidden_channels", type=int, default=CONFIG["hidden_channels"])
    ap.add_argument("--num_layers", type=int, default=CONFIG["num_layers"])
    ap.add_argument("--dropout", type=float, default=CONFIG["dropout"])
    ap.add_argument("--distance_threshold", type=float, default=CONFIG["distance_threshold"])
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output_dir", type=str, default="results/gnn/ready_models")
    return ap.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print("Device:", device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    merged_df, qsar_cols = load_data_with_features(args.csv_path, args.qsar_csv)
    print(f"Loaded training data: {len(merged_df)} samples")

    summary = {"timestamp": datetime.now().isoformat(), "models": []}

    selected_pairs = []
    if args.models:
        for item in args.models:
            if ":" not in item:
                raise ValueError(
                    f"Invalid --models entry '{item}'. Expected format ARCH:FEATURE "
                    f"(e.g. gat:Graph-only)."
                )
            arch, feature_name = item.split(":", 1)
            arch = arch.strip().lower()
            feature_name = feature_name.strip()

            if arch not in ARCHITECTURES:
                raise ValueError(
                    f"Unknown architecture '{arch}' in --models. "
                    f"Choose from: {ARCHITECTURES}"
                )
            if feature_name not in FEATURE_CONFIGS:
                raise ValueError(
                    f"Unknown feature set '{feature_name}' in --models. "
                    f"Choose from: {list(FEATURE_CONFIGS.keys())}"
                )
            selected_pairs.append((arch, feature_name))
    else:
        for feature_name in args.feature_sets:
            for arch in args.architectures:
                selected_pairs.append((arch, feature_name))

    # Keep output deterministic when duplicates are provided:
    # run once per unique pair while preserving first-seen order.
    unique_pairs = []
    seen = set()
    for pair in selected_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)

    for arch, feature_name in unique_pairs:
        f_cfg = FEATURE_CONFIGS[feature_name]
        ckpt_path, metrics = train_single_model(
            arch=arch,
            feature_name=feature_name,
            feature_cfg=f_cfg,
            df=merged_df,
            qsar_cols=qsar_cols,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        summary["models"].append(
            {
                "architecture": arch,
                "feature_set": feature_name,
                "checkpoint": ckpt_path,
                "metrics": {k: float(v) for k, v in metrics.items()},
            }
        )

    summary_path = out_dir / "ready_models_summary.json"
    with open(summary_path, "w") as f:
        import json

        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()


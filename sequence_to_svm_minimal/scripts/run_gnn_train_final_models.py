#!/usr/bin/env python3
"""
Train single GNN models (no CV) for each architecture/feature config,
and save checkpoints ready for use with `data_evaluation/compare_model_predictions.py`.

Typical usage after ``run_data_pipeline`` (writes ``<input_dir>/generated/``):

  python scripts/run_gnn_train_final_models.py path/to/generated

You may also pass the parent of ``generated/``; the script resolves ``generated/``
when the manifest lives there. Omit the path to use CONFIG / explicit CSV flags.

Configs mirror `run_gnn_comparison.py`:
- ESM (graph + ESM2)
- Geo (graph + Geo20 + ESM2)
- QSAR (graph + QSAR12 + ESM2)
- Combined (graph + Geo20 + QSAR12 + ESM2)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from gnn.data_utils import resolve_peptide_pdb_path
from gnn.models import PeptideGNN
from gnn.platt import (
    collect_margins_and_labels,
    default_platt_path,
    fit_platt,
    save_platt_json,
)
from gnn.train import run_training, evaluate
from gnn.extra_feature_scaler import ExtraFeatureRobustScaler, save_extra_feature_scaler


CONFIG = {
    "csv_path": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/geometric_features_clustered.csv",
    "pdb_dir": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced",
    "qsar_csv": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/qsar12_descriptors.csv",
    "esm2_csv": None,
    "esm2_amp_csv": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/esm2_amp.csv",
    "esm2_decoy_csv": "data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/esm2_decoy.csv",
    "seed": 42,
    "epochs": 300,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 30,
    "hidden_channels": 64,
    "num_layers": 3,
    "dropout": 0.2,
    "distance_threshold": 8.0,
    "label_smoothing": 0.08,
    "logit_penalty": 1e-4,
}

FEATURE_CONFIGS = {
    "ESM": {"use_geo": False, "use_qsar": False, "use_esm2": True},
    "Geo": {"use_geo": True, "use_qsar": False, "use_esm2": True},
    "QSAR": {"use_geo": False, "use_qsar": True, "use_esm2": True},
    "Combined": {"use_geo": True, "use_qsar": True, "use_esm2": True},
}

ARCHITECTURES = ["gcn", "gat", "egnn"]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_core_id(series: pd.Series) -> pd.Series:
    """Normalize IDs to a common core ID for AMP/DECOY joining."""
    s = series.astype(str)
    s = s.str.replace(r"^(AMP_|DECOY_)", "", regex=True)
    return s


def _load_esm2_table(esm2_csv: str | None, esm2_amp_csv: str | None, esm2_decoy_csv: str | None):
    """Load ESM2 embeddings from a combined CSV or AMP/DECOY CSV pair."""
    if esm2_csv:
        df = pd.read_csv(esm2_csv)
        if "peptide_id" in df.columns:
            id_col = "peptide_id"
        elif "seqIndex" in df.columns:
            id_col = "seqIndex"
        else:
            raise ValueError("ESM2 CSV must contain 'peptide_id' or 'seqIndex'")
        emb_cols = [c for c in df.columns if c.startswith("esm2_dim_")]
        if not emb_cols:
            raise ValueError("No ESM2 embedding columns found (expected prefix 'esm2_dim_')")
        out = df[[id_col] + emb_cols].copy()
        out["core_id"] = _normalize_core_id(out[id_col])
        return out[["core_id"] + emb_cols], emb_cols

    if esm2_amp_csv and esm2_decoy_csv:
        amp_df = pd.read_csv(esm2_amp_csv)
        decoy_df = pd.read_csv(esm2_decoy_csv)
        for d in (amp_df, decoy_df):
            if "seqIndex" not in d.columns:
                raise ValueError("ESM2 AMP/DECOY CSVs must contain 'seqIndex'")
        emb_cols = [c for c in amp_df.columns if c.startswith("esm2_dim_")]
        if not emb_cols:
            raise ValueError("No ESM2 embedding columns found (expected prefix 'esm2_dim_')")
        amp = amp_df[["seqIndex"] + emb_cols].copy()
        decoy = decoy_df[["seqIndex"] + emb_cols].copy()
        amp["core_id"] = _normalize_core_id(amp["seqIndex"])
        decoy["core_id"] = _normalize_core_id(decoy["seqIndex"])
        merged = pd.concat([amp[["core_id"] + emb_cols], decoy[["core_id"] + emb_cols]], axis=0, ignore_index=True)
        return merged.drop_duplicates(subset=["core_id"]), emb_cols

    return None, []


def load_data_with_features(csv_path: str,
                            qsar_csv: str,
                            esm2_csv: str | None = None,
                            esm2_amp_csv: str | None = None,
                            esm2_decoy_csv: str | None = None):
    """Load geometric CSV and merge QSAR and optional ESM2 embeddings."""
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

    esm2_cols = []
    esm2_df, esm2_cols = _load_esm2_table(esm2_csv, esm2_amp_csv, esm2_decoy_csv)
    if esm2_df is not None:
        merged_df["core_id"] = _normalize_core_id(merged_df["peptide_id"])
        merged_df = merged_df.merge(esm2_df, on="core_id", how="left")
        merged_df = merged_df.drop(columns=["core_id"])

    return merged_df, qsar_cols, esm2_cols


def create_feature_cols(use_geo: bool, use_qsar: bool, use_esm2: bool, qsar_cols, esm2_cols):
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
    if use_esm2 and esm2_cols:
        cols.extend(esm2_cols)
    return cols


class CustomPeptideDataset:
    """Same dataset as in run_gnn_comparison, but without CV."""

    def __init__(
        self,
        df,
        pdb_dir,
        feature_cols,
        distance_threshold: float = 8.0,
        tabular_scaler: ExtraFeatureRobustScaler | None = None,
    ):
        self.df = df
        self.pdb_dir = Path(pdb_dir)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold
        self.tabular_scaler = tabular_scaler

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

        pdb_file = row.get("pdb_file", None)
        pdb_path = resolve_peptide_pdb_path(self.pdb_dir, pdb_file, row["peptide_id"])
        if pdb_path is None:
            raise FileNotFoundError(
                f"PDB not found for peptide_id={row['peptide_id']!r} pdb_file={pdb_file!r} under {self.pdb_dir}"
            )

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
            if self.tabular_scaler is not None:
                extra = self.tabular_scaler.transform_row(row).astype(np.float32)
            else:
                extra = row[self.feature_cols].values.astype(np.float32)
                extra = np.nan_to_num(extra, nan=0.0)
            data.geo_features = torch.tensor(extra, dtype=torch.float32).unsqueeze(0)

        return data


def train_single_model(arch: str,
                       feature_name: str,
                       feature_cfg: dict,
                       df: pd.DataFrame,
                       qsar_cols,
                       esm2_cols,
                       args,
                       device: torch.device,
                       out_dir: Path):
    """Train a single model with a train/val split; save checkpoint and Platt JSON on val."""
    feature_cols = create_feature_cols(
        feature_cfg["use_geo"],
        feature_cfg["use_qsar"],
        feature_cfg.get("use_esm2", False),
        qsar_cols,
        esm2_cols,
    )

    labels = np.where(df["label"].values == 1, 1, 0)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.arange(len(labels)), labels))

    tabular_scaler = None
    if feature_cols and not args.no_tabular_robust_scaler:
        tabular_scaler = ExtraFeatureRobustScaler.fit(
            df.iloc[train_idx],
            feature_cols,
            balance_blocks=True,
        )

    dataset = CustomPeptideDataset(
        df,
        args.pdb_dir,
        feature_cols if feature_cols else None,
        args.distance_threshold,
        tabular_scaler=tabular_scaler,
    )

    from torch_geometric.loader import DataLoader

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    num_model_classes = 2
    train_y = labels[train_idx]
    counts = np.bincount(train_y, minlength=num_model_classes)
    if int(counts.min()) > 0:
        n_tr = int(train_y.shape[0])
        w = np.array(
            [n_tr / (num_model_classes * int(counts[c])) for c in range(num_model_classes)],
            dtype=np.float32,
        )
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        print(
            "Warning: training split has only one class; using unweighted cross-entropy.",
            flush=True,
        )
        class_weights = None

    geo_dim = len(feature_cols)
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
        label_smoothing=args.label_smoothing,
        logit_penalty=args.logit_penalty,
    )

    print("Validation metrics:")
    for k, v in best_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    model_name = f"{arch}_ready_{feature_name.replace('+', '_plus_')}.pt"
    ckpt_path = out_dir / model_name
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    if tabular_scaler is not None:
        scaler_path = ckpt_path.with_name(ckpt_path.stem + "_tabular_scaler.joblib")
        save_extra_feature_scaler(tabular_scaler, str(scaler_path))
        print(f"Saved tabular scaler: {scaler_path}")

    platt_path = default_platt_path(ckpt_path)
    margins, y_val = collect_margins_and_labels(model, val_loader, device)
    platt_payload = fit_platt(margins, y_val)
    if platt_payload is not None:
        save_platt_json(platt_path, platt_payload)
        print(
            f"Saved Platt calibration ({platt_payload['n_calib']} val samples): {platt_path}",
            flush=True,
        )
    else:
        print(
            "Platt calibration skipped (need both classes on the validation split).",
            flush=True,
        )

    return str(ckpt_path), best_metrics, str(platt_path) if platt_payload is not None else None


def resolve_final_train_paths(args: argparse.Namespace) -> None:
    from peptide_pipeline.manifest_paths import (
        gnn_final_training_paths_from_work_dir,
        resolve_generated_workspace,
    )

    pos = getattr(args, "generated", None)
    legacy = getattr(args, "pipeline_work_dir", None)
    input_dir = getattr(args, "input_dir", None)
    paths = [p for p in (pos, legacy, input_dir) if p]
    if len(paths) > 1 and len({Path(p).resolve() for p in paths}) > 1:
        raise SystemExit(
            "Use only one of: GENERATED (positional), --pipeline-work-dir, or --input-dir."
        )
    chosen = pos or legacy or input_dir
    bundle = None
    if chosen:
        workspace = resolve_generated_workspace(chosen)
        bundle = gnn_final_training_paths_from_work_dir(workspace)
        print("Pipeline workspace:", workspace, flush=True)
    if getattr(args, "csv_path", None) is None:
        args.csv_path = bundle["csv_path"] if bundle else CONFIG["csv_path"]
    if getattr(args, "pdb_dir", None) is None:
        args.pdb_dir = bundle["pdb_dir"] if bundle else CONFIG["pdb_dir"]
    if getattr(args, "qsar_csv", None) is None:
        args.qsar_csv = bundle["qsar_csv"] if bundle else CONFIG["qsar_csv"]
    if (
        args.esm2_csv is None
        and getattr(args, "esm2_amp_csv", None) is None
        and getattr(args, "esm2_decoy_csv", None) is None
    ):
        if bundle:
            args.esm2_csv = bundle["esm2_csv"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train single GNN models (no CV) for test-time inference.",
        epilog=(
            "Primary input: the pipeline generated/ folder (or parent containing generated/) "
            "with pipeline_manifest.json from run_data_pipeline."
        ),
    )
    ap.add_argument(
        "generated",
        nargs="?",
        default=None,
        metavar="GENERATED",
        help=(
            "Pipeline generated/ directory (pipeline_manifest.json inside), or a parent folder "
            "that contains generated/. Sets geometric CSV, PDB dir, QSAR12, ESM2 paths. "
            "Omit for CONFIG defaults or use --csv_path / overrides."
        ),
    )
    ap.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Same as positional GENERATED (kept for scripts and backward compatibility).",
    )
    ap.add_argument(
        "--pipeline-work-dir",
        type=str,
        default=None,
        dest="pipeline_work_dir",
        help="Same as positional GENERATED (alias used by other pipeline scripts).",
    )
    ap.add_argument("--csv_path", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--pdb_dir", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--qsar_csv", type=str, default=argparse.SUPPRESS)
    ap.add_argument(
        "--esm2_csv",
        type=str,
        default=None,
        help="Single merged ESM2 CSV (seqIndex or peptide_id + esm2_dim_*). Overrides CONFIG esm2 paths when set.",
    )
    ap.add_argument(
        "--esm2_amp_csv",
        type=str,
        default=None,
        help="Optional override for CONFIG esm2_amp_csv when not using --esm2_csv.",
    )
    ap.add_argument(
        "--esm2_decoy_csv",
        type=str,
        default=None,
        help="Optional override for CONFIG esm2_decoy_csv when not using --esm2_csv.",
    )
    ap.add_argument("--architectures", type=str, nargs="+", default=ARCHITECTURES, choices=ARCHITECTURES)
    ap.add_argument("--feature_sets", type=str, nargs="+", default=list(FEATURE_CONFIGS.keys()), choices=list(FEATURE_CONFIGS.keys()))
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit model selections as ARCH:FEATURE (e.g. "
            "gat:ESM gat:QSAR). If set, --architectures and "
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
    ap.add_argument(
        "--no_tabular_robust_scaler",
        action="store_true",
        help="Disable per-block RobustScaler + block balancing on concatenated extras (raw CSV values).",
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=CONFIG["label_smoothing"],
        help="Cross-entropy label smoothing (0 disables). Reduces extreme logit margins / softmax saturation.",
    )
    ap.add_argument(
        "--logit_penalty",
        type=float,
        default=CONFIG["logit_penalty"],
        help="Weight on mean(logits^2) added to training loss (0 disables). Softens raw score collapse.",
    )
    return ap.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    resolve_final_train_paths(args)
    set_seed(args.seed)
    device = get_device(args.device)

    print("Device:", device)

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)

    if args.esm2_csv is not None:
        esm2_csv_path = args.esm2_csv
        esm2_amp_path = None
        esm2_decoy_path = None
    else:
        esm2_csv_path = CONFIG.get("esm2_csv")
        esm2_amp_path = args.esm2_amp_csv if args.esm2_amp_csv is not None else CONFIG.get("esm2_amp_csv")
        esm2_decoy_path = args.esm2_decoy_csv if args.esm2_decoy_csv is not None else CONFIG.get("esm2_decoy_csv")
    merged_df, qsar_cols, esm2_cols = load_data_with_features(
        args.csv_path,
        args.qsar_csv,
        esm2_csv_path,
        esm2_amp_path,
        esm2_decoy_path,
    )
    if not esm2_cols:
        raise ValueError(
            "ESM2 embeddings are required for all feature configs, but no esm2_dim_* columns were loaded. "
            "Check CONFIG['esm2_csv'] or CONFIG['esm2_amp_csv']/CONFIG['esm2_decoy_csv']."
        )
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
        ckpt_path, metrics, platt_path = train_single_model(
            arch=arch,
            feature_name=feature_name,
            feature_cfg=f_cfg,
            df=merged_df,
            qsar_cols=qsar_cols,
            esm2_cols=esm2_cols,
            args=args,
            device=device,
            out_dir=out_dir,
        )
        entry = {
            "architecture": arch,
            "feature_set": feature_name,
            "checkpoint": ckpt_path,
            "metrics": {k: float(v) for k, v in metrics.items()},
        }
        if platt_path:
            entry["platt_calibration"] = platt_path
        summary["models"].append(entry)

    summary_path = out_dir / "ready_models_summary.json"
    with open(summary_path, "w") as f:
        import json

        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()


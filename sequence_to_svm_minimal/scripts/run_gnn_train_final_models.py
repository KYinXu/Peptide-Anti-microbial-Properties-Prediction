#!/usr/bin/env python3
"""
Train single GNN models (no CV) for each architecture/feature config,
and save checkpoints ready for use with `data_evaluation/compare_model_predictions.py`.
Each ``*.pt`` is accompanied by ``<same_stem>_gnn_meta.json`` (layout / hyperparameters)
so inference does not depend on matching ``compare_models.json`` node toggles.

Typical usage after ``run_data_pipeline`` (writes ``<input_dir>/generated/``):

  python scripts/run_gnn_train_final_models.py path/to/generated

You may also pass the parent of ``generated/``; the script resolves ``generated/``
when the manifest lives there. Omit the path to use ``configs/gnn_final_train.json`` (or ``--config``) / explicit CSV flags.

Configs define ``post_message_passing_tabular_presets`` (CSV columns fused after global pooling, before the classifier MLP). Node-level inputs (before pooling) come from ``node_feature_groups`` / ``--node-groups``. Example preset names:

- **ESM** — no extra tabular vector after pooling (graph embedding only; per-residue ESM on nodes is separate).
- **Geo** — append global geometric (Geo20) columns after pooling.
- **QSAR** — append QSAR12 columns after pooling.
- **Combined** — append Geo20 + QSAR12 after pooling.
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

from gnn.data_utils import (
    NodeFeatureGroups,
    node_feature_groups_from_cli,
    node_feature_groups_from_config_value,
    node_input_dim,
    resolve_peptide_pdb_path,
    wants_esm2_residue_nodes,
)
from gnn.models import PeptideGNN
from gnn.platt import (
    collect_margins_and_labels,
    default_platt_path,
    fit_platt,
    save_platt_json,
)
from gnn.train import run_training, evaluate
from gnn.extra_feature_scaler import ExtraFeatureRobustScaler, save_extra_feature_scaler
from gnn.checkpoint_meta import save_peptide_gnn_meta
from configs.load_config import argv_without_config_flags, load_gnn_final_train_bundle


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


def _load_esm2_table(esm2_csv: str | None):
    """Load ESM2 embeddings from a single merged CSV (peptide_id or seqIndex + esm2_dim_*)."""
    if not esm2_csv:
        return None, []
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


def load_data_with_features(csv_path: str, qsar_csv: str, esm2_csv: str | None = None):
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

    esm2_df, esm2_cols = _load_esm2_table(esm2_csv)
    if esm2_df is not None:
        merged_df["core_id"] = _normalize_core_id(merged_df["peptide_id"])
        merged_df = merged_df.merge(esm2_df, on="core_id", how="left")
        merged_df = merged_df.drop(columns=["core_id"])

    return merged_df, qsar_cols, esm2_cols


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

    def __init__(
        self,
        df,
        pdb_dir,
        feature_cols,
        distance_threshold: float = 8.0,
        tabular_scaler: ExtraFeatureRobustScaler | None = None,
        esm2_residue_dir: str | None = None,
        node_feature_groups: NodeFeatureGroups | None = None,
    ):
        self.df = df
        self.pdb_dir = Path(pdb_dir)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold
        self.tabular_scaler = tabular_scaler
        self.esm2_residue_dir = Path(esm2_residue_dir).resolve() if esm2_residue_dir else None
        self.node_feature_groups = node_feature_groups

        from gnn.data_utils import pdb_to_graph, parse_pdb, compute_node_features, compute_edges

        self.pdb_to_graph = pdb_to_graph
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        from gnn.data_utils import parse_pdb, compute_node_features, compute_edges, load_esm2_per_residue_tensor

        row = self.df.iloc[idx]

        pdb_file = row.get("pdb_file", None)
        pdb_path = resolve_peptide_pdb_path(self.pdb_dir, pdb_file, row["peptide_id"])
        if pdb_path is None:
            raise FileNotFoundError(
                f"PDB not found for peptide_id={row['peptide_id']!r} pdb_file={pdb_file!r} under {self.pdb_dir}"
            )

        aa_sequence, ca_coords, plddt_values = parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)

        x = compute_node_features(
            aa_sequence, plddt_values, n_residues, groups=self.node_feature_groups
        )
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

        if self.esm2_residue_dir is not None and wants_esm2_residue_nodes(self.node_feature_groups):
            esm = load_esm2_per_residue_tensor(self.esm2_residue_dir, row["peptide_id"])
            if int(esm.shape[0]) != int(data.num_nodes):
                raise ValueError(
                    f"ESM2 length {esm.shape[0]} != graph nodes {data.num_nodes} for peptide_id={row['peptide_id']!r}"
                )
            data.esm2_node = esm

        return data


def train_single_model(
    arch: str,
    feature_name: str,
    feature_cfg: dict,
    df: pd.DataFrame,
    qsar_cols,
    esm2_cols,
    args,
    device: torch.device,
    out_dir: Path,
    *,
    node_feature_groups: NodeFeatureGroups | None = None,
):
    """Train a single model with a train/val split; save checkpoint and Platt JSON on val."""
    feature_cols = create_feature_cols(
        feature_cfg["use_geo"],
        feature_cfg["use_qsar"],
        qsar_cols,
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

    want_esm2_nodes = wants_esm2_residue_nodes(node_feature_groups)
    esm2_dir = str(Path(args.esm2_residue_dir).resolve()) if want_esm2_nodes and args.esm2_residue_dir else None

    dataset = CustomPeptideDataset(
        df,
        args.pdb_dir,
        feature_cols if feature_cols else None,
        args.distance_threshold,
        tabular_scaler=tabular_scaler,
        esm2_residue_dir=esm2_dir,
        node_feature_groups=node_feature_groups,
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
    esm2_raw = len(esm2_cols) if want_esm2_nodes else 0
    in_ch = node_input_dim(node_feature_groups)
    model = PeptideGNN(
        architecture=arch,
        in_channels=in_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
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
    _meta_ng = node_feature_groups if node_feature_groups is not None else NodeFeatureGroups()
    meta_written = save_peptide_gnn_meta(
        ckpt_path,
        architecture=arch,
        node_feature_groups=_meta_ng,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling="mean_max",
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=args.esm2_hidden_dim,
    )
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved GNN layout meta: {meta_written}", flush=True)
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


def resolve_final_train_paths(args: argparse.Namespace, training_defaults: dict) -> None:
    from peptide_pipeline.manifest_paths import (
        gnn_final_training_paths_from_work_dir,
        resolve_generated_workspace,
    )

    pos = getattr(args, "generated", None)
    bundle = None
    if pos:
        workspace = resolve_generated_workspace(pos)
        bundle = gnn_final_training_paths_from_work_dir(workspace)
        print("Pipeline workspace:", workspace, flush=True)
    if getattr(args, "csv_path", None) is None:
        args.csv_path = bundle["csv_path"] if bundle else training_defaults["csv_path"]
    if getattr(args, "pdb_dir", None) is None:
        args.pdb_dir = bundle["pdb_dir"] if bundle else training_defaults["pdb_dir"]
    if getattr(args, "qsar_csv", None) is None:
        args.qsar_csv = bundle["qsar_csv"] if bundle else training_defaults["qsar_csv"]
    if getattr(args, "esm2_residue_dir", None) is None:
        if bundle and bundle.get("esm2_residue_dir"):
            args.esm2_residue_dir = bundle["esm2_residue_dir"]
        elif bundle and bundle.get("esm2_csv"):
            args.esm2_residue_dir = str(Path(bundle["esm2_csv"]).parent / "esm2_per_residue")
        else:

            def _first_resolved_path(*keys: str) -> Path | None:
                for key in keys:
                    val = getattr(args, key, None)
                    if val:
                        return Path(val).resolve()
                    tv = training_defaults.get(key)
                    if tv:
                        return Path(tv).resolve()
                return None

            anchor = _first_resolved_path(
                "esm2_csv",
                "qsar_csv",
                "csv_path",
            )
            if anchor is not None:
                args.esm2_residue_dir = str((anchor.parent / "esm2_per_residue").resolve())
    if args.esm2_csv is None and bundle:
        args.esm2_csv = bundle["esm2_csv"]


def parse_args():
    cfg_path, argv_rest = argv_without_config_flags(sys.argv[1:])
    (
        training_defaults,
        post_mp_tabular_presets,
        arch_list,
        node_groups_cfg,
        default_post_mp_tabular_presets,
    ) = load_gnn_final_train_bundle(cfg_path)
    preset_names = list(post_mp_tabular_presets.keys())
    default_post_mp = (
        default_post_mp_tabular_presets
        if default_post_mp_tabular_presets is not None
        else preset_names
    )

    ap = argparse.ArgumentParser(
        description="Train single GNN models (no CV) for test-time inference.",
        epilog=(
            "Primary input: the pipeline generated/ folder (or parent containing generated/) "
            "with pipeline_manifest.json from run_data_pipeline. "
            "Optional defaults file: pass --config PATH (configs/gnn_final_train.json when omitted)."
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
            "Omit to use config file defaults or --csv_path / overrides."
        ),
    )
    ap.add_argument("--csv_path", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--pdb_dir", type=str, default=argparse.SUPPRESS)
    ap.add_argument("--qsar_csv", type=str, default=argparse.SUPPRESS)
    ap.add_argument(
        "--esm2_csv",
        type=str,
        default=None,
        help="Merged ESM2 CSV (peptide_id or seqIndex + esm2_dim_*). Overrides config when set.",
    )
    ap.add_argument("--architectures", type=str, nargs="+", default=arch_list, choices=arch_list)
    ap.add_argument(
        "--post-mp-tabular-presets",
        "--feature-sets",
        type=str,
        nargs="+",
        default=default_post_mp,
        dest="post_mp_tabular_presets",
        metavar="PRESET",
        choices=preset_names,
        help=(
            "Which post-message-passing tabular presets to train (names from config "
            "post_message_passing_tabular_presets). Each preset selects Geo20 and/or QSAR12 "
            "columns concatenated after graph pooling. Default: "
            "default_post_mp_tabular_presets from config, or all preset names. "
            "--feature-sets is a deprecated alias."
        ),
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit model selections as ARCH:PRESET (e.g. "
            "gat:ESM gat:QSAR) where PRESET is a key from post_message_passing_tabular_presets. "
            "If set, --architectures, --post-mp-tabular-presets, and config "
            "default_post_mp_tabular_presets are ignored."
        ),
    )
    ap.add_argument("--epochs", type=int, default=training_defaults["epochs"])
    ap.add_argument("--batch_size", type=int, default=training_defaults["batch_size"])
    ap.add_argument("--lr", type=float, default=training_defaults["lr"])
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=training_defaults["patience"])
    ap.add_argument("--hidden_channels", type=int, default=training_defaults["hidden_channels"])
    ap.add_argument("--num_layers", type=int, default=training_defaults["num_layers"])
    ap.add_argument("--dropout", type=float, default=training_defaults["dropout"])
    ap.add_argument("--distance_threshold", type=float, default=training_defaults["distance_threshold"])
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=training_defaults["seed"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output_dir", type=str, default="results/gnn/ready_models")
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional subfolder name created under --output_dir for this run "
            "(default: timestamped run_YYYYmmdd_HHMMSS)."
        ),
    )
    ap.add_argument(
        "--no_tabular_robust_scaler",
        action="store_true",
        help="Disable per-block RobustScaler + block balancing on concatenated extras (raw CSV values).",
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=training_defaults["label_smoothing"],
        help="Cross-entropy label smoothing (0 disables). Reduces extreme logit margins / softmax saturation.",
    )
    ap.add_argument(
        "--logit_penalty",
        type=float,
        default=training_defaults["logit_penalty"],
        help="Weight on mean(logits^2) added to training loss (0 disables). Softens raw score collapse.",
    )
    ap.add_argument(
        "--esm2_residue_dir",
        type=str,
        default=None,
        help="Directory of {peptide_id}.pt per-residue ESM2 tensors (from esm_sequence_processor --per-residue-dir).",
    )
    ap.add_argument(
        "--esm2_hidden_dim",
        type=int,
        default=64,
        help="Project raw per-residue ESM2 (e.g. 1280-d) to this width before concatenating to node features.",
    )
    ap.add_argument(
        "--node-groups",
        type=str,
        default=None,
        help=(
            "Node feature blocks: comma tokens no_vae, no_onehot, no_pdb, no_esm2 (overrides "
            "configs/gnn_final_train.json node_feature_groups; no_esm2 disables per-residue ESM2 on graph nodes)."
        ),
    )
    args = ap.parse_args(argv_rest)
    return args, training_defaults, post_mp_tabular_presets, arch_list, node_groups_cfg


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _node_groups_as_dict(g: NodeFeatureGroups | None) -> dict[str, bool]:
    eff = g if g is not None else NodeFeatureGroups()
    return {
        "onehot": eff.onehot,
        "pdb_continuous": eff.pdb_continuous,
        "vae_table": eff.vae_table,
        "esm2_residue": eff.esm2_residue,
    }


def _collect_unique_training_pairs(args, post_mp_tabular_presets, architectures):
    selected_pairs = []
    if args.models:
        for item in args.models:
            if ":" not in item:
                raise ValueError(
                    f"Invalid --models entry '{item}'. Expected format ARCH:PRESET "
                    f"(e.g. gat:ESM)."
                )
            arch, preset_name = item.split(":", 1)
            arch = arch.strip().lower()
            preset_name = preset_name.strip()

            if arch not in architectures:
                raise ValueError(
                    f"Unknown architecture '{arch}' in --models. "
                    f"Choose from: {architectures}"
                )
            if preset_name not in post_mp_tabular_presets:
                raise ValueError(
                    f"Unknown post-MP tabular preset '{preset_name}' in --models. "
                    f"Choose from: {list(post_mp_tabular_presets.keys())}"
                )
            selected_pairs.append((arch, preset_name))
    else:
        for preset_name in args.post_mp_tabular_presets:
            for arch in args.architectures:
                selected_pairs.append((arch, preset_name))

    unique_pairs = []
    seen = set()
    for pair in selected_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def main():
    args, training_defaults, post_mp_tabular_presets, architectures, node_groups_cfg = parse_args()
    resolve_final_train_paths(args, training_defaults)
    set_seed(args.seed)
    device = get_device(args.device)

    node_feature_groups = (
        node_feature_groups_from_cli(args.node_groups)
        if args.node_groups
        else node_feature_groups_from_config_value(node_groups_cfg)
    )

    print("Device:", device, flush=True)
    print(
        "Node feature groups:",
        _node_groups_as_dict(node_feature_groups),
        f"(width {node_input_dim(node_feature_groups)})",
        flush=True,
    )

    base_out_dir = Path(args.output_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = base_out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    print("Output dir:", out_dir, flush=True)

    unique_pairs = _collect_unique_training_pairs(args, post_mp_tabular_presets, architectures)
    needs_merged_esm2 = wants_esm2_residue_nodes(node_feature_groups)

    if needs_merged_esm2:
        esm2_csv_path = args.esm2_csv if args.esm2_csv is not None else training_defaults.get("esm2_csv")
    else:
        esm2_csv_path = None

    merged_df, qsar_cols, esm2_cols = load_data_with_features(
        args.csv_path,
        args.qsar_csv,
        esm2_csv_path,
    )
    if needs_merged_esm2 and not esm2_cols:
        raise ValueError(
            "Per-residue ESM2 is enabled (node_feature_groups.esm2_residue) but no esm2_dim_* columns were merged. "
            "Check esm2_csv in configs/gnn_final_train.json or pass --esm2_csv."
        )
    rdir = Path(args.esm2_residue_dir) if args.esm2_residue_dir else None
    if needs_merged_esm2 and (rdir is None or not rdir.is_dir()):
        raise ValueError(
            f"Per-residue ESM2 directory missing or not a directory: {args.esm2_residue_dir!r}. "
            "Re-run ESM2 with models/esm_sequence_processor.py --per-residue-dir <dir> "
            "or use the data pipeline (writes work_dir/esm2_per_residue), or set node_feature_groups.esm2_residue "
            "to false / pass --node-groups no_esm2 to train without on-node ESM2."
        )
    print(f"Loaded training data: {len(merged_df)} samples")
    if needs_merged_esm2:
        print(f"Per-residue ESM2 dir: {rdir} (raw dim {len(esm2_cols)})", flush=True)
    else:
        print("Per-residue ESM2: off (node_feature_groups.esm2_residue false or --node-groups no_esm2).", flush=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "node_feature_groups": _node_groups_as_dict(node_feature_groups),
        "models": [],
    }

    for arch, preset_name in unique_pairs:
        f_cfg = post_mp_tabular_presets[preset_name]
        ckpt_path, metrics, platt_path = train_single_model(
            arch=arch,
            feature_name=preset_name,
            feature_cfg=f_cfg,
            df=merged_df,
            qsar_cols=qsar_cols,
            esm2_cols=esm2_cols,
            args=args,
            device=device,
            out_dir=out_dir,
            node_feature_groups=node_feature_groups,
        )
        entry = {
            "architecture": arch,
            "post_mp_tabular_preset": preset_name,
            "feature_set": preset_name,
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


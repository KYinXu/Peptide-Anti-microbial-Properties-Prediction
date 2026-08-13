#!/usr/bin/env python3
"""Run one SVM and one GNN over a processed peptide workspace.

A checkpoint base may contain both models. Explicit GNN and SVM paths override
the corresponding model from that base.

Updated version of compare_model_predictions.py script. Still relies on helpers from that
module but future plans are to isolate logic in new package and remove deprecated deps.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_model_predictions as inference
from configs.load_config import load_compare_models_config
from gnn.checkpoint_meta import load_peptide_gnn_meta
from gnn.data_utils import node_feature_groups_from_config_value
from gnn.extra_feature_scaler import load_extra_feature_scaler
from gnn_progress_csv import progress_csv_path
from peptide_pipeline.manifest_paths import resolve_generated_workspace

GNN_MODEL_NAME = "GNN"


class TeeStream:
    def __init__(self, console: TextIO, log: TextIO):
        self.console = console
        self.log = log

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


def start_run_log(workspace_input: str) -> Path:
    workspace = resolve_generated_workspace(workspace_input)
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_model_inference_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    print(f"Log: {log_path.resolve()}", flush=True)
    print("Command:", " ".join(sys.argv), flush=True)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SVM and one GNN on a pipeline workspace",
    )
    parser.add_argument(
        "workspace",
        help="Pipeline generated/ directory or its parent",
    )
    parser.add_argument(
        "--checkpoints-base",
        "--models-base",
        dest="checkpoints_base",
        help="Directory containing exactly one GNN checkpoint and one SVM bundle",
    )
    parser.add_argument(
        "--gnn-checkpoint",
        help="Explicit GNN .pt file; overrides --checkpoints-base",
    )
    parser.add_argument(
        "--architecture",
        choices=("gcn", "gat", "egnn"),
        default="gat",
    )
    parser.add_argument("--svm-pkl", "--svm_pkl", dest="svm_pkl")
    parser.add_argument("--svm-z-file", "--svm_z_file", dest="svm_z_file")
    parser.add_argument(
        "--svm-pipeline",
        "--svm_pipeline",
        dest="svm_pipeline",
        help="StandardScaler+SVC Pipeline joblib (no z-score file needed)",
    )
    parser.add_argument(
        "--svm-descriptor-csv",
        "--svm_descriptor_csv",
        dest="svm_descriptor_csv",
    )
    parser.add_argument(
        "--svm-only",
        action="store_true",
        help="Run SVM only (skip GNN checkpoint resolution and inference)",
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--checkpoint-every", type=int, default=4096)
    parser.add_argument("--fresh-progress", action="store_true")
    parser.add_argument("--no-gnn-platt", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--only-amp", action="store_true")
    parser.add_argument(
        "--output-csv",
        "--output_csv",
        dest="output_csv",
        help="Output path; defaults to <generated>/model_comparison_latest.csv",
    )
    return parser.parse_args()


def runtime_args(cli: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    args = argparse.Namespace(
        generated=cli.workspace,
        checkpoints_base=cli.checkpoints_base,
        architecture=cli.architecture,
        geo_csv=cfg["geo_csv"],
        pdb_dir=cfg["pdb_dir"],
        qsar_csv=cfg["qsar_csv"],
        esm2_csv=None,
        gnn_esm2_residue_dir=None,
        geometric_qsar_combined_csv=cfg["geometric_qsar_combined_csv"],
        legacy_pooled_esm_tabular=False,
        gnn_hidden=cfg["gnn_hidden"],
        gnn_layers=cfg["gnn_layers"],
        gnn_pooling=cfg["gnn_pooling"],
        svm_pkl=cfg["svm_pkl"],
        svm_z_file=cfg["svm_z_file"],
        svm_descriptor_csv=cfg["svm_descriptor_csv"],
    )
    return args


def configure_paths(
    cli: argparse.Namespace,
    args: argparse.Namespace,
) -> Path:
    workspace, _ = inference.apply_pipeline_generated_workspace(
        args,
        skip_svm_clear=cli.svm_descriptor_csv is not None,
    )
    if workspace is None:
        raise SystemExit("A pipeline workspace is required")
    if cli.checkpoints_base:
        inference.apply_checkpoints_base(args)
    for name in ("svm_pkl", "svm_z_file", "svm_descriptor_csv"):
        value = getattr(cli, name)
        if value:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    if cli.svm_pipeline:
        args.svm_pkl = str(Path(cli.svm_pipeline).expanduser().resolve())
        args.svm_z_file = None
    return workspace


def resolve_gnn_checkpoint(
    cli: argparse.Namespace,
) -> Path:
    if cli.gnn_checkpoint:
        raw = Path(cli.gnn_checkpoint).expanduser()
        candidates = [raw]
        if cli.checkpoints_base and not raw.is_absolute():
            candidates.append(Path(cli.checkpoints_base).expanduser() / raw)
        checkpoint = next((path.resolve() for path in candidates if path.is_file()), None)
        if checkpoint is not None:
            return checkpoint
        raise SystemExit(f"GNN checkpoint not found: {raw}")

    if not cli.checkpoints_base:
        raise SystemExit("Pass --gnn-checkpoint or --checkpoints-base")
    base = Path(cli.checkpoints_base).expanduser()
    roots = [base / folder for folder in ("", "gnn", "gnn_ready_models", "ready_models")]
    candidates = {
        path.resolve()
        for root in roots
        if root.is_dir()
        for path in root.glob("*.pt")
    }
    if len(candidates) != 1:
        found = ", ".join(sorted(path.name for path in candidates)) or "none"
        raise SystemExit(
            f"Expected exactly one GNN checkpoint under --checkpoints-base; found {found}. "
            "Select one with --gnn-checkpoint."
        )
    return candidates.pop()


def resolve_feature_columns(
    checkpoint: Path,
    args: argparse.Namespace,
    available_columns: dict[str, list[str]],
) -> list[str]:
    scaler_path = checkpoint.with_name(checkpoint.stem + "_tabular_scaler.joblib")
    if scaler_path.is_file():
        return list(load_extra_feature_scaler(str(scaler_path)).feature_cols)

    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    classifier_width = inference._classifier_input_dim_from_state_dict(state)
    pooled_width = inference._pool_dim_for_gnn_classifier(
        args.gnn_hidden,
        args.gnn_pooling,
    )
    feature_width = classifier_width - pooled_width
    if feature_width == 0:
        return []
    matches = {
        tuple(columns)
        for columns in available_columns.values()
        if len(columns) == feature_width
    }
    if len(matches) != 1:
        raise SystemExit(
            f"Cannot infer the {feature_width} GNN tabular feature columns from the workspace. "
            "Keep the checkpoint's _tabular_scaler.joblib beside its .pt file."
        )
    return list(matches.pop())


def validate_svm(args: argparse.Namespace) -> None:
    required = [args.svm_pkl, args.svm_descriptor_csv]
    if args.svm_z_file:
        required.append(args.svm_z_file)
    missing = [str(path) for path in required if not path or not Path(path).is_file()]
    if missing:
        raise SystemExit("Missing SVM input(s): " + ", ".join(missing))
    if not args.svm_z_file:
        try:
            import joblib
        except ImportError:
            from sklearn.externals import joblib as joblib
        est = joblib.load(args.svm_pkl)
        if not inference._is_sklearn_pipeline(est):
            raise SystemExit(
                "Legacy SVC checkpoints require --svm-z-file "
                "(or pass --svm-pipeline with a StandardScaler+SVC joblib)."
            )


def apply_checkpoint_metadata(
    cli: argparse.Namespace,
    args: argparse.Namespace,
    checkpoint: Path,
) -> None:
    metadata = load_peptide_gnn_meta(checkpoint)
    if not metadata:
        return
    cli.architecture = str(metadata.get("architecture", cli.architecture))
    args.architecture = cli.architecture
    args.gnn_hidden = int(metadata.get("hidden_channels", args.gnn_hidden))
    args.gnn_layers = int(metadata.get("num_layers", args.gnn_layers))
    args.gnn_pooling = str(metadata.get("pooling", args.gnn_pooling))
    print(f"Using GNN architecture and dimensions from {checkpoint.stem}_gnn_meta.json")


def run_svm(args: argparse.Namespace) -> dict:
    print("Running SVM...", flush=True)
    ids, pred, probability, distance = inference._load_svm_predictions(
        args.svm_descriptor_csv,
        args.svm_z_file,
        args.svm_pkl,
    )
    return {
        "ids": ids,
        "pred": pred,
        "probability": probability,
        "distance": distance,
    }


def run_gnn(
    cli: argparse.Namespace,
    args: argparse.Namespace,
    checkpoint: Path,
    master_csv: Path,
    feature_columns: list[str],
    model_name: str,
    workspace: Path,
    node_groups,
) -> dict:
    progress = progress_csv_path(
        workspace / "compare_progress",
        cli.architecture,
        model_name,
    )
    print(f"Running {cli.architecture.upper()} ({model_name})...", flush=True)
    values = inference._run_gnn_predictions(
        str(master_csv),
        args.pdb_dir,
        str(checkpoint),
        cli.architecture,
        args.gnn_hidden,
        args.gnn_layers,
        args.gnn_pooling,
        cli.batch_size,
        geometric_feature_cols=feature_columns,
        esm2_residue_dir=args.gnn_esm2_residue_dir,
        node_feature_groups=node_groups,
        use_gnn_platt=not cli.no_gnn_platt,
        progress_csv=progress,
        resume_progress=not cli.fresh_progress,
        checkpoint_every=cli.checkpoint_every,
        num_workers=cli.loader_workers,
    )
    ids, pred, probability, logit_amp, logit_nonamp, margin = values
    return {
        "ids": ids,
        "pred": pred,
        "probability": probability,
        "logit_amp": logit_amp,
        "logit_nonamp": logit_nonamp,
        "margin": margin,
    }


def run_models(cli: argparse.Namespace, svm_call, gnn_call) -> tuple[dict, dict]:
    if cli.sequential:
        return svm_call(), gnn_call()
    print("Running SVM and GNN concurrently.", flush=True)
    with ThreadPoolExecutor(max_workers=1) as executor:
        svm_future = executor.submit(svm_call)
        gnn_result = gnn_call()
        return svm_future.result(), gnn_result


def prediction_frame(name: str, result: dict, raw: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "peptide_id": pd.Series(result["ids"], dtype="string"),
            f"{name}_pred": result["pred"],
            f"{name}_prob_AMP": result["probability"],
            f"{name}_confidence": np.maximum(
                result["probability"],
                1.0 - result["probability"],
            ),
            **raw,
        }
    )
    frame["peptide_id"] = frame["peptide_id"].str.strip()
    return frame.drop_duplicates("peptide_id", keep="first")


def add_z_score(frame: pd.DataFrame, name: str, raw_column: str) -> None:
    raw = pd.to_numeric(frame[raw_column], errors="coerce")
    finite = raw[np.isfinite(raw)]
    if finite.empty:
        frame[f"{name}_score_z"] = np.nan
        return
    standard_deviation = finite.std(ddof=0)
    frame[f"{name}_score_z"] = (
        (raw - finite.mean()) / standard_deviation
        if standard_deviation > 0
        else np.where(raw.notna(), 0.0, np.nan)
    )


def build_output(
    args: argparse.Namespace,
    model_name: str,
    svm: dict,
    gnn: dict | None,
) -> pd.DataFrame:
    source = pd.read_csv(args.geo_csv)
    id_column = "peptide_id" if "peptide_id" in source else "name"
    base_columns = [id_column] + (["sequence"] if "sequence" in source else [])
    output = source[base_columns].rename(columns={id_column: "peptide_id"}).copy()
    output["peptide_id"] = output["peptide_id"].astype(str).str.strip()

    svm_frame = prediction_frame(
        "SVM",
        svm,
        {
            "SVM_hyperplane_distance": svm["distance"],
            "SVM_distance": svm["distance"],
        },
    )
    output = output.merge(svm_frame, on="peptide_id", how="left", sort=False)
    add_z_score(output, "SVM", "SVM_distance")
    model_names = ["SVM"]
    if gnn is not None:
        gnn_frame = prediction_frame(
            model_name,
            gnn,
            {
                f"{model_name}_logit_AMP": gnn["logit_amp"],
                f"{model_name}_logit_nonAMP": gnn["logit_nonamp"],
                f"{model_name}_logit_margin": gnn["margin"],
            },
        )
        output = output.merge(gnn_frame, on="peptide_id", how="left", sort=False)
        add_z_score(output, model_name, f"{model_name}_logit_margin")
        model_names.append(model_name)
    columns = ["peptide_id"] + (["sequence"] if "sequence" in output else [])
    columns += inference._ordered_result_column_names(model_names)
    return output[[column for column in columns if column in output]]


def main() -> int:
    cli = parse_args()
    start_run_log(cli.workspace)
    cfg = load_compare_models_config(None)
    args = runtime_args(cli, cfg)
    workspace = configure_paths(cli, args)
    validate_svm(args)

    if cli.svm_only:
        print("SVM-only mode: skipping GNN.", flush=True)
        svm = run_svm(args)
        output = build_output(args, GNN_MODEL_NAME, svm, None)
        if cli.only_amp:
            output = output[output["SVM_pred"] == 1]
    else:
        checkpoint = resolve_gnn_checkpoint(cli)
        apply_checkpoint_metadata(cli, args, checkpoint)
        master_csv, available_columns, _ = inference._build_gnn_feature_master(args, workspace)
        if master_csv is None:
            raise SystemExit("Workspace has no usable geometric feature CSV")
        feature_columns = resolve_feature_columns(checkpoint, args, available_columns)
        node_groups = node_feature_groups_from_config_value(cfg.get("node_feature_groups"))
        svm_call = lambda: run_svm(args)
        gnn_call = lambda: run_gnn(
            cli,
            args,
            checkpoint,
            master_csv,
            feature_columns,
            GNN_MODEL_NAME,
            workspace,
            node_groups,
        )
        svm, gnn = run_models(cli, svm_call, gnn_call)
        output = build_output(args, GNN_MODEL_NAME, svm, gnn)
        if cli.only_amp:
            output = output[
                (output["SVM_pred"] == 1) | (output["GNN_pred"] == 1)
            ]

    destination = (
        Path(cli.output_csv).expanduser()
        if cli.output_csv
        else workspace / "model_comparison_latest.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Saved {len(output):,} predictions to {destination.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

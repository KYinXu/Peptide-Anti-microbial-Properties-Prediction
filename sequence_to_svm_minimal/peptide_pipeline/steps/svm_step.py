from __future__ import annotations

import sys
from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.constants import CANONICAL_WINDOWS_SIDECAR
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_svm(ctx: RunContext, cfg: RunConfig) -> Path | None:
    if not cfg.with_svm:
        return None

    if not cfg.svm_aaindex or not cfg.svm_model_pkl or not cfg.svm_scaler_csv:
        raise ValueError(
            "--with-svm requires --svm-aaindex, --svm-model-pkl, --svm-scaler-csv",
        )
    svm_out = Path(cfg.svm_output_dir) if cfg.svm_output_dir else (ctx.work_dir / "svm_out")
    svm_out.mkdir(parents=True, exist_ok=True)
    seqs_path = ctx.canonical
    win_side = ctx.inputs_dir / CANONICAL_WINDOWS_SIDECAR
    if cfg.uses_windowing() and not cfg.window_expand_canonical and win_side.is_file():
        seqs_path = win_side
    cmd = [
        ctx.py,
        str(ctx.run_svm_script),
        "--seqs",
        str(seqs_path),
        "--aaindex",
        str(Path(cfg.svm_aaindex).resolve()),
        "--output-dir",
        str(svm_out),
        "--model-pkl",
        str(Path(cfg.svm_model_pkl).resolve()),
        "--scaler-csv",
        str(Path(cfg.svm_scaler_csv).resolve()),
    ]
    pred_file = svm_out / "descriptors_PREDICTIONS.csv"
    if not (cfg.skip_if_exists and pred_file.is_file()):
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "run_sequence_svm", "cmd": cmd})
    svm_preds = svm_out / "descriptors_PREDICTIONS.csv"
    if not cfg.dry_run and not svm_preds.is_file():
        print(
            f"Warning: expected SVM predictions at {svm_preds}; geometric build may omit SVM columns.",
            file=sys.stderr,
        )
    return svm_preds


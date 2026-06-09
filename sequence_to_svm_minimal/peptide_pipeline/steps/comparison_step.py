from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_compare_model_predictions(ctx: RunContext, cfg: RunConfig) -> None:
<<<<<<< HEAD
    """Run compare_model_predictions on the workspace (Platt on by default; see --no-gnn-platt)."""
    out_d = (
        Path(cfg.final_gnn_output_dir).resolve()
        if cfg.final_gnn_output_dir
        else (ctx.work_dir / "gnn_ready_models").resolve()
    )
    # If final-model training writes into timestamped subfolders, pick the newest folder with checkpoints.
    if out_d.is_dir() and not list(out_d.glob("*.pt")):
        subdirs = [p for p in out_d.iterdir() if p.is_dir()]
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for sd in subdirs:
            if list(sd.glob("*.pt")):
                out_d = sd
                break
=======
    """Run compare_model_predictions on the workspace (models from compare config / checkpoints-base)."""
    compare_models = cfg.compare_models
    if cfg.features_only and compare_models == "all":
        compare_models = "svm"
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    cmd = [
        ctx.py,
        str(ctx.compare_script),
        str(ctx.work_dir),
<<<<<<< HEAD
        "--gnn-checkpoints-dir",
        str(out_d),
        "--architecture",
        cfg.compare_gnn_architecture,
    ]
    if cfg.no_gnn_platt:
        cmd.append("--no-gnn-platt")
=======
        "--compare-models",
        compare_models,
    ]
    if cfg.checkpoints_base:
        cmd.extend(["--checkpoints-base", str(Path(cfg.checkpoints_base).resolve())])
    if cfg.train_final_gnn or compare_models != "svm":
        out_d = (
            Path(cfg.final_gnn_output_dir).resolve()
            if cfg.final_gnn_output_dir
            else (ctx.work_dir / "gnn_ready_models").resolve()
        )
        if out_d.is_dir() and not list(out_d.glob("*.pt")):
            subdirs = [p for p in out_d.iterdir() if p.is_dir()]
            subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for sd in subdirs:
                if list(sd.glob("*.pt")):
                    out_d = sd
                    break
        cmd.extend(["--gnn-checkpoints-dir", str(out_d)])
    if cfg.no_gnn_platt:
        cmd.append("--no-gnn-platt")
    if cfg.train_final_gnn:
        cmd.extend(["--architecture", cfg.compare_gnn_architecture])
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "compare_model_predictions", "cmd": cmd})

from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_legacy_gnn(ctx: RunContext, cfg: RunConfig, geo_csv: Path) -> None:
    m = ctx.manifest.get("geometric_features")
    if m and Path(m).resolve() != Path(geo_csv).resolve():
        raise RuntimeError(
            f"Manifest geometric_features {m!r} does not match training CSV {geo_csv!r}"
        )
    cmd = [
        ctx.py,
        str(ctx.legacy_train_script),
        str(ctx.work_dir),
        "--architecture",
        cfg.legacy_gnn_architecture,
        "--epochs",
        str(cfg.legacy_gnn_epochs),
    ]
    run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "run_gnn_training", "cmd": cmd})


def step_final_gnn(ctx: RunContext, cfg: RunConfig, geo_csv: Path) -> None:
    m = ctx.manifest.get("geometric_features")
    if m and Path(m).resolve() != Path(geo_csv).resolve():
        raise RuntimeError(
            f"Manifest geometric_features {m!r} does not match training CSV {geo_csv!r}"
        )
    out_d = cfg.final_gnn_output_dir or str(ctx.work_dir / "gnn_ready_models")
    cmd = [
        ctx.py,
        str(ctx.final_train_script),
        str(ctx.work_dir),
        "--output_dir",
        str(out_d),
    ]
    if cfg.final_gnn_epochs is not None:
        cmd.extend(["--epochs", str(cfg.final_gnn_epochs)])
    run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "run_gnn_train_final_models", "cmd": cmd})

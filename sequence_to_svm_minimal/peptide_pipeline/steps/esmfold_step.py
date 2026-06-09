from __future__ import annotations

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_esmfold(ctx: RunContext, cfg: RunConfig) -> None:
    if cfg.is_train_mode():
        if ctx.canonical_amp is None or ctx.canonical_decoy is None:
            raise RuntimeError("Train mode requires canonical AMP/DECOY inputs.")
        cmd = [
            ctx.py,
            str(ctx.esmfold_script),
            "--amp-file",
            str(ctx.canonical_amp),
            "--decoy-file",
            str(ctx.canonical_decoy),
            "--output",
            str(ctx.structures_dir),
            "--max-length",
            str(cfg.esmfold_max_length),
        ]
    else:
        cmd = [
            ctx.py,
            str(ctx.esmfold_script),
            "--amp-file",
            str(ctx.canonical),
            "--output",
            str(ctx.structures_dir),
            "--unlabeled",
            "--max-length",
            str(cfg.esmfold_max_length),
        ]
    if cfg.reset_esmfold:
        cmd.append("--reset")
    if cfg.esmfold_device:
        cmd.extend(["--device", cfg.esmfold_device])
    run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "esmfold", "cmd": cmd})

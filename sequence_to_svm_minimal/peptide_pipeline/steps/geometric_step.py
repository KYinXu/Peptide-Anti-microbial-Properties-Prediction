from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_geometric(ctx: RunContext, cfg: RunConfig, svm_preds: Path | None) -> None:
    cmd = [
        ctx.py,
        str(ctx.build_geo_script),
        "--pdb-dir",
        str(ctx.structures_dir),
        "--output",
        str(ctx.geo_csv),
    ]
    if cfg.is_blind_mode():
        cmd.append("--unlabeled")
    else:
        # In labeled mode, make results_log discovery explicit so label/sequence are attached.
        cmd.extend(["--results-log", str(ctx.structures_dir / "results_log.csv")])
    if svm_preds is not None and svm_preds.is_file():
        cmd.extend(["--svm-predictions", str(svm_preds)])
    skip = cfg.skip_if_exists and ctx.geo_csv.is_file()
    if not skip:
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "build_geometric_features", "cmd": cmd})

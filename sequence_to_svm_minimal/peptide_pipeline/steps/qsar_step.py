from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_qsar(ctx: RunContext, cfg: RunConfig, geo_for_qsar: Path) -> None:
    cmd = [
        ctx.py,
        str(ctx.gen_qsar_script),
        "--input",
        str(geo_for_qsar),
        "--output",
        str(ctx.qsar_csv),
    ]
    skip = cfg.skip_if_exists and ctx.qsar_csv.is_file()
    if not skip:
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "generate_qsar_features", "cmd": cmd})
    ctx.manifest["qsar12_descriptors"] = str(ctx.qsar_csv)

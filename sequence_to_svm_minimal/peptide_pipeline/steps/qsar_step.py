from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


<<<<<<< HEAD
<<<<<<< HEAD
def step_qsar(ctx: RunContext, cfg: RunConfig, geo_for_qsar: Path) -> None:
=======
def step_qsar(ctx: RunContext, cfg: RunConfig, input_path: Path) -> None:
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
def step_qsar(ctx: RunContext, cfg: RunConfig, geo_for_qsar: Path) -> None:
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    cmd = [
        ctx.py,
        str(ctx.gen_qsar_script),
        "--input",
<<<<<<< HEAD
<<<<<<< HEAD
        str(geo_for_qsar),
=======
        str(input_path),
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
        str(geo_for_qsar),
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        "--output",
        str(ctx.qsar_csv),
    ]
    skip = cfg.skip_if_exists and ctx.qsar_csv.is_file()
    if not skip:
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "generate_qsar_features", "cmd": cmd})
    ctx.manifest["qsar12_descriptors"] = str(ctx.qsar_csv)

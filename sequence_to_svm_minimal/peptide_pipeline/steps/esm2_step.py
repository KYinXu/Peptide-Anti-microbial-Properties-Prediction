from __future__ import annotations

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.esm2_post import add_peptide_id_to_esm2_csv
from peptide_pipeline.steps.exec import run_command


def _esm2_device(cfg: RunConfig) -> str:
    if cfg.esm2_device:
        return cfg.esm2_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def step_esm2(ctx: RunContext, cfg: RunConfig) -> None:
    dev = _esm2_device(cfg)
    per_residue = ctx.work_dir / "esm2_per_residue"
    cmd = [
        ctx.py,
        str(ctx.esm2_script),
        "--input",
        str(ctx.canonical),
        "--output",
        str(ctx.esm2_csv),
        "--mode",
        "embeddings",
        "--device",
        dev,
        "--max-length",
        str(cfg.esm2_max_length),
        "--per-residue-dir",
        str(per_residue),
    ]
    skip = cfg.skip_if_exists and ctx.esm2_csv.is_file()
    if not skip:
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
        if not cfg.dry_run and ctx.esm2_csv.is_file():
            add_peptide_id_to_esm2_csv(ctx.esm2_csv)
    ctx.manifest["steps"].append({"name": "esm_sequence_processor", "cmd": cmd})
    ctx.manifest["esm2_embeddings"] = str(ctx.esm2_csv)
    ctx.manifest["esm2_per_residue"] = str(per_residue)

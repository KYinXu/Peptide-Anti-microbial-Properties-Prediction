from __future__ import annotations

from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.exec import run_command


def step_cluster(ctx: RunContext, cfg: RunConfig) -> Path:
    """Return geometric CSV path to use for QSAR (clustered or base)."""
    if not cfg.with_cluster:
        return ctx.geo_csv

    if cfg.cluster_run_cdhit:
        cmd = [
            ctx.py,
            str(ctx.prepare_clusters_script),
            "--run-cdhit",
            "--input",
            str(ctx.geo_csv),
            "--output",
            str(ctx.geo_clustered),
            "--cdhit-path",
            cfg.cdhit_path,
            "--cdhit-identity",
            str(cfg.cdhit_identity),
        ]
    else:
        cmd = [
            ctx.py,
            str(ctx.prepare_clusters_script),
            "--simple-clusters",
            "--input",
            str(ctx.geo_csv),
            "--output",
            str(ctx.geo_clustered),
            "--identity",
            str(cfg.cluster_simple_identity),
        ]
    skip = cfg.skip_if_exists and ctx.geo_clustered.is_file()
    if not skip:
        run_command(cmd, root=ctx.root, dry_run=cfg.dry_run)
    ctx.manifest["steps"].append({"name": "prepare_clusters", "cmd": cmd})
    return ctx.geo_clustered

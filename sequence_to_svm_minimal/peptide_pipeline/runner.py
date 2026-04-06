"""Execute the default data pipeline from RunConfig."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext
from peptide_pipeline.steps.cluster_step import step_cluster
from peptide_pipeline.steps.esm2_step import step_esm2
from peptide_pipeline.steps.esmfold_step import step_esmfold
from peptide_pipeline.steps.geometric_step import step_geometric
from peptide_pipeline.steps.normalize import normalize_to_canonical
from peptide_pipeline.steps.qsar_step import step_qsar
from peptide_pipeline.steps.svm_step import step_svm
from peptide_pipeline.steps.train_step import step_final_gnn, step_legacy_gnn


def run_pipeline(cfg: RunConfig) -> int:
    inp = cfg.input_path.resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    ctx = RunContext.from_config(cfg, py_executable=sys.executable)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    ctx.inputs_dir.mkdir(parents=True, exist_ok=True)
    print("Workspace:", ctx.work_dir, flush=True)

    if not cfg.dry_run:
        st = normalize_to_canonical(inp, ctx.canonical, min_len=cfg.min_len, max_len=cfg.max_len)
        ctx.manifest["normalization"] = st
        ctx.manifest["canonical_seqs"] = str(ctx.canonical)
        if st["n_written"] == 0:
            print("No sequences after normalization.", file=sys.stderr)
            return 1
    else:
        ctx.manifest["canonical_seqs"] = str(ctx.canonical)
        ctx.manifest["normalization"] = {"dry_run": True, "note": "normalize skipped; commands show intended paths"}

    skip_esmfold = cfg.skip_if_exists and (ctx.structures_dir / "results_log.csv").is_file()
    if not skip_esmfold:
        step_esmfold(ctx, cfg)

    try:
        svm_preds = step_svm(ctx, cfg)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    step_geometric(ctx, cfg, svm_preds)

    geo_for_qsar = step_cluster(ctx, cfg)

    if not cfg.skip_qsar:
        step_qsar(ctx, cfg, geo_for_qsar)

    if not cfg.skip_esm2:
        step_esm2(ctx, cfg)

    ctx.manifest["geometric_features"] = str(geo_for_qsar if cfg.with_cluster else ctx.geo_csv)
    ctx.manifest["structures_dir"] = str(ctx.structures_dir)

    geo_train = ctx.geo_clustered if cfg.with_cluster else ctx.geo_csv

    manifest_path = ctx.work_dir / "pipeline_manifest.json"
    if not cfg.dry_run and (cfg.train_legacy_gnn or cfg.train_final_gnn):
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(ctx.manifest, f, indent=2)

    if cfg.train_legacy_gnn:
        step_legacy_gnn(ctx, cfg, geo_train)

    if cfg.train_final_gnn:
        if cfg.skip_qsar or cfg.skip_esm2:
            print(
                "--train-final-gnn requires QSAR and ESM2 outputs (do not use --skip-qsar / --skip-esm2).",
                file=sys.stderr,
            )
            return 1
        step_final_gnn(ctx, cfg, geo_train)

    if not cfg.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(ctx.manifest, f, indent=2)
        print(f"Wrote manifest: {manifest_path}")

    print("\nDone. Outputs under:", ctx.work_dir)
    return 0

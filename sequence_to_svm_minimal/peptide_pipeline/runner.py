"""Execute the default data pipeline from RunConfig."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from peptide_pipeline.config import RunConfig
from peptide_pipeline.constants import CANONICAL_WINDOWS_SIDECAR
from peptide_pipeline.context import RunContext
from peptide_pipeline.inference_samples import build_inference_samples, inference_sequences_path
from peptide_pipeline.steps.cluster_step import step_cluster
from peptide_pipeline.steps.esm2_step import step_esm2
from peptide_pipeline.steps.esmfold_step import step_esmfold
from peptide_pipeline.steps.geometric_step import step_geometric
from peptide_pipeline.sequence_io import read_sequence_records, write_canonical
from peptide_pipeline.steps.normalize import normalize_to_canonical
from peptide_pipeline.steps.qsar_step import step_qsar
from peptide_pipeline.steps.svm_step import step_svm
from peptide_pipeline.steps.comparison_step import step_compare_model_predictions
from peptide_pipeline.steps.train_step import step_final_gnn, step_legacy_gnn
from peptide_pipeline.steps.window_aggregate_step import step_window_aggregate
from peptide_pipeline.windowing import expand_records_to_windows


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
        if cfg.is_train_mode() and cfg.uses_windowing():
            print("--mode train does not support --window-min-len/--window-max-len.", file=sys.stderr)
            return 2

        if cfg.is_train_mode():
            assert ctx.canonical_amp is not None and ctx.canonical_decoy is not None
            assert ctx.amp_input_path is not None and ctx.decoy_input_path is not None

            st_amp = normalize_to_canonical(
                ctx.amp_input_path, ctx.canonical_amp, min_len=cfg.min_len, max_len=cfg.max_len
            )
            st_decoy = normalize_to_canonical(
                ctx.decoy_input_path, ctx.canonical_decoy, min_len=cfg.min_len, max_len=cfg.max_len
            )
            ctx.manifest["canonical_amp_seqs"] = str(ctx.canonical_amp)
            ctx.manifest["canonical_decoy_seqs"] = str(ctx.canonical_decoy)
            # Some downstream steps (ESM2, optional SVM) use a single canonical file.
            # In train mode, also produce the combined file for compatibility.
            combined = []
            combined.extend(read_sequence_records(ctx.canonical_amp))
            combined.extend(read_sequence_records(ctx.canonical_decoy))
            write_canonical(ctx.canonical, combined)
            ctx.manifest["canonical_seqs"] = str(ctx.canonical)
            ctx.manifest["normalization"] = {
                "mode": "train",
                "amp": st_amp,
                "decoy": st_decoy,
            }
            if st_amp["n_written"] == 0 or st_decoy["n_written"] == 0:
                print("No sequences after normalization (amp or decoy).", file=sys.stderr)
                return 1
        elif cfg.uses_windowing():
            assert cfg.window_min_len is not None and cfg.window_max_len is not None
            if cfg.window_expand_canonical:
                records = [(pid, seq) for pid, seq in read_sequence_records(inp) if seq]
            else:
                st_norm = normalize_to_canonical(
                    inp, ctx.canonical, min_len=cfg.min_len, max_len=cfg.max_len
                )
                if st_norm["n_written"] == 0:
                    print("No sequences after normalization.", file=sys.stderr)
                    return 1
                records = [(pid, seq) for pid, seq in read_sequence_records(ctx.canonical) if seq]
            expanded, windows = expand_records_to_windows(
                records,
                min_len=cfg.window_min_len,
                max_len=cfg.window_max_len,
                stride=cfg.window_stride,
            )
            if not expanded:
                print(
                    "No sliding windows produced (check parent lengths vs --window-min-len).",
                    file=sys.stderr,
                )
                return 1
            if cfg.window_expand_canonical:
                write_canonical(ctx.canonical, expanded)
            else:
                # Parents already in ctx.canonical from normalize_to_canonical.
                win_side = ctx.inputs_dir / CANONICAL_WINDOWS_SIDECAR
                write_canonical(win_side, expanded)
            wpath = ctx.inputs_dir / "window_map.csv"
            with open(wpath, "w", newline="", encoding="utf-8") as wf:
                writer = csv.DictWriter(
                    wf,
                    fieldnames=[
                        "seqIndex",
                        "peptide_id",
                        "window_id",
                        "parent_id",
                        "start",
                        "length",
                        "sequence",
                    ],
                )
                writer.writeheader()
                for w in windows:
                    writer.writerow(
                        {
                            "seqIndex": w.seq_index,
                            "peptide_id": w.peptide_id,
                            "window_id": w.window_id,
                            "parent_id": w.parent_id,
                            "start": w.start,
                            "length": w.length,
                            "sequence": w.window_seq,
                        }
                    )
            ctx.manifest["canonical_seqs"] = str(ctx.canonical)
            if cfg.window_expand_canonical:
                ctx.manifest["normalization"] = {
                    "format": "windowed_txt",
                    "n_written": len(expanded),
                    "n_parents": len({w.parent_id for w in windows}),
                }
            else:
                ctx.manifest["normalization"] = {
                    "format": "windowed_txt_parents",
                    "n_written": len(records),
                    "n_parents": len(records),
                    "n_window_rows": len(expanded),
                }
            ctx.manifest["windowing"] = {
                "window_map": str(wpath),
                "n_windows": len(windows),
                "min_len": cfg.window_min_len,
                "max_len": cfg.window_max_len,
                "stride": cfg.window_stride,
                "expand_canonical": cfg.window_expand_canonical,
                "esmfold_sequences": "windows" if cfg.window_expand_canonical else "parents",
            }
            if not cfg.window_expand_canonical:
                ctx.manifest["windowing"]["canonical_windows_expanded"] = str(
                    (ctx.inputs_dir / CANONICAL_WINDOWS_SIDECAR).resolve()
                )
        else:
            st = normalize_to_canonical(inp, ctx.canonical, min_len=cfg.min_len, max_len=cfg.max_len)
            ctx.manifest["normalization"] = st
            ctx.manifest["canonical_seqs"] = str(ctx.canonical)
            if st["n_written"] == 0:
                print("No sequences after normalization.", file=sys.stderr)
                return 1
    else:
        if cfg.is_train_mode():
            assert ctx.canonical_amp is not None and ctx.canonical_decoy is not None
            ctx.manifest["canonical_amp_seqs"] = str(ctx.canonical_amp)
            ctx.manifest["canonical_decoy_seqs"] = str(ctx.canonical_decoy)
            ctx.manifest["canonical_seqs"] = str(ctx.canonical)
            ctx.manifest["normalization"] = {
                "dry_run": True,
                "mode": "train",
                "note": "normalize skipped; commands show intended paths",
            }
        else:
            ctx.manifest["canonical_seqs"] = str(ctx.canonical)
            if cfg.uses_windowing():
                ctx.manifest["normalization"] = {
                    "dry_run": True,
                    "note": "windowed canonical skipped; commands show intended paths",
                }
            else:
                ctx.manifest["normalization"] = {"dry_run": True, "note": "normalize skipped; commands show intended paths"}

    if cfg.features_only:
        if cfg.compare_models not in ("all", "svm"):
            print("--features-only supports --compare-models all or svm.", file=sys.stderr)
            return 2
        seq_input = inference_sequences_path(ctx, cfg)
        if not cfg.dry_run:
            try:
                build_inference_samples(ctx, cfg)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 1
        ctx.manifest["geometric_features"] = str(ctx.geo_csv)
        ctx.manifest["inference_samples"] = str(ctx.geo_csv)
        if not cfg.skip_qsar:
            step_qsar(ctx, cfg, seq_input)
        if cfg.run_compare:
            step_compare_model_predictions(ctx, cfg)
        if not cfg.dry_run:
            step_window_aggregate(ctx, cfg, None)
            ctx.manifest["features_only"] = True
            ctx.manifest["work_dir"] = str(ctx.work_dir.resolve())
            ctx.manifest["geometric_features"] = str(ctx.geo_csv.resolve())
            if ctx.qsar_csv.is_file():
                ctx.manifest["qsar12_descriptors"] = str(ctx.qsar_csv.resolve())
            manifest_path = ctx.work_dir / "pipeline_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(ctx.manifest, f, indent=2)
            print(f"Wrote manifest: {manifest_path}")
        print("\nDone. Outputs under:", ctx.work_dir)
        return 0

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
        if not cfg.skip_model_comparison:
            step_compare_model_predictions(ctx, cfg)

    if not cfg.dry_run:
        step_window_aggregate(ctx, cfg, svm_preds)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(ctx.manifest, f, indent=2)
        print(f"Wrote manifest: {manifest_path}")

    print("\nDone. Outputs under:", ctx.work_dir)
    return 0

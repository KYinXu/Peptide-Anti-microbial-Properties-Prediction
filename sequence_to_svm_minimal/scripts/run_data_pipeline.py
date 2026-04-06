#!/usr/bin/env python3
"""
Single-entry orchestrator: txt/FASTA sequences → ESMFold → geometric features → QSAR-12 → ESM-2.

Default path matches unlabeled ESMFold + build_geometric_features --unlabeled.
Optional: clustering, SVM merge (rebuild geometry), legacy/final GNN training.

Run from repository root sequence_to_svm_minimal:
  python scripts/run_data_pipeline.py --input path/to/seqs.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts" / "data_generation"))
from pipeline_input_normalize import normalize_to_canonical  # noqa: E402


def _subprocess_env() -> dict[str, str]:
    """
    Child processes (ESMFold, torch, numpy): MKL_THREADING_LAYER=INTEL often conflicts
    with libgomp on Linux/WSL; GNU is compatible with mixed OpenMP stacks.
    """
    env = os.environ.copy()
    cur = (env.get("MKL_THREADING_LAYER") or "").strip().upper()
    if cur in ("", "INTEL"):
        env["MKL_THREADING_LAYER"] = "GNU"
    return env


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=_subprocess_env())


def _add_peptide_id_to_esm2_csv(csv_path: Path, *, esmfold_id_prefix: str = "SEQ") -> None:
    """Add peptide_id column so merge with geometric CSV (SEQ_<idx>) works."""
    df = pd.read_csv(csv_path)
    if "peptide_id" in df.columns:
        return
    if "seqIndex" not in df.columns:
        raise ValueError(f"{csv_path}: expected seqIndex column")
    sid = df["seqIndex"].astype(str).str.strip()
    df["peptide_id"] = sid.apply(lambda s: s if s.startswith(f"{esmfold_id_prefix}_") else f"{esmfold_id_prefix}_{s}")
    df.to_csv(csv_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run sequence → structure → features pipeline (see DATA_PROCESSING.md).")
    ap.add_argument("--input", "-i", type=str, required=True, help="Txt-like sequences or FASTA")
    ap.add_argument(
        "--work-dir",
        "-w",
        type=str,
        default=None,
        help="Output workspace (default: <input_dir>/generated/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands only (skips writing canonical_seqs and manifest; input not parsed for length)",
    )
    ap.add_argument("--skip-if-exists", action="store_true", help="Skip steps whose outputs already exist")
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--reset-esmfold", action="store_true", help="Pass --reset to ESMFold checkpoint")
    ap.add_argument("--esmfold-max-length", type=int, default=200)
    ap.add_argument("--esmfold-device", type=str, choices=["cuda", "cpu"], default=None)

    ap.add_argument("--skip-qsar", action="store_true")
    ap.add_argument("--skip-esm2", action="store_true")
    ap.add_argument("--with-cluster", action="store_true")
    ap.add_argument("--cluster-simple-identity", type=float, default=0.80, help="--identity for prepare_clusters --simple-clusters")
    ap.add_argument("--cluster-run-cdhit", action="store_true", help="Use prepare_clusters --run-cdhit instead of simple")
    ap.add_argument("--cdhit-path", type=str, default="cd-hit")
    ap.add_argument("--cdhit-identity", type=float, default=0.40)

    ap.add_argument("--with-svm", action="store_true")
    ap.add_argument("--svm-aaindex", type=str, default=None)
    ap.add_argument("--svm-model-pkl", type=str, default=None)
    ap.add_argument("--svm-scaler-csv", type=str, default=None)
    ap.add_argument("--svm-output-dir", type=str, default=None, help="Default: work_dir/svm_out")

    ap.add_argument("--esm2-device", type=str, choices=["cuda", "cpu"], default=None)
    ap.add_argument("--esm2-max-length", type=int, default=400)

    ap.add_argument("--train-legacy-gnn", action="store_true")
    ap.add_argument("--legacy-gnn-architecture", type=str, default="gcn", choices=["gcn", "gat", "egnn"])
    ap.add_argument("--legacy-gnn-epochs", type=int, default=100)

    ap.add_argument("--train-final-gnn", action="store_true")
    ap.add_argument("--final-gnn-output-dir", type=str, default=None)
    ap.add_argument("--final-gnn-epochs", type=int, default=None)

    args = ap.parse_args()

    inp = Path(args.input).resolve()
    if not inp.is_file():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 1

    default_work = inp.parent / "generated"
    work = Path(args.work_dir).resolve() if args.work_dir else default_work
    work.mkdir(parents=True, exist_ok=True)
    print("Workspace:", work, flush=True)
    inputs_dir = work / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    canonical = inputs_dir / "canonical_seqs.txt"

    manifest: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(inp),
        "work_dir": str(work),
        "steps": [],
    }

    # 1) Normalize
    if not args.dry_run:
        st = normalize_to_canonical(inp, canonical, min_len=args.min_len, max_len=args.max_len)
        manifest["normalization"] = st
        manifest["canonical_seqs"] = str(canonical)
        if st["n_written"] == 0:
            print("No sequences after normalization.", file=sys.stderr)
            return 1
    else:
        manifest["canonical_seqs"] = str(canonical)
        manifest["normalization"] = {"dry_run": True, "note": "normalize skipped; commands show intended paths"}

    py = sys.executable
    structures_dir = work / "structures"
    geo_csv = work / "geometric_features.csv"
    geo_clustered = work / "geometric_features_clustered.csv"
    qsar_csv = work / "qsar12_descriptors.csv"
    esm2_csv = work / "esm2_embeddings.csv"

    esmfold = ROOT / "models" / "run_esmfold_peptides.py"
    build_geo = ROOT / "scripts" / "data_generation" / "build_geometric_features.py"
    gen_qsar = ROOT / "scripts" / "data_generation" / "generate_qsar_features.py"
    esm2_script = ROOT / "models" / "esm_sequence_processor.py"
    prepare_cl = ROOT / "nn_pipeline" / "prepare_clusters.py"
    run_svm = ROOT / "scripts" / "data_generation" / "run_sequence_svm.py"
    legacy_train = ROOT / "scripts" / "run_gnn_training.py"
    final_train = ROOT / "scripts" / "run_gnn_train_final_models.py"

    def exists_skip(path: Path) -> bool:
        return args.skip_if_exists and path.is_file()

    # 2) ESMFold (resume via checkpoint inside script; optional skip if outputs present)
    skip_esmfold = args.skip_if_exists and (structures_dir / "results_log.csv").is_file()
    if not skip_esmfold:
        ef_cmd = [
            py,
            str(esmfold),
            "--amp-file",
            str(canonical),
            "--output",
            str(structures_dir),
            "--unlabeled",
            "--max-length",
            str(args.esmfold_max_length),
        ]
        if args.reset_esmfold:
            ef_cmd.append("--reset")
        if args.esmfold_device:
            ef_cmd.extend(["--device", args.esmfold_device])
        _run(ef_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "esmfold", "cmd": ef_cmd})

    # 3) Geometric features (base)
    svm_preds_path: Path | None = None
    if args.with_svm:
        if not args.svm_aaindex or not args.svm_model_pkl or not args.svm_scaler_csv:
            print("--with-svm requires --svm-aaindex, --svm-model-pkl, --svm-scaler-csv", file=sys.stderr)
            return 1
        svm_out = Path(args.svm_output_dir) if args.svm_output_dir else (work / "svm_out")
        svm_out.mkdir(parents=True, exist_ok=True)
        svm_cmd = [
            py,
            str(run_svm),
            "--seqs",
            str(canonical),
            "--aaindex",
            str(Path(args.svm_aaindex).resolve()),
            "--output-dir",
            str(svm_out),
            "--model-pkl",
            str(Path(args.svm_model_pkl).resolve()),
            "--scaler-csv",
            str(Path(args.svm_scaler_csv).resolve()),
        ]
        pred_file = svm_out / "descriptors_PREDICTIONS.csv"
        if not (args.skip_if_exists and pred_file.is_file()):
            _run(svm_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "run_sequence_svm", "cmd": svm_cmd})
        svm_preds_path = svm_out / "descriptors_PREDICTIONS.csv"
        if not args.dry_run and not svm_preds_path.is_file():
            print(
                f"Warning: expected SVM predictions at {svm_preds_path}; geometric build may omit SVM columns.",
                file=sys.stderr,
            )

    geo_cmd = [
        py,
        str(build_geo),
        "--pdb-dir",
        str(structures_dir),
        "--output",
        str(geo_csv),
        "--unlabeled",
    ]
    if svm_preds_path and svm_preds_path.is_file():
        geo_cmd.extend(["--svm-predictions", str(svm_preds_path)])
    if not exists_skip(geo_csv):
        _run(geo_cmd, dry_run=args.dry_run)
    manifest["steps"].append({"name": "build_geometric_features", "cmd": geo_cmd})

    geo_for_qsar = geo_csv
    if args.with_cluster:
        if args.cluster_run_cdhit:
            cl_cmd = [
                py,
                str(prepare_cl),
                "--run-cdhit",
                "--input",
                str(geo_csv),
                "--output",
                str(geo_clustered),
                "--cdhit-path",
                args.cdhit_path,
                "--cdhit-identity",
                str(args.cdhit_identity),
            ]
        else:
            cl_cmd = [
                py,
                str(prepare_cl),
                "--simple-clusters",
                "--input",
                str(geo_csv),
                "--output",
                str(geo_clustered),
                "--identity",
                str(args.cluster_simple_identity),
            ]
        if not exists_skip(geo_clustered):
            _run(cl_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "prepare_clusters", "cmd": cl_cmd})
        geo_for_qsar = geo_clustered

    if not args.skip_qsar:
        qsar_cmd = [
            py,
            str(gen_qsar),
            "--input",
            str(geo_for_qsar),
            "--output",
            str(qsar_csv),
        ]
        if not exists_skip(qsar_csv):
            _run(qsar_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "generate_qsar_features", "cmd": qsar_cmd})
        manifest["qsar12_descriptors"] = str(qsar_csv)

    if not args.skip_esm2:
        if args.esm2_device:
            dev = args.esm2_device
        else:
            try:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        esm2_cmd = [
            py,
            str(esm2_script),
            "--input",
            str(canonical),
            "--output",
            str(esm2_csv),
            "--mode",
            "embeddings",
            "--device",
            dev,
            "--max-length",
            str(args.esm2_max_length),
        ]
        if not exists_skip(esm2_csv):
            _run(esm2_cmd, dry_run=args.dry_run)
            if not args.dry_run and esm2_csv.is_file():
                _add_peptide_id_to_esm2_csv(esm2_csv)
        manifest["steps"].append({"name": "esm_sequence_processor", "cmd": esm2_cmd})
        manifest["esm2_embeddings"] = str(esm2_csv)

    manifest["geometric_features"] = str(geo_for_qsar if args.with_cluster else geo_csv)
    manifest["structures_dir"] = str(structures_dir)

    if args.train_legacy_gnn:
        csv_p = geo_clustered if args.with_cluster else geo_csv
        lt_cmd = [
            py,
            str(legacy_train),
            "--csv_path",
            str(csv_p),
            "--pdb_dir",
            str(structures_dir),
            "--architecture",
            args.legacy_gnn_architecture,
            "--epochs",
            str(args.legacy_gnn_epochs),
        ]
        _run(lt_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "run_gnn_training", "cmd": lt_cmd})

    if args.train_final_gnn:
        if args.skip_qsar or args.skip_esm2:
            print("--train-final-gnn requires QSAR and ESM2 outputs (do not use --skip-qsar / --skip-esm2).", file=sys.stderr)
            return 1
        csv_p = geo_clustered if args.with_cluster else geo_csv
        out_d = args.final_gnn_output_dir or str(work / "gnn_ready_models")
        ft_cmd = [
            py,
            str(final_train),
            "--csv_path",
            str(csv_p),
            "--pdb_dir",
            str(structures_dir),
            "--qsar_csv",
            str(qsar_csv),
            "--esm2_csv",
            str(esm2_csv),
            "--output_dir",
            str(out_d),
        ]
        if args.final_gnn_epochs is not None:
            ft_cmd.extend(["--epochs", str(args.final_gnn_epochs)])
        _run(ft_cmd, dry_run=args.dry_run)
        manifest["steps"].append({"name": "run_gnn_train_final_models", "cmd": ft_cmd})

    manifest_path = work / "pipeline_manifest.json"
    if not args.dry_run:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest: {manifest_path}")

    print("\nDone. Outputs under:", work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

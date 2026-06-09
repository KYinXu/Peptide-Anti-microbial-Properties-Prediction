#!/usr/bin/env python3
"""
CLI for the modular data pipeline (peptide_pipeline package).

Run from sequence_to_svm_minimal:
  python scripts/run_data_pipeline.py --input path/to/seqs.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.load_config import merge_pipeline_config_paths
from peptide_pipeline.config import RunConfig
from peptide_pipeline.runner import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run sequence -> structure -> features pipeline (see DATA_PROCESSING.md).",
        epilog=(
            "JSON presets: use `--config PATH` / `-c PATH` before other flags (repeatable; later files override). "
            "CLI flags override merged JSON. See configs/pipeline_defaults.json and configs/windowed_*.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["blind", "train"],
        help=(
            "Pipeline mode. blind: single unlabeled input (--input). "
            "train: labeled AMP+DECOY inputs (--amp-input/--decoy-input). "
            "If omitted, inferred from provided inputs."
        ),
    )
<<<<<<< HEAD
    ap.add_argument("--input", "-i", type=str, default=None, help="(blind) Txt-like sequences or FASTA")
    ap.add_argument("--amp-input", type=str, default=None, help="(train) AMP sequences (txt-like or FASTA)")
    ap.add_argument("--decoy-input", type=str, default=None, help="(train) Decoy sequences (txt-like or FASTA)")
=======
    ap.add_argument("input_positional", nargs="?", default=None, help="(blind) Optional alias for --input")
    ap.add_argument("--input", "-i", type=str, default=None, help="(blind) TXT-like, FASTA, or CSV sequences")
    ap.add_argument("--amp-input", type=str, default=None, help="(train) AMP sequences (TXT-like, FASTA, or CSV)")
    ap.add_argument("--decoy-input", type=str, default=None, help="(train) Decoy sequences (TXT-like, FASTA, or CSV)")
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
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
    cluster_g = ap.add_mutually_exclusive_group()
    cluster_g.add_argument(
        "--with-cluster",
        action="store_true",
        default=None,
        dest="with_cluster",
        help="Enable clustering step (default: auto-enabled in --mode train; off in --mode blind).",
    )
    cluster_g.add_argument(
        "--no-cluster",
        action="store_false",
        default=None,
        dest="with_cluster",
        help="Disable clustering step (overrides the --mode train default).",
    )
    ap.add_argument(
        "--cluster-simple-identity",
        type=float,
        default=0.80,
        help="--identity for prepare_clusters --simple-clusters",
    )
    ap.add_argument("--cluster-run-cdhit", action="store_true", help="Use prepare_clusters --run-cdhit instead of simple")
    ap.add_argument("--cdhit-path", type=str, default="cd-hit")
    ap.add_argument("--cdhit-identity", type=float, default=0.40)

<<<<<<< HEAD
    ap.add_argument("--with-svm", action="store_true")
    ap.add_argument("--svm-aaindex", type=str, default=None)
    ap.add_argument("--svm-model-pkl", type=str, default=None)
    ap.add_argument("--svm-scaler-csv", type=str, default=None)
    ap.add_argument("--svm-output-dir", type=str, default=None, help="Default: work_dir/svm_out")
=======
    ap.add_argument(
        "--features-only",
        "--svm-only",
        action="store_true",
        dest="features_only",
        help=(
            "Prepare inference inputs only: normalize/window, write geometric_features.csv "
            "(peptide_id + sequence), and qsar12_descriptors.csv. Skips ESMFold, ESM2, and GNN. "
            "Run compare_model_predictions.py on the workspace for model scoring "
            "(or pass --run-compare)."
        ),
    )
    ap.add_argument(
        "--run-compare",
        action="store_true",
        help=(
            "After the pipeline, run compare_model_predictions.py on the workspace. "
            "Model paths come from configs/compare_models.json and/or --checkpoints-base."
        ),
    )
    ap.add_argument(
        "--checkpoints-base",
        type=str,
        default=None,
        help=(
            "Forwarded to compare_model_predictions: directory with svm_qsar12_model.pkl, "
            "svm_qsar12_zscores.txt, and optional GNN .pt files."
        ),
    )
    ap.add_argument(
        "--compare-models",
        type=str,
        default="all",
        choices=["all", "svm", "gnn"],
        help="Forwarded to compare_model_predictions (default: svm when using --features-only).",
    )
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)

    ap.add_argument("--esm2-device", type=str, choices=["cuda", "cpu"], default=None)
    ap.add_argument("--esm2-max-length", type=int, default=400)

    ap.add_argument("--train-legacy-gnn", action="store_true")
    ap.add_argument("--legacy-gnn-architecture", type=str, default="gcn", choices=["gcn", "gat", "egnn"])
    ap.add_argument("--legacy-gnn-epochs", type=int, default=100)

    ap.add_argument("--train-final-gnn", action="store_true")
    ap.add_argument("--final-gnn-output-dir", type=str, default=None)
    ap.add_argument("--final-gnn-epochs", type=int, default=None)
    ap.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help=(
            "With --train-final-gnn, skip compare_model_predictions.py at the end "
            "(default: run comparison on the workspace using Platt when *_platt.json exists)."
        ),
    )
    ap.add_argument(
        "--no-gnn-platt",
        action="store_true",
        help="Forwarded to compare_model_predictions: use GNN softmax instead of Platt scaling.",
    )
    ap.add_argument(
        "--compare-gnn-architecture",
        type=str,
        default="gat",
        choices=["gcn", "gat", "egnn"],
        help="GNN backbone for the post-training model comparison step (default: gat).",
    )
    ap.add_argument(
        "--window-min-len",
        type=int,
        default=None,
        dest="window_min_len",
        help="With --window-max-len: sliding window minimum length (aa) per parent sequence.",
    )
    ap.add_argument(
        "--window-max-len",
        type=int,
        default=None,
        dest="window_max_len",
        help="With --window-min-len: sliding window maximum length (aa) per parent sequence.",
    )
    ap.add_argument(
        "--window-stride",
        type=int,
        default=1,
        dest="window_stride",
        help="Sliding window stride (default: 1). Requires --window-min-len and --window-max-len.",
    )
    ap.add_argument(
        "--window-expand-canonical",
        action="store_true",
        help=(
            "With window min/max: fold each window separately (canonical_seqs.txt = all windows). "
            "Default without this flag: ESMFold/ESM2 on full parents in canonical_seqs.txt; "
            "window_map.csv plus inputs/canonical_windows_expanded.txt list windows (e.g. for SVM)."
        ),
    )
    return ap


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        "-c",
        action="append",
        default=None,
        metavar="PATH",
        help="JSON defaults for pipeline flags (repeatable; later files override earlier).",
    )
    pre_args, rest = pre.parse_known_args()
    defaults = merge_pipeline_config_paths(pre_args.config)
    parser = _build_parser()
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(rest)
    wm, wx = args.window_min_len, args.window_max_len
    if (wm is None) != (wx is None):
        print(
            "Use both --window-min-len and --window-max-len together (or omit both).",
            file=sys.stderr,
        )
        return 2
    if wm is not None:
        if args.window_stride <= 0 or wm <= 0 or wx <= 0:
            print(
                "--window-min-len, --window-max-len, and --window-stride must be positive.",
                file=sys.stderr,
            )
            return 2
        if wm > wx:
            print("--window-min-len cannot be greater than --window-max-len.", file=sys.stderr)
            return 2
<<<<<<< HEAD
=======
    if args.features_only and (args.train_legacy_gnn or args.train_final_gnn):
        print("--features-only cannot be combined with GNN training flags.", file=sys.stderr)
        return 2
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    cfg = RunConfig.from_args(args)
    return run_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CLI for the modular data pipeline (peptide_pipeline package).

Run from sequence_to_svm_minimal:
  python scripts/run_data_pipeline.py --input path/to/seqs.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptide_pipeline.config import RunConfig
from peptide_pipeline.runner import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run sequence -> structure -> features pipeline (see DATA_PROCESSING.md).")
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
    ap.add_argument("--input", "-i", type=str, default=None, help="(blind) Txt-like sequences or FASTA")
    ap.add_argument("--amp-input", type=str, default=None, help="(train) AMP sequences (txt-like or FASTA)")
    ap.add_argument("--decoy-input", type=str, default=None, help="(train) Decoy sequences (txt-like or FASTA)")
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
    ap.add_argument(
        "--cluster-simple-identity",
        type=float,
        default=0.80,
        help="--identity for prepare_clusters --simple-clusters",
    )
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
    return ap


def _config_file_arg_defaults(config_path: str | None) -> dict:
    if not config_path:
        return {}
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    out = dict(raw)
    if "input_path" in out:
        out["input"] = out.pop("input_path")
    return out


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "-c", type=str, default=None)
    pre_args, rest = pre.parse_known_args()
    defaults = _config_file_arg_defaults(pre_args.config)
    parser = _build_parser()
    if defaults:
        parser.set_defaults(**defaults)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="JSON object with defaults for pipeline flags (use input_path or input, work_dir, booleans, etc.). CLI overrides the file.",
    )
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
    cfg = RunConfig.from_args(args)
    return run_pipeline(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

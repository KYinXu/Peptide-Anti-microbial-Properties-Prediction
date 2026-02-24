#!/usr/bin/env python3
"""
Generate a comparison plot of MLP and GNN learning curves.

Loads MLP history from PNAS or fusion evaluation JSONs and/or GNN curves from
a run directory, then overlays them in one figure (train/val loss and val metric).
Skips missing data; works with only MLP or only GNN inputs.

Usage:
    python scripts/generate_mlp_gnn_comparison.py --mlp-history results/evaluation/final_train_history.json --gnn-curves-dir results/gnn/curves/run_20260202_142709 --output results/gnn/mlp_gnn_comparison.png
    python scripts/generate_mlp_gnn_comparison.py --mlp-history results/evaluation/mlp_combined_round_1.json --mlp-history results/evaluation/final_train_history.json --gnn-curves-dir results/gnn/curves/run_20260202_142709
    python scripts/generate_mlp_gnn_comparison.py --gnn-curves-dir results/gnn/curves/run_20260202_142709 --output results/gnn/curves_comparison.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.evaluation import (
    load_mlp_history_from_json,
    load_gnn_curves_from_run,
    plot_learning_curves_comparison,
)


def _label_from_path(path: Path) -> str:
    stem = path.stem
    if "final_train" in stem or "final" in stem:
        return "MLP (final)"
    if "round_" in stem:
        return stem.replace("_", " ").title()
    return stem.replace("_", " ").title()


def main():
    parser = argparse.ArgumentParser(
        description="Plot MLP vs GNN learning curves comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mlp-history",
        type=Path,
        action="append",
        default=[],
        dest="mlp_histories",
        help="Path to MLP history JSON (PNAS or fusion). Can be repeated.",
    )
    parser.add_argument(
        "--gnn-curves-dir",
        type=Path,
        default=None,
        help="Path to GNN run directory (e.g. results/gnn/curves/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: results/gnn/mlp_gnn_comparison.png or next to gnn-curves-dir)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MLP vs GNN",
        help="Plot title",
    )
    args = parser.parse_args()

    series_list = []

    for p in args.mlp_histories or []:
        path = Path(p).resolve()
        if not path.is_file():
            print(f"Warning: MLP history not found, skipping: {path}", file=sys.stderr)
            continue
        history = load_mlp_history_from_json(path)
        if history is not None:
            series_list.append((_label_from_path(path), history))

    if args.gnn_curves_dir is not None:
        gnn_dir = Path(args.gnn_curves_dir).resolve()
        if gnn_dir.is_dir():
            gnn_series = load_gnn_curves_from_run(gnn_dir)
            series_list.extend(gnn_series)
        else:
            print(f"Warning: GNN curves dir not found: {gnn_dir}", file=sys.stderr)

    if not series_list:
        print("No curve data loaded. Provide --mlp-history and/or --gnn-curves-dir.", file=sys.stderr)
        sys.exit(0)

    if args.output is not None:
        out_path = Path(args.output).resolve()
    elif args.gnn_curves_dir is not None:
        gnn_dir = Path(args.gnn_curves_dir).resolve()
        out_path = gnn_dir.parent / "mlp_gnn_comparison.png"
    else:
        out_path = Path("results/gnn/mlp_gnn_comparison.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if plot_learning_curves_comparison(series_list, out_path, title=args.title):
        print(f"Saved: {out_path}")
        for label, history in series_list:
            if history.get("val_auc_roc"):
                best_epoch = 1 + max(range(len(history["val_auc_roc"])), key=lambda i: history["val_auc_roc"][i])
                best_auc = max(history["val_auc_roc"])
                print(f"  {label}: best val AUC {best_auc:.4f} at epoch {best_epoch}")
    else:
        print("Failed to generate comparison plot.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

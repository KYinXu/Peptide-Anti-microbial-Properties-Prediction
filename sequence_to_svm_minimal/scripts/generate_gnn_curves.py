#!/usr/bin/env python3
"""
Generate evaluation plots from an existing GNN comparison run.

Reads curve CSVs and val_probs JSONs from a run directory (e.g. results/gnn/curves/run_*),
optionally a comparison JSON, and writes learning-curve, ROC/PR, and CV summary plots.
Skips any missing data.

Usage:
    python scripts/generate_gnn_curves.py --curves-dir results/gnn/curves/run_20260202_142709
    python scripts/generate_gnn_curves.py --curves-dir results/gnn/curves/run_20260202_142709 --comparison-json results/gnn/gnn_comparison_20260202_142709.json
    python scripts/generate_gnn_curves.py --curves-dir results/gnn/curves/run_20260202_142709 --output-dir results/gnn/figures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.evaluation import generate_plots_from_gnn_run


def main():
    parser = argparse.ArgumentParser(
        description="Generate GNN evaluation plots from existing curve data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--curves-dir",
        type=Path,
        required=True,
        help="Path to GNN run curves directory (e.g. results/gnn/curves/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=None,
        help="Optional path to gnn_comparison_*.json for CV summary bar chart",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: curves_dir.parent/evaluation_plots/run_name)",
    )
    args = parser.parse_args()

    curves_dir = args.curves_dir.resolve()
    if not curves_dir.is_dir():
        print(f"Error: curves directory not found: {curves_dir}", file=sys.stderr)
        sys.exit(1)

    comparison = args.comparison_json.resolve() if args.comparison_json else None
    if comparison is not None and not comparison.is_file():
        print(f"Warning: comparison JSON not found, skipping CV summary: {comparison}", file=sys.stderr)
        comparison = None

    out_dir = args.output_dir.resolve() if args.output_dir else None
    generated = generate_plots_from_gnn_run(
        curves_base_dir=curves_dir,
        comparison_json_path=comparison,
        output_dir=out_dir,
    )
    if not generated:
        print("No plots generated (missing or invalid data).", file=sys.stderr)
        sys.exit(0)
    print("Generated plots:")
    for p in generated:
        print(f"  {p}")


if __name__ == "__main__":
    main()

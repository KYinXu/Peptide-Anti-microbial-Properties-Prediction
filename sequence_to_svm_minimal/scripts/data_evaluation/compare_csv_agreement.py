#!/usr/bin/env python3
"""
Compare two SVM prediction CSVs by matching identical sequences and calculating
the percentage agreement of their SVM_prob_AMP and SVM_hyperplane_distance scores.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Compare SVM predictions between two CSVs based on identical sequences."
    )
    parser.add_argument("--csv1", type=Path, required=True, help="Path to first CSV")
    parser.add_argument("--csv2", type=Path, required=True, help="Path to second CSV")
    parser.add_argument("--seq-col", type=str, default="sequence", help="Column name to match sequences on (default: 'sequence')")
    parser.add_argument("--tol", type=float, default=1e-5, help="Tolerance for floating point comparison (default: 1e-5)")

    args = parser.parse_args()

    if not args.csv1.is_file():
        raise FileNotFoundError(f"File not found: {args.csv1}")
    if not args.csv2.is_file():
        raise FileNotFoundError(f"File not found: {args.csv2}")

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    if args.seq_col not in df1.columns or args.seq_col not in df2.columns:
        raise ValueError(f"Match column '{args.seq_col}' not found in both CSVs.")

    for col in ["SVM_prob_AMP", "SVM_hyperplane_distance"]:
        if col not in df1.columns:
            raise ValueError(f"Column '{col}' not found in {args.csv1.name}")
        if col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in {args.csv2.name}")

    # Drop duplicates to compare unique sequences and avoid cross-join explosion
    df1_unique = df1.drop_duplicates(subset=[args.seq_col]).copy()
    df2_unique = df2.drop_duplicates(subset=[args.seq_col]).copy()

    merged = pd.merge(df1_unique, df2_unique, on=args.seq_col, suffixes=('_1', '_2'))

    n_common = len(merged)
    print("=" * 60)
    print("CSV AGREEMENT REPORT")
    print("=" * 60)
    print(f"File 1: {args.csv1.name} ({len(df1):,} rows)")
    print(f"File 2: {args.csv2.name} ({len(df2):,} rows)")
    print(f"Common unique sequences found: {n_common:,}")
    print("-" * 60)

    if n_common == 0:
        print("No common sequences found to compare.")
        return

    # Compare probability
    prob1 = pd.to_numeric(merged["SVM_prob_AMP_1"], errors="coerce")
    prob2 = pd.to_numeric(merged["SVM_prob_AMP_2"], errors="coerce")
    prob_diff = np.abs(prob1 - prob2)
    prob_exact = (prob1 == prob2).sum()
    prob_tol = (prob_diff <= args.tol).sum()

    # Compare distance
    dist1 = pd.to_numeric(merged["SVM_hyperplane_distance_1"], errors="coerce")
    dist2 = pd.to_numeric(merged["SVM_hyperplane_distance_2"], errors="coerce")
    dist_diff = np.abs(dist1 - dist2)
    dist_exact = (dist1 == dist2).sum()
    dist_tol = (dist_diff <= args.tol).sum()

    print("SVM_prob_AMP Agreement:")
    print(f"  Exact matches:       {prob_exact:,} ({prob_exact / n_common:.2%})")
    print(f"  Matches (tol={args.tol}): {prob_tol:,} ({prob_tol / n_common:.2%})")
    print(f"  Max discrepancy:     {prob_diff.max():.2e}")
    print()
    print("SVM_hyperplane_distance Agreement:")
    print(f"  Exact matches:       {dist_exact:,} ({dist_exact / n_common:.2%})")
    print(f"  Matches (tol={args.tol}): {dist_tol:,} ({dist_tol / n_common:.2%})")
    print(f"  Max discrepancy:     {dist_diff.max():.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()

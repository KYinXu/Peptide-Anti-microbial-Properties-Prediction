#!/usr/bin/env python3
"""Check the fixed geometric features for quality."""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    csv_path = Path(__file__).parent / "data" / "training_dataset" / "geometric_features_fixed.csv"
    
    print("\n" + "=" * 70)
    print("  GEOMETRIC FEATURES QUALITY CHECK (FIXED)")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    print(f"\n📊 Loaded: {csv_path.name}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Check for NaN/Inf
    print("\n" + "-" * 70)
    print("  Missing Values and Infinities")
    print("-" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    nan_count = df[numeric_cols].isna().sum()
    inf_count = df[numeric_cols].apply(lambda x: np.isinf(x).sum())
    
    problem_cols = nan_count[nan_count > 0].index.tolist() + inf_count[inf_count > 0].index.tolist()
    
    if problem_cols:
        print("\n⚠️  Columns with issues:")
        for col in set(problem_cols):
            print(f"   {col}: {nan_count.get(col, 0)} NaN, {inf_count.get(col, 0)} Inf")
    else:
        print("\n✅ No NaN or Inf values in numeric columns!")
    
    # Check secondary structure
    print("\n" + "-" * 70)
    print("  Secondary Structure Analysis")
    print("-" * 70)
    
    if 'fraction_helix' in df.columns:
        print(f"\n  fraction_helix:")
        print(f"    Mean:   {df['fraction_helix'].mean():.4f}")
        print(f"    Median: {df['fraction_helix'].median():.4f}")
        print(f"    Std:    {df['fraction_helix'].std():.4f}")
        print(f"    Min:    {df['fraction_helix'].min():.4f}")
        print(f"    Max:    {df['fraction_helix'].max():.4f}")
        
        # Distribution bins
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        labels = ['0-10%', '10-30%', '30-50%', '50-70%', '70-90%', '90-100%']
        df['helix_bin'] = pd.cut(df['fraction_helix'], bins=bins, labels=labels)
        print(f"\n  Helix fraction distribution:")
        for label in labels:
            count = (df['helix_bin'] == label).sum()
            pct = 100 * count / len(df)
            bar = '█' * int(pct / 2)
            print(f"    {label:8s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    if 'fraction_sheet' in df.columns:
        print(f"\n  fraction_sheet:")
        print(f"    Mean:   {df['fraction_sheet'].mean():.4f}")
        print(f"    Max:    {df['fraction_sheet'].max():.4f}")
    
    if 'fraction_coil' in df.columns:
        print(f"\n  fraction_coil:")
        print(f"    Mean:   {df['fraction_coil'].mean():.4f}")
        print(f"    Max:    {df['fraction_coil'].max():.4f}")
    
    if 'ss_method' in df.columns:
        print(f"\n  ss_method values:")
        print(f"    {df['ss_method'].value_counts().to_dict()}")
    
    # Compare AMP vs DECOY
    print("\n" + "-" * 70)
    print("  AMP vs DECOY Comparison")
    print("-" * 70)
    
    if 'label' in df.columns:
        amp = df[df['label'] == 1]
        decoy = df[df['label'] == -1]
        
        compare_cols = ['fraction_helix', 'fraction_sheet', 'fraction_coil', 
                        'mean_plddt', 'radius_of_gyration', 'net_charge', 'mean_hydrophobicity']
        
        print(f"\n  {'Feature':<25} {'AMP (mean)':<12} {'DECOY (mean)':<12} {'Diff':<10}")
        print("  " + "-" * 60)
        
        for col in compare_cols:
            if col in df.columns:
                amp_mean = amp[col].mean()
                decoy_mean = decoy[col].mean()
                diff = amp_mean - decoy_mean
                print(f"  {col:<25} {amp_mean:>10.4f}  {decoy_mean:>10.4f}  {diff:>+8.4f}")
    
    # Curvature check
    print("\n" + "-" * 70)
    print("  Curvature/Torsion Check")
    print("-" * 70)
    
    curv_cols = [c for c in df.columns if 'curvature' in c or 'torsion' in c]
    for col in curv_cols:
        print(f"\n  {col}:")
        print(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
        print(f"    Mean:  {df[col].mean():.4f} ± {df[col].std():.4f}")
    
    print("\n" + "=" * 70)
    print("  CHECK COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

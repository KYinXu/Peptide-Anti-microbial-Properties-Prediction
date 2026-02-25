#!/usr/bin/env python3
"""Quick error check for geometric features CSV."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the CSV
csv_path = Path(__file__).parent / "data/training_dataset/geometric_features.csv"
df = pd.read_csv(csv_path)

print("="*60)
print("  GEOMETRIC FEATURES ERROR CHECK")
print("="*60)

print(f"\n📊 Dataset Shape: {df.shape[0]} samples × {df.shape[1]} columns")

# Check for missing values
print("\n--- Missing Values Check ---")
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) == 0:
    print("✅ No missing values")
else:
    print("❌ Missing values found:")
    for col, count in missing_cols.items():
        print(f"   {col}: {count} missing ({100*count/len(df):.1f}%)")

# Check for infinite values
print("\n--- Infinite Values Check ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_count = 0
for col in numeric_cols:
    n_inf = np.isinf(df[col]).sum()
    if n_inf > 0:
        print(f"❌ {col}: {n_inf} infinite values")
        inf_count += n_inf
if inf_count == 0:
    print("✅ No infinite values")

# Check for NaN in numeric columns
print("\n--- NaN in Numeric Columns ---")
nan_count = 0
for col in numeric_cols:
    n_nan = df[col].isna().sum()
    if n_nan > 0:
        print(f"❌ {col}: {n_nan} NaN values")
        nan_count += n_nan
if nan_count == 0:
    print("✅ No NaN in numeric columns")

# Value range checks
print("\n--- Value Range Validation ---")

# pLDDT should be 0-1
if "plddt_mean" in df.columns:
    pmin, pmax = df["plddt_mean"].min(), df["plddt_mean"].max()
    if pmin >= 0 and pmax <= 1:
        print(f"✅ plddt_mean in valid range [0,1]: [{pmin:.3f}, {pmax:.3f}]")
    else:
        print(f"❌ plddt_mean out of range: [{pmin:.3f}, {pmax:.3f}]")

# Fractions should be 0-1
for col in ["fraction_helix", "fraction_sheet", "fraction_coil", "fraction_hydrophobic_sasa"]:
    if col in df.columns:
        cmin, cmax = df[col].min(), df[col].max()
        if cmin >= -0.001 and cmax <= 1.001:  # small tolerance
            print(f"✅ {col} in [0,1]: [{cmin:.3f}, {cmax:.3f}]")
        else:
            print(f"❌ {col} out of range: [{cmin:.3f}, {cmax:.3f}]")

# Positive values check
for col in ["radius_gyration", "total_sasa", "length", "curvature_mean"]:
    if col in df.columns:
        cmin = df[col].min()
        if cmin >= 0:
            print(f"✅ {col} is non-negative: min={cmin:.3f}")
        else:
            print(f"❌ {col} has negative values: min={cmin:.3f}")

# Length sanity check
if "length" in df.columns:
    print(f"\n--- Peptide Length Distribution ---")
    print(f"   Range: {int(df['length'].min())} - {int(df['length'].max())} residues")
    print(f"   Mean: {df['length'].mean():.1f} ± {df['length'].std():.1f}")

# Label distribution
if "label" in df.columns:
    print(f"\n--- Label Distribution ---")
    for label in sorted(df["label"].unique()):
        count = (df["label"] == label).sum()
        pct = 100 * count / len(df)
        name = "AMP" if label == 1 else "DECOY"
        print(f"   {name} (label={label}): {count} samples ({pct:.1f}%)")

# Statistical summary of key features
print("\n--- Feature Statistics ---")
key_features = ["plddt_mean", "radius_gyration", "net_charge", "mean_hydrophobicity", "curvature_mean"]
for feat in key_features:
    if feat in df.columns:
        print(f"   {feat}: {df[feat].mean():.3f} ± {df[feat].std():.3f} (range: {df[feat].min():.3f} to {df[feat].max():.3f})")

# Check for duplicates
print(f"\n--- Duplicate Check ---")
dup_ids = df["peptide_id"].duplicated().sum()
if dup_ids == 0:
    print("✅ No duplicate peptide IDs")
else:
    print(f"❌ {dup_ids} duplicate peptide IDs")

dup_seqs = df["sequence"].duplicated().sum()
if dup_seqs == 0:
    print("✅ No duplicate sequences")
else:
    print(f"⚠️  {dup_seqs} duplicate sequences (may be intentional)")

# Compare AMP vs DECOY
print(f"\n--- AMP vs DECOY Comparison ---")
amp = df[df["label"] == 1]
decoy = df[df["label"] == -1]

compare_feats = ["plddt_mean", "radius_gyration", "net_charge", "mean_hydrophobicity", "fraction_helix"]
for feat in compare_feats:
    if feat in df.columns:
        amp_mean = amp[feat].mean()
        decoy_mean = decoy[feat].mean()
        diff = amp_mean - decoy_mean
        print(f"   {feat}: AMP={amp_mean:.3f}, DECOY={decoy_mean:.3f}, Δ={diff:+.3f}")

print("\n" + "="*60)
print("  ✅ ERROR CHECK COMPLETE")
print("="*60)

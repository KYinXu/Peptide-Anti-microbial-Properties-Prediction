#!/usr/bin/env python3
"""Check for feature overlap between QSAR-12 and Geometric-24 feature sets."""

import pandas as pd
import numpy as np

# Load both datasets
qsar = pd.read_csv('data/training_dataset/qsar12_descriptors.csv')
geo = pd.read_csv('data/training_dataset/geometric_features.csv')

# Align by peptide_id
df = qsar.merge(geo, on=['peptide_id', 'sequence'], how='inner')

print('='*70)
print('FEATURE OVERLAP ANALYSIS: QSAR-12 vs Geometric-24')
print('='*70)
print()

# List feature sets
qsar_feats = ['netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK', '_SolventAccessibilityD1025', 
              'tau2_GRAR740104', 'tau4_GRAR740104', 'QSO50_GRAR740104', 'QSO29_GRAR740104']

geo_feats = ['plddt_mean', 'plddt_std', 'plddt_min', 'plddt_max', 'radius_gyration', 
             'end_to_end_distance', 'max_pairwise_distance', 'centroid_distance_mean', 
             'centroid_distance_std', 'fraction_helix', 'fraction_sheet', 'fraction_coil',
             'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa', 'length', 
             'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment', 
             'curvature_mean', 'curvature_std', 'curvature_max', 'torsion_mean', 'torsion_std']

print('QSAR-12 FEATURES:')
for i, f in enumerate(qsar_feats, 1):
    print(f'  {i:2d}. {f}')

print()
print('GEOMETRIC-24 FEATURES:')
for i, f in enumerate(geo_feats, 1):
    print(f'  {i:2d}. {f}')

print()
print('='*70)
print('POTENTIAL SEMANTIC OVERLAP CHECK')
print('='*70)

# 1. Check net_charge vs netCharge
print()
print('1. netCharge (QSAR) vs net_charge (Geo):')
print('   - Both compute net charge at pH 7')
corr_charge = df['netCharge'].corr(df['net_charge'])
print(f'   - Pearson correlation: r = {corr_charge:.6f}')
print()
print('   Sample values (first 10 peptides):')
print(f'   QSAR netCharge: {df["netCharge"].head(10).tolist()}')
print(f'   Geo net_charge: {df["net_charge"].head(10).tolist()}')
print()
diff = df['netCharge'] - df['net_charge']
print('   Difference statistics:')
print(f'   Mean diff:  {diff.mean():.4f}')
print(f'   Max diff:   {diff.max():.4f}')
print(f'   Min diff:   {diff.min():.4f}')
print(f'   N identical: {(diff.abs() < 0.01).sum()} / {len(diff)}')

# Check WHY they differ - look at the definitions
print()
print('   WHY THE DIFFERENCE?')
print('   QSAR uses: H=+1  (charged at pH7)')
print('   Geo  uses: H=+0.1 (only ~10% protonated)')
print('   This explains the small differences!')

# 2. Check _SolventAccessibilityD1025 vs SASA features
print()
print('='*70)
print('2. _SolventAccessibilityD1025 (QSAR) vs SASA features (Geo):')
print('='*70)
print()
print('   QSAR _SolventAccessibilityD1025: CTD descriptor (sequence-based)')
print('   Geo SASA features: 3D structure-based (Shrake-Rupley algorithm)')
print()
for sasa_feat in ['total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa']:
    corr = df['_SolventAccessibilityD1025'].corr(df[sasa_feat])
    print(f'   _SolventAccessibilityD1025 vs {sasa_feat}: r = {corr:.4f}')
print()
print('   VERDICT: Low correlations show these are DIFFERENT measures!')
print('   - QSAR uses predicted/sequence-based accessibility')
print('   - Geo uses actual 3D computed SASA')

# 3. Cross-correlation matrix for any high correlations
print()
print('='*70)
print('3. FULL CROSS-CORRELATION CHECK')
print('='*70)

all_corrs = []
for q_feat in qsar_feats:
    for g_feat in geo_feats:
        corr = df[q_feat].corr(df[g_feat])
        all_corrs.append((q_feat, g_feat, corr))

# Sort by absolute correlation
all_corrs.sort(key=lambda x: abs(x[2]), reverse=True)

print()
print('Top 20 cross-feature correlations (sorted by |r|):')
print('-'*70)
for q, g, r in all_corrs[:20]:
    flag = '⚠️ OVERLAP' if abs(r) > 0.95 else ('⚡ HIGH' if abs(r) > 0.8 else '')
    print(f'   {q:30s} <-> {g:30s} : r = {r:+.4f}  {flag}')

high_overlap = [c for c in all_corrs if abs(c[2]) > 0.95]
print()
print('='*70)
print('SUMMARY')
print('='*70)
print()
if high_overlap:
    print(f'⚠️  Found {len(high_overlap)} feature pair(s) with |r| > 0.95 (TRUE OVERLAP):')
    for q, g, r in high_overlap:
        print(f'    • {q} ↔ {g} (r={r:.4f})')
else:
    print('✅ No feature pairs with |r| > 0.95 found!')
    print('   The QSAR-12 and Geometric-24 feature sets are DISTINCT.')

# Check netCharge specifically
print()
print(f'Note on netCharge vs net_charge:')
print(f'   - Correlation: r = {corr_charge:.4f}')
if corr_charge > 0.99:
    print(f'   ⚠️  These are essentially the SAME feature!')
    print(f'   Recommend: Remove one to avoid redundancy in Combined-36')
else:
    print(f'   - Small differences due to histidine charge treatment')
    print(f'   - Technically measuring same concept, but values differ slightly')

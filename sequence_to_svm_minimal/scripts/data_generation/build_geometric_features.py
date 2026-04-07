#!/usr/bin/env python3
"""
Build Geometric Features from ESMFold PDB Files

Converts ESMFold-predicted peptide structures into fixed-length geometric 
feature vectors for MIC regression.

Input:
    - Directory of PDB files (from ESMFold)
    - Optional: results_log.csv with sequences and labels
    - Optional: SVM predictions and QSAR descriptors

Output:
    - CSV/Parquet file with one row per peptide
    - Feature names saved to separate file
    
Usage:
    python build_geometric_features.py --pdb-dir data/training_dataset/structures/ \\
                                       --output features/geometric_features.csv
                                       
    python build_geometric_features.py --pdb-dir data/training_dataset/structures/ \\
                                       --results-log data/training_dataset/structures/results_log.csv \\
                                       --output features/geometric_features.parquet
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from features.geometric_features import (
    extract_all_features,
    get_feature_names,
    parse_pdb_structure
)


def find_pdb_files(pdb_dir: Path) -> list:
    """Recursively find all PDB files in directory."""
    pdb_files = []
    
    for ext in ['*.pdb', '*.PDB']:
        pdb_files.extend(pdb_dir.rglob(ext))
    
    return sorted(pdb_files)


def load_results_log(results_path: Path) -> pd.DataFrame:
    """Load results log with sequence and label information."""
    if not results_path.exists():
        return None
    
    df = pd.read_csv(results_path)
    
    # Create lookup by unique_id
    if 'unique_id' in df.columns:
        return df.set_index('unique_id')
    elif 'peptide_id' in df.columns:
        return df.set_index('peptide_id')
    else:
        return df


def load_svm_predictions(svm_path: Path) -> pd.DataFrame:
    """Load SVM prediction file."""
    if not svm_path or not svm_path.exists():
        return None
    
    df = pd.read_csv(svm_path)
    
    # Expected columns: seqIndex, prediction, distToMargin, P(-1), P(+1)
    if 'seqIndex' in df.columns:
        return df.set_index('seqIndex')
    
    return df


def load_qsar_descriptors(qsar_path: Path) -> pd.DataFrame:
    """Load QSAR descriptors file."""
    if not qsar_path or not qsar_path.exists():
        return None
    
    return pd.read_csv(qsar_path, index_col=0)


def extract_peptide_id_from_path(pdb_path: Path) -> str:
    """Extract peptide ID from PDB filename."""
    # Handle formats like: AMP_1.pdb, DECOY_42.pdb, structure_123.pdb
    stem = pdb_path.stem
    return stem


def process_single_pdb(
    pdb_path: Path,
    results_lookup: pd.DataFrame = None,
    svm_lookup: pd.DataFrame = None,
    qsar_lookup: pd.DataFrame = None,
    pdb_anchor_dir: Path | None = None,
) -> dict:
    """
    Process a single PDB file and extract all features.
    
    Args:
        pdb_path: Path to PDB file
        results_lookup: DataFrame with sequence/label info
        svm_lookup: DataFrame with SVM predictions
        qsar_lookup: DataFrame with QSAR descriptors
        
    Returns:
        Dictionary of features or None if failed
    """
    peptide_id = extract_peptide_id_from_path(pdb_path)
    
    # Get sequence and label from results log
    sequence = None
    label = None
    
    if results_lookup is not None and peptide_id in results_lookup.index:
        row = results_lookup.loc[peptide_id]
        sequence = row.get('sequence', None)
        label = row.get('label', None)
    
    # Get SVM outputs
    svm_sigma = None
    svm_prob = None
    
    if svm_lookup is not None:
        # Try to match by peptide ID or index
        try:
            idx = int(peptide_id.split('_')[-1])
            if idx in svm_lookup.index:
                svm_row = svm_lookup.loc[idx]
                svm_sigma = svm_row.get('distToMargin', None)
                svm_prob = svm_row.get('P(+1)', None)
        except (ValueError, KeyError):
            pass
    
    # Get QSAR descriptors
    qsar_descriptors = None
    
    if qsar_lookup is not None:
        try:
            idx = int(peptide_id.split('_')[-1])
            if idx in qsar_lookup.index:
                qsar_descriptors = qsar_lookup.loc[idx].values[:12].tolist()
        except (ValueError, KeyError):
            pass
    
    # Extract features
    try:
        features = extract_all_features(
            pdb_path=str(pdb_path),
            peptide_id=peptide_id,
            sequence=sequence,
            svm_sigma=svm_sigma,
            svm_prob=svm_prob,
            qsar_descriptors=qsar_descriptors
        )
        
        # Add label if available
        if label is not None:
            features['label'] = label
        
        if pdb_anchor_dir is not None:
            try:
                features["pdb_file"] = str(pdb_path.resolve().relative_to(pdb_anchor_dir.resolve()))
            except ValueError:
                features["pdb_file"] = str(pdb_path.name)
        else:
            features["pdb_file"] = str(pdb_path.name)
        
        return features
        
    except Exception as e:
        print(f"  ❌ Error processing {pdb_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract geometric features from ESMFold PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_geometric_features.py --pdb-dir structures/ --output features.csv
  
  # With results log (includes sequences and labels)
  python build_geometric_features.py --pdb-dir structures/ \\
      --results-log structures/results_log.csv \\
      --output features.csv
      
  # Include SVM and QSAR features
  python build_geometric_features.py --pdb-dir structures/ \\
      --svm-predictions predictionsParameters/descriptors_PREDICTIONS.csv \\
      --qsar-descriptors predictionsParameters/descriptors.csv \\
      --output features.parquet

  # Unlabeled dataset (no AMP/decoy distinction)
  python build_geometric_features.py --pdb-dir structures/ --output features.csv --unlabeled
        """
    )
    
    parser.add_argument('--pdb-dir', '-p', type=str, required=True,
                        help='Directory containing PDB files (searched recursively)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output file path (.csv or .parquet)')
    parser.add_argument('--results-log', '-r', type=str, default=None,
                        help='Path to results_log.csv with sequences and labels')
    parser.add_argument('--svm-predictions', type=str, default=None,
                        help='Path to SVM predictions CSV')
    parser.add_argument('--qsar-descriptors', type=str, default=None,
                        help='Path to QSAR descriptors CSV')
    parser.add_argument('--save-feature-names', action='store_true',
                        help='Save feature names to JSON file')
    parser.add_argument('--unlabeled', action='store_true',
                        help='Unlabeled dataset: no AMP/decoy distinction; do not require or expect labels')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("  Geometric Feature Extraction Pipeline")
    print("="*60)
    print(f"  PDB directory: {args.pdb_dir}")
    print(f"  Output: {args.output}")
    print()
    
    # Find PDB files
    pdb_dir = Path(args.pdb_dir)
    if not pdb_dir.exists():
        print(f"❌ PDB directory not found: {pdb_dir}")
        sys.exit(1)
    
    pdb_anchor = pdb_dir.resolve()
    pdb_files = find_pdb_files(pdb_dir)
    print(f"📂 Found {len(pdb_files)} PDB files")
    
    if len(pdb_files) == 0:
        print("❌ No PDB files found")
        sys.exit(1)
    
    # Load optional data sources (results_log supplies sequence + label)
    results_lookup = None
    def _try_load_results(path: Path):
        if path.exists():
            return load_results_log(path)
        return None
    if args.results_log:
        results_path = Path(args.results_log)
        results_lookup = _try_load_results(results_path)
        if results_lookup is not None:
            print(f"📋 Loaded results log: {len(results_lookup)} entries")
        elif not results_path.exists():
            for candidate in [pdb_dir / "results_log.csv", pdb_dir.parent / "results_log.csv"]:
                results_lookup = _try_load_results(candidate)
                if results_lookup is not None:
                    print(f"📋 Loaded results log from {candidate}: {len(results_lookup)} entries")
                    break
    else:
        for candidate in [pdb_dir / "results_log.csv", pdb_dir.parent / "results_log.csv"]:
            results_lookup = _try_load_results(candidate)
            if results_lookup is not None:
                print(f"📋 Auto-loaded results log from {candidate}: {len(results_lookup)} entries")
                break
    
    svm_lookup = None
    if args.svm_predictions:
        svm_lookup = load_svm_predictions(Path(args.svm_predictions))
        if svm_lookup is not None:
            print(f"🔮 Loaded SVM predictions: {len(svm_lookup)} entries")
    
    qsar_lookup = None
    if args.qsar_descriptors:
        qsar_lookup = load_qsar_descriptors(Path(args.qsar_descriptors))
        if qsar_lookup is not None:
            print(f"🧪 Loaded QSAR descriptors: {len(qsar_lookup)} entries")
    
    # Process all PDB files
    print(f"\n{'='*60}")
    print("  Extracting Features")
    print(f"{'='*60}")
    
    all_features = []
    failed_count = 0
    
    for pdb_path in tqdm(pdb_files, desc="Processing", unit="pdb"):
        features = process_single_pdb(
            pdb_path=pdb_path,
            results_lookup=results_lookup,
            svm_lookup=svm_lookup,
            qsar_lookup=qsar_lookup,
            pdb_anchor_dir=pdb_anchor,
        )
        
        if features is not None:
            all_features.append(features)
        else:
            failed_count += 1
    
    print(f"\n✅ Successfully processed: {len(all_features)}")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count}")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns: ID columns first, then features, then metadata
    id_cols = ['peptide_id', 'sequence', 'pdb_file', 'label']
    feature_cols = get_feature_names(include_optional=True)
    
    # Get actual columns that exist
    id_cols_present = [c for c in id_cols if c in df.columns]
    feature_cols_present = [c for c in feature_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in id_cols_present + feature_cols_present]
    
    ordered_cols = id_cols_present + feature_cols_present + other_cols
    df = df[ordered_cols]
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    
    print(f"\n📁 Saved features to: {output_path}")
    print(f"   Shape: {df.shape[0]} samples × {df.shape[1]} columns")
    if not args.unlabeled:
        if "label" not in df.columns:
            print("   ⚠️  No 'label' column. Pass --results-log or use --pdb-dir so results_log.csv is found")
            print("      (script checks pdb_dir and pdb_dir.parent for results_log.csv).")
        elif df["label"].isna().all():
            print("   ⚠️  All labels are NaN. PDB peptide_id (filename stem) may not match results_log IDs.")
    elif args.unlabeled:
        print("   📋 Unlabeled mode: no label column required.")
    if "sequence" not in df.columns and results_lookup is None:
        print("   ⚠️  No 'sequence' column. results_log.csv is required for sequence and label.")
    
    # Save feature names
    if args.save_feature_names:
        names_path = output_path.with_suffix('.feature_names.json')
        with open(names_path, 'w') as f:
            json.dump({
                'feature_names': feature_cols_present,
                'all_columns': list(df.columns),
                'created': datetime.now().isoformat()
            }, f, indent=2)
        print(f"📝 Saved feature names to: {names_path}")
    
    # Print feature summary
    print(f"\n{'='*60}")
    print("  Feature Summary")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\n📊 Numeric features: {len(numeric_cols)}")
    
    # Group by category
    categories = {
        'pLDDT': ['plddt_mean', 'plddt_std', 'plddt_min', 'plddt_max'],
        'Compactness': ['radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
                       'centroid_distance_mean', 'centroid_distance_std'],
        'Secondary Structure': ['fraction_helix', 'fraction_sheet', 'fraction_coil', 'ss_method'],
        'SASA': ['total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa'],
        'Sequence': ['length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment'],
        'Curvature': ['curvature_mean', 'curvature_std', 'curvature_max', 'torsion_mean', 'torsion_std'],
        'SVM': ['svm_sigma', 'svm_prob_positive'],
        'QSAR': [c for c in df.columns if c.startswith('qsar_')]
    }
    
    for cat, cols in categories.items():
        present = [c for c in cols if c in df.columns]
        if present:
            print(f"   {cat}: {len(present)} features")
    
    # Print sample statistics
    print(f"\n📈 Sample Statistics:")
    if 'length' in df.columns:
        print(f"   Peptide lengths: {df['length'].min():.0f} - {df['length'].max():.0f} residues")
    if 'plddt_mean' in df.columns:
        print(f"   Mean pLDDT: {df['plddt_mean'].mean():.3f} ± {df['plddt_mean'].std():.3f}")
    if 'fraction_helix' in df.columns:
        print(f"   Avg helix fraction: {df['fraction_helix'].mean():.3f}")
    if 'label' in df.columns and not args.unlabeled:
        label_counts = df['label'].value_counts()
        print(f"   Label distribution: {dict(label_counts)}")
    
    print("\n✅ Done!")
    print()


if __name__ == "__main__":
    main()

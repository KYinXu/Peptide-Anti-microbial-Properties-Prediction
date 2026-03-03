#!/usr/bin/env python3
"""
Prepare sequences for CD-HIT clustering and parse cluster results.

This module:
1. Converts sequences to FASTA format for CD-HIT
2. Parses CD-HIT cluster output
3. Assigns cluster IDs for GroupKFold splitting

Usage:
    # One-shot: run CD-HIT and produce geometric_features_clustered.csv (requires cd-hit on PATH)
    python prepare_clusters.py --run-cdhit -i geometric_features.csv -o geometric_features_clustered.csv

    # Or manual steps:
    python prepare_clusters.py --generate-fasta -i geometric_features.csv -o sequences.fasta
    # cd-hit -i sequences.fasta -o clusters -c 0.40 -n 2 -M 16000
    python prepare_clusters.py --parse-clusters --clstr-file clusters.clstr -i geometric_features.csv -o geometric_features_clustered.csv
"""

import argparse
import shutil
import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re


def generate_fasta(features_csv: Path, output_fasta: Path) -> int:
    """
    Convert features CSV to FASTA format for CD-HIT.
    
    Args:
        features_csv: Path to geometric_features.csv (must have peptide_id and sequence columns)
        output_fasta: Output FASTA file path
        
    Returns:
        Number of sequences written
    """
    df = pd.read_csv(features_csv)
    if 'sequence' not in df.columns:
        raise ValueError(
            "CSV has no 'sequence' column. Use build_geometric_features with --results-log to include sequences."
        )
    id_col = 'peptide_id' if 'peptide_id' in df.columns else df.columns[0]
    
    with open(output_fasta, 'w') as f:
        for _, row in df.iterrows():
            peptide_id = row[id_col]
            sequence = row['sequence']
            f.write(f">{peptide_id}\n{sequence}\n")
    
    print(f"✅ Wrote {len(df)} sequences to {output_fasta}")
    return len(df)


def parse_cdhit_clusters(clstr_file: Path) -> Dict[str, int]:
    """
    Parse CD-HIT .clstr file to extract cluster assignments.
    
    Args:
        clstr_file: Path to .clstr file from CD-HIT
        
    Returns:
        Dictionary mapping peptide_id → cluster_id
    """
    cluster_map = {}
    current_cluster = -1
    
    with open(clstr_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>Cluster'):
                # New cluster: ">Cluster 0"
                current_cluster = int(line.split()[1])
            elif line:
                # Sequence line: "0	22aa, >AMP_123... *"
                # Extract the sequence ID between > and ...
                match = re.search(r'>(\S+)\.\.\.', line)
                if match:
                    peptide_id = match.group(1)
                    cluster_map[peptide_id] = current_cluster
    
    return cluster_map


def add_clusters_to_features(features_csv: Path, clstr_file: Path, 
                              output_csv: Path) -> pd.DataFrame:
    """
    Add cluster assignments to features CSV.
    
    Args:
        features_csv: Original geometric_features.csv
        clstr_file: CD-HIT cluster file
        output_csv: Output path for clustered features
        
    Returns:
        DataFrame with cluster_id column
    """
    df = pd.read_csv(features_csv)
    cluster_map = parse_cdhit_clusters(clstr_file)
    
    # Add cluster IDs
    df['cluster_id'] = df['peptide_id'].map(cluster_map)
    
    # Check for unmapped sequences
    unmapped = df['cluster_id'].isna().sum()
    if unmapped > 0:
        print(f"⚠️  {unmapped} sequences not found in cluster file")
        unmapped_mask = df['cluster_id'].isna()
        max_cluster = df['cluster_id'].max()
        start = int(max_cluster) + 1 if pd.notna(max_cluster) else 0
        df.loc[unmapped_mask, 'cluster_id'] = range(start, start + unmapped)
    
    df['cluster_id'] = df['cluster_id'].astype(int)
    
    # Save
    df.to_csv(output_csv, index=False)
    
    n_clusters = df['cluster_id'].nunique()
    print(f"✅ Added cluster IDs to {len(df)} sequences")
    print(f"   Total clusters: {n_clusters}")
    print(f"   Cluster sizes: min={df['cluster_id'].value_counts().min()}, "
          f"max={df['cluster_id'].value_counts().max()}, "
          f"median={df['cluster_id'].value_counts().median():.0f}")
    print(f"   Saved to: {output_csv}")
    
    return df


def create_simple_clusters(features_csv: Path, output_csv: Path, 
                           identity_threshold: float = 0.80) -> pd.DataFrame:
    """
    Create simple sequence-based clusters without CD-HIT.
    
    Uses a greedy approach: for each unclustered sequence, create a new cluster
    and add all sequences with >threshold identity to it.
    If the CSV has no 'sequence' column, assigns one cluster per row (no grouping).
    
    Args:
        features_csv: Input features CSV
        output_csv: Output path
        identity_threshold: Sequence identity threshold (0-1)
        
    Returns:
        DataFrame with cluster_id column
    """
    from difflib import SequenceMatcher
    
    df = pd.read_csv(features_csv)
    if 'sequence' not in df.columns:
        print("⚠️  No 'sequence' column in CSV; assigning one cluster per row (no sequence-based grouping).")
        df['cluster_id'] = np.arange(len(df))
        df.to_csv(output_csv, index=False)
        print(f"✅ Wrote {len(df)} rows with cluster_id to {output_csv}")
        return df
    
    sequences = df['sequence'].tolist()
    n = len(sequences)
    
    print(f"🔄 Creating simple clusters at {identity_threshold*100:.0f}% identity...")
    print(f"   This may take a while for {n} sequences...")
    
    cluster_ids = [-1] * n
    current_cluster = 0
    
    for i in range(n):
        if cluster_ids[i] >= 0:
            continue  # Already assigned
            
        # Start new cluster with this sequence
        cluster_ids[i] = current_cluster
        seq_i = sequences[i]
        
        # Find all similar sequences
        for j in range(i + 1, n):
            if cluster_ids[j] >= 0:
                continue
            
            seq_j = sequences[j]
            
            # Quick length filter
            len_ratio = min(len(seq_i), len(seq_j)) / max(len(seq_i), len(seq_j))
            if len_ratio < identity_threshold:
                continue
            
            # Compute sequence identity
            identity = SequenceMatcher(None, seq_i, seq_j).ratio()
            
            if identity >= identity_threshold:
                cluster_ids[j] = current_cluster
        
        current_cluster += 1
        
        if current_cluster % 50 == 0:
            print(f"   Processed {current_cluster} clusters...")
    
    df['cluster_id'] = cluster_ids
    df.to_csv(output_csv, index=False)
    
    n_clusters = df['cluster_id'].nunique()
    print(f"✅ Created {n_clusters} clusters")
    print(f"   Cluster sizes: min={df['cluster_id'].value_counts().min()}, "
          f"max={df['cluster_id'].value_counts().max()}")
    print(f"   Saved to: {output_csv}")
    
    return df


def run_cdhit_pipeline(
    features_csv: Path,
    output_csv: Path,
    identity: float = 0.40,
    cdhit_cmd: str = "cd-hit",
    word_size: int = 2,
    memory_mb: int = 16000,
    fallback_to_simple: bool = True,
) -> pd.DataFrame:
    """
    Generate FASTA, run CD-HIT, parse .clstr, write geometric_features_clustered.csv.
    If CD-HIT is not found and fallback_to_simple is True, uses create_simple_clusters instead.
    """
    def _cdhit_available(cmd: str) -> bool:
        return bool(shutil.which(cmd)) or (Path(cmd).is_file() if cmd else False)

    if not _cdhit_available(cdhit_cmd):
        if fallback_to_simple:
            print("CD-HIT not found; using built-in sequence clustering (--simple-clusters style).")
            return create_simple_clusters(features_csv, output_csv, identity_threshold=0.80)
        raise FileNotFoundError(
            f"CD-HIT not found: {cdhit_cmd}. Install CD-HIT, pass --cdhit-path, or use --simple-clusters."
        )

    work_dir = output_csv.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "sequences_cdhit.fasta"
    cluster_stem = work_dir / "clusters"
    clstr_path = Path(str(cluster_stem) + ".clstr")

    print("Step 1: Generating FASTA for CD-HIT...")
    generate_fasta(features_csv, fasta_path)

    print("Step 2: Running CD-HIT...")
    cmd = [
        cdhit_cmd,
        "-i", str(fasta_path),
        "-o", str(cluster_stem),
        "-c", str(identity),
        "-n", str(word_size),
        "-M", str(memory_mb),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"CD-HIT failed with return code {result.returncode}")
    if not clstr_path.exists():
        raise FileNotFoundError(f"CD-HIT did not produce {clstr_path}")

    print("Step 3: Parsing clusters and writing output...")
    return add_clusters_to_features(features_csv, clstr_path, output_csv)


def main():
    parser = argparse.ArgumentParser(description="Prepare sequence clusters for training")
    
    # Mode selection
    parser.add_argument('--generate-fasta', action='store_true',
                        help='Generate FASTA file for CD-HIT')
    parser.add_argument('--parse-clusters', action='store_true',
                        help='Parse CD-HIT output and add to features')
    parser.add_argument('--simple-clusters', action='store_true',
                        help='Create simple clusters without CD-HIT')
    parser.add_argument('--run-cdhit', action='store_true',
                        help='Generate FASTA, run CD-HIT, parse .clstr, write clustered CSV (all in one)')
    
    # Input/output
    parser.add_argument('--input', '-i', type=Path,
                        help='Input features CSV')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output file path')
    parser.add_argument('--clstr-file', type=Path,
                        help='CD-HIT .clstr file (for --parse-clusters)')
    parser.add_argument('--identity', type=float, default=0.80,
                        help='Sequence identity threshold for --simple-clusters (default: 0.80)')
    parser.add_argument('--cdhit-path', type=str, default='cd-hit',
                        help='Path to cd-hit executable (default: cd-hit from PATH)')
    parser.add_argument('--cdhit-identity', type=float, default=0.40,
                        help='CD-HIT -c threshold when using --run-cdhit (default: 0.40)')
    
    args = parser.parse_args()
    
    if args.run_cdhit:
        if not args.input or not args.output:
            parser.error("--run-cdhit requires --input and --output")
        run_cdhit_pipeline(
            args.input,
            args.output,
            identity=args.cdhit_identity,
            cdhit_cmd=args.cdhit_path,
        )
    elif args.generate_fasta:
        if not args.input or not args.output:
            parser.error("--generate-fasta requires --input and --output")
        generate_fasta(args.input, args.output)
        
    elif args.parse_clusters:
        if not args.input or not args.output or not args.clstr_file:
            parser.error("--parse-clusters requires --input, --output, and --clstr-file")
        add_clusters_to_features(args.input, args.clstr_file, args.output)
        
    elif args.simple_clusters:
        if not args.input or not args.output:
            parser.error("--simple-clusters requires --input and --output")
        create_simple_clusters(args.input, args.output, args.identity)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

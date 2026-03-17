#!/usr/bin/env python3
"""
Convert raw test CSV to the pipeline format expected by ESMFold.
"""
import argparse
import re
import pandas as pd
from pathlib import Path

def clean_name(name):
    """Clean the name to be safe for filenames and IDs."""
    name = str(name).strip()
    # Replace problematic characters
    name = name.replace(' ', '_').replace('?', '')
    # Replace any non-alphanumeric characters with underscore
    name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
    # Condense multiple underscores
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def main():
    parser = argparse.ArgumentParser(description="Convert test.csv to ESMFold format")
    parser.add_argument("--input", "-i", type=Path, default=Path("data/test/test.csv"), help="Path to input test.csv")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/test/test_seqs.txt"), help="Path to output text file")
    
    args = parser.parse_args()
    
    print(f"Reading {args.input}...")
    
    # Load the CSV, taking only the first two columns (ignoring the empty commas)
    try:
        df = pd.read_csv(args.input, usecols=[0, 1], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, usecols=[0, 1], encoding='latin1')
        
    df.columns = ['name', 'seq']
    
    # Drop any rows where sequence is empty
    df = df.dropna(subset=['seq'])
    
    # Write to the format ESMFold expects (name and sequence separated by a tab)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for _, row in df.iterrows():
            name = clean_name(row['name'])
            seq = str(row['seq']).strip()
            
            # ESMFold expects: ID Sequence (separated by whitespace)
            f.write(f"{name}\t{seq}\n")
            
    print(f"✅ Saved {len(df)} sequences to {args.output}")
    print("\nYou can now run ESMFold on these sequences (unlabeled, no AMP/decoy) with:")
    print(f"python models/run_esmfold_peptides.py --amp-file {args.output} --output {args.output.parent / 'structures'} --unlabeled")

if __name__ == "__main__":
    main()

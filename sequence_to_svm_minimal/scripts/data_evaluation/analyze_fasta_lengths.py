#!/usr/bin/env python3
"""
Script to analyze and visualize the sequence length distribution
of given FASTA files.
"""

import argparse
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def parse_fasta_lengths(fasta_path: Path) -> list[int]:
    """Parse a FASTA file and return a list of sequence lengths."""
    lengths = []
    current_len = 0
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_len > 0:
                    lengths.append(current_len)
                    current_len = 0
            else:
                current_len += len(line.replace(" ", "").replace("\n", ""))
        
        # Don't forget the last sequence
        if current_len > 0:
            lengths.append(current_len)
            
    return lengths

def main():
    parser = argparse.ArgumentParser(description="Plot sequence length distribution from FASTA files.")
    parser.add_argument("--fasta1", type=Path, required=True, help="Path to the first FASTA file (e.g., AMPs)")
    parser.add_argument("--label1", type=str, default="AMPs", help="Label for the first FASTA file")
    parser.add_argument("--fasta2", type=Path, required=False, help="Path to the second FASTA file (e.g., Decoys)")
    parser.add_argument("--label2", type=str, default="Decoys", help="Label for the second FASTA file")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Path to save the output plot")
    parser.add_argument("--title", type=str, default="Sequence Length Distribution", help="Title of the plot")
    
    args = parser.parse_args()
    
    if not args.fasta1.exists():
        print(f"Error: {args.fasta1} not found.")
        return
        
    lengths1 = parse_fasta_lengths(args.fasta1)
    print(f"Parsed {len(lengths1)} sequences from {args.fasta1.name}")
    print(f"  {args.label1} lengths: min={min(lengths1)}, max={max(lengths1)}, mean={sum(lengths1)/len(lengths1):.2f}")
    
    data_to_plot = {args.label1: lengths1}
    
    if args.fasta2:
        if not args.fasta2.exists():
            print(f"Error: {args.fasta2} not found.")
            return
        lengths2 = parse_fasta_lengths(args.fasta2)
        print(f"Parsed {len(lengths2)} sequences from {args.fasta2.name}")
        print(f"  {args.label2} lengths: min={min(lengths2)}, max={max(lengths2)}, mean={sum(lengths2)/len(lengths2):.2f}")
        data_to_plot[args.label2] = lengths2
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Using seaborn for better aesthetics
    sns.set_theme(style="whitegrid")
    
    for label, lengths in data_to_plot.items():
        sns.histplot(lengths, kde=True, label=f"{label} (n={len(lengths)})", alpha=0.5, bins=30)
    
    plt.title(args.title, fontsize=14)
    plt.xlabel("Sequence Length (Amino Acids)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    # Add summary statistics text box
    stats_text = ""
    for label, lengths in data_to_plot.items():
        stats_text += f"{label}:\n  Mean: {sum(lengths)/len(lengths):.1f}\n  Median: {sorted(lengths)[len(lengths)//2]}\n  Range: {min(lengths)}-{max(lengths)}\n\n"
        
    plt.annotate(stats_text.strip(), xy=(0.95, 0.5), xycoords='axes fraction', 
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 ha='right', va='center')
                 
    plt.tight_layout()
    
    # Setup output path
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"analyze_fasta_lengths_{timestamp}.png"
    else:
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    main()

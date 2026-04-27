"""
Batch ESMFold Structure Prediction with Resume Capability

Processes large datasets of peptide sequences through ESMFold with:
- Individual PDB file saves per sequence
- Checkpoint/resume functionality (survives interruptions)
- Progress tracking with detailed logging
- GPU memory management

Designed for the alpha helical SVM dataset (~18K sequences)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence


def load_checkpoint(checkpoint_file):
    """Load progress checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {
        'completed_indices': [],
        'failed_indices': [],
        'last_processed': None,
        'start_time': datetime.now().isoformat(),
        'total_time_seconds': 0
    }


def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save progress checkpoint"""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def parse_sequence_file(input_file):
    """
    Parse sequence file in SVM format:
    1 MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN
    2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG

    Lines whose sequence is not entirely standard 20 amino acids are skipped.

    Returns list of (index, sequence) tuples (uppercase canonical letters).
    """
    sequences = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(None, 1)
            if len(parts) == 2:
                idx, seq = parts[0].strip(), parts[1].strip()
            elif len(parts) == 1:
                seq = parts[0].strip()
                idx = str(len(sequences) + 1)
            else:
                continue

            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                continue

            sequences.append((idx, canon))

    return sequences


def load_esmfold_model(device="cuda"):
    """Load ESMFold model with memory optimizations"""
    from transformers import EsmForProteinFolding
    
    print(f"\n{'='*60}")
    print(f"  Loading ESMFold Model")
    print(f"{'='*60}")
    
    # Try local model first
    local_model_path = Path(__file__).parent / "esmfold_v1_local"
    
    # Load in FP16 for memory efficiency
    load_dtype = torch.float16 if device == "cuda" else torch.float32
    
    if local_model_path.exists():
        print(f"✅ Loading from local: {local_model_path}")
        model = EsmForProteinFolding.from_pretrained(
            str(local_model_path),
            local_files_only=True,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )
    else:
        print("⚠️  Local model not found, downloading from HuggingFace...")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )
    
    model = model.to(device)
    model.eval()
    
    if device == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Loaded in FP16 | GPU: {mem_used:.1f}/{mem_total:.1f} GB")
    
    return model


def predict_single_structure(model, sequence, device="cuda"):
    """Predict structure for a single sequence"""
    if device == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        pdb_string = model.infer_pdb(sequence)
    
    return pdb_string


def estimate_time(total, completed, elapsed_seconds):
    """Estimate remaining time based on progress"""
    if completed == 0:
        return "calculating..."
    
    avg_time_per_seq = elapsed_seconds / completed
    remaining = total - completed
    eta_seconds = remaining * avg_time_per_seq
    
    return str(timedelta(seconds=int(eta_seconds)))


def main():
    parser = argparse.ArgumentParser(
        description="Batch ESMFold with resume capability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire dataset (with automatic resume on interruption)
  python batch_esmfold.py --input seqs.txt --output structures/

  # Process first 100 sequences only
  python batch_esmfold.py --input seqs.txt --output structures/ --limit 100

  # Start from specific sequence index
  python batch_esmfold.py --input seqs.txt --output structures/ --start 500

  # Reset and start fresh (ignore checkpoint)
  python batch_esmfold.py --input seqs.txt --output structures/ --reset
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input sequence file (SVM format)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for PDB files')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of sequences to process')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start from sequence index (0-based)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset checkpoint and start fresh')
    parser.add_argument('--max-length', type=int, default=1000,
                        help='Maximum sequence length (default: 1000)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "🧬" * 30)
    print("   Batch ESMFold Structure Prediction")
    print("🧬" * 30)
    print(f"   Resume capability: ENABLED")
    print()
    
    # Check GPU
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint file in output directory
    checkpoint_file = output_dir / "checkpoint.json"
    
    # Load or reset checkpoint
    if args.reset and checkpoint_file.exists():
        print("🔄 Resetting checkpoint...")
        checkpoint_file.unlink()
    
    checkpoint = load_checkpoint(checkpoint_file)
    completed_set = set(checkpoint['completed_indices'])
    
    # Load sequences
    print(f"\n{'='*60}")
    print(f"  Loading Sequences")
    print(f"{'='*60}")
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    all_sequences = parse_sequence_file(args.input)
    print(f"✅ Total sequences in file: {len(all_sequences)}")
    
    # Apply start/limit filters
    sequences = all_sequences[args.start:]
    if args.limit:
        sequences = sequences[:args.limit]
    
    print(f"   Processing range: {args.start} to {args.start + len(sequences) - 1}")
    print(f"   Sequences to process: {len(sequences)}")
    
    # Filter out already completed
    sequences_to_process = [
        (idx, seq) for idx, seq in sequences 
        if idx not in completed_set
    ]
    
    print(f"   Already completed: {len(completed_set)}")
    print(f"   Remaining: {len(sequences_to_process)}")
    
    if not sequences_to_process:
        print("\n✅ All sequences already processed!")
        sys.exit(0)
    
    # Load model
    model = load_esmfold_model(args.device)
    
    # Process sequences
    print(f"\n{'='*60}")
    print(f"  Predicting Structures")
    print(f"{'='*60}")
    
    # Results log
    results_file = output_dir / "results_log.csv"
    results_exist = results_file.exists()
    
    batch_start_time = time.time()
    successful = 0
    failed = 0
    
    # Open results file in append mode
    with open(results_file, 'a') as results_f:
        # Write header if new file
        if not results_exist:
            results_f.write("seqIndex,sequence,length,status,pdb_file,time_seconds,timestamp\n")
        
        # Progress bar
        pbar = tqdm(sequences_to_process, desc="Folding", unit="seq")
        
        for idx, seq in pbar:
            # Check length
            if len(seq) > args.max_length:
                result = {
                    'seqIndex': idx,
                    'sequence': seq,
                    'length': len(seq),
                    'status': 'skipped_too_long',
                    'pdb_file': '',
                    'time_seconds': 0,
                    'timestamp': datetime.now().isoformat()
                }
                checkpoint['failed_indices'].append(idx)
                failed += 1
            else:
                # Predict structure
                try:
                    start_time = time.time()
                    pdb_string = predict_single_structure(model, seq, args.device)
                    elapsed = time.time() - start_time
                    
                    # Save PDB file
                    pdb_file = output_dir / f"structure_{idx}.pdb"
                    with open(pdb_file, 'w') as f:
                        f.write(pdb_string)
                    
                    result = {
                        'seqIndex': idx,
                        'sequence': seq,
                        'length': len(seq),
                        'status': 'success',
                        'pdb_file': str(pdb_file),
                        'time_seconds': round(elapsed, 2),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    checkpoint['completed_indices'].append(idx)
                    successful += 1
                    
                except Exception as e:
                    result = {
                        'seqIndex': idx,
                        'sequence': seq,
                        'length': len(seq),
                        'status': f'error: {str(e)[:100]}',
                        'pdb_file': '',
                        'time_seconds': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                    checkpoint['failed_indices'].append(idx)
                    failed += 1
                    tqdm.write(f"❌ Error on {idx}: {str(e)[:50]}")
            
            # Write result immediately (no data loss on crash)
            results_f.write(
                f"{result['seqIndex']},"
                f"{result['sequence']},"
                f"{result['length']},"
                f"{result['status']},"
                f"{result['pdb_file']},"
                f"{result['time_seconds']},"
                f"{result['timestamp']}\n"
            )
            results_f.flush()
            
            # Update checkpoint
            checkpoint['last_processed'] = idx
            checkpoint['total_time_seconds'] = time.time() - batch_start_time
            save_checkpoint(checkpoint_file, checkpoint)
            
            # Update progress bar
            total_done = successful + failed
            elapsed_total = time.time() - batch_start_time
            eta = estimate_time(len(sequences_to_process), total_done, elapsed_total)
            pbar.set_postfix({
                'done': f"{successful}✓ {failed}✗",
                'eta': eta
            })
    
    # Final summary
    total_time = time.time() - batch_start_time
    
    print(f"\n{'='*60}")
    print("  ✅ Batch Processing Complete!")
    print(f"{'='*60}")
    print(f"   Successful: {successful}")
    print(f"   Failed:     {failed}")
    print(f"   Total time: {timedelta(seconds=int(total_time))}")
    print(f"   Output dir: {output_dir}")
    print(f"   Results:    {results_file}")
    print(f"   Checkpoint: {checkpoint_file}")
    print()
    
    # Summary statistics
    if successful > 0:
        avg_time = total_time / successful
        print(f"   Avg time per sequence: {avg_time:.2f}s")
    
    print("\n💡 To resume if interrupted: Run the same command again!")
    print("   The script will automatically skip completed sequences.\n")


if __name__ == "__main__":
    main()

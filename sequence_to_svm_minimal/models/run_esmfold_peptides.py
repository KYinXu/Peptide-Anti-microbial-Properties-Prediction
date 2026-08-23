"""
ESMFold Inference for Peptide Dataset

Runs ESMFold structure prediction on AMP and decoy peptide sequences with:
- Individual PDB file saves per sequence (IMMEDIATELY after folding)
- Checkpoint/resume functionality (survives interruptions)
- Class labels preserved (AMP=+1, decoy=-1)
- Progress tracking with detailed logging

SAFEGUARDS:
1. Each PDB saved immediately after folding - no data loss on crash
2. Results CSV written with flush() after each sequence
3. Checkpoint JSON updated after each sequence
4. Can resume from any interruption point
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from peptide_pipeline.esmfold_sequences import (
    ParseStats,
    iter_esmfold_inputs,
    summarize_esmfold_work,
)


def load_checkpoint(checkpoint_file):
    """Load progress checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {
        'completed_ids': [],
        'failed_ids': [],
        'last_processed': None,
        'start_time': datetime.now().isoformat(),
        'total_time_seconds': 0,
        'amp_completed': 0,
        'decoy_completed': 0
    }


def save_checkpoint(checkpoint_file, checkpoint_data):
    """Save progress checkpoint IMMEDIATELY"""
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, separators=(",", ":"))


def _input_iter_kwargs(args):
    amp = args.amp_file if os.path.exists(args.amp_file) else None
    decoy = args.decoy_file if os.path.exists(args.decoy_file) else None
    return {
        "unlabeled": args.unlabeled,
        "amp_file": amp,
        "decoy_file": decoy,
        "amp_only": args.amp_only,
        "decoy_only": args.decoy_only,
    }


def _write_result_row(results_f, result):
    results_f.write(
        f"{result['unique_id']},"
        f"{result['original_idx']},"
        f"{result['sequence']},"
        f"{result['length']},"
        f"{result['label']},"
        f"{result['status']},"
        f"{result['pdb_file']},"
        f"{result['time_seconds']},"
        f"{result['timestamp']}\n"
    )
    results_f.flush()


def _process_one_sequence(
    unique_id,
    orig_idx,
    seq,
    label,
    *,
    args,
    sequences_dir,
    amp_dir,
    decoy_dir,
    model,
    output_dir,
):
    if args.unlabeled:
        pdb_subdir = sequences_dir
        class_name = "seq"
    elif label == 1:
        pdb_subdir = amp_dir
        class_name = "AMP"
    else:
        pdb_subdir = decoy_dir
        class_name = "DECOY"

    if len(seq) > args.max_length:
        result = {
            "unique_id": unique_id,
            "original_idx": orig_idx,
            "sequence": seq,
            "length": len(seq),
            "label": label,
            "status": "skipped_too_long",
            "pdb_file": "",
            "time_seconds": 0,
            "timestamp": datetime.now().isoformat(),
        }
        return result, class_name, False

    start_time = time.time()
    pdb_string = predict_single_structure(model, seq, args.device)
    elapsed = time.time() - start_time
    pdb_file = pdb_subdir / f"{unique_id}.pdb"
    with open(pdb_file, "w") as f:
        f.write(pdb_string)
    del pdb_string
    result = {
        "unique_id": unique_id,
        "original_idx": orig_idx,
        "sequence": seq,
        "length": len(seq),
        "label": label,
        "status": "success",
        "pdb_file": str(pdb_file.relative_to(output_dir)),
        "time_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }
    return result, class_name, True


def load_esmfold_model(device="cuda"):
    """Load ESMFold model with memory optimizations"""
    from transformers import EsmForProteinFolding
    
    print(f"\n{'='*60}")
    print(f"  Loading ESMFold Model")
    print(f"{'='*60}")
    
    local_model_path = Path(__file__).parent / "esmfold_v1_local"
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
        description="ESMFold Inference for Peptide Dataset (AMP + Decoy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on peptide dataset
  python run_esmfold_peptides.py --output structures/

  # Run with custom input files
  python run_esmfold_peptides.py \\
      --amp-file data/training_dataset/seqs_AMP.txt \\
      --decoy-file data/training_dataset/seqs_decoy_subsample.txt \\
      --output structures/

  # Resume after interruption (just run same command)
  python run_esmfold_peptides.py --output structures/

  # Reset and start fresh
  python run_esmfold_peptides.py --output structures/ --reset

  # Unlabeled: bare lines → SEQ_n; two-field with numeric id → SEQ_n; accession-style id → <id>.pdb
  python run_esmfold_peptides.py --amp-file data/test_seqs.txt --output structures/ --unlabeled
        """
    )
    
    script_dir = Path(__file__).parent.parent
    default_amp = script_dir / "data" / "training_dataset" / "seqs_AMP.txt"
    default_decoy = script_dir / "data" / "training_dataset" / "seqs_decoy_subsample.txt"
    
    parser.add_argument('--amp-file', '-a', type=str,
                        default=str(default_amp),
                        help='Input AMP sequences file')
    parser.add_argument('--decoy-file', '-d', type=str,
                        default=str(default_decoy),
                        help='Input decoy sequences file')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for PDB files')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--reset', action='store_true',
                        help='Reset checkpoint and start fresh')
    parser.add_argument('--max-length', type=int, default=200,
                        help='Maximum sequence length (default: 200)')
    parser.add_argument('--amp-only', action='store_true',
                        help='Only process AMP sequences')
    parser.add_argument('--decoy-only', action='store_true',
                        help='Only process decoy sequences')
    parser.add_argument('--unlabeled', action='store_true',
                        help='Unlabeled dataset: single sequence file, no AMP/decoy; PDBs in sequences/; bare lines → SEQ_n; numeric id + seq → SEQ_n; non-numeric id → <id>.pdb; label=0 in results_log')
    
    args = parser.parse_args()
    
    print("\n" + "🧬" * 30)
    print("   ESMFold Peptide Structure Inference")
    print("🧬" * 30)
    print(f"   Resume capability: ENABLED")
    print(f"   Immediate save: ENABLED (no data loss)")
    print()
    
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = Path(args.output)
    sequences_dir = amp_dir = decoy_dir = None
    if args.unlabeled:
        sequences_dir = output_dir / "sequences"
        output_dir.mkdir(parents=True, exist_ok=True)
        sequences_dir.mkdir(exist_ok=True)
    else:
        amp_dir = output_dir / "AMP"
        decoy_dir = output_dir / "DECOY"
        output_dir.mkdir(parents=True, exist_ok=True)
        amp_dir.mkdir(exist_ok=True)
        decoy_dir.mkdir(exist_ok=True)
    
    checkpoint_file = output_dir / "checkpoint.json"
    
    if args.reset and checkpoint_file.exists():
        print("🔄 Resetting checkpoint...")
        checkpoint_file.unlink()
    
    checkpoint = load_checkpoint(checkpoint_file)
    completed_set = set(checkpoint['completed_ids'])
    
    print(f"\n{'='*60}")
    print(f"  Loading Sequences")
    print(f"{'='*60}")

    if args.unlabeled:
        if not os.path.exists(args.amp_file):
            print(f"❌ Sequence file not found: {args.amp_file}")
            sys.exit(1)
        print("✅ Unlabeled input (streamed; no class distinction)")
    else:
        if not args.decoy_only and not os.path.exists(args.amp_file):
            print(f"❌ AMP file not found: {args.amp_file}")
            sys.exit(1)
        if not args.amp_only and not os.path.exists(args.decoy_file):
            print(f"❌ Decoy file not found: {args.decoy_file}")
            sys.exit(1)
        if not args.decoy_only:
            print(f"✅ AMP file: {args.amp_file}")
        if not args.amp_only:
            print(f"✅ Decoy file: {args.decoy_file}")

    parse_stats = ParseStats()
    summary = summarize_esmfold_work(
        iter_esmfold_inputs(stats=parse_stats, **_input_iter_kwargs(args)),
        completed_ids=completed_set,
        max_length=args.max_length,
    )

    if parse_stats.n_skipped_invalid:
        print(
            f"   Skipped {parse_stats.n_skipped_invalid} lines (non-standard letters or X; "
            "only standard 20 amino acids accepted)"
        )

    print("   Mode: stream from disk (one sequence in memory at a time)")
    print(f"   Total sequences: {summary.n_valid}")
    print(f"   Already completed: {len(completed_set)}")
    print(f"   Remaining: {summary.n_remaining}")
    print(f"   Foldable (len ≤ {args.max_length}): {summary.n_foldable}")

    if summary.n_remaining == 0:
        print("\n✅ All sequences already processed!")
        sys.exit(0)

    model = None
    if summary.n_foldable > 0:
        model = load_esmfold_model(args.device)
    else:
        print("   No foldable sequences; skipping ESMFold model load")

    print(f"\n{'='*60}")
    print(f"  Running ESMFold Inference")
    print(f"{'='*60}")

    results_file = output_dir / "results_log.csv"
    results_exist = results_file.exists()

    batch_start_time = time.time()
    successful = 0
    failed = 0

    with open(results_file, "a") as results_f:
        if not results_exist:
            results_f.write(
                "unique_id,original_idx,sequence,length,label,status,pdb_file,time_seconds,timestamp\n"
            )

        pbar = tqdm(total=summary.n_remaining, desc="Folding", unit="seq")
        try:
            for unique_id, orig_idx, seq, label in iter_esmfold_inputs(**_input_iter_kwargs(args)):
                if unique_id in completed_set:
                    continue
                try:
                    result, class_name, ok = _process_one_sequence(
                        unique_id,
                        orig_idx,
                        seq,
                        label,
                        args=args,
                        sequences_dir=sequences_dir,
                        amp_dir=amp_dir,
                        decoy_dir=decoy_dir,
                        model=model,
                        output_dir=output_dir,
                    )
                except Exception as e:
                    class_name = "seq" if args.unlabeled else ("AMP" if label == 1 else "DECOY")
                    result = {
                        "unique_id": unique_id,
                        "original_idx": orig_idx,
                        "sequence": seq,
                        "length": len(seq),
                        "label": label,
                        "status": f"error: {str(e)[:100]}",
                        "pdb_file": "",
                        "time_seconds": 0,
                        "timestamp": datetime.now().isoformat(),
                    }
                    checkpoint["failed_ids"].append(unique_id)
                    failed += 1
                    tqdm.write(f"❌ Error on {unique_id}: {str(e)[:50]}")
                    ok = False

                if ok:
                    checkpoint["completed_ids"].append(unique_id)
                    completed_set.add(unique_id)
                    if args.unlabeled:
                        checkpoint["sequences_completed"] = checkpoint.get("sequences_completed", 0) + 1
                    elif label == 1:
                        checkpoint["amp_completed"] = checkpoint.get("amp_completed", 0) + 1
                    else:
                        checkpoint["decoy_completed"] = checkpoint.get("decoy_completed", 0) + 1
                    successful += 1
                elif result["status"] == "skipped_too_long":
                    checkpoint["failed_ids"].append(unique_id)
                    failed += 1

                _write_result_row(results_f, result)

                checkpoint["last_processed"] = unique_id
                checkpoint["total_time_seconds"] = time.time() - batch_start_time
                save_checkpoint(checkpoint_file, checkpoint)

                total_done = successful + failed
                elapsed_total = time.time() - batch_start_time
                eta = estimate_time(summary.n_remaining, total_done, elapsed_total)
                pbar.update(1)
                pbar.set_postfix({
                    "done": f"{successful}✓ {failed}✗",
                    "eta": eta,
                    "class": class_name,
                })
        finally:
            pbar.close()
    
    total_time = time.time() - batch_start_time
    
    print(f"\n{'='*60}")
    print("  ✅ ESMFold Inference Complete!")
    print(f"{'='*60}")
    print(f"   Successful: {successful}")
    print(f"   Failed:     {failed}")
    print(f"   Total time: {timedelta(seconds=int(total_time))}")
    print(f"\n   Output structure:")
    print(f"   ├── {output_dir}/")
    if args.unlabeled:
        print(f"   │   ├── sequences/     ({checkpoint.get('sequences_completed', 0)} structures)")
    else:
        print(f"   │   ├── AMP/           ({checkpoint.get('amp_completed', 0)} structures)")
        print(f"   │   ├── DECOY/         ({checkpoint.get('decoy_completed', 0)} structures)")
    print(f"   │   ├── results_log.csv")
    print(f"   │   └── checkpoint.json")
    print()
    
    if successful > 0:
        avg_time = total_time / successful
        print(f"   Avg time per sequence: {avg_time:.2f}s")
    
    print("\n💡 To resume if interrupted: Run the same command again!\n")


if __name__ == "__main__":
    main()

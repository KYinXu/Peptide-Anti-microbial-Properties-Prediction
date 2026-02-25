#!/usr/bin/env python3
"""
All-in-one pipeline:
- Input: sequences file (2-column: seqIndex, sequence), aaindex dir, model pkl, scaler csv
- Steps: generate descriptors (Python 3 generator) -> run SVM predictions
- Output: descriptors.csv, descriptors_PREDICTIONS_unsorted.csv, descriptors_PREDICTIONS.csv inside output dir
"""
import argparse
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC_PY = os.path.join(ROOT, "descriptors", "descripGen_12_py3.py")
PRED_PY = os.path.join(ROOT, "predictionsParameters", "predictSVC.py")


def run(cmd_list, cwd: str):
    """Run a command given as a list of arguments."""
    print(f"[RUN] {' '.join(cmd_list)}")
    proc = subprocess.run(cmd_list, cwd=cwd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def main():
    ap = argparse.ArgumentParser(description="All-in-one GenomeClassifier pipeline (descriptors + SVM)")
    ap.add_argument("--seqs", required=True, help="Path to sequences file (2 columns: seqIndex sequence)")
    ap.add_argument("--aaindex", required=True, help="Path to aaindex directory (contains aaindex1/2/3)")
    ap.add_argument("--output-dir", required=True, help="Directory to write descriptors / predictions")
    ap.add_argument("--model-pkl", required=True, help="Path to svc.pkl (legacy pre-0.18 pickle)")
    ap.add_argument("--scaler-csv", required=True, help="Path to Z_score_mean_std__intersect_noflip.csv")
    ap.add_argument("--start", type=int, default=1, help="1-based start index in seqs file (default: 1)")
    ap.add_argument("--stop", type=int, default=None, help="1-based stop index in seqs file (default: all)")
    args = ap.parse_args()

    # Resolve absolute paths
    seqs = os.path.abspath(args.seqs)
    aaindex = os.path.abspath(args.aaindex)
    outdir = os.path.abspath(args.output_dir)
    model = os.path.abspath(args.model_pkl)
    scaler = os.path.abspath(args.scaler_csv)

    os.makedirs(outdir, exist_ok=True)

    # Determine stop index
    if args.stop is None:
        with open(seqs, 'r') as f:
            n_lines = sum(1 for _ in f)
        stop = n_lines
    else:
        stop = args.stop

    # 1) Generate descriptors
    cmd_desc = [
        sys.executable,
        DESC_PY,
        os.path.normpath(aaindex),
        seqs,
        os.path.normpath(outdir),
        str(args.start),
        str(stop)
    ]
    run(cmd_desc, cwd=ROOT)

    # 2) Run SVM predictions
    descriptors_csv = os.path.join(outdir, "descriptors.csv")
    cmd_pred = [
        sys.executable,
        PRED_PY,
        descriptors_csv,
        scaler,
        model
    ]
    run(cmd_pred, cwd=ROOT)

    # 3) Report outputs
    print("\nDONE.")
    print(f"Descriptors: {descriptors_csv}")
    print(f"Predictions (unsorted): {os.path.join(outdir, 'descriptors_PREDICTIONS_unsorted.csv')}")
    print(f"Predictions (sorted):   {os.path.join(outdir, 'descriptors_PREDICTIONS.csv')}")


if __name__ == "__main__":
    main()


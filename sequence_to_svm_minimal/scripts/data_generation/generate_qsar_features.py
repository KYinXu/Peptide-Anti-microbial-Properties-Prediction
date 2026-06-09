#!/usr/bin/env python3
"""
Generate QSAR-12 descriptor features for a peptide dataset.

Input: CSV with 'sequence' column (and optional 'peptide_id' or 'name'), or a text file
       with one sequence per line or "id\tsequence" lines.
Output: CSV with peptide_id, sequence, and 12 QSAR columns.

Usage:
    python generate_qsar_features.py --input data.csv --output qsar12_descriptors.csv
    python generate_qsar_features.py -i sequences.txt -o qsar.csv
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# QSAR-12 column names for output
QSAR_COLUMNS = [
    'netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK',
    '_SolventAccessibilityD1025', 'tau2_GRAR740104', 'tau4_GRAR740104',
    'QSO50_GRAR740104', 'QSO29_GRAR740104'
]

CHARGE_DICT = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0, "H": 1, "I": 0,
    "K": 1, "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1, "S": 0,
    "T": 0, "V": 0, "W": 0, "Y": 0
}


def _compute_one(seq: str, pid: str) -> dict:
    from propy.PyPro import GetProDes
    from propy import ProCheck

    if ProCheck.ProteinCheck(seq) == 0:
        raise ValueError(f"Invalid sequence: {seq}")

    Des = GetProDes(seq)
    row = {'peptide_id': pid, 'sequence': seq}

    row['netCharge'] = sum(CHARGE_DICT.get(x, 0) for x in seq)

    dpc = Des.GetDPComp()
    for handle in ['FC', 'LW', 'DP', 'NK', 'AE']:
        row[handle] = round(dpc.get(handle, 0), 2)

    n_m = sum(1 for x in seq if x == 'M')
    n_k = sum(1 for x in seq if x == 'K')
    row['pcMK'] = 0 if n_m == 0 else n_m / (n_m + n_k) if (n_m + n_k) > 0 else 0

    ctd = Des.GetCTD()
    row['_SolventAccessibilityD1025'] = ctd.get('_SolventAccessibilityD1025', 0)

    try:
        socn = Des.GetSOCN(maxlag=30)
        seq_len = len(seq)
        row['tau2_GRAR740104'] = socn.get('tau2', 0) / (seq_len - 2) if seq_len > 2 else 0
        row['tau4_GRAR740104'] = socn.get('tau4', 0) / (seq_len - 4) if seq_len > 4 else 0
    except Exception:
        row['tau2_GRAR740104'] = 0
        row['tau4_GRAR740104'] = 0

    try:
        qso = Des.GetQSO(maxlag=30, weight=0.05)
        row['QSO50_GRAR740104'] = qso.get('QSO50', 0)
        row['QSO29_GRAR740104'] = qso.get('QSO29', 0)
    except Exception:
        row['QSO50_GRAR740104'] = 0
        row['QSO29_GRAR740104'] = 0

    return row


def _empty_row(pid: str, seq: str) -> dict:
    row = {'peptide_id': pid, 'sequence': seq}
    for name in QSAR_COLUMNS:
        row[name] = 0
    return row


def compute_qsar12(sequences: List[str], peptide_ids: List[str]) -> pd.DataFrame:
    results = []
    n = len(sequences)
    for i, (seq, pid) in enumerate(zip(sequences, peptide_ids)):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{n}...")
        try:
            results.append(_compute_one(seq, pid))
        except Exception:
            results.append(_empty_row(pid, seq))
    return pd.DataFrame(results)


def load_input(path: Path) -> Tuple[List[str], List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(path)
        seq_col = 'sequence' if 'sequence' in df.columns else df.columns[1]
        id_col = None
        for c in ('peptide_id', 'name', 'id', 'ID'):
            if c in df.columns:
                id_col = c
                break
        if id_col is None:
            id_col = df.columns[0]
        sequences = df[seq_col].astype(str).str.strip().tolist()
        peptide_ids = df[id_col].astype(str).tolist()
        return sequences, peptide_ids

    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    ids = []
    seqs = []
    for i, line in enumerate(lines):
        if '\t' in line:
            part = line.split('\t', 1)
            ids.append(part[0].strip())
            seqs.append(part[1].strip())
        elif ' ' in line:
            part = line.split(None, 1)
            if len(part) == 2 and part[1] and all(c.isalpha() for c in part[1].upper() if c.isalpha()):
                ids.append(part[0].strip())
                seqs.append(part[1].strip())
            else:
                ids.append(f"seq_{i+1}")
                seqs.append(line)
        else:
            ids.append(f"seq_{i+1}")
            seqs.append(line)
    return seqs, ids


def main():
    parser = argparse.ArgumentParser(description="Generate QSAR-12 features for a peptide dataset")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input CSV (with 'sequence' column) or text file (one sequence per line or id\\tseq)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path for QSAR-12 descriptors")
    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    sequences, peptide_ids = load_input(args.input)
    if not sequences:
        print("No sequences found in input.", file=sys.stderr)
        sys.exit(1)
    print(f"Computing QSAR-12 for {len(sequences)} sequences...")
    df = compute_qsar12(sequences, peptide_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

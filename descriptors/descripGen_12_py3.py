"""
Python 3 version of descriptors/descripGen_12.py

Generates descriptors using the vendored Python 3-converted propy located at
`descriptors/propy3_src/`.

IN:     aaindexDirPath   - directory containing aaindex1, aaindex2, aaindex3
        filename        - text file: each non-empty line is either
                          "index sequence" (index int, rest = sequence) or
                          "sequence" only (1-based line index assigned in file order).
                          Lines starting with # are skipped.
        outputFolder    - directory where descriptors.csv will be written
        startIndex      - 1-based index of first row to process
        stopIndex       - 1-based index of last row to process (inclusive)

OUT:    <outputFolder>/descriptors.csv

Notes:
- This script mirrors the original Python 2 implementation for identical outputs.
- It auto-adds `descriptors/propy3_src` to sys.path so users do not need to set PYTHONPATH.
"""

import os
import sys
import time
import math
import random
from typing import List, Tuple

import numpy as np
import numpy.matlib

# Auto-add vendored, converted propy3 package to import path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer vendored converted package (exact semantics)
_VENDOR_PARENT = os.path.join(_THIS_DIR, "propy3_vendor")
_VENDOR_INIT = os.path.join(_VENDOR_PARENT, "propy", "__init__.py")
if os.path.exists(_VENDOR_INIT) and _VENDOR_PARENT not in sys.path:
    sys.path.insert(0, _VENDOR_PARENT)

# Fallback: older converted tree used earlier in this repo
_PROPY3_PATH = os.path.join(_THIS_DIR, "propy3_src")
if os.path.exists(_PROPY3_PATH) and _PROPY3_PATH not in sys.path:
    sys.path.insert(0, _PROPY3_PATH)

import propy  # type: ignore
from propy import ProCheck  # type: ignore
from propy import AAIndex  # type: ignore
from propy.PyPro import GetProDes  # type: ignore


def _usage() -> None:
    print(f"USAGE: {sys.argv[0]}  <aaindexDirPath> <filename> <outputFolder> <startIndex> <stopIndex>")
    print("       aaindexDirPath    = directory containing aaindex1, aaindex2, aaindex3")
    print("       filename          = per line: 'index sequence' (index int) OR one sequence only (auto index)")
    print("       outputFolder      = directory path to write descriptors.csv (must be writable)")
    print("       startIndex        = 1-based index of first row to process")
    print("       stopIndex         = 1-based index of last row to process (inclusive)")


def unique(a: List[str]) -> List[str]:
    return list(set(a))


def intersect(a: List[str], b: List[str]) -> List[str]:
    return list(set(a) & set(b))


def union(a: List[str], b: List[str]) -> List[str]:
    return list(set(a) | set(b))


def descripGen_bespoke(proseq: str, aap_dict_grar740104: dict) -> Tuple[List[str], np.ndarray]:
    v_lambda = 7    # tiers for PseAAC; must be < min peptide length
    v_nlag = 30     # max aa separation for sequence-order features
    v_weight = 0.05 # PseAAC weight

    descNames: List[str] = []
    descValues = np.empty([0,])

    if ProCheck.ProteinCheck(proseq) == 0:
        print(f"ERROR: protein sequence {proseq} is invalid")
        sys.exit(1)

    seqLength = len(proseq)

    Des = GetProDes(proseq)

    # 1: netCharge
    chargeDict = {"A":0, "C":0, "D":-1, "E":-1, "F":0, "G":0, "H":1, "I":0, "K":1, "L":0, "M":0, "N":0, "P":0, "Q":0, "R":1, "S":0, "T":0, "V":0, "W":0, "Y":0}
    netCharge = sum([chargeDict.get(x, 0) for x in proseq])
    descNames.append('netCharge')
    descValues = np.append(descValues, netCharge)

    # 2-6: FC, LW, DP, NK, AE
    # Force legacy two-decimal formatting to avoid tiny 0.01 drift vs Py2
    dpc = Des.GetDPComp()
    for handle in ['FC', 'LW', 'DP', 'NK', 'AE']:
        descNames.append(handle)
        val = float('%.2f' % dpc[handle])
        descValues = np.append(descValues, val)

    # 7: pcMK
    pp = 'M'
    qq = 'K'
    Npp = sum(1 for x in proseq if x == pp)
    Nqq = sum(1 for x in proseq if x == qq)
    pc_pp_qq = 0 if Npp == 0 else float(Npp) / float(Npp + Nqq)
    descNames.append('pc' + pp + qq)
    descValues = np.append(descValues, pc_pp_qq)

    # 8: _SolventAccessibilityD1025
    ctd = Des.GetCTD()
    for handle in ['_SolventAccessibilityD1025']:
        descNames.append(handle)
        descValues = np.append(descValues, ctd[handle])

    # 9-10: tau2_GRAR740104, tau4_GRAR740104
    prop = 'GRAR740104'
    # Reuse preloaded AAIndex dict (very expensive to parse repeatedly)
    socn_p = Des.GetSOCNp(maxlag=v_nlag, distancematrix=aap_dict_grar740104)
    for handle in ['tau2', 'tau4']:
        delta = float(handle[3:])
        if delta > (seqLength - 1):
            value = 0
        else:
            value = socn_p[handle] / float(seqLength - delta)
        descNames.append(handle + '_' + prop)
        descValues = np.append(descValues, value)

    # 11-12: QSO50_GRAR740104, QSO29_GRAR740104
    prop = 'GRAR740104'
    # Reuse the same preloaded dict here as well
    qso_p = Des.GetQSOp(maxlag=v_nlag, weight=v_weight, distancematrix=aap_dict_grar740104)
    for handle in ['QSO50', 'QSO29']:
        descNames.append(handle + '_' + prop)
        descValues = np.append(descValues, qso_p[handle])

    return descNames, descValues


# main
if __name__ == "__main__":
    if len(sys.argv) != 6:
        _usage()
        sys.exit(1)

    aaindex_path = sys.argv[1]
    inFile = sys.argv[2]
    outputFolder = str(sys.argv[3])
    startIndex = int(sys.argv[4])
    stopIndex = int(sys.argv[5])

    # load sequences
    indices: List[int] = []
    seqs: List[str] = []
    with open(inFile, 'r') as seqsFile:
        for line in seqsFile:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                try:
                    a = int(parts[0])
                except ValueError:
                    print(f"ERROR: two-field line must start with integer index, got: {line[:80]!r}")
                    sys.exit(1)
                b = str(parts[1]).strip()
            else:
                a = len(indices) + 1
                b = str(parts[0]).strip()
            indices.append(a)
            seqs.append(b)

    if startIndex < 1:
        print("ERROR: startIndex < 1")
        sys.exit(1)
    if stopIndex > len(indices):
        print(f"ERROR: stopIndex > # seqs in {inFile}")
        sys.exit(1)
    if startIndex > stopIndex:
        print("ERROR: startIndex >= stopIndex")
        sys.exit(1)

    # v_lambda must be < min seq length (same constraint as original)
    v_lambda = 7
    if v_lambda >= len(min(seqs, key=len)):
        print(f"ERROR: v_lambda = {v_lambda} >= minimum seq length in {inFile}")
        sys.exit(1)

    t_start = time.perf_counter()

    # Preload the AAIndex property used by SOCNp and QSOp once (major speedup)
    try:
        aap_dict_grar740104 = AAIndex.GetAAIndex23('GRAR740104', path=aaindex_path)
    except Exception as e:
        print(f"ERROR: failed to load AAIndex GRAR740104 from {aaindex_path}: {e}")
        sys.exit(1)

    out_path = os.path.join(outputFolder, "descriptors.csv")
    with open(out_path, 'w') as fout:
        print(f"Commencing descriptor generation for selected sequences in {inFile}")
        for SEQIDX in range(startIndex - 1, stopIndex):
            # progress (reduce print frequency to cut IO overhead)
            if ((SEQIDX + 1) % 200) == 0 or SEQIDX == (startIndex - 1):
                print(f"    Processing sequence {SEQIDX + 1} ({stopIndex - SEQIDX - 1} more sequences remain)")

            proseq = seqs[SEQIDX]
            seqIndex = indices[SEQIDX]
            if ProCheck.ProteinCheck(proseq) == 0:
                print(f"ERROR: protein sequence {proseq} is invalid")
                sys.exit(1)

            # compute descriptors
            descNames, descValues = descripGen_bespoke(proseq, aap_dict_grar740104)

            # write header once
            if SEQIDX == (startIndex - 1):
                fout.write('seqIndex,')
                for ii in range(0, len(descNames)):
                    fout.write(f"{descNames[ii]}")
                    if ii < (len(descNames) - 1):
                        fout.write(',')
                    else:
                        fout.write('\n')

            # write values
            fout.write(f"{seqIndex},")
            for ii in range(0, len(descValues)):
                fout.write(f"{descValues[ii]}")
                if ii < (len(descNames) - 1):
                    fout.write(',')
                else:
                    fout.write('\n')

    print("DONE!")

    t_stop = time.perf_counter()
    total = (t_stop - t_start)
    denom = (stopIndex - startIndex + 1)
    per = (total / denom) if denom > 0 else float('nan')
    print(f"Elapsed time      = {total:.2f} s")
    print(f"Time per sequence = {per:.2f} s")

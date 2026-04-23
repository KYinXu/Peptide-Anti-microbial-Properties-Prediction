#!/usr/bin/env python3
"""
Run the data pipeline on a single sequence string or a small input file.

Forwards all extra arguments to ``scripts/run_data_pipeline.py`` (e.g. ``--window-min-len``).

From ``sequence_to_svm_minimal``::

  python scripts/analyze_individual_samples.py --sequence "ACDEFGHIKLMNPQRSTVWY" -w data/my_one_off/generated \\
      --window-min-len 10 --window-max-len 15 --window-stride 1

  python scripts/analyze_individual_samples.py --input path/to/seqs.txt -w data/my_run/generated
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convenience wrapper: one sequence or file → run_data_pipeline.py",
    )
    ap.add_argument(
        "--sequence",
        "-s",
        type=str,
        default=None,
        help="Inline peptide sequence (one sample). Mutually exclusive with --input.",
    )
    ap.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Existing sequence file (TXT/FASTA). Mutually exclusive with --sequence.",
    )
    ap.add_argument(
        "--work-dir",
        "-w",
        type=str,
        required=True,
        help="Pipeline workspace directory (same as run_data_pipeline --work-dir).",
    )
    args, rest = ap.parse_known_args()
    if (args.sequence is None) == (args.input is None):
        print("Provide exactly one of --sequence / -s or --input / -i.", file=sys.stderr)
        return 2

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.sequence is not None:
        inp = work / "individual_input.txt"
        inp.write_text(args.sequence.strip() + "\n", encoding="utf-8")
    else:
        inp = Path(args.input).resolve()
        if not inp.is_file():
            print(f"Input not found: {inp}", file=sys.stderr)
            return 1

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_data_pipeline.py"),
        "--input",
        str(inp),
        "--work-dir",
        str(work),
    ]
    cmd.extend(rest)
    return int(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())

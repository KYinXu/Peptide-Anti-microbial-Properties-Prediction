from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Matches auto indices from bare sequence lines (1, 2, 10); not 02264 or Q6FI13
_AUTO_INDEX = re.compile(r"^[1-9]\d*$")


def add_peptide_id_to_esm2_csv(csv_path: Path, *, esmfold_id_prefix: str = "SEQ") -> None:
    df = pd.read_csv(csv_path)
    if "peptide_id" in df.columns:
        return
    if "seqIndex" not in df.columns:
        raise ValueError(f"{csv_path}: expected seqIndex column")
    sid = df["seqIndex"].astype(str).str.strip()

    def _pid(s: str) -> str:
        if s.startswith(f"{esmfold_id_prefix}_"):
            return s
        if _AUTO_INDEX.fullmatch(s):
            return f"{esmfold_id_prefix}_{s}"
        return s

    df["peptide_id"] = sid.map(_pid)
    df.to_csv(csv_path, index=False)

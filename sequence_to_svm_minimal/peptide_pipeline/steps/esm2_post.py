from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_peptide_id_to_esm2_csv(csv_path: Path, *, esmfold_id_prefix: str = "SEQ") -> None:
    df = pd.read_csv(csv_path)
    if "peptide_id" in df.columns:
        return
    if "seqIndex" not in df.columns:
        raise ValueError(f"{csv_path}: expected seqIndex column")
    sid = df["seqIndex"].astype(str).str.strip()
    df["peptide_id"] = sid.apply(
        lambda s: s if s.startswith(f"{esmfold_id_prefix}_") else f"{esmfold_id_prefix}_{s}"
    )
    df.to_csv(csv_path, index=False)

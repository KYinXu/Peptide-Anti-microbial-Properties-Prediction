"""Recover parent flanks from candidates table + proteome FASTA."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from data_normalizer.shared.records import normalize_sequence

from .variants.common import ParentRecord, SubsetRecord


CANDIDATE_COLUMNS = (
    "peptide_id",
    "sequence",
    "source_protein_id",
    "start",
    "end",
)


def load_candidate_index(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidates table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path, usecols=lambda name: name in CANDIDATE_COLUMNS)
    else:
        raise ValueError(f"Candidates table must be CSV or parquet; got {path.suffix}")

    missing = [name for name in CANDIDATE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Candidates table missing column(s): {missing}")

    out = frame[list(CANDIDATE_COLUMNS)].copy()
    out["peptide_id"] = out["peptide_id"].astype(str).str.strip()
    out["sequence"] = out["sequence"].map(lambda value: normalize_sequence(str(value), "candidate"))
    out["source_protein_id"] = out["source_protein_id"].astype(str).str.strip()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    if out["peptide_id"].duplicated().any():
        duplicates = out.loc[out["peptide_id"].duplicated(), "peptide_id"].unique().tolist()
        raise ValueError(f"Candidates table has duplicate peptide_id values: {duplicates[:5]}")
    return out.set_index("peptide_id", drop=False)


def load_fasta_sequences(path: Path, protein_ids: Iterable[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Proteome FASTA not found: {path}")
    wanted = {str(protein_id).strip() for protein_id in protein_ids}
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None and current_id in wanted:
                    sequences[current_id] = "".join(chunks).upper()
                    if len(sequences) == len(wanted):
                        break
                header = line[1:].strip()
                current_id = header.split()[0] if header else "record"
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None and current_id in wanted and current_id not in sequences:
            sequences[current_id] = "".join(chunks).upper()
    return sequences


def recover_parents(
    subset: Iterable[SubsetRecord],
    candidates: pd.DataFrame,
    proteome_fasta: Path,
) -> list[ParentRecord]:
    subset_list = list(subset)
    if not subset_list:
        return []

    missing_ids = [record.id for record in subset_list if record.id not in candidates.index]
    if missing_ids:
        raise KeyError(
            "Subset id(s) missing from candidates table: "
            + ", ".join(missing_ids[:5])
            + ("..." if len(missing_ids) > 5 else "")
        )

    protein_ids = [
        str(candidates.loc[record.id, "source_protein_id"])
        for record in subset_list
    ]
    proteins = load_fasta_sequences(proteome_fasta, protein_ids)
    missing_proteins = sorted(set(protein_ids) - set(proteins))
    if missing_proteins:
        raise KeyError(
            "source_protein_id(s) missing from FASTA: "
            + ", ".join(missing_proteins[:5])
            + ("..." if len(missing_proteins) > 5 else "")
        )

    parents: list[ParentRecord] = []
    for record in subset_list:
        row = candidates.loc[record.id]
        candidate_sequence = str(row["sequence"])
        if candidate_sequence != record.sequence:
            raise ValueError(
                f"{record.id}: subset sequence does not match candidates table "
                f"({candidate_sequence!r} vs {record.sequence!r})."
            )
        start = int(row["start"])
        end = int(row["end"])
        protein_id = str(row["source_protein_id"])
        protein = proteins[protein_id]
        if start < 0 or end > len(protein) or start >= end:
            raise ValueError(
                f"{record.id}: invalid coordinates start={start} end={end} "
                f"for protein length {len(protein)}."
            )
        slice_sequence = protein[start:end]
        if slice_sequence != record.sequence:
            raise ValueError(
                f"{record.id}: FASTA slice {protein_id}[{start}:{end}] does not match sequence."
            )
        parents.append(
            ParentRecord(
                id=record.id,
                sequence=record.sequence,
                source_protein_id=protein_id,
                start=start,
                end=end,
                left_flank=protein[:start],
                right_flank=protein[end:],
                extras=dict(record.extras),
            )
        )
    return parents


def iter_recovered_parents(
    subset: Iterable[SubsetRecord],
    candidates_table: Path,
    proteome_fasta: Path,
) -> Iterator[ParentRecord]:
    candidates = load_candidate_index(candidates_table)
    yield from recover_parents(subset, candidates, proteome_fasta)

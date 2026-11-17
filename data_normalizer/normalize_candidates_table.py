#!/usr/bin/env python3
"""Recover proteome coordinates for a peptide table into a length_analysis candidates table.

Peptide tables report 1-based inclusive coordinates against whatever search
database produced them; `length_analysis` slices `protein[start:end]` out of a
FASTA keyed by full header token. This rewrites one into the other and drops
rows whose peptide cannot be located in the proteome, so the emitted
candidates table and sequence CSV always agree.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_normalizer.normalize_fasta import iter_normalized_fasta
from data_normalizer.shared.records import NormalizedSequenceRecord, normalize_sequence
from data_normalizer.shared.writers import write_normalized_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTEOME_FASTA = (
    ROOT / "length_analysis" / "uniprotkb_UP000005640_AND_model_organis_2026_06_27.fasta"
)
CANDIDATE_COLUMNS = ["peptide_id", "sequence", "source_protein_id", "start", "end"]
SEQUENCE_COLUMNS = ("sequence", "Sequence")
PROTEIN_COLUMNS = ("source_protein_id", "leading_razor_protein", "Leading razor protein")
START_COLUMNS = ("start", "Start position")
END_COLUMNS = ("end", "End position")
SUBSET_EXTRA_COLUMNS = {"Gene names": "gene_names", "Leading razor protein": "leading_razor_protein"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map a peptide table onto a proteome FASTA and write the candidates table "
            "plus the matching sequence CSV that length_analysis --mode extension needs."
        )
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Peptide table CSV with sequence, protein, and start/end columns.")
    parser.add_argument("--proteome-fasta", type=Path, default=DEFAULT_PROTEOME_FASTA, help="Proteome FASTA the coordinates refer to (default: bundled human UP000005640).")
    parser.add_argument("--output", "-o", type=Path, help="Candidates table path (default: <input directory>/candidates_table.csv).")
    parser.add_argument("--sequences-output", type=Path, help="Sequence CSV of the located peptides (default: <candidates table directory>/sequences_located.csv).")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.parent / "candidates_table.csv"
    if args.sequences_output is None:
        args.sequences_output = args.output.parent / "sequences_located.csv"
    return args


def load_protein_index(fasta: Path) -> dict[str, tuple[str, str]]:
    """Map both the full header token and the bare accession to (token, sequence)."""
    index: dict[str, tuple[str, str]] = {}
    for record in iter_normalized_fasta(fasta):
        entry = (record.id, record.sequence)
        index.setdefault(record.id, entry)
        parts = record.id.split("|")
        if len(parts) >= 2:
            index.setdefault(parts[1], entry)
    return index


def resolve_column(fieldnames: list[str], options: tuple[str, ...], label: str) -> str:
    for option in options:
        if option in fieldnames:
            return option
    raise ValueError(f"Peptide table has no {label} column; expected one of {list(options)}.")


def as_int(value: str | None) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def locate_peptide(
    sequence: str,
    protein_key: str,
    start: int | None,
    end: int | None,
    index: dict[str, tuple[str, str]],
) -> tuple[str, int, int, str]:
    key = protein_key.split("|")[1] if "|" in protein_key else protein_key
    entry = index.get(protein_key) or index.get(key)
    if entry is None:
        raise LookupError(f"protein {protein_key!r} is not in the FASTA")
    token, protein = entry
    if start is not None and end is not None:
        if protein[start:end] == sequence:
            return token, start, end, "as-given"
        if protein[start - 1 : end] == sequence:
            return token, start - 1, end, "one-based"
    offset = protein.find(sequence)
    if offset < 0:
        raise LookupError(f"peptide is absent from {token}")
    return token, offset, offset + len(sequence), "searched"


def recover_candidates(
    table: Path, index: dict[str, tuple[str, str]]
) -> tuple[list[dict[str, object]], list[NormalizedSequenceRecord], dict[str, int], list[tuple[str, str]]]:
    if not table.is_file():
        raise FileNotFoundError(f"Peptide table not found: {table}")
    with table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        sequence_column = resolve_column(fieldnames, SEQUENCE_COLUMNS, "sequence")
        protein_column = resolve_column(fieldnames, PROTEIN_COLUMNS, "protein")
        start_column = resolve_column(fieldnames, START_COLUMNS, "start")
        end_column = resolve_column(fieldnames, END_COLUMNS, "end")
        extra_columns = {
            source: target for source, target in SUBSET_EXTRA_COLUMNS.items() if source in fieldnames
        }

        candidates: list[dict[str, object]] = []
        sequences: list[NormalizedSequenceRecord] = []
        counts: dict[str, int] = {}
        dropped: list[tuple[str, str]] = []
        for row_number, row in enumerate(reader, start=1):
            peptide_id = (row.get("id") or "").strip() or f"SEQ_{row_number}"
            sequence = normalize_sequence(row.get(sequence_column, ""), peptide_id)
            try:
                token, start, end, how = locate_peptide(
                    sequence,
                    (row.get(protein_column) or "").strip(),
                    as_int(row.get(start_column)),
                    as_int(row.get(end_column)),
                    index,
                )
            except LookupError as error:
                dropped.append((peptide_id, str(error)))
                continue
            counts[how] = counts.get(how, 0) + 1
            candidates.append(
                {
                    "peptide_id": peptide_id,
                    "sequence": sequence,
                    "source_protein_id": token,
                    "start": start,
                    "end": end,
                }
            )
            sequences.append(
                NormalizedSequenceRecord(
                    id=peptide_id,
                    sequence=sequence,
                    extras={target: (row.get(source) or "").strip() for source, target in extra_columns.items()},
                )
            )
    return candidates, sequences, counts, dropped


def write_candidates_table(path: Path, candidates: list[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)
    return len(candidates)


def main() -> int:
    args = parse_args()
    try:
        index = load_protein_index(args.proteome_fasta)
        candidates, sequences, counts, dropped = recover_candidates(args.input, index)
        if not candidates:
            raise ValueError("No peptide could be located in the proteome.")
        write_candidates_table(args.output, candidates)
        write_normalized_csv(args.sequences_output, sequences)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    located = len(candidates)
    print(f"Located {located} peptide(s) in {args.proteome_fasta.name}")
    for how in ("as-given", "one-based", "searched"):
        if counts.get(how):
            print(f"   {counts[how]} by {how} coordinates")
    if dropped:
        print(f"Dropped {len(dropped)} peptide(s):")
        for peptide_id, reason in dropped[:10]:
            print(f"   {peptide_id}: {reason}")
        if len(dropped) > 10:
            print(f"   ... and {len(dropped) - 10} more")
    print(f"Saved candidates table to {args.output.resolve()}")
    print(f"Saved {len(sequences)} located sequence row(s) to {args.sequences_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

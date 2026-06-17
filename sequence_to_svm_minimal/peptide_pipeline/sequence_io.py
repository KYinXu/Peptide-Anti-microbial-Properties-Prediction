<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
"""Shared TXT/FASTA parsing and canonical line writing (ESMFold / ESM-2 compatible)."""

from __future__ import annotations

<<<<<<< HEAD
=======
"""Shared TXT/FASTA/CSV parsing and canonical line writing (ESMFold / ESM-2 compatible)."""

from __future__ import annotations

import csv
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
from pathlib import Path

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence


def _iter_fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                header = line[1:].strip()
                current_id = header.split()[0] if header else str(len(records) + 1)
                chunks = []
            else:
                chunks.append(line.strip())
        if current_id is not None:
            records.append((current_id, "".join(chunks)))
    return records


def is_fasta_suffix(path: Path) -> bool:
    return path.suffix.lower() in (".fa", ".fasta", ".faa")


<<<<<<< HEAD
<<<<<<< HEAD
=======
def is_csv_suffix(path: Path) -> bool:
    return path.suffix.lower() == ".csv"


def _find_csv_column(fieldnames: list[str | None], name: str) -> str | None:
    target = name.lower()
    for field in fieldnames:
        if field is not None and field.strip().lower() == target:
            return field
    return None


def _iter_csv_records(path: Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        name_col = _find_csv_column(reader.fieldnames or [], "name")
        seq_col = _find_csv_column(reader.fieldnames or [], "seq")
        if name_col is None or seq_col is None:
            raise ValueError(f"CSV input must contain 'name' and 'seq' columns: {path}")
        return [
            ((row.get(name_col) or "").strip(), (row.get(seq_col) or "").strip())
            for row in reader
        ]


>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
def read_sequence_records(
    path: Path, invalid_stats: dict | None = None
) -> list[tuple[str, str]]:
    """
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    Parse sequences from TXT or FASTA.

    TXT: blank lines and full-line ``#`` comments skipped; each line is ``id seq`` or bare ``seq``
    (auto 1..n index). FASTA: only when suffix is .fa/.fasta/.faa (same as normalize_to_canonical).
<<<<<<< HEAD
=======
    Parse sequences from TXT, FASTA, or CSV.

    TXT: blank lines and full-line ``#`` comments skipped; each line is ``id seq`` or bare ``seq``
    (auto 1..n index). FASTA: only when suffix is .fa/.fasta/.faa (same as normalize_to_canonical).
    CSV: only when suffix is .csv, with ``name`` and ``seq`` columns.
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    Records whose sequence is not entirely standard 20 amino acids (after uppercasing) are dropped.
    If ``invalid_stats`` is passed, it is updated with key ``n_skipped_invalid`` (incremented per
    dropped record).
    """
    path = Path(path)
    records: list[tuple[str, str]] = []

    def _skip_invalid() -> None:
        if invalid_stats is not None:
            invalid_stats["n_skipped_invalid"] = invalid_stats.get("n_skipped_invalid", 0) + 1

    if is_fasta_suffix(path):
        for rid, seq in _iter_fasta_records(path):
            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                _skip_invalid()
                continue
            records.append((rid, canon))
        return records

<<<<<<< HEAD
<<<<<<< HEAD
=======
    if is_csv_suffix(path):
        for rid, seq in _iter_csv_records(path):
            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                _skip_invalid()
                continue
            records.append((rid, canon))
        return records

>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    def _strip_inline_comment(s: str) -> str:
        """
        Allow inline comments in TXT inputs.

        Examples:
          "id ACDEFG  # comment" -> "id ACDEFG"
          "ACDEFG # comment"     -> "ACDEFG"
        """
        if "#" not in s:
            return s
        return s.split("#", 1)[0].rstrip()

    auto_i = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = _strip_inline_comment(line.strip())
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(None, 1)
            if len(parts) == 2:
                idx, seq = parts[0].strip(), parts[1].strip()
            elif len(parts) == 1:
                auto_i += 1
                idx, seq = str(auto_i), parts[0].strip()
            else:
                continue
            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                _skip_invalid()
                continue
            records.append((idx, canon))
    return records


def write_canonical(path: Path, records: list[tuple[str, str]]) -> None:
    """Write ``id sequence`` lines (one per record)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{rid} {seq}" for rid, seq in records]
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines))
        if lines:
            out.write("\n")

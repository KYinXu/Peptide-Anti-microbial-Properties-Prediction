"""Shared TXT/FASTA parsing and canonical line writing (ESMFold / ESM-2 compatible)."""

from __future__ import annotations

from pathlib import Path

from peptide_pipeline.aa_sanitize import sanitize_for_esm2


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


def read_sequence_records(path: Path) -> list[tuple[str, str]]:
    """
    Parse sequences from TXT or FASTA.

    TXT: blank lines and full-line ``#`` comments skipped; each line is ``id seq`` or bare ``seq``
    (auto 1..n index). FASTA: only when suffix is .fa/.fasta/.faa (same as normalize_to_canonical).

    Returns ``(peptide_id, sanitized_sequence)`` per record (sanitized via ``sanitize_for_esm2``).
    """
    path = Path(path)
    records: list[tuple[str, str]] = []

    if is_fasta_suffix(path):
        for rid, seq in _iter_fasta_records(path):
            seq = sanitize_for_esm2(seq.replace(" ", ""))
            records.append((rid, seq))
        return records

    auto_i = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = line.strip()
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
            seq = sanitize_for_esm2(seq.replace(" ", ""))
            records.append((idx, seq))
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

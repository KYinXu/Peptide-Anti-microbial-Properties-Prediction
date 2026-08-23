"""Shared TXT/FASTA parsing and canonical line writing (ESMFold / ESM-2 compatible)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence


def _iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    current_id: str | None = None
    chunks: list[str] = []
    n_emitted = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(chunks)
                    n_emitted += 1
                header = line[1:].strip()
                current_id = header.split()[0] if header else str(n_emitted + 1)
                chunks = []
            else:
                chunks.append(line.strip())
        if current_id is not None:
            yield current_id, "".join(chunks)


def is_fasta_suffix(path: Path) -> bool:
    return path.suffix.lower() in (".fa", ".fasta", ".faa")


def _strip_inline_comment(s: str) -> str:
    if "#" not in s:
        return s
    return s.split("#", 1)[0].rstrip()


def iter_sequence_records(
    path: Path, invalid_stats: dict | None = None
) -> Iterator[tuple[str, str]]:
    """Yield ``(id, sequence)`` records without materializing the full file."""
    path = Path(path)

    def _skip_invalid() -> None:
        if invalid_stats is not None:
            invalid_stats["n_skipped_invalid"] = invalid_stats.get("n_skipped_invalid", 0) + 1

    if is_fasta_suffix(path):
        for rid, seq in _iter_fasta_records(path):
            canon = canonical_standard_aa_sequence(seq)
            if canon is None:
                _skip_invalid()
                continue
            yield rid, canon
        return

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
            yield idx, canon


def read_sequence_records(
    path: Path, invalid_stats: dict | None = None
) -> list[tuple[str, str]]:
    """
    Parse sequences from TXT or FASTA.

    TXT: blank lines and full-line ``#`` comments skipped; each line is ``id seq`` or bare ``seq``
    (auto 1..n index). FASTA: only when suffix is .fa/.fasta/.faa (same as normalize_to_canonical).

    Records whose sequence is not entirely standard 20 amino acids (after uppercasing) are dropped.
    If ``invalid_stats`` is passed, it is updated with key ``n_skipped_invalid`` (incremented per
    dropped record).
    """
    return list(iter_sequence_records(path, invalid_stats=invalid_stats))


def write_canonical(path: Path, records: Iterable[tuple[str, str]]) -> None:
    """Write ``id sequence`` lines one record at a time (accepts lists or generators)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            for rid, seq in records:
                out.write(f"{rid} {seq}\n")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def concatenate_canonical_files(dest: Path, sources: Iterable[Path]) -> None:
    """Copy canonical ``id seq`` files into ``dest`` without loading them all at once."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            for src in sources:
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line:
                            continue
                        out.write(line if line.endswith("\n") else f"{line}\n")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

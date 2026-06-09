"""FASTA parsing, standard amino-acid filtering, and batch writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from proteome_candidate_generator.progress import progress_iter

STANDARD_AA_20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class BatchFile:
    index: int
    path: Path
    n_records: int


@dataclass(frozen=True)
class PreprocessResult:
    records: list[ProteinRecord]
    batches: list[BatchFile]
    stats: dict[str, int]


<<<<<<< HEAD
def canonical_standard_sequence(seq: str) -> str | None:
    clean = seq.replace(" ", "").replace("\t", "").upper()
    if not clean:
        return None
    if any(aa not in STANDARD_AA_20 for aa in clean):
=======
def canonical_standard_sequence(seq: str, *, require_standard_aa_20: bool = True) -> str | None:
    clean = seq.replace(" ", "").replace("\t", "").upper()
    if not clean:
        return None
    if require_standard_aa_20 and any(aa not in STANDARD_AA_20 for aa in clean):
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
        return None
    return clean


def iter_fasta_records(path: Path) -> Iterator[ProteinRecord]:
    current_id: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield ProteinRecord(current_id, "".join(chunks))
                header = line[1:].strip()
                current_id = header.split()[0] if header else "record"
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        yield ProteinRecord(current_id, "".join(chunks))


def read_valid_proteins(
    path: Path,
    *,
    limit: int | None = None,
<<<<<<< HEAD
=======
    require_standard_aa_20: bool = True,
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    show_progress: bool = False,
) -> tuple[list[ProteinRecord], dict[str, int]]:
    records: list[ProteinRecord] = []
    stats = {"total": 0, "valid": 0, "skipped_invalid": 0}
    source = iter_fasta_records(path)
    if show_progress:
        source = progress_iter(source, desc="Reading FASTA", total=limit)
    for raw in source:
        if limit is not None and stats["total"] >= limit:
            break
        stats["total"] += 1
<<<<<<< HEAD
        sequence = canonical_standard_sequence(raw.sequence)
=======
        sequence = canonical_standard_sequence(raw.sequence, require_standard_aa_20=require_standard_aa_20)
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
        if sequence is None:
            stats["skipped_invalid"] += 1
            continue
        records.append(ProteinRecord(raw.protein_id, sequence))
        stats["valid"] += 1
    return records, stats


def write_fasta(records: list[ProteinRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(f">{record.protein_id}\n")
            for start in range(0, len(record.sequence), 80):
                handle.write(record.sequence[start : start + 80] + "\n")


def write_batches(records: list[ProteinRecord], batches_dir: Path, *, batch_size: int) -> list[BatchFile]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches_dir.mkdir(parents=True, exist_ok=True)
    batches: list[BatchFile] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        index = len(batches) + 1
        path = batches_dir / f"batch_{index:05d}.fasta"
        write_fasta(chunk, path)
        batches.append(BatchFile(index=index, path=path, n_records=len(chunk)))
    return batches


def preprocess_fasta(
    input_path: Path,
    batches_dir: Path,
    *,
    batch_size: int,
    limit: int | None = None,
<<<<<<< HEAD
    show_progress: bool = False,
) -> PreprocessResult:
    records, stats = read_valid_proteins(input_path, limit=limit, show_progress=show_progress)
=======
    require_standard_aa_20: bool = True,
    show_progress: bool = False,
) -> PreprocessResult:
    records, stats = read_valid_proteins(
        input_path,
        limit=limit,
        require_standard_aa_20=require_standard_aa_20,
        show_progress=show_progress,
    )
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    batches = write_batches(records, batches_dir, batch_size=batch_size)
    stats["batches"] = len(batches)
    return PreprocessResult(records=records, batches=batches, stats=stats)

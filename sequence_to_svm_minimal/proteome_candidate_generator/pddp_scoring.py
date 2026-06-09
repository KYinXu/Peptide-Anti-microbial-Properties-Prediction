"""Paper-style AMP contribution scoring for PDDP candidate filtering."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from proteome_candidate_generator.fasta import canonical_standard_sequence, iter_fasta_records

STANDARD_AA = tuple("ACDEFGHIKLMNPQRSTVWY")


class SequenceScorer(Protocol):
    def score_sequence(self, sequence: str) -> float:
        ...


@dataclass(frozen=True)
class ActivityScoreMatrix:
    scores: dict[int, dict[str, float]]

    @property
    def max_position(self) -> int:
        return max(self.scores) if self.scores else 0

    def score_sequence(self, sequence: str) -> float:
        seq = canonical_standard_sequence(sequence)
        if seq is None:
            raise ValueError(f"Invalid peptide sequence: {sequence!r}")
        if not self.scores:
            raise ValueError("Score matrix is empty")
        window = self.max_position
        if len(seq) <= window:
            return _score_aligned(seq, self.scores)
        return max(_score_aligned(seq[start : start + window], self.scores) for start in range(0, len(seq) - window + 1))


def _score_aligned(sequence: str, scores: dict[int, dict[str, float]]) -> float:
    total = 0.0
    for offset, aa in enumerate(sequence, start=1):
        try:
            total += scores[offset][aa]
        except KeyError as exc:
            raise ValueError(f"Score matrix missing value for position {offset}, residue {aa}") from exc
    return total


def load_score_matrix(path: Path) -> ActivityScoreMatrix:
    """Load a long or wide amino-acid contribution matrix CSV/TSV.

    Long format columns: position, amino_acid, score.
    Wide format: first column is position; amino-acid columns are A/C/D/...
    """
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"Score matrix is empty: {path}")
    normalized = {_norm(key): key for key in rows[0]}
    if {"position", "aminoacid", "score"}.issubset(normalized):
        return ActivityScoreMatrix(_load_long_matrix(rows, normalized))
    return ActivityScoreMatrix(_load_wide_matrix(rows))


def is_mapp_database(path: Path) -> bool:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(handle, dialect=dialect)
        headers = {_norm(header) for header in (reader.fieldnames or [])}
    return "sequence" in headers and "treatment" in headers


@dataclass(frozen=True)
class MappReferenceEntry:
    sequence: str
    treatment_total: float
    treatment: str


@dataclass(frozen=True)
class MappReferenceScorer:
    entries: dict[str, MappReferenceEntry]

    def score_sequence(self, sequence: str) -> float:
        seq = canonical_standard_sequence(sequence)
        if seq is None:
            raise ValueError(f"Invalid peptide sequence: {sequence!r}")
        entry = self.entries.get(seq)
        return entry.treatment_total if entry is not None else 0.0


def load_mapp_reference_scorer(path: Path) -> MappReferenceScorer:
    rows = _read_table(path)
    if not rows:
        raise ValueError(f"MAPP database is empty: {path}")
    normalized = {_norm(key): key for key in rows[0]}
    seq_col = normalized.get("sequence")
    if seq_col is None:
        raise ValueError(f"MAPP database must include a Sequence column: {path}")
    treatment_col = normalized.get("treatment")
    entries: dict[str, MappReferenceEntry] = {}
    for row in rows:
        seq = canonical_standard_sequence(row.get(seq_col, ""))
        if seq is None:
            continue
        treatment = row.get(treatment_col, "") if treatment_col else ""
        total = _parse_treatment_total(treatment)
        existing = entries.get(seq)
        if existing is None or total > existing.treatment_total:
            entries[seq] = MappReferenceEntry(seq, total, treatment)
    if not entries:
        raise ValueError(f"MAPP database contains no valid peptide sequences: {path}")
    return MappReferenceScorer(entries)


def _parse_treatment_total(value: str) -> float:
    if not value:
        return 1.0
    total = 0.0
    saw_numeric = False
    for item in value.split(";"):
        if ":" not in item:
            continue
        _, raw = item.rsplit(":", 1)
        try:
            total += float(raw)
            saw_numeric = True
        except ValueError:
            continue
    return total if saw_numeric else 1.0


def _read_table(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return list(csv.DictReader(handle, dialect=dialect))


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _load_long_matrix(rows: list[dict[str, str]], normalized: dict[str, str]) -> dict[int, dict[str, float]]:
    pos_col = normalized["position"]
    aa_col = normalized["aminoacid"]
    score_col = normalized["score"]
    scores: dict[int, dict[str, float]] = {}
    for row in rows:
        pos = int(float(row[pos_col]))
        aa = row[aa_col].strip().upper()
        if aa not in STANDARD_AA:
            raise ValueError(f"Invalid amino acid in score matrix: {aa!r}")
        scores.setdefault(pos, {})[aa] = float(row[score_col])
    _validate_positions(scores)
    return scores


def _load_wide_matrix(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    headers = list(rows[0])
    pos_col = next((h for h in headers if _norm(h) in {"position", "pos", "index"}), headers[0])
    aa_cols = [h for h in headers if h != pos_col and h.strip().upper() in STANDARD_AA]
    if not aa_cols:
        raise ValueError("Wide score matrix must include amino-acid columns")
    scores: dict[int, dict[str, float]] = {}
    for row in rows:
        pos = int(float(row[pos_col]))
        scores[pos] = {col.strip().upper(): float(row[col]) for col in aa_cols}
    _validate_positions(scores)
    return scores


def _validate_positions(scores: dict[int, dict[str, float]]) -> None:
    if not scores:
        raise ValueError("Score matrix contains no values")
    expected = set(range(1, max(scores) + 1))
    missing_positions = sorted(expected - set(scores))
    if missing_positions:
        raise ValueError(f"Score matrix missing positions: {missing_positions[:10]}")
    missing_values = {
        pos: sorted(set(STANDARD_AA) - set(values))
        for pos, values in scores.items()
        if set(STANDARD_AA) - set(values)
    }
    if missing_values:
        pos, aas = next(iter(missing_values.items()))
        raise ValueError(f"Score matrix missing residues at position {pos}: {aas}")


def load_known_amp_sequences(paths: list[Path]) -> list[str]:
    sequences: list[str] = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
            for record in iter_fasta_records(path):
                seq = canonical_standard_sequence(record.sequence)
                if seq is not None:
                    sequences.append(seq)
            continue
        sequences.extend(_load_text_or_csv_sequences(path))
    return sequences


def _load_text_or_csv_sequences(path: Path) -> list[str]:
    rows = _read_table(path) if path.suffix.lower() in {".csv", ".tsv"} else None
    if rows is not None:
        headers = list(rows[0]) if rows else []
        seq_col = next((h for h in headers if _norm(h) in {"sequence", "seq", "peptide"}), headers[-1] if headers else None)
        if seq_col is None:
            return []
        return [
            seq
            for row in rows
            if (seq := canonical_standard_sequence(row.get(seq_col, ""))) is not None
        ]
    out: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            seq = canonical_standard_sequence(parts[-1])
            if seq is not None:
                out.append(seq)
    return out


def compute_nonzero_mean_threshold(sequences: list[str], matrix: ActivityScoreMatrix) -> float:
    scores = [matrix.score_sequence(seq) for seq in sequences]
    nonzero = [score for score in scores if score != 0]
    if not nonzero:
        raise ValueError("Known AMP sequences produced no nonzero scores")
    return sum(nonzero) / len(nonzero)

"""Candidate peptide expansion, filtering, ranking, and outputs."""

from __future__ import annotations

import csv
import heapq
import math
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from proteome_candidate_generator.cleavage import ProteinCleavageSites
from proteome_candidate_generator.fasta import ProteinRecord
from proteome_candidate_generator.pddp_scoring import SequenceScorer, net_charge, hydrophobicity, HYDROPHOBIC_AA, POSITIVE_AA, NEGATIVE_AA
from proteome_candidate_generator.progress import progress_iter

@dataclass(frozen=True)
class CandidatePeptide:
    peptide_id: str
    sequence: str
    source_protein_id: str
    start: int
    end: int
    length: int
    net_charge: int
    hydrophobicity: float
    hydrophobic_moment: float
    rank_score: float
    left_cleavage_probability: float
    right_cleavage_probability: float
    predicted_cleavage_probability: float
    pddp_score: float | None = None
    score_threshold: float | None = None
    passes_score_threshold: bool | None = None
    passes_cationic_cterm: bool | None = None


@dataclass(frozen=True)
class CandidateStats:
    expanded: int
    duplicate_sequences: int
    failed_filters: int
    retained: int
    score_filtered: int = 0
    overlap_removed: int = 0
    cterm_filtered: int = 0


def hydrophobic_moment(sequence: str, *, angle_degrees: float = 100.0) -> float:
    angle = math.radians(angle_degrees)
    x_total = 0.0
    y_total = 0.0
    for index, aa in enumerate(sequence):
        value = 1.0 if aa in HYDROPHOBIC_AA else 0.0
        x_total += value * math.cos(index * angle)
        y_total += value * math.sin(index * angle)
    return math.sqrt((x_total * x_total) + (y_total * y_total)) / len(sequence)


def has_cationic_cterm(sequence: str, residues: str = "KRH") -> bool:
    return bool(sequence) and sequence[-1] in set(residues.upper())


def _boundaries(
    sites: ProteinCleavageSites,
    *,
    include_terminal_boundaries: bool,
) -> list[int]:
    values = set(sites.sites)
    if include_terminal_boundaries:
        values.update((0, sites.length))
    return sorted(values)


def _boundary_probability(sites: ProteinCleavageSites, boundary: int) -> float:
    if boundary == 0 or boundary == sites.length:
        return 1.0
    return sites.site_probabilities.get(boundary, 0.0)


def _iter_raw_candidates(
    record: ProteinRecord,
    sites: ProteinCleavageSites,
    *,
    min_len: int,
    max_len: int,
    include_terminal_boundaries: bool,
):
    boundaries = _boundaries(sites, include_terminal_boundaries=include_terminal_boundaries)
    for left_index, start in enumerate(boundaries):
        for end in boundaries[left_index + 1 :]:
            length = end - start
            if length > max_len:
                break
            if length < min_len:
                continue
            yield start, end, record.sequence[start:end]


def _base_candidate(
    *,
    sequence: str,
    source_protein_id: str,
    start: int,
    end: int,
    sites: ProteinCleavageSites,
    rank_score: float,
    pddp_score: float | None = None,
    score_threshold: float | None = None,
    passes_score_threshold: bool | None = None,
    passes_cationic_cterm: bool | None = None,
) -> CandidatePeptide:
    charge = net_charge(sequence)
    hydro = hydrophobicity(sequence)
    left_prob = _boundary_probability(sites, start)
    right_prob = _boundary_probability(sites, end)
    return CandidatePeptide(
        peptide_id="",
        sequence=sequence,
        source_protein_id=source_protein_id,
        start=start,
        end=end,
        length=len(sequence),
        net_charge=charge,
        hydrophobicity=hydro,
        hydrophobic_moment=hydrophobic_moment(sequence),
        rank_score=rank_score,
        left_cleavage_probability=left_prob,
        right_cleavage_probability=right_prob,
        predicted_cleavage_probability=min(left_prob, right_prob),
        pddp_score=pddp_score,
        score_threshold=score_threshold,
        passes_score_threshold=passes_score_threshold,
        passes_cationic_cterm=passes_cationic_cterm,
    )


def generate_candidates(
    records: list[ProteinRecord],
    sites_by_protein: dict[str, ProteinCleavageSites],
    *,
    min_len: int,
    max_len: int,
    min_charge: int,
    min_hydrophobicity: float,
    top_n: int | None,
    include_terminal_boundaries: bool,
    show_progress: bool = False,
) -> tuple[list[CandidatePeptide], CandidateStats]:
    seen: set[str] = set()
    retained: list[CandidatePeptide] = []
    heap: list[tuple[float, int, CandidatePeptide]] = []
    expanded = duplicate_sequences = failed_filters = counter = 0

    record_iter = records
    if show_progress:
        record_iter = progress_iter(records, desc="Expanding and filtering candidates", total=len(records))
    for record in record_iter:
        sites = sites_by_protein.get(record.protein_id)
        if sites is None:
            continue
        for start, end, sequence in _iter_raw_candidates(
            record,
            sites,
            min_len=min_len,
            max_len=max_len,
            include_terminal_boundaries=include_terminal_boundaries,
        ):
            expanded += 1
            if sequence in seen:
                duplicate_sequences += 1
                continue
            seen.add(sequence)
            charge = net_charge(sequence)
            hydro = hydrophobicity(sequence)
            if charge < min_charge or hydro < min_hydrophobicity:
                failed_filters += 1
                continue
            moment = hydrophobic_moment(sequence)
            candidate = _base_candidate(
                sequence=sequence,
                source_protein_id=record.protein_id,
                start=start,
                end=end,
                sites=sites,
                rank_score=charge * moment,
            )
            counter += 1
            if top_n is None:
                retained.append(candidate)
            elif len(heap) < top_n:
                heapq.heappush(heap, (candidate.rank_score, counter, candidate))
            elif candidate.rank_score > heap[0][0]:
                heapq.heapreplace(heap, (candidate.rank_score, counter, candidate))

    if top_n is not None:
        retained = [item[2] for item in sorted(heap, key=lambda row: row[0], reverse=True)]
    else:
        retained.sort(key=lambda row: row.rank_score, reverse=True)
    final = [_with_peptide_id(candidate, index + 1) for index, candidate in enumerate(retained)]
    stats = CandidateStats(
        expanded=expanded,
        duplicate_sequences=duplicate_sequences,
        failed_filters=failed_filters,
        retained=len(final),
    )
    return final, stats


def generate_paper_candidates(
    records: list[ProteinRecord],
    sites_by_protein: dict[str, ProteinCleavageSites],
    *,
    min_len: int,
    max_len: int,
    scorer: SequenceScorer,
    score_threshold: float,
    require_cationic_cterm: bool,
    cationic_cterm_residues: str,
    overlap_policy: str,
    include_terminal_boundaries: bool,
    show_progress: bool = False,
    finalize_outputs: tuple[Path, Path, str] | None = None,
) -> tuple[list[CandidatePeptide], CandidateStats]:
    expanded = score_filtered = overlap_removed = cterm_filtered = duplicate_sequences = 0

    record_iter = records
    if show_progress:
        record_iter = progress_iter(records, desc="Expanding and paper-scoring candidates", total=len(records))

    seen_sequences: set[str] = set()
    spill = _CandidateSpillStore()

    try:
        for record in record_iter:
            sites = sites_by_protein.get(record.protein_id)
            if sites is None:
                continue

            protein_retained: list[CandidatePeptide] = []
            for start, end, sequence in _iter_raw_candidates(
                record,
                sites,
                min_len=min_len,
                max_len=max_len,
                include_terminal_boundaries=include_terminal_boundaries,
            ):
                expanded += 1
                score = scorer.score_sequence(sequence)
                if score <= score_threshold:
                    score_filtered += 1
                    continue
                protein_retained.append(
                    _base_candidate(
                        sequence=sequence,
                        source_protein_id=record.protein_id,
                        start=start,
                        end=end,
                        sites=sites,
                        rank_score=score,
                        pddp_score=score,
                        score_threshold=score_threshold,
                        passes_score_threshold=True,
                        passes_cationic_cterm=None,
                    )
                )

            protein_non_overlapping, removed = _apply_overlap_policy(protein_retained, overlap_policy)
            overlap_removed += removed

            for candidate in protein_non_overlapping:
                if require_cationic_cterm:
                    passed = has_cationic_cterm(candidate.sequence, cationic_cterm_residues)
                    if not passed:
                        cterm_filtered += 1
                        continue
                    candidate = _replace_candidate(candidate, passes_cationic_cterm=True)
                else:
                    candidate = _replace_candidate(candidate, passes_cationic_cterm=None)

                if candidate.sequence in seen_sequences:
                    duplicate_sequences += 1
                    continue

                seen_sequences.add(candidate.sequence)
                spill.add(candidate)

        stats = CandidateStats(
            expanded=expanded,
            duplicate_sequences=duplicate_sequences,
            failed_filters=score_filtered + cterm_filtered,
            retained=spill.count,
            score_filtered=score_filtered,
            overlap_removed=overlap_removed,
            cterm_filtered=cterm_filtered,
        )
        if finalize_outputs is not None:
            table_stem, txt_path, output_format = finalize_outputs
            spill.write_sorted_outputs(table_stem, txt_path, output_format=output_format)
            return [], stats
        return spill.to_list_with_ids(), stats
    finally:
        spill.close()


def _apply_overlap_policy(candidates: list[CandidatePeptide], policy: str) -> tuple[list[CandidatePeptide], int]:
    if policy == "keep_all":
        return candidates, 0
    if policy == "top_score":
        return _remove_overlapping_candidates(
            candidates,
            sort_key=lambda row: (
                row.pddp_score or float("-inf"),
                row.predicted_cleavage_probability,
                -row.length,  # Break ties by preferring shorter peptides
            ),
        )
    if policy == "longest":
        return _remove_overlapping_candidates(
            candidates,
            sort_key=lambda row: (
                row.length,
                row.pddp_score or float("-inf"),
                row.predicted_cleavage_probability,
            ),
        )
    raise ValueError(f"Unknown overlap policy: {policy}")


def _remove_overlapping_lower_scores(candidates: list[CandidatePeptide]) -> tuple[list[CandidatePeptide], int]:
    return _remove_overlapping_candidates(
        candidates,
        sort_key=lambda row: (
            row.pddp_score or float("-inf"),
            row.predicted_cleavage_probability,
            -row.length,
        ),
    )


def _remove_overlapping_candidates(candidates, *, sort_key) -> tuple[list[CandidatePeptide], int]:
    selected: list[CandidatePeptide] = []
    removed = 0
    by_protein: dict[str, list[CandidatePeptide]] = {}
    for candidate in candidates:
        by_protein.setdefault(candidate.source_protein_id, []).append(candidate)
    for protein_candidates in by_protein.values():
        chosen: list[CandidatePeptide] = []
        for candidate in sorted(
            protein_candidates,
            key=sort_key,
            reverse=True,
        ):
            if any(_overlaps(candidate, existing) for existing in chosen):
                removed += 1
                continue
            chosen.append(candidate)
        selected.extend(chosen)
    return selected, removed


def _overlaps(left: CandidatePeptide, right: CandidatePeptide) -> bool:
    return left.source_protein_id == right.source_protein_id and left.start < right.end and right.start < left.end


def _replace_candidate(candidate: CandidatePeptide, **updates: object) -> CandidatePeptide:
    values = asdict(candidate)
    values.update(updates)
    return CandidatePeptide(**values)


def _with_peptide_id(candidate: CandidatePeptide, index: int) -> CandidatePeptide:
    values = asdict(candidate)
    values["peptide_id"] = f"PEP_{index}"
    return CandidatePeptide(**values)


_CANDIDATE_FIELDNAMES = list(CandidatePeptide.__dataclass_fields__)
_SPILL_COMMIT_INTERVAL = 10_000


def _optional_bool_to_db(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool_from_db(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _candidate_to_spill_row(candidate: CandidatePeptide) -> tuple[object, ...]:
    return (
        candidate.peptide_id,
        candidate.sequence,
        candidate.source_protein_id,
        candidate.start,
        candidate.end,
        candidate.length,
        candidate.net_charge,
        candidate.hydrophobicity,
        candidate.hydrophobic_moment,
        candidate.rank_score,
        candidate.left_cleavage_probability,
        candidate.right_cleavage_probability,
        candidate.predicted_cleavage_probability,
        candidate.pddp_score,
        candidate.score_threshold,
        _optional_bool_to_db(candidate.passes_score_threshold),
        _optional_bool_to_db(candidate.passes_cationic_cterm),
    )


def _candidate_from_spill_row(row: sqlite3.Row) -> CandidatePeptide:
    return CandidatePeptide(
        peptide_id=row["peptide_id"],
        sequence=row["sequence"],
        source_protein_id=row["source_protein_id"],
        start=row["start"],
        end=row["end"],
        length=row["length"],
        net_charge=row["net_charge"],
        hydrophobicity=row["hydrophobicity"],
        hydrophobic_moment=row["hydrophobic_moment"],
        rank_score=row["rank_score"],
        left_cleavage_probability=row["left_cleavage_probability"],
        right_cleavage_probability=row["right_cleavage_probability"],
        predicted_cleavage_probability=row["predicted_cleavage_probability"],
        pddp_score=row["pddp_score"],
        score_threshold=row["score_threshold"],
        passes_score_threshold=_optional_bool_from_db(row["passes_score_threshold"]),
        passes_cationic_cterm=_optional_bool_from_db(row["passes_cationic_cterm"]),
    )


class _CandidateSpillStore:
    def __init__(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "candidates.sqlite"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        self.count = 0

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peptide_id TEXT,
                sequence TEXT NOT NULL,
                source_protein_id TEXT NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                length INTEGER NOT NULL,
                net_charge INTEGER NOT NULL,
                hydrophobicity REAL NOT NULL,
                hydrophobic_moment REAL NOT NULL,
                rank_score REAL NOT NULL,
                left_cleavage_probability REAL NOT NULL,
                right_cleavage_probability REAL NOT NULL,
                predicted_cleavage_probability REAL NOT NULL,
                pddp_score REAL,
                score_threshold REAL,
                passes_score_threshold INTEGER,
                passes_cationic_cterm INTEGER
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX idx_candidates_pddp_score ON candidates (pddp_score DESC, id ASC)"
        )

    def add(self, candidate: CandidatePeptide) -> None:
        self._conn.execute(
            """
            INSERT INTO candidates (
                peptide_id, sequence, source_protein_id, start, end, length,
                net_charge, hydrophobicity, hydrophobic_moment, rank_score,
                left_cleavage_probability, right_cleavage_probability,
                predicted_cleavage_probability, pddp_score, score_threshold,
                passes_score_threshold, passes_cationic_cterm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _candidate_to_spill_row(candidate),
        )
        self.count += 1
        if self.count % _SPILL_COMMIT_INTERVAL == 0:
            self._conn.commit()

    def iter_sorted(self) -> Iterator[CandidatePeptide]:
        self._conn.commit()
        cursor = self._conn.execute(
            """
            SELECT *
            FROM candidates
            ORDER BY COALESCE(pddp_score, -1e308) DESC, id ASC
            """
        )
        for row in cursor:
            yield _candidate_from_spill_row(row)

    def to_list_with_ids(self) -> list[CandidatePeptide]:
        return [_with_peptide_id(candidate, index + 1) for index, candidate in enumerate(self.iter_sorted())]

    def write_sorted_outputs(self, table_stem: Path, txt_path: Path, *, output_format: str) -> Path:
        csv_path = table_stem.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_handle, txt_path.open(
            "w", encoding="utf-8"
        ) as txt_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=_CANDIDATE_FIELDNAMES)
            writer.writeheader()
            for index, candidate in enumerate(self.iter_sorted(), start=1):
                final_candidate = _with_peptide_id(candidate, index)
                writer.writerow(asdict(final_candidate))
                txt_handle.write(f"{final_candidate.peptide_id} {final_candidate.sequence}\n")

        if output_format == "parquet" or (output_format == "auto" and _can_write_parquet()):
            parquet_path = table_stem.with_suffix(".parquet")
            _csv_to_parquet_chunked(csv_path, parquet_path)
            return parquet_path
        return csv_path

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


def _csv_to_parquet_chunked(csv_path: Path, parquet_path: Path, *, chunksize: int = 100_000) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


def write_candidates_csv(candidates: list[CandidatePeptide], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CandidatePeptide.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def write_candidates_table(candidates: list[CandidatePeptide], path: Path, *, output_format: str) -> Path:
    if output_format == "parquet" or (output_format == "auto" and _can_write_parquet()):
        table_path = path.with_suffix(".parquet")
        _write_parquet(candidates, table_path)
        return table_path
    table_path = path.with_suffix(".csv")
    write_candidates_csv(candidates, table_path)
    return table_path


def resolve_candidates_table_path(table_stem: Path, output_format: str) -> Path:
    if output_format == "parquet" or (output_format == "auto" and _can_write_parquet()):
        return table_stem.with_suffix(".parquet")
    return table_stem.with_suffix(".csv")


def _can_write_parquet() -> bool:
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


def _write_parquet(candidates: list[CandidatePeptide], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(candidate) for candidate in candidates]).to_parquet(path, index=False)


def write_pipeline_txt(candidates: list[CandidatePeptide], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(f"{candidate.peptide_id} {candidate.sequence}\n")

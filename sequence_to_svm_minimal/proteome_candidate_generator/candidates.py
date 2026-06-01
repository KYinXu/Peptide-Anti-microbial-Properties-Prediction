"""Candidate peptide expansion, filtering, ranking, and outputs."""

from __future__ import annotations

import csv
import heapq
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from proteome_candidate_generator.cleavage import ProteinCleavageSites
from proteome_candidate_generator.fasta import ProteinRecord
from proteome_candidate_generator.pddp_scoring import SequenceScorer
from proteome_candidate_generator.progress import progress_iter

HYDROPHOBIC_AA = frozenset("AILMFVPG")
POSITIVE_AA = frozenset("RK")
NEGATIVE_AA = frozenset("DE")


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


def net_charge(sequence: str) -> int:
    return sum(1 for aa in sequence if aa in POSITIVE_AA) - sum(1 for aa in sequence if aa in NEGATIVE_AA)


def hydrophobicity(sequence: str) -> float:
    return sum(1 for aa in sequence if aa in HYDROPHOBIC_AA) / len(sequence)


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
    include_terminal_boundaries: bool,
    show_progress: bool = False,
) -> tuple[list[CandidatePeptide], CandidateStats]:
    retained: list[CandidatePeptide] = []
    expanded = score_filtered = 0

    record_iter = records
    if show_progress:
        record_iter = progress_iter(records, desc="Expanding and paper-scoring candidates", total=len(records))
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
            score = scorer.score_sequence(sequence)
            if score <= score_threshold:
                score_filtered += 1
                continue
            retained.append(
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

    non_overlapping, overlap_removed = _remove_overlapping_lower_scores(retained)
    final_candidates = non_overlapping
    cterm_filtered = 0
    if require_cationic_cterm:
        filtered: list[CandidatePeptide] = []
        for candidate in final_candidates:
            passed = has_cationic_cterm(candidate.sequence, cationic_cterm_residues)
            candidate = _replace_candidate(candidate, passes_cationic_cterm=passed)
            if passed:
                filtered.append(candidate)
            else:
                cterm_filtered += 1
        final_candidates = filtered
    else:
        final_candidates = [
            _replace_candidate(candidate, passes_cationic_cterm=None)
            for candidate in final_candidates
        ]

    final_candidates.sort(key=lambda row: (row.pddp_score or float("-inf")), reverse=True)
    final = [_with_peptide_id(candidate, index + 1) for index, candidate in enumerate(final_candidates)]
    stats = CandidateStats(
        expanded=expanded,
        duplicate_sequences=0,
        failed_filters=score_filtered + cterm_filtered,
        retained=len(final),
        score_filtered=score_filtered,
        overlap_removed=overlap_removed,
        cterm_filtered=cterm_filtered,
    )
    return final, stats


def _remove_overlapping_lower_scores(candidates: list[CandidatePeptide]) -> tuple[list[CandidatePeptide], int]:
    selected: list[CandidatePeptide] = []
    removed = 0
    by_protein: dict[str, list[CandidatePeptide]] = {}
    for candidate in candidates:
        by_protein.setdefault(candidate.source_protein_id, []).append(candidate)
    for protein_candidates in by_protein.values():
        chosen: list[CandidatePeptide] = []
        for candidate in sorted(
            protein_candidates,
            key=lambda row: (row.pddp_score or float("-inf"), row.predicted_cleavage_probability, row.length),
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

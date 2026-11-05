"""Enumerate mutation variants for parent peptides."""

from __future__ import annotations

import random
from typing import Iterable, Iterator

from .common import MutantRecord, SubsetRecord
from .rules import (
    SCHEMES,
    build_blosum_table,
    eligible_positions,
    sample_replacement,
    substitution_count,
)


DEFAULT_P_LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_REPLICATES = 500
DEFAULT_SEED = 0


def generate_mutants(
    parents: Iterable[SubsetRecord],
    *,
    p_levels: Iterable[float] = DEFAULT_P_LEVELS,
    replicates: int = DEFAULT_REPLICATES,
    schemes: Iterable[str] = SCHEMES,
    seed: int = DEFAULT_SEED,
    blosum_temperature: float = 1.0,
) -> Iterator[MutantRecord]:
    blosum_table = build_blosum_table(blosum_temperature)
    p_levels = list(p_levels)
    schemes = list(schemes)
    for scheme in schemes:
        if scheme not in SCHEMES:
            raise ValueError(f"Unknown mutation scheme: {scheme}")

    for parent in parents:
        yield _parent_record(parent)
        for scheme in schemes:
            positions = eligible_positions(parent.sequence, scheme, blosum_table)
            for p in p_levels:
                k_requested = substitution_count(p, len(parent.sequence))
                k_applied = min(k_requested, len(positions))
                if k_applied == 0:
                    continue
                for replicate in range(replicates):
                    rng = random.Random(
                        f"{seed}|{parent.id}|{scheme}|{round(p * 100)}|{replicate}"
                    )
                    yield _mutate(
                        parent,
                        scheme=scheme,
                        p=p,
                        replicate=replicate,
                        positions=positions,
                        k_requested=k_requested,
                        k_applied=k_applied,
                        rng=rng,
                        blosum_table=blosum_table,
                    )


def _mutate(
    parent: SubsetRecord,
    *,
    scheme: str,
    p: float,
    replicate: int,
    positions: list[int],
    k_requested: int,
    k_applied: int,
    rng: random.Random,
    blosum_table,
) -> MutantRecord:
    chosen = sorted(rng.sample(positions, k_applied))
    residues = list(parent.sequence)
    edits = []
    for index in chosen:
        origin = parent.sequence[index]
        replacement = sample_replacement(rng, scheme, origin, blosum_table)
        residues[index] = replacement
        edits.append(f"{origin}{index + 1}{replacement}")
    return MutantRecord(
        mutant_id=f"{parent.id}_{scheme}_p{round(p * 100):02d}_r{replicate:03d}",
        parent_id=parent.id,
        parent_sequence=parent.sequence,
        sequence="".join(residues),
        scheme=scheme,
        p=p,
        replicate=replicate,
        k_requested=k_requested,
        k_applied=k_applied,
        mutations=";".join(edits),
        extras=dict(parent.extras),
    )


def _parent_record(parent: SubsetRecord) -> MutantRecord:
    return MutantRecord(
        mutant_id=f"{parent.id}_parent",
        parent_id=parent.id,
        parent_sequence=parent.sequence,
        sequence=parent.sequence,
        scheme="parent",
        p=0.0,
        replicate=0,
        k_requested=0,
        k_applied=0,
        mutations="",
        extras=dict(parent.extras),
    )

"""Dataclasses for parent peptides and mutation variants."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubsetRecord:
    id: str
    sequence: str
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MutantRecord:
    mutant_id: str
    parent_id: str
    parent_sequence: str
    sequence: str
    scheme: str
    p: float
    replicate: int
    k_requested: int
    k_applied: int
    mutations: str
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)

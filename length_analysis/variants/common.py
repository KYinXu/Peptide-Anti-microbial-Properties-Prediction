"""Dataclasses for parent peptides and length-extension variants."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubsetRecord:
    id: str
    sequence: str
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentRecord:
    id: str
    sequence: str
    source_protein_id: str
    start: int
    end: int
    left_flank: str
    right_flank: str
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class VariantRecord:
    variant_id: str
    parent_id: str
    sequence: str
    left_added: int
    right_added: int
    core_length: int
    source_protein_id: str
    start: int
    end: int
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def variant_length(self) -> int:
        return len(self.sequence)

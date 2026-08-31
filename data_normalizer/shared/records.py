"""Shared normalized sequence dataset contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol


REQUIRED_COLUMNS = ["id", "sequence"]


@dataclass(frozen=True)
class NormalizedSequenceRecord:
    id: str
    sequence: str
    extras: dict[str, str] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, str]:
        row = {"id": self.id, "sequence": self.sequence}
        row.update(self.extras)
        return row


class RecordNormalizer(Protocol):
    """Converts one raw input format into normalized sequence records."""

    def iter_records(self) -> Iterable[NormalizedSequenceRecord]:
        """Yield normalized records with at least id and sequence."""


def normalize_sequence(sequence: str, record_id: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"{record_id}: empty sequence.")
    return normalized

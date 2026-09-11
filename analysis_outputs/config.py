"""Standard configuration recorded by analysis run manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfiguration:
    output_mode: str
    effective: Mapping[str, Any]
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_mode.strip():
            raise ValueError("output_mode must be non-empty.")

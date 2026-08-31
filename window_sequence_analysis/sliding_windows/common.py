"""Shared data structures for compact sliding-window analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WindowRecord:
    start: int
    end: int
    length: int
    sequence: str


@dataclass(frozen=True)
class WindowScores:
    p_amp: np.ndarray
    hyperplane_distance: np.ndarray


@dataclass(frozen=True)
class ProfileConfig:
    min_len: int = 10
    max_len: int = 35
    stride: int = 1
    batch_starts: int = 64
    precision: int = 6


@dataclass
class BestWindow:
    p_amp: float = float("-inf")
    hyperplane_distance: float = float("nan")
    start: int = -1
    end: int = -1
    length: int = 0
    sequence: str = ""


class WindowScorer(Protocol):
    """Model adapter interface used by the sliding profile code."""

    def score(self, windows: list[WindowRecord]) -> WindowScores:
        """Return one P(AMP) and hyperplane-distance value per input window."""

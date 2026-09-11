"""Model-independent aggregation of window scores into residue profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .common import PROFILE_AGGREGATIONS, ProfileAggregation, WindowRecord


@dataclass
class ResidueProfileAccumulator:
    aggregation: ProfileAggregation
    values: dict[str, np.ndarray]
    coverage: dict[str, np.ndarray]

    @classmethod
    def create(
        cls,
        length: int,
        metric_names: Sequence[str],
        aggregation: ProfileAggregation,
    ) -> "ResidueProfileAccumulator":
        if aggregation not in PROFILE_AGGREGATIONS:
            raise ValueError(f"Unsupported profile aggregation: {aggregation}")
        initial = 0.0 if aggregation == "mean" else float("-inf")
        return cls(
            aggregation=aggregation,
            values={
                name: np.full(length, initial, dtype=np.float64)
                for name in metric_names
            },
            coverage={
                name: np.zeros(length, dtype=np.int32)
                for name in metric_names
            },
        )

    def update(
        self,
        windows: Sequence[WindowRecord],
        metric_scores: Mapping[str, np.ndarray],
    ) -> None:
        self._validate_metrics(windows, metric_scores)
        for window_index, window in enumerate(windows):
            for name, scores in metric_scores.items():
                score = float(scores[window_index])
                if not np.isfinite(score):
                    continue
                target = self.values[name][window.start : window.end]
                if self.aggregation == "mean":
                    target += score
                else:
                    np.maximum(target, score, out=target)
                self.coverage[name][window.start : window.end] += 1

    def profiles(self) -> dict[str, np.ndarray]:
        return {
            name: self._finalize_metric(name)
            for name in self.values
        }

    def _finalize_metric(self, name: str) -> np.ndarray:
        covered = self.coverage[name] > 0
        profile = np.full(len(covered), np.nan, dtype=np.float64)
        if self.aggregation == "mean":
            profile[covered] = self.values[name][covered] / self.coverage[name][covered]
        else:
            profile[covered] = self.values[name][covered]
        return profile

    def _validate_metrics(
        self,
        windows: Sequence[WindowRecord],
        metric_scores: Mapping[str, np.ndarray],
    ) -> None:
        expected = set(self.values)
        supplied = set(metric_scores)
        if supplied != expected:
            raise ValueError(
                f"Metric names do not match accumulator: expected {sorted(expected)}, "
                f"received {sorted(supplied)}."
            )
        mismatched = [name for name, scores in metric_scores.items() if len(scores) != len(windows)]
        if mismatched:
            raise ValueError(
                f"Window and score counts do not match for metric(s): {', '.join(sorted(mismatched))}."
            )

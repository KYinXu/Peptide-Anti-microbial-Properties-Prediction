"""Model-independent aggregation of window scores into residue profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .common import PROFILE_AGGREGATIONS, ProfileAggregation, WindowRecord


@dataclass
class ResidueProfileAccumulator:
    aggregation: ProfileAggregation
    sum_values: dict[str, np.ndarray]
    max_values: dict[str, np.ndarray]
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
        return cls(
            aggregation=aggregation,
            sum_values={
                name: np.zeros(length, dtype=np.float64)
                for name in metric_names
            },
            max_values={
                name: np.full(length, float("-inf"), dtype=np.float64)
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
                
                if self.aggregation in ("mean", "both"):
                    self.sum_values[name][window.start : window.end] += score
                
                if self.aggregation in ("max", "both"):
                    target_max = self.max_values[name][window.start : window.end]
                    np.maximum(target_max, score, out=target_max)
                    
                self.coverage[name][window.start : window.end] += 1

    def profiles(self) -> dict[str, np.ndarray]:
        result = {}
        for name in self.coverage:
            covered = self.coverage[name] > 0
            
            if self.aggregation in ("mean", "both"):
                mean_profile = np.full(len(covered), np.nan, dtype=np.float64)
                mean_profile[covered] = self.sum_values[name][covered] / self.coverage[name][covered]
                result[f"{name}_mean"] = mean_profile
                
            if self.aggregation in ("max", "both"):
                max_profile = np.full(len(covered), np.nan, dtype=np.float64)
                max_profile[covered] = self.max_values[name][covered]
                result[f"{name}_max"] = max_profile
                
        return result

    def _validate_metrics(
        self,
        windows: Sequence[WindowRecord],
        metric_scores: Mapping[str, np.ndarray],
    ) -> None:
        expected = set(self.coverage)
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

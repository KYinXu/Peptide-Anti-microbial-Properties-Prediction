"""Low-level residue-profile accumulation for one sliding-window analysis."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from window_sequence_analysis.sliding_windows.common import BestWindow, ProfileConfig, SequenceRecord, WindowRecord


def update_profiles(
    windows: list[WindowRecord],
    p_amp: np.ndarray,
    distance: np.ndarray,
    p_amp_sum: np.ndarray,
    distance_sum: np.ndarray,
    coverage: np.ndarray,
    best: BestWindow,
) -> None:
    for window, probability, hyperplane_distance in zip(windows, p_amp, distance):
        p_amp_sum[window.start : window.end] += probability
        distance_sum[window.start : window.end] += hyperplane_distance
        coverage[window.start : window.end] += 1
        if probability > best.p_amp:
            best.p_amp = float(probability)
            best.hyperplane_distance = float(hyperplane_distance)
            best.start = window.start
            best.end = window.end
            best.length = window.length
            best.sequence = window.sequence


def finalize_positions(
    start: int,
    end: int,
    p_amp_sum: np.ndarray,
    distance_sum: np.ndarray,
    coverage: np.ndarray,
    p_amp_profile: np.ndarray,
    distance_profile: np.ndarray,
) -> None:
    if end < start:
        return
    slc = slice(start, end + 1)
    covered = coverage[slc] > 0
    p_values = np.full(end - start + 1, np.nan, dtype=np.float64)
    d_values = np.full(end - start + 1, np.nan, dtype=np.float64)
    p_values[covered] = p_amp_sum[slc][covered] / coverage[slc][covered]
    d_values[covered] = distance_sum[slc][covered] / coverage[slc][covered]
    p_amp_profile[slc] = p_values
    distance_profile[slc] = d_values


def build_output_row(
    record: SequenceRecord,
    p_amp_profile: np.ndarray,
    distance_profile: np.ndarray,
    coverage: np.ndarray,
    best: BestWindow,
    window_count: int,
    config: ProfileConfig,
) -> dict[str, Any]:
    max_p_amp_index = finite_argmax(p_amp_profile)
    max_distance_index = finite_argmax(distance_profile)
    row: dict[str, Any] = {
        "id": record.id,
        "sequence_length": len(record.sequence),
        "window_min_len": config.min_len,
        "window_max_len": config.max_len,
        "stride": config.stride,
        "window_count": window_count,
        "profile_aggregation": "mean_over_covering_windows",
        "profile_index_base": 1,
        "profile_delimiter": ";",
        "p_amp_mean": finite_mean(p_amp_profile),
        "p_amp_max_residue_mean": finite_value(p_amp_profile, max_p_amp_index),
        "p_amp_max_residue_1based": none_or_one_based(max_p_amp_index),
        "hyperplane_distance_mean": finite_mean(distance_profile),
        "hyperplane_distance_max_residue_mean": finite_value(distance_profile, max_distance_index),
        "hyperplane_distance_max_residue_1based": none_or_one_based(max_distance_index),
        "best_window_p_amp": none_if_not_finite(best.p_amp),
        "best_window_hyperplane_distance": none_if_not_finite(best.hyperplane_distance),
        "best_window_start_0based": None if best.start < 0 else best.start,
        "best_window_end_0based_exclusive": None if best.end < 0 else best.end,
        "best_window_start_1based": None if best.start < 0 else best.start + 1,
        "best_window_end_1based_inclusive": None if best.end < 0 else best.end,
        "best_window_length": None if best.length == 0 else best.length,
        "best_window_sequence": best.sequence,
        "coverage_profile": format_int_profile(coverage),
        "p_amp_mean_profile": format_float_profile(p_amp_profile, config.precision),
        "hyperplane_distance_mean_profile": format_float_profile(distance_profile, config.precision),
    }
    for key, value in record.extras.items():
        row[key if key not in row else f"label_{key}"] = value
    return row


def finite_argmax(values: np.ndarray) -> int | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    finite_indices = np.flatnonzero(finite)
    return int(finite_indices[np.argmax(values[finite])])


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def finite_value(values: np.ndarray, index: int | None) -> float | None:
    if index is None:
        return None
    value = float(values[index])
    return value if np.isfinite(value) else None


def none_or_one_based(index: int | None) -> int | None:
    return None if index is None else index + 1


def none_if_not_finite(value: float) -> float | None:
    return value if np.isfinite(value) else None


def format_float_profile(values: np.ndarray, precision: int) -> str:
    return ";".join("nan" if not np.isfinite(value) else f"{float(value):.{precision}g}" for value in values)


def format_int_profile(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in values)

"""Profile summaries and output rows for sliding-window analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import BestWindow, ProfileConfig, SequenceRecord, WindowRecord


def update_best_window(
    windows: list[WindowRecord],
    p_amp: np.ndarray,
    distance: np.ndarray,
    best: BestWindow,
) -> None:
    for window, probability, hyperplane_distance in zip(windows, p_amp, distance):
        if probability > best.p_amp:
            best.p_amp = float(probability)
            best.hyperplane_distance = float(hyperplane_distance)
            best.start = window.start
            best.end = window.end
            best.length = window.length
            best.sequence = window.sequence


def build_output_row(
    record: SequenceRecord,
    profiles: dict[str, np.ndarray],
    best: BestWindow,
    window_count: int,
    config: ProfileConfig,
) -> dict[str, Any]:
    # Use max profile if available, fallback to mean
    p_amp_profile = profiles.get("p_amp_max", profiles.get("p_amp_mean"))
    distance_profile = profiles.get("hyperplane_distance_max", profiles.get("hyperplane_distance_mean"))
    
    max_p_amp_index = finite_argmax(p_amp_profile)
    max_distance_index = finite_argmax(distance_profile)
    
    p_amp_serialized = format_float_profile(p_amp_profile, config.precision)
    distance_serialized = format_float_profile(distance_profile, config.precision)
    
    p_amp_mean_serialized = format_float_profile(profiles["p_amp_mean"], config.precision) if "p_amp_mean" in profiles else None
    distance_mean_serialized = format_float_profile(profiles["hyperplane_distance_mean"], config.precision) if "hyperplane_distance_mean" in profiles else None
    
    p_amp_max_serialized = format_float_profile(profiles["p_amp_max"], config.precision) if "p_amp_max" in profiles else None
    distance_max_serialized = format_float_profile(profiles["hyperplane_distance_max"], config.precision) if "hyperplane_distance_max" in profiles else None

    row: dict[str, Any] = {
        "id": record.id,
        "sequence_length": len(record.sequence),
        "window_min_len": config.min_len,
        "window_max_len": config.max_len,
        "stride": config.stride,
        "window_count": window_count,
        "profile_aggregation": f"{config.aggregation}_over_covering_windows",
        "profile_index_base": 1,
        "profile_delimiter": ";",
        "p_amp_mean": finite_mean(profiles.get("p_amp_mean", p_amp_profile)),
        "p_amp_max_residue_index": none_or_one_based(max_p_amp_index),
        "hyperplane_distance_mean": finite_mean(profiles.get("hyperplane_distance_mean", distance_profile)),
        "hyperplane_distance_max_residue_index": none_or_one_based(max_distance_index),
        "best_window_p_amp": none_if_not_finite(best.p_amp),
        "best_window_hyperplane_distance": none_if_not_finite(best.hyperplane_distance),
        "best_window_start_0based": None if best.start < 0 else best.start,
        "best_window_end_0based_exclusive": None if best.end < 0 else best.end,
        "best_window_start_index": None if best.start < 0 else best.start + 1,
        "best_window_end_index_inclusive": None if best.end < 0 else best.end,
        "best_window_length": None if best.length == 0 else best.length,
        "best_window_sequence": best.sequence,
        "p_amp_mean_profile": p_amp_mean_serialized,
        "hyperplane_distance_mean_profile": distance_mean_serialized,
        "p_amp_max_profile": p_amp_max_serialized,
        "hyperplane_distance_max_profile": distance_max_serialized,
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

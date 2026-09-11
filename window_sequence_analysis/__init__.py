"""Compact window sequence analysis tools."""

from .data_loader import NormalizedSequenceDataset
from .models import SvmWindowScorer
from .sliding_windows import (
    PROFILE_AGGREGATIONS,
    ProfileAggregation,
    ProfileConfig,
    ResidueProfileAccumulator,
    SequenceRecord,
    WindowConfig,
    WindowRecord,
    WindowScorer,
    WindowScores,
    build_progress_reporter,
    profile_sequence,
    run_raw_window_analysis,
    run_window_profile_analysis,
)

__all__ = [
    "NormalizedSequenceDataset",
    "PROFILE_AGGREGATIONS",
    "ProfileAggregation",
    "ProfileConfig",
    "ResidueProfileAccumulator",
    "SequenceRecord",
    "SvmWindowScorer",
    "WindowRecord",
    "WindowScorer",
    "WindowScores",
    "build_progress_reporter",
    "profile_sequence",
    "run_raw_window_analysis",
    "run_window_profile_analysis",
    "WindowConfig",
]

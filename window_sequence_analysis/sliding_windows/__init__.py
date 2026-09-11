"""Model-agnostic sliding-window profile generation."""

from .aggregation import ResidueProfileAccumulator
from .common import (
    PROFILE_AGGREGATIONS,
    ProfileAggregation,
    ProfileConfig,
    SequenceRecord,
    WindowConfig,
    WindowRecord,
    WindowScorer,
    WindowScores,
)
from .progress import build_progress_reporter
from .raw import build_raw_rows, iter_raw_window_rows
from .runner import run_raw_window_analysis, run_window_profile_analysis
from .sequence import iter_window_batches, profile_sequence, windows_at_start

__all__ = [
    "build_progress_reporter",
    "build_raw_rows",
    "iter_raw_window_rows",
    "iter_window_batches",
    "profile_sequence",
    "PROFILE_AGGREGATIONS",
    "ProfileAggregation",
    "ProfileConfig",
    "ResidueProfileAccumulator",
    "run_raw_window_analysis",
    "run_window_profile_analysis",
    "SequenceRecord",
    "WindowConfig",
    "WindowRecord",
    "WindowScorer",
    "WindowScores",
    "windows_at_start",
]

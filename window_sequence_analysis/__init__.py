"""Compact window sequence analysis tools."""

from .data_loader import NormalizedSequenceDataset
from .models import SvmWindowScorer
from .sliding_windows import (
    ProfileConfig,
    SequenceRecord,
    WindowRecord,
    WindowScorer,
    WindowScores,
    build_progress_reporter,
    profile_sequence,
    run_window_profile_analysis,
)

__all__ = [
    "NormalizedSequenceDataset",
    "ProfileConfig",
    "SequenceRecord",
    "SvmWindowScorer",
    "WindowRecord",
    "WindowScorer",
    "WindowScores",
    "build_progress_reporter",
    "profile_sequence",
    "run_window_profile_analysis",
]

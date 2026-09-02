"""Model-agnostic sliding-window profile generation."""

from .common import ProfileConfig, SequenceRecord, WindowRecord, WindowScorer, WindowScores
from .progress import build_progress_reporter
from .runner import run_window_profile_analysis
from .sequence import iter_window_batches, profile_sequence, windows_at_start

__all__ = [
    "build_progress_reporter",
    "iter_window_batches",
    "profile_sequence",
    "ProfileConfig",
    "run_window_profile_analysis",
    "SequenceRecord",
    "WindowRecord",
    "WindowScorer",
    "WindowScores",
    "windows_at_start",
]

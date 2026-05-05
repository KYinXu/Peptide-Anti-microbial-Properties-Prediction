"""Repository root (sequence_to_svm_minimal/)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Windowed blind mode (parent canonical): expanded windows for SVM / joins only.
CANONICAL_WINDOWS_SIDECAR = "canonical_windows_expanded.txt"

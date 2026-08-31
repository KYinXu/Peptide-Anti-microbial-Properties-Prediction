"""Optional terminal progress reporting for sliding-window runs."""

from __future__ import annotations

import sys
from typing import Protocol, TextIO


class ProgressReporter(Protocol):
    def update(self, count: int, record_id: str) -> None:
        """Report that another sequence has been processed."""

    def close(self) -> None:
        """Flush or finalize progress output."""


class NullProgressReporter:
    def update(self, count: int, record_id: str) -> None:
        return

    def close(self) -> None:
        return


class TerminalProgressReporter:
    def __init__(self, every: int = 1, stream: TextIO | None = None) -> None:
        if every < 1:
            raise ValueError("Progress interval must be at least 1.")
        self.every = every
        self.stream = sys.stderr if stream is None else stream

    def update(self, count: int, record_id: str) -> None:
        if count % self.every != 0:
            return
        print(f"Processed {count} sequence(s); latest id={record_id}", file=self.stream, flush=True)

    def close(self) -> None:
        self.stream.flush()


def build_progress_reporter(*, quiet: bool = False, every: int = 1) -> ProgressReporter:
    if quiet:
        return NullProgressReporter()
    return TerminalProgressReporter(every=every)

"""Optional tqdm progress reporting for sliding-window runs."""

from __future__ import annotations

from typing import Protocol

from tqdm import tqdm


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


class TqdmProgressReporter:
    def __init__(self, *, total: int | None = None) -> None:
        self._bar = tqdm(
            total=total,
            desc="Profiling sequences",
            unit="seq",
            dynamic_ncols=True,
        )

    def update(self, count: int, record_id: str) -> None:
        self._bar.set_postfix_str(f"id={record_id}", refresh=False)
        self._bar.update(1)

    def close(self) -> None:
        self._bar.close()


def build_progress_reporter(*, quiet: bool = False, total: int | None = None) -> ProgressReporter:
    if quiet:
        return NullProgressReporter()
    return TqdmProgressReporter(total=total)

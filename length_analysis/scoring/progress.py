"""Two-level tqdm progress for length runs: stages completed, variants within a stage.

Unlike the mutation sweep, every variant of every parent is folded and scored in one
batch (ESMFold reloads on each ``ensure_processed_data`` call, so a per-parent loop
would reload it once per peptide). Peptide-level progress therefore does not exist to
report; the stage bar stands in for it, and the variant bar carries the parent id of
the variant in flight so the current peptide is still visible.

Bars sit at positions 1 and 2, leaving position 0 to the ``Folding sequences`` bar
that ``gnn_minimal`` opens during the fold stage.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

from tqdm import tqdm

from .runner import NullProgressReporter


STAGES = ("folding", "descriptors", "scoring")


class TqdmProgressReporter:
    def __init__(self, *, labels: list[str] | None = None) -> None:
        self._labels = labels or []
        self._phase = None
        self._stages = tqdm(
            total=len(STAGES),
            desc="Stages",
            unit="stage",
            position=1,
            dynamic_ncols=True,
        )
        self._variants = tqdm(
            total=None,
            desc="  pending",
            unit="var",
            position=2,
            dynamic_ncols=True,
            leave=False,
        )

    def set_labels(self, labels: list[str]) -> None:
        """Parent id per variant, in scoring order, for the variant bar's postfix."""
        self._labels = labels

    def stage(self, phase: str, done: int, total: int) -> None:
        bar = self._variants
        if phase != self._phase:
            if self._phase is not None:
                self._stages.update(1)
            self._phase = phase
            bar.reset(total=total)
            bar.set_description_str(f"  {phase}")
            self._stages.set_postfix_str(phase, refresh=False)
        if bar.total != total:
            bar.reset(total=total)
            bar.set_description_str(f"  {phase}")
        if self._labels and done:
            bar.set_postfix_str(self._labels[min(done, len(self._labels)) - 1], refresh=False)
        bar.update(max(done - bar.n, 0))

    def quieted(self):
        """gnn_minimal prints progress text per 100 rows, which would break up the bars.

        Its own ``Folding sequences`` bar writes to stderr and still shows through.
        """
        return redirect_stdout(io.StringIO())

    def update(self, count: int, label: str = "") -> None:
        self._stages.set_postfix_str(f"writing {label}", refresh=False)

    def close(self) -> None:
        if self._phase is not None:
            self._stages.update(self._stages.total - self._stages.n)
        self._variants.close()
        self._stages.close()


def build_length_progress(*, quiet: bool = False, labels: list[str] | None = None):
    if quiet:
        return NullProgressReporter()
    return TqdmProgressReporter(labels=labels)

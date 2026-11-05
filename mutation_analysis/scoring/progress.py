"""Two-level tqdm progress for mutation runs: parents completed, current parent scored.

Bars write to stderr, so stdout stays clean for the run summary. The inner bar's total
is unknown until a parent's mutants are generated, so the first report sets it.
"""

from __future__ import annotations

from tqdm import tqdm


class NullMutationProgress:
    def start_parent(self, parent_id: str):
        return None

    def finish_parent(self, parent_id: str, counted: int) -> None:
        return

    def close(self) -> None:
        return


class TqdmMutationProgress:
    """``per_parent`` off for pooled runs, where workers report nothing until done."""

    def __init__(self, *, total_parents: int, per_parent: bool = True) -> None:
        self._parents = tqdm(
            total=total_parents,
            desc="Peptides",
            unit="pep",
            position=0,
            dynamic_ncols=True,
        )
        self._mutants = (
            tqdm(
                total=None,
                desc="  pending",
                unit="mut",
                position=1,
                dynamic_ncols=True,
                leave=False,
            )
            if per_parent
            else None
        )

    def start_parent(self, parent_id: str):
        bar = self._mutants
        if bar is None:
            return None
        bar.reset(total=None)
        bar.set_description_str(f"  {parent_id}")
        bar.set_postfix_str("", refresh=False)

        def report(phase: str, done: int, total: int) -> None:
            if bar.total != total:
                bar.reset(total=total)
                bar.set_description_str(f"  {parent_id}")
            bar.set_postfix_str(phase, refresh=False)
            bar.update(max(done - bar.n, 0))

        return report

    def finish_parent(self, parent_id: str, counted: int) -> None:
        self._parents.set_postfix_str(f"{parent_id}: {counted} mutant(s)", refresh=False)
        self._parents.update(1)

    def close(self) -> None:
        if self._mutants is not None:
            self._mutants.close()
        self._parents.close()


def build_mutation_progress(
    *, quiet: bool = False, total_parents: int = 0, per_parent: bool = True
):
    if quiet:
        return NullMutationProgress()
    return TqdmMutationProgress(total_parents=total_parents, per_parent=per_parent)

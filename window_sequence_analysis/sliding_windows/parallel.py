"""Process-pool helpers for per-sequence window profiling."""

from __future__ import annotations

from multiprocessing import get_context
from typing import Any, Callable, Iterable, Iterator

from .common import ProfileConfig, SequenceRecord, WindowScorer
from .sequence import profile_sequence


_scorer: WindowScorer | None = None
_config: ProfileConfig | None = None


def init_profile_worker(scorer_factory: Callable[[], WindowScorer], config: ProfileConfig) -> None:
    global _scorer, _config
    _scorer = scorer_factory()
    _config = config


def profile_record_worker(record: SequenceRecord) -> dict[str, Any]:
    if _scorer is None or _config is None:
        raise RuntimeError("Worker was not initialized.")
    return profile_sequence(record, _scorer, _config)


def iter_parallel_profile_rows(
    records: Iterable[SequenceRecord],
    scorer_factory: Callable[[], WindowScorer],
    config: ProfileConfig,
    workers: int,
) -> Iterator[dict[str, Any]]:
    context = get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_profile_worker,
        initargs=(scorer_factory, config),
    ) as pool:
        yield from pool.imap_unordered(profile_record_worker, records, chunksize=1)

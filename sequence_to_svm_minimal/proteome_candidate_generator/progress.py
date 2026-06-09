"""Small progress helpers with a no-dependency fallback."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress_iter(items: Iterable[T], *, desc: str, total: int | None = None) -> Iterator[T]:
    try:
        from tqdm import tqdm
    except ImportError:
        yield from _fallback_progress(items, desc=desc, total=total)
        return
    yield from tqdm(items, desc=desc, total=total, unit="item")


def _fallback_progress(items: Iterable[T], *, desc: str, total: int | None) -> Iterator[T]:
    print(f"{desc}...", flush=True)
    step = max(1, (total or 100) // 20)
    for index, item in enumerate(items, start=1):
        if index == 1 or index % step == 0:
            suffix = f"/{total}" if total is not None else ""
            print(f"{desc}: {index}{suffix}", flush=True)
        yield item
    print(f"{desc}: done", flush=True)

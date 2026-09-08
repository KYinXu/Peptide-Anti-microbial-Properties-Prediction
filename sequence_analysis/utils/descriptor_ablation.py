"""Utilities for experimental QSAR descriptor ablation."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import TypeVar


T = TypeVar("T", int, float)


def split_descriptor_tokens(values: Iterable[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or ():
        for token in str(value).replace(",", " ").split():
            name = token.strip()
            if name:
                tokens.append(name)
    return tokens


def parse_null_descriptor_names(
    values: Iterable[str] | None,
    valid_names: Iterable[str],
) -> tuple[str, ...]:
    valid = tuple(valid_names)
    valid_set = set(valid)
    unknown = sorted(set(split_descriptor_tokens(values)) - valid_set)
    if unknown:
        raise ValueError(
            "Unknown descriptor(s) for ablation: "
            + ", ".join(unknown)
            + ". Valid descriptors: "
            + ", ".join(valid)
        )

    seen: set[str] = set()
    parsed: list[str] = []
    for name in split_descriptor_tokens(values):
        if name not in seen:
            seen.add(name)
            parsed.append(name)
    return tuple(parsed)


def null_descriptor_values(
    values: MutableMapping[str, T],
    descriptor_names: Iterable[str],
) -> MutableMapping[str, T]:
    for name in descriptor_names:
        values[name] = 0.0  # type: ignore[assignment]
    return values


def null_descriptor_frame(df, descriptor_names: Iterable[str]):
    names = tuple(descriptor_names)
    if not names:
        return df
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValueError(f"Cannot null descriptor(s) missing from table: {', '.join(missing)}")
    out = df.copy()
    for name in names:
        out[name] = 0.0
    return out

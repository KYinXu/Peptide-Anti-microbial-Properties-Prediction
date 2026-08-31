"""Parse FASTA labels into ID, optional name, keyed fields, and raw text."""

from __future__ import annotations

import re
from dataclasses import dataclass


FIELD_START = re.compile(r"^([A-Za-z][A-Za-z0-9_.:-]*)=(.*)$")


@dataclass(frozen=True)
class FastaLabel:
    id: str
    name: str | None
    fields: dict[str, str]
    rawtext: str


def parse_fasta_label(header: str) -> FastaLabel:
    rawtext = header.strip()
    if not rawtext:
        raise ValueError("FASTA header is missing a sequence ID.")

    parts = rawtext.split()
    record_id = parts[0]
    name_tokens: list[str] = []
    fields: dict[str, str] = {}
    active_key: str | None = None
    active_values: list[str] = []

    for token in parts[1:]:
        match = FIELD_START.match(token)
        if match:
            flush_field(fields, active_key, active_values)
            active_key = match.group(1)
            active_values = [match.group(2)] if match.group(2) else []
        elif active_key is None:
            name_tokens.append(token)
        else:
            active_values.append(token)

    flush_field(fields, active_key, active_values)
    name = " ".join(name_tokens).strip() or None
    return FastaLabel(id=record_id, name=name, fields=fields, rawtext=rawtext)


def flush_field(fields: dict[str, str], key: str | None, values: list[str]) -> None:
    if key is None:
        return
    fields[key] = " ".join(values).strip()

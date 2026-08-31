"""Writers for normalized sequence datasets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from data_normalizer.shared.records import NormalizedSequenceRecord, REQUIRED_COLUMNS


def write_normalized_csv(path: Path, records: Iterable[NormalizedSequenceRecord]) -> int:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted({key for row in rows for key in row.extras})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS + extra_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())
    return len(rows)

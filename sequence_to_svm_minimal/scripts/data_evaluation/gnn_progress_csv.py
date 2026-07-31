"""Per-model GNN inference progress CSV (batch write + resume)."""

from __future__ import annotations

import csv
import re
from pathlib import Path

PROGRESS_COLUMNS = (
    "peptide_id",
    "pred",
    "prob_amp",
    "logit_amp",
    "logit_nonamp",
    "logit_margin",
)

_SAFE = re.compile(r"[^\w.\-]+")


def safe_progress_stem(architecture: str, model_name: str) -> str:
    raw = f"{architecture}_{model_name}".strip().lower()
    return _SAFE.sub("_", raw).strip("_") or "model"


def progress_csv_path(progress_dir: Path, architecture: str, model_name: str) -> Path:
    return Path(progress_dir) / f"{safe_progress_stem(architecture, model_name)}_predictions.csv"


def truncate_incomplete_last_line(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        return False
    cut = raw.rfind(b"\n")
    if cut < 0:
        path.write_bytes(b"")
    else:
        path.write_bytes(raw[: cut + 1])
    return True


def load_progress_rows(path: Path) -> dict[str, dict[str, float | int | str]]:
    """Map peptide_id -> prediction fields from a partial progress CSV."""
    out: dict[str, dict[str, float | int | str]] = {}
    if not path.is_file() or path.stat().st_size == 0:
        return out
    truncate_incomplete_last_line(path)
    if path.stat().st_size == 0:
        return out
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "peptide_id" not in reader.fieldnames:
            raise ValueError(f"{path}: expected header with peptide_id")
        for row in reader:
            if not row or not row.get("peptide_id"):
                continue
            if any(row.get(c) in (None, "") for c in PROGRESS_COLUMNS if c != "peptide_id"):
                break
            pid = str(row["peptide_id"]).strip()
            out[pid] = {
                "peptide_id": pid,
                "pred": int(float(row["pred"])),
                "prob_amp": float(row["prob_amp"]),
                "logit_amp": float(row["logit_amp"]),
                "logit_nonamp": float(row["logit_nonamp"]),
                "logit_margin": float(row["logit_margin"]),
            }
    return out


def append_progress_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PROGRESS_COLUMNS), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r[c] for c in PROGRESS_COLUMNS})
        fh.flush()

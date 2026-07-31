#!/usr/bin/env python3
"""
Rebuild pooled esm2_embeddings.csv from existing esm2_per_residue/*.pt files.

Use when ESM-2 finished writing per-residue tensors but died before the pooled CSV
(or before pipeline_manifest.json). Does not re-run the ESM-2 model.

Supports:
  - batched CSV writes with periodic flush
  - resume: skip seqIndexes already present in a partial --output CSV

Run from sequence_to_svm_minimal:
  python scripts/rebuild_esm2_csv_from_per_residue.py \\
    --per-residue-dir data/proteomes/gnn_predictions/generated/esm2_per_residue \\
    --output data/proteomes/gnn_predictions/generated/esm2_embeddings.csv

Then resume the pipeline with --skip-if-exists (ESM2 skips once the CSV exists).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_AUTO_INDEX = re.compile(r"^[1-9]\d*$")


def _peptide_id(seq_index: str, *, prefix: str = "SEQ") -> str:
    s = str(seq_index).strip()
    if s.startswith(f"{prefix}_"):
        return s
    if _AUTO_INDEX.fullmatch(s):
        return f"{prefix}_{s}"
    return s


def _safe_stem(seq_index: str) -> str:
    s = str(seq_index).strip().replace("\\", "/")
    return s.replace("/", "_").replace(":", "_")


def _load_pt(path: Path) -> tuple[str, object]:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "embedding" not in obj:
        raise ValueError(f"{path}: expected dict with 'embedding' key")
    sid = str(obj.get("seqIndex", path.stem)).strip()
    return sid, obj["embedding"]


def _mean_pool_row(embedding) -> list[float]:
    import torch

    t = embedding if isinstance(embedding, torch.Tensor) else torch.as_tensor(embedding)
    vec = t.detach().float().mean(0).reshape(-1)
    return [float(x) for x in vec.tolist()]


def _make_header(*, dim: int, add_peptide_id: bool) -> list[str]:
    header = ["seqIndex"]
    if add_peptide_id:
        header.append("peptide_id")
    header.extend(f"esm2_dim_{i}" for i in range(dim))
    return header


def _format_row(sid: str, vals: list[float], *, add_peptide_id: bool) -> list[str]:
    row = [sid]
    if add_peptide_id:
        row.append(_peptide_id(sid))
    row.extend(f"{x:.8g}" for x in vals)
    return row


def _truncate_incomplete_last_line(path: Path) -> bool:
    """If the file does not end with newline, drop the truncated last line. Returns True if truncated."""
    raw = path.read_bytes()
    if not raw:
        return False
    if raw.endswith(b"\n"):
        return False
    cut = raw.rfind(b"\n")
    if cut < 0:
        path.write_bytes(b"")
    else:
        path.write_bytes(raw[: cut + 1])
    return True


def _load_done_from_csv(
    output: Path,
    *,
    expected_ncols: int | None,
    add_peptide_id: bool,
) -> tuple[set[str], int | None]:
    """
    Return (done_seqIndexes, embedding_dim_or_None).
    Embedding dim is inferred from header when present.
    """
    done: set[str] = set()
    dim: int | None = None
    if not output.is_file() or output.stat().st_size == 0:
        return done, dim

    _truncate_incomplete_last_line(output)
    if output.stat().st_size == 0:
        return done, dim

    with output.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return done, dim

        if not header or header[0] != "seqIndex":
            raise ValueError(
                f"{output}: cannot resume — expected header starting with seqIndex, got {header[:3]!r}"
            )
        has_pid = len(header) > 1 and header[1] == "peptide_id"
        if has_pid != add_peptide_id:
            raise ValueError(
                f"{output}: peptide_id column mismatch vs --no-peptide-id; use --fresh to rebuild"
            )
        dim_cols = [c for c in header if c.startswith("esm2_dim_")]
        dim = len(dim_cols)
        if expected_ncols is not None and len(header) != expected_ncols:
            raise ValueError(
                f"{output}: header has {len(header)} cols, expected {expected_ncols}; use --fresh"
            )

        for row in reader:
            if not row:
                continue
            if len(row) < len(header):
                # incomplete row mid-file — stop marking done after last full row
                break
            done.add(str(row[0]).strip())

    return done, dim


def _infer_dim_from_pt(path: Path) -> tuple[str, int]:
    sid, emb = _load_pt(path)
    return sid, len(_mean_pool_row(emb))


def _log(msg: str) -> None:
    print(msg, flush=True)


def rebuild_pooled_csv(
    per_residue_dir: Path,
    output: Path,
    *,
    add_peptide_id: bool = True,
    limit: int | None = None,
    batch_size: int = 256,
    resume: bool = True,
    log_every: int = 1000,
) -> dict:
    _log(f"Listing *.pt under {per_residue_dir} (can take a while on /mnt/c)...")
    files = sorted(per_residue_dir.glob("*.pt"))
    if limit is not None:
        files = files[: max(0, limit)]
    _log(f"Found {len(files)} .pt files")
    if not files:
        raise FileNotFoundError(f"No .pt files under {per_residue_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    dim: int | None = None
    if resume and output.is_file() and output.stat().st_size > 0:
        _log(f"Reading partial CSV for resume: {output}")
        done, dim = _load_done_from_csv(output, expected_ncols=None, add_peptide_id=add_peptide_id)
        _log(f"Resume: {len(done)} rows already in {output}")
    elif output.is_file() and not resume:
        output.unlink()
        _log(f"--fresh: removed existing {output}")

    if dim is None:
        _log(f"Inferring embedding dim from {files[0].name}...")
        _, dim = _infer_dim_from_pt(files[0])

    header = _make_header(dim=dim, add_peptide_id=add_peptide_id)
    done_stems = {_safe_stem(s) for s in done}
    pending = [p for p in files if p.stem not in done_stems]
    n_skipped = len(files) - len(pending)
    _log(
        f"Pending {len(pending)} / {len(files)} "
        f"(skipped {n_skipped}, dim={dim}, batch_size={batch_size})"
    )

    if not pending:
        return {
            "n_pt_files": len(files),
            "n_written": 0,
            "n_skipped": n_skipped,
            "embedding_dim": dim,
            "output": str(output.resolve()),
            "bytes": output.stat().st_size if output.is_file() else 0,
            "resumed": bool(done),
        }

    append = output.is_file() and output.stat().st_size > 0 and bool(done)
    mode = "a" if append else "w"
    n_written = 0
    n_seen = 0
    batch: list[list[str]] = []
    use_tqdm = sys.stderr.isatty()

    def _flush(writer: csv.writer, fh) -> None:
        nonlocal n_written, batch
        if not batch:
            return
        writer.writerows(batch)
        fh.flush()
        n_written += len(batch)
        batch = []

    with output.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not append:
            writer.writerow(header)
            fh.flush()

        if use_tqdm:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    pending,
                    desc="rebuild pooled CSV",
                    unit="pt",
                    total=len(pending),
                    file=sys.stderr,
                )
            except ImportError:
                iterator = pending
                use_tqdm = False
        else:
            iterator = pending
            _log("Non-TTY (e.g. nohup): logging every "
                 f"{log_every} files instead of tqdm")

        for path in iterator:
            sid, emb = _load_pt(path)
            n_seen += 1
            if sid in done:
                continue
            vals = _mean_pool_row(emb)
            if len(vals) != dim:
                raise ValueError(
                    f"{path}: embedding dim {len(vals)} != expected {dim}"
                )
            batch.append(_format_row(sid, vals, add_peptide_id=add_peptide_id))
            done.add(sid)
            if len(batch) >= batch_size:
                _flush(writer, fh)
                if use_tqdm and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(written=n_written, refresh=False)
            if not use_tqdm and (n_seen % log_every == 0 or n_seen == len(pending)):
                _log(
                    f"progress {n_seen}/{len(pending)} "
                    f"({100.0 * n_seen / len(pending):.1f}%) "
                    f"written={n_written} csv_bytes={output.stat().st_size}"
                )

        _flush(writer, fh)

    _log(f"Done writing new rows: {n_written}")
    return {
        "n_pt_files": len(files),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "embedding_dim": dim,
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "resumed": bool(done) and n_skipped > 0,
        "batch_size": batch_size,
    }


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rebuild esm2_embeddings.csv from esm2_per_residue/*.pt (no model inference).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/rebuild_esm2_csv_from_per_residue.py \\\n"
            "    --per-residue-dir data/proteomes/gnn_predictions/generated/esm2_per_residue \\\n"
            "    --output data/proteomes/gnn_predictions/generated/esm2_embeddings.csv\n"
            "\n"
            "Safe to re-run: resumes by skipping seqIndexes already in --output.\n"
            "Use --fresh to overwrite. Tune --batch-size for flush frequency.\n"
            "\n"
            "Then:\n"
            "  python scripts/run_data_pipeline.py --mode blind \\\n"
            "    --input data/proteomes/gnn_predictions/final_candidates.txt \\\n"
            "    --work-dir data/proteomes/gnn_predictions/generated --skip-if-exists\n"
        ),
    )
    ap.add_argument(
        "--per-residue-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory of {seqIndex}.pt tensors written by esm_sequence_processor",
    )
    ap.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Destination pooled CSV (esm2_embeddings.csv)",
    )
    ap.add_argument(
        "--no-peptide-id",
        action="store_true",
        help="Do not add peptide_id column (pipeline post-step can add later)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of .pt files (smoke test)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rows to buffer before flush (default: 256)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite --output instead of resuming from a partial CSV",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="When stdout is not a TTY (nohup), print a progress line every N files (default: 1000)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pr = args.per_residue_dir.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not pr.is_dir():
        print(f"Not a directory: {pr}", file=sys.stderr, flush=True)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be >= 1", file=sys.stderr, flush=True)
        return 2
    if args.log_every < 1:
        print("--log-every must be >= 1", file=sys.stderr, flush=True)
        return 2
    _log(f"rebuild_esm2_csv_from_per_residue starting → {out}")
    try:
        stats = rebuild_pooled_csv(
            pr,
            out,
            add_peptide_id=not args.no_peptide_id,
            limit=args.limit,
            batch_size=args.batch_size,
            resume=not args.fresh,
            log_every=args.log_every,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        print(str(e), file=sys.stderr, flush=True)
        return 1

    _log(
        f"Wrote {stats['n_written']} new rows "
        f"(skipped {stats['n_skipped']}, dim={stats['embedding_dim']}) → {stats['output']} "
        f"({stats['bytes'] / (1024 ** 2):.1f} MiB)"
    )
    _log("Resume pipeline with --skip-if-exists once this CSV is in the work-dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

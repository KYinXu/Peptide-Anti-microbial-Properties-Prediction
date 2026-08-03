#!/usr/bin/env python3
"""
Extract a clean pipeline work-dir subset by matching sequences from a reference
(CSV with Sequence column, or id+sequence TXT) against an existing generated/
workspace — without re-running ESMFold / ESM2.

Copies matching:
  - structures (PDBs + filtered results_log + checkpoint.json)
  - inputs/canonical_seqs.txt
  - geometric_features.csv / qsar12_descriptors.csv / esm2_embeddings.csv (filtered)
  - esm2_per_residue/{id}.pt when present
  - pipeline_manifest.json (paths rewritten to the new work-dir)

Match key is **sequence** (standard AA, uppercased), not ID — so SEQ_* and
legacy accession IDs both work when the sequence is in the reference set.

Example:
  python scripts/extract_workspace_by_sequences.py \\
    --source-work-dir data/test/mass_spec/generated \\
    --reference data/test/mass_spec/dedup_MAPP.csv \\
    --output-work-dir data/test/mass_spec_dedup/generated \\
    --prefer-id-prefix SEQ
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_DG = Path(__file__).resolve().parent / "data_generation"
if str(_DG) not in sys.path:
    sys.path.insert(0, str(_DG))

from peptide_pipeline.aa_sanitize import canonical_standard_aa_sequence
from prepare_test_data import DEFAULT_SEQUENCE_ALIASES, resolve_column

PIPELINE_CSV_CANDIDATES = (
    "geometric_features.csv",
    "qsar12_descriptors.csv",
    "esm2_embeddings.csv",
    "compare_geo_qsar_merged.csv",
    "model_comparison_latest.csv",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _canon(seq: object) -> str | None:
    if seq is None or (isinstance(seq, float) and pd.isna(seq)):
        return None
    return canonical_standard_aa_sequence(str(seq))


def load_reference_sequences(path: Path, *, sequence_col: str | None) -> set[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    seqs: set[str] = set()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        col = sequence_col or resolve_column(
            df.columns, list(DEFAULT_SEQUENCE_ALIASES), required=True, kind="sequence"
        )
        assert col is not None
        for v in df[col]:
            c = _canon(v)
            if c:
                seqs.add(c)
    else:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                parts = raw.split(None, 1)
                seq = parts[1] if len(parts) == 2 else parts[0]
                c = _canon(seq)
                if c:
                    seqs.add(c)
    if not seqs:
        raise ValueError(f"No valid sequences in reference: {path}")
    return seqs


def load_id_sequence_map(path: Path | None) -> dict[str, str]:
    """Parse ``id sequence`` lines → {id: canon_seq}."""
    out: dict[str, str] = {}
    if path is None or not path.is_file():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(None, 1)
            if len(parts) != 2:
                continue
            pid, seq = parts[0].strip(), parts[1].strip()
            c = _canon(seq)
            if c:
                out[pid] = c
    return out


def _safe_stem(pid: str) -> str:
    return str(pid).strip().replace("\\", "/").replace("/", "_").replace(":", "_")


def collect_keep_ids(
    *,
    reference: set[str],
    results_log: Path | None,
    id_seq_maps: list[dict[str, str]],
    prefer_prefix: str | None,
) -> tuple[set[str], dict[str, str], dict[str, list[str]]]:
    """
    Returns:
      keep_ids,
      id_to_seq,
      seq_to_ids (all ids seen for each sequence)
    """
    id_to_seq: dict[str, str] = {}
    seq_to_ids: dict[str, list[str]] = defaultdict(list)

    def _add(pid: str, seq: str) -> None:
        if seq not in reference:
            return
        pid = str(pid).strip()
        if not pid:
            return
        if pid not in id_to_seq:
            id_to_seq[pid] = seq
            seq_to_ids[seq].append(pid)

    for m in id_seq_maps:
        for pid, seq in m.items():
            _add(pid, seq)

    if results_log is not None and results_log.is_file() and results_log.stat().st_size > 0:
        df = pd.read_csv(results_log)
        id_col = "unique_id" if "unique_id" in df.columns else None
        seq_col = "sequence" if "sequence" in df.columns else None
        if id_col and seq_col:
            for _, row in df.iterrows():
                c = _canon(row[seq_col])
                if c:
                    _add(str(row[id_col]), c)

    keep: set[str] = set()
    for seq, ids in seq_to_ids.items():
        if prefer_prefix:
            preferred = [i for i in ids if i.startswith(prefer_prefix)]
            if preferred:
                keep.update(preferred)
                continue
        keep.update(ids)

    return keep, id_to_seq, dict(seq_to_ids)


def _find_pdb(structures_dir: Path, pid: str, pdb_file_hint: str | None) -> Path | None:
    candidates: list[Path] = []
    if pdb_file_hint:
        hint = Path(str(pdb_file_hint))
        candidates.append(structures_dir / hint)
        candidates.append(structures_dir / hint.name)
    stem = _safe_stem(pid)
    candidates.extend(
        [
            structures_dir / "sequences" / f"{stem}.pdb",
            structures_dir / f"{stem}.pdb",
            structures_dir / "AMP" / f"{stem}.pdb",
            structures_dir / "DECOY" / f"{stem}.pdb",
        ]
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def filter_csv_by_ids_or_seq(
    src: Path,
    dst: Path,
    *,
    keep_ids: set[str],
    reference: set[str],
) -> int:
    if not src.is_file() or src.stat().st_size == 0:
        return 0
    df = pd.read_csv(src)
    id_col = next((c for c in ("peptide_id", "unique_id", "seqIndex", "name") if c in df.columns), None)
    seq_col = next((c for c in ("sequence", "Sequence") if c in df.columns), None)
    mask = pd.Series(False, index=df.index)
    if id_col:
        mask |= df[id_col].astype(str).isin(keep_ids)
    if seq_col:
        mask |= df[seq_col].map(lambda s: _canon(s) in reference if pd.notna(s) else False)
    out = df.loc[mask].copy()
    if out.empty:
        return 0
    # Deduplicate on id if possible (keep first)
    if id_col:
        out = out.drop_duplicates(subset=[id_col], keep="first")
    elif seq_col:
        out = out.drop_duplicates(subset=[seq_col], keep="first")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(out)


def write_results_log(
    src_log: Path | None,
    dst_log: Path,
    *,
    keep_ids: set[str],
    id_to_seq: dict[str, str],
    copied_pdbs: dict[str, str],
) -> int:
    rows: list[dict] = []
    seen: set[str] = set()
    if src_log is not None and src_log.is_file() and src_log.stat().st_size > 0:
        df = pd.read_csv(src_log)
        for _, row in df.iterrows():
            pid = str(row.get("unique_id", "")).strip()
            if pid not in keep_ids or pid in seen:
                continue
            if pid not in copied_pdbs:
                continue
            c = _canon(row.get("sequence"))
            if c is None:
                c = id_to_seq.get(pid)
            if c is None:
                continue
            rows.append(
                {
                    "unique_id": pid,
                    "original_idx": row.get("original_idx", pid),
                    "sequence": c,
                    "length": len(c),
                    "label": row.get("label", 0),
                    "status": "success",
                    "pdb_file": copied_pdbs[pid],
                    "time_seconds": row.get("time_seconds", ""),
                    "timestamp": row.get("timestamp", ""),
                }
            )
            seen.add(pid)
    for pid, rel in copied_pdbs.items():
        if pid in seen:
            continue
        c = id_to_seq.get(pid)
        if not c:
            continue
        rows.append(
            {
                "unique_id": pid,
                "original_idx": pid,
                "sequence": c,
                "length": len(c),
                "label": 0,
                "status": "success",
                "pdb_file": rel,
                "time_seconds": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        seen.add(pid)

    dst_log.parent.mkdir(parents=True, exist_ok=True)
    with dst_log.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "unique_id",
                "original_idx",
                "sequence",
                "length",
                "label",
                "status",
                "pdb_file",
                "time_seconds",
                "timestamp",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def rewrite_manifest(src: Path | None, dst: Path, new_work: Path, *, n_written: int) -> None:
    meta: dict = {}
    if src is not None and src.is_file():
        try:
            meta = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    new_work = new_work.resolve()
    meta["work_dir"] = str(new_work)
    meta["extracted_utc"] = datetime.now(timezone.utc).isoformat()
    meta["canonical_seqs"] = str(new_work / "inputs" / "canonical_seqs.txt")
    meta["structures_dir"] = str(new_work / "structures")
    for key, name in (
        ("geometric_features", "geometric_features.csv"),
        ("qsar12_descriptors", "qsar12_descriptors.csv"),
        ("esm2_embeddings", "esm2_embeddings.csv"),
        ("esm2_per_residue", "esm2_per_residue"),
    ):
        p = new_work / name
        if p.exists():
            meta[key] = str(p)
    norm = meta.get("normalization") if isinstance(meta.get("normalization"), dict) else {}
    norm = dict(norm)
    norm["n_written"] = n_written
    norm["note"] = "subset extracted by extract_workspace_by_sequences.py"
    meta["normalization"] = norm
    # Drop stale absolute step cmds; keep names only
    if "steps" in meta and isinstance(meta["steps"], list):
        meta["steps"] = [{"name": s.get("name"), "extracted": True} for s in meta["steps"] if isinstance(s, dict)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def extract_workspace(
    source: Path,
    output: Path,
    reference_seqs: set[str],
    *,
    prefer_id_prefix: str | None = "SEQ",
    dry_run: bool = False,
) -> dict:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output work-dirs must differ")

    structures_src = source / "structures"
    results_log = structures_src / "results_log.csv"
    canonical = source / "inputs" / "canonical_seqs.txt"

    id_maps = [
        load_id_sequence_map(canonical),
        load_id_sequence_map(source.parent / "candidates.txt"),
    ]
    # Also mine geometric_features for id↔seq
    geo = source / "geometric_features.csv"
    if geo.is_file() and geo.stat().st_size > 0:
        gdf = pd.read_csv(geo, usecols=lambda c: c in {"peptide_id", "sequence"})
        if "peptide_id" in gdf.columns and "sequence" in gdf.columns:
            m = {}
            for _, row in gdf.iterrows():
                c = _canon(row["sequence"])
                if c:
                    m[str(row["peptide_id"])] = c
            id_maps.append(m)

    keep_ids, id_to_seq, seq_to_ids = collect_keep_ids(
        reference=reference_seqs,
        results_log=results_log if results_log.is_file() else None,
        id_seq_maps=id_maps,
        prefer_prefix=prefer_id_prefix,
    )
    _log(
        f"Reference sequences: {len(reference_seqs)}\n"
        f"Matched keep_ids: {len(keep_ids)} "
        f"(sequences with ≥1 id: {sum(1 for s, ids in seq_to_ids.items() if ids)})"
    )
    if not keep_ids:
        raise ValueError("No IDs matched reference sequences in the source workspace")

    # PDB hints from results_log
    pdb_hints: dict[str, str] = {}
    if results_log.is_file() and results_log.stat().st_size > 0:
        rdf = pd.read_csv(results_log)
        if "unique_id" in rdf.columns and "pdb_file" in rdf.columns:
            for _, row in rdf.iterrows():
                pid = str(row["unique_id"]).strip()
                if pid in keep_ids and pd.notna(row["pdb_file"]):
                    pdb_hints[pid] = str(row["pdb_file"])

    copied_pdbs: dict[str, str] = {}
    missing_pdb: list[str] = []
    for pid in sorted(keep_ids):
        src_pdb = _find_pdb(structures_src, pid, pdb_hints.get(pid))
        if src_pdb is None:
            missing_pdb.append(pid)
            continue
        rel = f"sequences/{_safe_stem(pid)}.pdb"
        if not dry_run:
            dst_pdb = output / "structures" / rel
            dst_pdb.parent.mkdir(parents=True, exist_ok=True)
            if not dst_pdb.exists():
                shutil.copy2(src_pdb, dst_pdb)
        copied_pdbs[pid] = rel

    _log(f"PDBs copied: {len(copied_pdbs)}  missing: {len(missing_pdb)}")

    # Restrict keep_ids to those with structures when possible
    keep_with_pdb = set(copied_pdbs) if copied_pdbs else keep_ids

    stats = {
        "reference_sequences": len(reference_seqs),
        "keep_ids": len(keep_ids),
        "pdbs_copied": len(copied_pdbs),
        "pdbs_missing": len(missing_pdb),
        "csv_rows": {},
        "esm2_pt_copied": 0,
        "dry_run": dry_run,
        "output_work_dir": str(output),
    }

    if dry_run:
        return stats

    # canonical_seqs for copied ids (stable SEQ order when possible)
    def _sort_key(pid: str) -> tuple:
        if pid.startswith("SEQ_"):
            try:
                return (0, int(pid.split("_", 1)[1]))
            except ValueError:
                return (0, pid)
        return (1, pid)

    out_canon = output / "inputs" / "canonical_seqs.txt"
    out_canon.parent.mkdir(parents=True, exist_ok=True)
    n_canon = 0
    with out_canon.open("w", encoding="utf-8") as fh:
        for pid in sorted(keep_with_pdb, key=_sort_key):
            seq = id_to_seq.get(pid)
            if not seq:
                continue
            fh.write(f"{pid} {seq}\n")
            n_canon += 1
    stats["canonical_seqs"] = n_canon

    n_log = write_results_log(
        results_log if results_log.is_file() else None,
        output / "structures" / "results_log.csv",
        keep_ids=keep_with_pdb,
        id_to_seq=id_to_seq,
        copied_pdbs=copied_pdbs,
    )
    stats["results_log_rows"] = n_log

    ckpt = {
        "completed_ids": sorted(keep_with_pdb),
        "failed_ids": [],
        "sequences_completed": len(keep_with_pdb),
        "extracted_from": str(source),
        "note": "Written by extract_workspace_by_sequences.py",
    }
    (output / "structures" / "checkpoint.json").write_text(
        json.dumps(ckpt, indent=2) + "\n", encoding="utf-8"
    )

    for name in PIPELINE_CSV_CANDIDATES:
        src = source / name
        if not src.is_file() or src.stat().st_size == 0:
            continue
        n = filter_csv_by_ids_or_seq(
            src, output / name, keep_ids=keep_with_pdb, reference=reference_seqs
        )
        stats["csv_rows"][name] = n
        _log(f"  {name}: {n} rows")

    # esm2 per-residue
    esm_src = source / "esm2_per_residue"
    esm_dst = output / "esm2_per_residue"
    n_pt = 0
    if esm_src.is_dir():
        esm_dst.mkdir(parents=True, exist_ok=True)
        for pid in keep_with_pdb:
            stem = _safe_stem(pid)
            src_pt = esm_src / f"{stem}.pt"
            if src_pt.is_file():
                shutil.copy2(src_pt, esm_dst / src_pt.name)
                n_pt += 1
    stats["esm2_pt_copied"] = n_pt
    _log(f"  esm2_per_residue .pt copied: {n_pt}")

    rewrite_manifest(
        source / "pipeline_manifest.json",
        output / "pipeline_manifest.json",
        output,
        n_written=n_canon,
    )

    report = output / "extract_report.json"
    report.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote report: {report}")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Extract pipeline work-dir subset matching reference sequences (no re-fold).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--source-work-dir",
        type=Path,
        required=True,
        help="Existing generated/ workspace (e.g. data/test/mass_spec/generated)",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="dedup_MAPP.csv (Sequence col) or id+sequence TXT (e.g. candidates.txt)",
    )
    ap.add_argument(
        "--output-work-dir",
        type=Path,
        required=True,
        help="New generated/ folder to create",
    )
    ap.add_argument(
        "--sequence-col",
        type=str,
        default=None,
        help="Sequence column name when --reference is CSV (default: auto)",
    )
    ap.add_argument(
        "--prefer-id-prefix",
        type=str,
        default="SEQ",
        help="When multiple IDs share a sequence, prefer this prefix (default: SEQ). Empty = keep all.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print match counts; do not copy files",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    src = args.source_work_dir.expanduser().resolve()
    out = args.output_work_dir.expanduser().resolve()
    if not src.is_dir():
        print(f"Source work-dir not found: {src}", file=sys.stderr)
        return 1
    try:
        ref = load_reference_sequences(args.reference, sequence_col=args.sequence_col)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1
    prefer = args.prefer_id_prefix.strip() or None
    try:
        stats = extract_workspace(
            src,
            out,
            ref,
            prefer_id_prefix=prefer,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    _log(
        f"\nDone. matched_ids={stats['keep_ids']} pdbs={stats['pdbs_copied']} "
        f"missing_pdb={stats['pdbs_missing']} → {stats['output_work_dir']}"
    )
    if not args.dry_run:
        _log(
            "You can continue with --skip-if-exists on the new work-dir for any missing steps "
            "(e.g. ESM2 if .pt/csv were incomplete)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

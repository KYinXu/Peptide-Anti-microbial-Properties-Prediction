#!/usr/bin/env python3
"""
GNN (+ optional SVM) inference on sliding windows from a blind ``run_data_pipeline`` workspace.

Reads ``inputs/window_map.csv``, crops each window from the parent ESMFold PDB, extracts Geo-20
on the fragment (same columns as ``compare_model_predictions.py``), slices per-residue ESM2 from
the parent ``.pt`` files, and runs the same pretrained checkpoints.

QSAR-12: default ``--qsar-mode parent`` copies descriptors from ``qsar12_descriptors.csv`` by
``parent_id``. Use ``--qsar-mode per-window`` to recompute from each window sequence (slow).

Example (from ``sequence_to_svm_minimal``)::

  python scripts/data_evaluation/compare_model_predictions_windowed.py path/to/generated \\
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
      --gnn-checkpoints-dir checkpoints/latest

  python scripts/data_evaluation/compare_model_predictions_windowed.py path/to/generated \\
      --max-windows 200 --gnn-checkpoints-dir checkpoints/latest
<<<<<<< HEAD
=======
      --checkpoints-base checkpoints/latest

Model defaults inherit from ``configs/compare_models.json``; optional overrides in
``configs/compare_models_windowed.json``.
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

<<<<<<< HEAD
<<<<<<< HEAD
from configs.load_config import argv_without_config_flags, load_compare_models_config
=======
from configs.load_config import argv_without_config_flags, load_compare_models_windowed_config
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
from configs.load_config import argv_without_config_flags, load_compare_models_config
>>>>>>> f255595470cd527a24de3b686587977fb372fb16


def _load_cmpred():
    p = _ROOT / "scripts" / "data_evaluation" / "compare_model_predictions.py"
    spec = importlib.util.spec_from_file_location("cmpred", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_window_pdb_subset(parent_pdb: Path, out_pdb: Path, start: int, length: int) -> None:
    from Bio.PDB import PDBIO, PDBParser, Select

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("w", str(parent_pdb))
    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        raise ValueError(f"No chain in PDB: {parent_pdb}")
    chain = chains[0]
    ordered = [r for r in chain.get_residues() if r.id[0] == " "]
    end = start + length
    if start < 0 or end > len(ordered):
        raise ValueError(
            f"Window [{start}:{end}) out of range for {parent_pdb} ({len(ordered)} residues in first chain)"
        )
    subset = ordered[start:end]
    allowed = {(r.parent.id, r.id) for r in subset}

    class _Sub(Select):
        def accept_residue(self, residue):
            return (residue.parent.id, residue.id) in allowed

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_pdb), _Sub())


def _geo_row_for_window(
    parent_pdb: Path,
    start: int,
    length: int,
    peptide_id: str,
    window_seq: str,
    tmp_path: Path,
    geo_cols: list[str],
) -> dict[str, float]:
    from features.geometric_features import extract_all_features

    _write_window_pdb_subset(parent_pdb, tmp_path, start, length)
    feat = extract_all_features(str(tmp_path), peptide_id=peptide_id, sequence=window_seq)
    out: dict[str, float] = {}
    for c in geo_cols:
        v = feat.get(c, 0.0)
        try:
            out[c] = float(v)
        except (TypeError, ValueError):
            out[c] = 0.0
    return out


def _load_generate_qsar():
    p = _ROOT / "scripts" / "data_generation" / "generate_qsar_features.py"
    spec = importlib.util.spec_from_file_location("genqsar", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _build_window_master(
    *,
    resolve_peptide_pdb_path,
    read_canonical_sequences,
    ws: Path,
    wm: pd.DataFrame,
    geo_parent: pd.DataFrame,
    qsar_parent: pd.DataFrame,
    pdb_dir: Path,
    geo_cols: list[str],
    qsar_cols: list[str],
    qsar_mode: str,
    max_windows: int | None,
    tmp_dir: Path,
) -> pd.DataFrame:
    if "parent_id" not in wm.columns or "start" not in wm.columns or "length" not in wm.columns:
        raise SystemExit("window_map.csv must include parent_id, start, length")

    pdb_by_parent = {}
    if "peptide_id" in geo_parent.columns and "pdb_file" in geo_parent.columns:
        for _, r in geo_parent.iterrows():
            pdb_by_parent[str(r["peptide_id"]).strip()] = r["pdb_file"]

    canon_path = ws / "inputs" / "canonical_seqs.txt"
    parent_seqs = read_canonical_sequences(canon_path) if canon_path.is_file() else {}

    wm2 = wm.copy()
    if max_windows is not None and max_windows > 0:
        wm2 = wm2.iloc[: int(max_windows)].copy()

    if qsar_mode == "per-window":
        genq = _load_generate_qsar()
        compute_qsar12 = genq.compute_qsar12

        pids = [str(x) for x in wm2["peptide_id"].astype(str)]
        seqs = [str(x).strip() for x in wm2["sequence"]]
        qdf = compute_qsar12(seqs, pids)
        qsar_by_pid = qdf.set_index("peptide_id")
    else:
        if qsar_parent.empty:
            raise SystemExit("qsar_mode=parent requires a non-empty qsar12_descriptors.csv")
        qsar_by_pid = qsar_parent.set_index(
            "peptide_id" if "peptide_id" in qsar_parent.columns else qsar_parent.columns[0]
        )

    rows: list[dict] = []
    tmp_pdb = tmp_dir / "window_fragment.pdb"

    for n_done, (_, wr) in enumerate(wm2.iterrows(), start=1):
        pid = str(wr["peptide_id"]).strip()
        parent_id = str(wr["parent_id"]).strip()
        start = int(wr["start"])
        length = int(wr["length"])
        wseq = str(wr["sequence"]).strip().upper()

        pdb_rel = pdb_by_parent.get(parent_id)
        pdb_path = resolve_peptide_pdb_path(pdb_dir, pdb_rel, parent_id)
        if pdb_path is None:
            raise SystemExit(f"No PDB for parent_id={parent_id!r} (pdb_file={pdb_rel!r})")

        if parent_seqs:
            pseq = parent_seqs.get(parent_id, "")
            if pseq and wseq != pseq[start : start + length]:
                raise SystemExit(
                    f"Window sequence mismatch for {pid}: map vs canonical parent slice "
                    f"(parent={parent_id!r}, start={start}, len={length})"
                )

        gvec = _geo_row_for_window(pdb_path, start, length, pid, wseq, tmp_pdb, geo_cols)

        if qsar_mode == "per-window":
            qrow = qsar_by_pid.loc[pid]
        else:
            if parent_id not in qsar_by_pid.index:
                raise SystemExit(f"parent_id {parent_id!r} missing from QSAR CSV index")
            qrow = qsar_by_pid.loc[parent_id]

        row: dict = {
            "peptide_id": pid,
            "parent_id": parent_id,
            "window_start": start,
            "window_length": length,
            "sequence": wseq,
            "label": 0,
            "pdb_file": pdb_rel
            if pdb_rel is not None and not (isinstance(pdb_rel, float) and pd.isna(pdb_rel))
            else f"{parent_id}.pdb",
        }
        if "seqIndex" in wm2.columns:
            row["seqIndex"] = int(wr["seqIndex"])
        if "window_id" in wm2.columns:
            row["window_id"] = str(wr["window_id"])
        row.update(gvec)
        for c in qsar_cols:
            try:
                row[c] = float(qrow[c])
            except Exception:
                row[c] = 0.0
        rows.append(row)

        if n_done % 200 == 0:
            print(f"  Built {n_done}/{len(wm2)} window feature rows…", flush=True)

    return pd.DataFrame(rows)


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
def main() -> int:
    cfg_path, argv_rest = argv_without_config_flags(sys.argv[1:])
    cfg = load_compare_models_config(cfg_path)

    ap = argparse.ArgumentParser(
        description="GNN/SVM comparison on pipeline sliding windows (window_map.csv).",
        epilog="Run from sequence_to_svm_minimal. Requires inputs/window_map.csv in GENERATED.",
<<<<<<< HEAD
=======
def _lookup_qsar_row(qsar_by_pid: pd.DataFrame, peptide_id: str, seq_index: int | None) -> pd.Series:
    keys = [str(peptide_id).strip()]
    if seq_index is not None:
        si = int(seq_index)
        keys.extend([f"SEQ_{si}", f"seq_{si}", str(si)])
    for key in keys:
        if key in qsar_by_pid.index:
            return qsar_by_pid.loc[key]
    raise KeyError(peptide_id)


def _master_from_features_only(
    wm: pd.DataFrame,
    qsar: pd.DataFrame,
    *,
    max_windows: int | None,
) -> pd.DataFrame:
    """Merge window_map with per-window qsar12 (no PDB / ESM2)."""
    if qsar.empty:
        raise SystemExit("features-only workspace requires qsar12_descriptors.csv")

    wm2 = wm.copy()
    if max_windows is not None and max_windows > 0:
        wm2 = wm2.iloc[: int(max_windows)].copy()

    id_col = "peptide_id" if "peptide_id" in qsar.columns else qsar.columns[0]
    qsar_idx = qsar.copy()
    qsar_idx["_pid"] = qsar_idx[id_col].astype(str).str.strip()
    qsar_by_pid = qsar_idx.set_index("_pid")

    rows: list[dict] = []
    for _, wr in wm2.iterrows():
        pid = str(wr["peptide_id"]).strip()
        row: dict = {
            "peptide_id": pid,
            "parent_id": str(wr["parent_id"]).strip(),
            "window_start": int(wr["start"]),
            "window_length": int(wr["length"]),
            "sequence": str(wr["sequence"]).strip(),
        }
        if "seqIndex" in wm2.columns:
            row["seqIndex"] = int(wr["seqIndex"])
        if "window_id" in wm2.columns:
            row["window_id"] = str(wr["window_id"])
        seq_index = int(wr["seqIndex"]) if "seqIndex" in wm2.columns else None
        try:
            qrow = _lookup_qsar_row(qsar_by_pid, pid, seq_index)
        except KeyError:
            raise SystemExit(
                f"peptide_id {pid!r} (seqIndex={seq_index}) missing from qsar12_descriptors.csv"
            ) from None
        for c in qsar_by_pid.columns:
            if c in ("_pid", "sequence"):
                continue
            row[c] = qrow[c]
        rows.append(row)
    return pd.DataFrame(rows)


def _svm_row_index(
    results: dict,
    model_name: str,
    pid: str,
    seq_index: int | None,
    svm_id_to_row: dict[str, int] | None,
) -> int | None:
    if model_name == "SVM" and svm_id_to_row is not None:
        j = svm_id_to_row.get(str(pid).strip())
        if j is not None:
            return j
        if seq_index is not None:
            j = svm_id_to_row.get(str(int(seq_index)))
            if j is not None:
                return j
    r = results[model_name]
    if pid in r["ids"]:
        return r["ids"].index(pid)
    return None


def main() -> int:
    cfg_path, argv_rest = argv_without_config_flags(sys.argv[1:])
    cfg = load_compare_models_windowed_config(cfg_path)

    ap = argparse.ArgumentParser(
        description="GNN/SVM comparison on pipeline sliding windows (window_map.csv).",
        epilog=(
            "Run from sequence_to_svm_minimal. Requires inputs/window_map.csv in GENERATED. "
            "JSON: inherits configs/compare_models.json; optional configs/compare_models_windowed.json "
            "and --config PATH (later overrides earlier)."
        ),
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    )
    ap.add_argument(
        "generated",
        type=str,
        help="Pipeline generated/ directory (or parent containing generated/)",
    )
    ap.add_argument(
        "--gnn-checkpoints-dir",
        type=str,
        default=None,
        help="Folder with {arch}_ready_*.pt (same as compare_model_predictions.py)",
    )
    ap.add_argument(
        "--checkpoints-base",
        type=str,
        default=None,
        dest="checkpoints_base",
        help="Optional base dir for GNN + SVM artifacts (same semantics as compare script)",
    )
    ap.add_argument("--architecture", type=str, default=cfg["architecture"], choices=["gcn", "gat", "egnn"])
    ap.add_argument("--gnn_hidden", type=int, default=cfg["gnn_hidden"])
    ap.add_argument("--gnn_layers", type=int, default=cfg["gnn_layers"])
    ap.add_argument("--gnn_pooling", type=str, default=cfg["gnn_pooling"])
    ap.add_argument("--batch_size", type=int, default=min(32, cfg.get("batch_size", 32)))
<<<<<<< HEAD
<<<<<<< HEAD
=======
    ap.add_argument(
        "--compare-models",
        type=str,
        default=cfg.get("compare_models", "all"),
        choices=["all", "svm", "gnn"],
        help="Which model families to run (default: all, or compare_models_windowed.json).",
    )
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    ap.add_argument("--max-windows", type=int, default=None, help="Process only the first N windows (debug)")
    ap.add_argument(
        "--qsar-mode",
        type=str,
        choices=["parent", "per-window"],
<<<<<<< HEAD
<<<<<<< HEAD
        default="parent",
=======
        default=str(cfg.get("qsar_mode", "parent")),
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
        default="parent",
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        help="parent: copy QSAR-12 from qsar12_descriptors by parent_id; per-window: recompute (slow)",
    )
    ap.add_argument("--no-gnn-platt", action="store_true")
    ap.add_argument(
        "--node-groups",
        type=str,
        default=None,
        dest="node_groups",
        help="Same as compare_model_predictions.py (no_esm2, …)",
    )
    ap.add_argument("--output-csv", type=str, default=None, help="Primary output path (default under GENERATED)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--svm_descriptor_csv",
        type=str,
        default=None,
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        help="Optional descriptors.csv (e.g. svm_out/descriptors.csv) for QSAR-SVM block",
    )
    ap.add_argument("--svm_z_file", type=str, default=None)
    ap.add_argument("--svm_pkl", type=str, default=None)
<<<<<<< HEAD
=======
        help="QSAR descriptor CSV for SVM (default: generated/window_qsar12_descriptors.csv)",
    )
    ap.add_argument("--svm_z_file", type=str, default=cfg.get("svm_z_file"))
    ap.add_argument("--svm_pkl", type=str, default=cfg.get("svm_pkl"))
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    args = ap.parse_args(argv_rest)

    from gnn.data_utils import (
        NodeFeatureGroups,
        node_feature_groups_from_cli,
        node_feature_groups_from_config_value,
        node_input_dim,
        read_canonical_sequences,
        resolve_peptide_pdb_path,
    )
<<<<<<< HEAD
<<<<<<< HEAD
    from peptide_pipeline.manifest_paths import load_pipeline_manifest, resolve_generated_workspace
=======
    from peptide_pipeline.manifest_paths import (
        load_pipeline_manifest,
        normalize_manifest_path,
        resolve_generated_workspace,
    )
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
    from peptide_pipeline.manifest_paths import load_pipeline_manifest, resolve_generated_workspace
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    cmp = _load_cmpred()

    node_feature_groups = (
        node_feature_groups_from_cli(args.node_groups)
        if args.node_groups
        else node_feature_groups_from_config_value(cfg.get("node_feature_groups"))
    )
    _nfg = node_feature_groups if node_feature_groups is not None else NodeFeatureGroups()
    print(
        f"GNN node groups: onehot={_nfg.onehot} pdb={_nfg.pdb_continuous} vae={_nfg.vae_table} "
        f"esm2_residue={_nfg.esm2_residue} (base width {node_input_dim(node_feature_groups)})",
        flush=True,
    )

    ws = resolve_generated_workspace(Path(args.generated))
    m = load_pipeline_manifest(ws)
    wmap_path = ws / "inputs" / "window_map.csv"
    if not wmap_path.is_file():
        raise SystemExit(f"Missing {wmap_path} — run the pipeline with --window-min-len and --window-max-len.")

    wm = pd.read_csv(wmap_path)
    if wm.empty:
        raise SystemExit("window_map.csv is empty.")

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    geo_path = Path(m["geometric_features"])
    qsar_path = Path(m["qsar12_descriptors"])
    pdb_dir = Path(m["structures_dir"])
    esm_dir = m.get("esm2_per_residue")
    if not esm_dir:
        emb = Path(m["esm2_embeddings"])
        esm_dir = str((emb.parent / "esm2_per_residue").resolve())
    esm_dir = Path(esm_dir)
<<<<<<< HEAD
=======
    geo_key = m.get("geometric_features") or m.get("inference_samples")
    if not geo_key:
        raise SystemExit("Manifest missing geometric_features — run run_data_pipeline.py first.")
    qsar_key = m.get("qsar12_descriptors")
    if not qsar_key:
        raise SystemExit("Manifest missing qsar12_descriptors — run run_data_pipeline.py first.")

    geo_path = normalize_manifest_path(geo_key).resolve()
    qsar_path = normalize_manifest_path(qsar_key).resolve()
    structures_key = m.get("structures_dir")
    pdb_dir = (
        normalize_manifest_path(structures_key).resolve()
        if structures_key
        else None
    )
    has_structures = pdb_dir is not None and pdb_dir.is_dir()
    features_only_ws = bool(m.get("features_only")) or not has_structures

    esm_dir: Path | None = None
    esm_key = m.get("esm2_per_residue")
    if esm_key:
        esm_dir = normalize_manifest_path(esm_key).resolve()
    elif m.get("esm2_embeddings"):
        emb = normalize_manifest_path(m["esm2_embeddings"]).resolve()
        esm_dir = (emb.parent / "esm2_per_residue").resolve()

    compare_models = str(args.compare_models)
    if features_only_ws and compare_models in ("all", "gnn"):
        print(
            "Workspace has no structures (features-only pipeline). Running SVM only; "
            "use the full pipeline (ESMFold + ESM2) for windowed GNN.",
            flush=True,
        )
        compare_models = "svm"
    elif not has_structures and compare_models in ("all", "gnn"):
        raise SystemExit(
            "Manifest has no structures_dir. Run the full data pipeline with ESMFold, "
            "or pass --compare-models svm for QSAR-SVM on precomputed window descriptors."
        )

    run_svm = compare_models in ("all", "svm")
    run_gnn = compare_models in ("all", "gnn") and has_structures
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    geo_parent = pd.read_csv(geo_path)
    qsar_parent = pd.read_csv(qsar_path) if qsar_path.is_file() else pd.DataFrame()

    ns = argparse.Namespace(
        generated=str(ws),
        geo_csv=str(geo_path),
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        pdb_dir=str(pdb_dir),
        qsar_csv=str(qsar_path) if qsar_path.is_file() else "",
        esm2_csv=str(Path(m["esm2_embeddings"])) if m.get("esm2_embeddings") else None,
        gnn_esm2_residue_dir=str(esm_dir),
        geometric_qsar_combined_csv=str(ws / "window_compare_geo_qsar_merged.csv"),
        svm_descriptor_csv=args.svm_descriptor_csv or "",
<<<<<<< HEAD
=======
        pdb_dir=str(pdb_dir) if pdb_dir is not None else "",
        qsar_csv=str(qsar_path) if qsar_path.is_file() else "",
        esm2_csv=str(normalize_manifest_path(m["esm2_embeddings"]).resolve())
        if m.get("esm2_embeddings")
        else None,
        gnn_esm2_residue_dir=str(esm_dir) if esm_dir is not None else None,
        geometric_qsar_combined_csv=str(ws / "window_compare_geo_qsar_merged.csv"),
        svm_descriptor_csv="",
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        svm_z_file=args.svm_z_file or "",
        svm_pkl=args.svm_pkl or "",
        architecture=args.architecture,
        esm_only_pt=cfg["esm_only_pt"],
        esm_geo_pt=cfg["esm_geo_pt"],
        esm_qsar_pt=cfg["esm_qsar_pt"],
        esm_combined_pt=cfg["esm_combined_pt"],
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_pooling=args.gnn_pooling,
        batch_size=args.batch_size,
        no_gnn_platt=args.no_gnn_platt,
        legacy_pooled_esm_tabular=False,
    )

    if args.checkpoints_base:
        ns.checkpoints_base = str(Path(args.checkpoints_base).expanduser().resolve())
        cmp.apply_checkpoints_base(ns)
    if args.gnn_checkpoints_dir:
        gdir = Path(args.gnn_checkpoints_dir).expanduser().resolve()
        if not gdir.is_dir():
            raise SystemExit(f"--gnn-checkpoints-dir is not a directory: {gdir}")
        cmp._set_gnn_paths_from_dir(ns, gdir)
        print(f"GNN checkpoints: {gdir}", flush=True)
    cmp._try_apply_ready_models_summary(ns.architecture, ns, ws)

    _geo_present = [c for c in cmp.GNN_GEO_COLS if c in geo_parent.columns]
    geo_cols = _geo_present if _geo_present else list(cmp.GNN_GEO_COLS)
    qsar_cols = list(cmp.GNN_QSAR_COLS)

    print(f"Workspace: {ws}", flush=True)
    print(f"Windows: {len(wm)} rows in window_map.csv", flush=True)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    with tempfile.TemporaryDirectory(prefix="win_geo_") as td:
        tmp_dir = Path(td)
        print("Building per-window tabular features (cropped PDB → Geo-20)…", flush=True)
        master = _build_window_master(
            resolve_peptide_pdb_path=resolve_peptide_pdb_path,
            read_canonical_sequences=read_canonical_sequences,
            ws=ws,
            wm=wm,
            geo_parent=geo_parent,
            qsar_parent=qsar_parent,
            pdb_dir=pdb_dir,
            geo_cols=geo_cols,
            qsar_cols=qsar_cols,
            qsar_mode=args.qsar_mode,
            max_windows=args.max_windows,
            tmp_dir=tmp_dir,
        )
<<<<<<< HEAD
=======

    if features_only_ws or not run_gnn:
        print("Using per-window QSAR from workspace (no PDB cropping).", flush=True)
        master = _master_from_features_only(wm, qsar_parent, max_windows=args.max_windows)
    else:
        with tempfile.TemporaryDirectory(prefix="win_geo_") as td:
            tmp_dir = Path(td)
            print("Building per-window tabular features (cropped PDB → Geo-20)…", flush=True)
            master = _build_window_master(
                resolve_peptide_pdb_path=resolve_peptide_pdb_path,
                read_canonical_sequences=read_canonical_sequences,
                ws=ws,
                wm=wm,
                geo_parent=geo_parent,
                qsar_parent=qsar_parent,
                pdb_dir=pdb_dir,
                geo_cols=geo_cols,
                qsar_cols=qsar_cols,
                qsar_mode=args.qsar_mode,
                max_windows=args.max_windows,
                tmp_dir=tmp_dir,
            )
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    master_path = ws / "window_gnn_feature_master.csv"
    master.to_csv(master_path, index=False)
    print(f"Wrote {master_path} ({len(master)} rows)", flush=True)

<<<<<<< HEAD
<<<<<<< HEAD
=======
    window_qsar_path = ws / "window_qsar12_descriptors.csv"
    qsar_out_cols = ["peptide_id", "sequence"] + [c for c in cmp.GNN_QSAR_COLS if c in master.columns]
    master[qsar_out_cols].to_csv(window_qsar_path, index=False)
    _svm_desc_cli = any(
        a == "--svm_descriptor_csv" or a.startswith("--svm_descriptor_csv=")
        for a in argv_rest
    )
    if _svm_desc_cli and args.svm_descriptor_csv:
        ns.svm_descriptor_csv = str(Path(args.svm_descriptor_csv).expanduser().resolve())
    else:
        ns.svm_descriptor_csv = str(window_qsar_path.resolve())

>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    gnn_master_path = Path(ns.geometric_qsar_combined_csv)
    master.to_csv(gnn_master_path, index=False)

    geo_present = [c for c in cmp.GNN_GEO_COLS if c in master.columns]
    qsar_present = [c for c in cmp.GNN_QSAR_COLS if c in master.columns]
    mode_cols = {
        "esm": [],
        "geo20": geo_present,
        "qsar12": qsar_present,
        "combined32": geo_present + qsar_present,
    }

    canonical_ids = master["peptide_id"].astype(str).tolist()
    seq_by_id = dict(zip(master["peptide_id"].astype(str), master["sequence"]))

    results: dict = {}
    feature_models = [
        ("ESM-only", ns.esm_only_pt, "esm"),
        ("ESM+Geo20", ns.esm_geo_pt, "geo20"),
        ("ESM+QSAR12", ns.esm_qsar_pt, "qsar12"),
        ("ESM+Combined32", ns.esm_combined_pt, "combined32"),
    ]

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    for name, path, feat_mode in feature_models:
        if not path or not Path(path).exists():
            continue
        geom_cols = mode_cols.get(feat_mode, [])
        if feat_mode != "esm" and not geom_cols:
            if feat_mode in ("qsar12", "combined32"):
                print(f"Skipping {name}: missing tabular columns for mode {feat_mode}", flush=True)
            continue
        print(f"Running {ns.architecture.upper()} ({name}) on {len(master)} windows…", flush=True)
        ids, preds, prob_amp, logit_amp, logit_nonamp, logit_margin = cmp._run_gnn_predictions(
            str(gnn_master_path),
            str(pdb_dir),
            path,
            ns.architecture,
            ns.gnn_hidden,
            ns.gnn_layers,
            ns.gnn_pooling,
            ns.batch_size,
            geometric_feature_cols=geom_cols,
            esm2_residue_dir=str(esm_dir),
            canonical_seqs_path=str(ws / "inputs" / "canonical_seqs.txt"),
            node_feature_groups=node_feature_groups,
            use_gnn_platt=not ns.no_gnn_platt,
        )
        results[name] = {
            "ids": ids,
            "pred": preds,
            "prob_amp": prob_amp,
            "confidence": cmp._confidence(prob_amp),
            "logit_amp": logit_amp,
            "logit_nonamp": logit_nonamp,
            "logit_margin": logit_margin,
        }

    svm_seqindex_to_row: dict[str, int] | None = None
    if ns.svm_pkl and ns.svm_descriptor_csv and ns.svm_z_file:
<<<<<<< HEAD
=======
    if run_gnn:
        for name, path, feat_mode in feature_models:
            if not path or not Path(path).exists():
                continue
            geom_cols = mode_cols.get(feat_mode, [])
            if feat_mode != "esm" and not geom_cols:
                if feat_mode in ("qsar12", "combined32"):
                    print(f"Skipping {name}: missing tabular columns for mode {feat_mode}", flush=True)
                continue
            print(f"Running {ns.architecture.upper()} ({name}) on {len(master)} windows…", flush=True)
            ids, preds, prob_amp, logit_amp, logit_nonamp, logit_margin = cmp._run_gnn_predictions(
                str(gnn_master_path),
                str(pdb_dir),
                path,
                ns.architecture,
                ns.gnn_hidden,
                ns.gnn_layers,
                ns.gnn_pooling,
                ns.batch_size,
                geometric_feature_cols=geom_cols,
                esm2_residue_dir=str(esm_dir),
                canonical_seqs_path=str(ws / "inputs" / "canonical_seqs.txt"),
                node_feature_groups=node_feature_groups,
                use_gnn_platt=not ns.no_gnn_platt,
            )
            results[name] = {
                "ids": ids,
                "pred": preds,
                "prob_amp": prob_amp,
                "confidence": cmp._confidence(prob_amp),
                "logit_amp": logit_amp,
                "logit_nonamp": logit_nonamp,
                "logit_margin": logit_margin,
            }

    svm_id_to_row: dict[str, int] | None = None
    if run_svm and ns.svm_pkl and ns.svm_descriptor_csv and ns.svm_z_file:
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        print("Running SVM…", flush=True)
        ids, preds, prob_amp, distance = cmp._load_svm_predictions(
            ns.svm_descriptor_csv, ns.svm_z_file, ns.svm_pkl
        )
        results["SVM"] = {
            "ids": ids,
            "pred": preds,
            "prob_amp": prob_amp,
            "confidence": cmp._confidence(prob_amp),
            "distance": distance,
        }
<<<<<<< HEAD
<<<<<<< HEAD
        svm_seqindex_to_row = {}
        for j, sid in enumerate(ids):
            svm_seqindex_to_row[str(sid).strip()] = j
=======
        svm_id_to_row = {str(sid).strip(): j for j, sid in enumerate(ids)}
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
        svm_seqindex_to_row = {}
        for j, sid in enumerate(ids):
            svm_seqindex_to_row[str(sid).strip()] = j
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    if not results:
        print("No models ran (missing checkpoints or tabular columns).", flush=True)
        return 1

    names = list(results.keys())
    score_z_by_model: dict[str, np.ndarray] = {}
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    for m in names:
        r = results[m]
        raw = cmp._raw_for_benchmark_z(m, r)
        if (
            m == "SVM"
            and svm_seqindex_to_row is not None
            and "seqIndex" in master.columns
        ):
            aligned = np.full(len(canonical_ids), np.nan, dtype=np.float64)
            for wi in range(len(canonical_ids)):
                si = str(int(master.iloc[wi]["seqIndex"]))
                j = svm_seqindex_to_row.get(si)
                if j is not None:
                    aligned[wi] = float(raw[j])
            z, _, _, _ = cmp._zscore_aligned_to_ids(canonical_ids, canonical_ids, aligned)
        else:
            z, _, _, _ = cmp._zscore_aligned_to_ids(canonical_ids, r["ids"], raw)
        score_z_by_model[m] = z
<<<<<<< HEAD
=======
    for mname in names:
        r = results[mname]
        raw = cmp._raw_for_benchmark_z(mname, r)
        z, _, _, _ = cmp._zscore_aligned_to_ids(canonical_ids, r["ids"], raw)
        score_z_by_model[mname] = z
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16

    out_rows: list[dict] = []
    for idx, pid in enumerate(canonical_ids):
        mr = master.iloc[idx]
        row = {
            "parent_id": mr["parent_id"],
            "start": int(mr["window_start"]),
            "length": int(mr["window_length"]),
            "peptide_id": pid,
            "sequence": seq_by_id.get(str(pid).strip(), pd.NA),
        }
        if "seqIndex" in master.columns:
            row["seqIndex"] = int(mr["seqIndex"])
        if "window_id" in master.columns:
            row["window_id"] = mr["window_id"]
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        for m in names:
            r = results[m]
            i = None
            if m == "SVM" and svm_seqindex_to_row is not None and "seqIndex" in master.columns:
                si = str(int(master.iloc[idx]["seqIndex"]))
                i = svm_seqindex_to_row.get(si)
            if i is None and pid in r["ids"]:
                i = r["ids"].index(pid)
            if i is not None:
                row[f"{m}_pred"] = int(r["pred"][i])
                row[f"{m}_confidence"] = float(r["confidence"][i])
                row[f"{m}_prob_AMP"] = float(r["prob_amp"][i])
                zv = score_z_by_model[m][idx]
                row[f"{m}_score_z"] = float(zv) if np.isfinite(zv) else None
                if m == "SVM":
                    d = float(r["distance"][i]) if np.isfinite(r["distance"][i]) else None
                    row[f"{m}_hyperplane_distance"] = d
                    row[f"{m}_distance"] = d
                else:
                    row[f"{m}_logit_AMP"] = float(r["logit_amp"][i])
                    row[f"{m}_logit_nonAMP"] = float(r["logit_nonamp"][i])
                    row[f"{m}_logit_margin"] = float(r["logit_margin"][i])
            elif i is None:
                row[f"{m}_pred"] = None
                row[f"{m}_confidence"] = None
                row[f"{m}_prob_AMP"] = None
                row[f"{m}_score_z"] = None
                if m == "SVM":
                    row[f"{m}_hyperplane_distance"] = None
                    row[f"{m}_distance"] = None
                else:
                    row[f"{m}_logit_AMP"] = None
                    row[f"{m}_logit_nonAMP"] = None
                    row[f"{m}_logit_margin"] = None
<<<<<<< HEAD
=======
        seq_idx = int(mr["seqIndex"]) if "seqIndex" in master.columns else None
        for mname in names:
            r = results[mname]
            i = _svm_row_index(results, mname, pid, seq_idx, svm_id_to_row)
            if i is not None:
                row[f"{mname}_pred"] = int(r["pred"][i])
                row[f"{mname}_confidence"] = float(r["confidence"][i])
                row[f"{mname}_prob_AMP"] = float(r["prob_amp"][i])
                zv = score_z_by_model[mname][idx]
                row[f"{mname}_score_z"] = float(zv) if np.isfinite(zv) else None
                if mname == "SVM":
                    d = float(r["distance"][i]) if np.isfinite(r["distance"][i]) else None
                    row[f"{mname}_hyperplane_distance"] = d
                    row[f"{mname}_distance"] = d
                else:
                    row[f"{mname}_logit_AMP"] = float(r["logit_amp"][i])
                    row[f"{mname}_logit_nonAMP"] = float(r["logit_nonamp"][i])
                    row[f"{mname}_logit_margin"] = float(r["logit_margin"][i])
            else:
                row[f"{mname}_pred"] = None
                row[f"{mname}_confidence"] = None
                row[f"{mname}_prob_AMP"] = None
                row[f"{mname}_score_z"] = None
                if mname == "SVM":
                    row[f"{mname}_hyperplane_distance"] = None
                    row[f"{mname}_distance"] = None
                else:
                    row[f"{mname}_logit_AMP"] = None
                    row[f"{mname}_logit_nonAMP"] = None
                    row[f"{mname}_logit_margin"] = None
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    preferred = ["seqIndex", "window_id", "parent_id", "start", "length", "peptide_id", "sequence"]
    front = [c for c in preferred if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in front]
    out_df = out_df[front + rest]

    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        primary = Path(args.output_csv) if args.output_csv else ws / f"model_comparison_windowed_{ts}.csv"
        primary.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(primary, index=False)
        snap = ws / "model_comparison_windowed_latest.csv"
        out_df.to_csv(snap, index=False)
        print(f"\nSaved: {primary.resolve()}", flush=True)
        print(f"Latest: {snap.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

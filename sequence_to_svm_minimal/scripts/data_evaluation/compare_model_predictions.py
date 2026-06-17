#!/usr/bin/env python3
"""
Compare model predictions on unlabeled test data (SVM, GCN, GAT, EGNN).

Outputs: per-sample predictions, P(AMP), confidence, raw classifier scores (SVM
``decision_function`` as ``SVM_hyperplane_distance`` / ``SVM_distance``; GNN
``logit_AMP``, ``logit_nonAMP``, ``logit_margin``), and per-model z-scores of those
raw scores within the run (GNN margin = logit_AMP − logit_nonAMP). GNN P(AMP) uses Platt scaling by default when ``<checkpoint_stem>_platt.json`` sits
next to the ``.pt`` file (written by ``run_gnn_train_final_models`` on the validation
split); otherwise plain softmax. Pass ``--no-gnn-platt`` to force softmax even if a
Platt file exists. Writes a
timestamped CSV plus ``model_comparison_latest.csv`` (under GENERATED, or under
``results/comparisons/`` when not using a workspace). Includes ``sequence`` from
``--geo_csv`` when that file has a ``sequence`` column. No ground-truth metrics (data is unlabeled).

Typical usage after ``run_data_pipeline`` + ``run_gnn_train_final_models`` (checkpoints in
``generated/gnn_ready_models/``)::

  python scripts/data_evaluation/compare_model_predictions.py path/to/generated

If weights live elsewhere (e.g. ``checkpoints/``)::

  python scripts/data_evaluation/compare_model_predictions.py path/to/generated \\
      --gnn-checkpoints-dir path/to/checkpoints

One tree for both GNN and QSAR-SVM artifacts::

  python scripts/data_evaluation/compare_model_predictions.py path/to/generated \\
      --checkpoints-base path/to/my_checkpoints

Expected under ``my_checkpoints``: GNN weights in ``gnn/``, ``gnn_ready_models/``, or flat;
SVM as ``svm_qsar12_model.pkl`` and ``svm_qsar12_zscores.txt`` (optionally under ``svm/``).
Descriptors for SVM default to the pipeline ``--qsar_csv`` when present.

You may pass the parent directory that contains ``generated/``. This mode reads
``pipeline_manifest.json`` (including ``esm2_embeddings`` / ``esm2_per_residue``) and skips the legacy QSAR-SVM block
unless you omit the positional and pass ``--svm_pkl`` / ``--svm_z_file`` / ``--svm_descriptor_csv``
explicitly, or use ``--checkpoints-base``. GNN checkpoints from ``run_gnn_train_final_models``
use graph-level Geo/QSAR tabular columns; per-residue ESM2 is loaded from ``esm2_per_residue/`` (see manifest).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.load_config import argv_without_config_flags, load_compare_models_config, repo_root
from gnn.checkpoint_meta import resolve_node_layout_for_checkpoint
from gnn.data_utils import (
    NodeFeatureGroups,
    node_feature_groups_from_cli,
    node_feature_groups_from_config_value,
    node_input_dim,
)

_ROOT = repo_root()

def _comparison_snapshot_csv_path(ws: Path | None) -> Path:
    """Stable path overwritten each run so tools can open a fixed filename."""
    if ws is not None:
        return ws / "model_comparison_latest.csv"
    comp = _ROOT / "results" / "comparisons"
    comp.mkdir(parents=True, exist_ok=True)
    return comp / "model_comparison_latest.csv"


def _ordered_result_column_names(model_names: list[str]) -> list[str]:
    """CSV column order: pred / prob / confidence, raw classifier scores, then per-run z-scores."""
    cols: list[str] = []
    for m in model_names:
        cols.extend([f"{m}_pred", f"{m}_prob_AMP", f"{m}_confidence"])
        if m == "SVM":
            cols.extend([f"{m}_hyperplane_distance", f"{m}_distance", f"{m}_score_z"])
        else:
            cols.extend(
                [
                    f"{m}_logit_AMP",
                    f"{m}_logit_nonAMP",
                    f"{m}_logit_margin",
                    f"{m}_score_z",
                ]
            )
    return cols


def _write_comparison_metadata_json(
    *,
    json_path: Path,
    args: argparse.Namespace,
    ws: Path | None,
    results: dict,
    mode_cols: dict[str, list[str]] | None,
    gnn_master_path: Path | None,
    canonical_count: int,
) -> None:
    """Write a compact metadata JSON next to the comparison outputs."""
    def _p(v: object) -> str | None:
        if v is None:
            return None
        try:
            s = str(v)
        except Exception:
            return None
        if not s:
            return None
        return s

    names = list(results.keys())
    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "workspace_generated": _p(ws.resolve()) if ws is not None else None,
        "architecture": getattr(args, "architecture", None),
        "inputs": {
            "geo_csv": _p(getattr(args, "geo_csv", None)),
            "pdb_dir": _p(getattr(args, "pdb_dir", None)),
            "qsar_csv": _p(getattr(args, "qsar_csv", None)),
            "esm2_csv": _p(getattr(args, "esm2_csv", None)),
            "gnn_feature_master_csv": _p(gnn_master_path.resolve()) if gnn_master_path is not None else None,
            "canonical_peptide_count": int(canonical_count),
        },
        "checkpoints": {
            "svm_pkl": _p(getattr(args, "svm_pkl", None)),
            "svm_z_file": _p(getattr(args, "svm_z_file", None)),
            "svm_descriptor_csv": _p(getattr(args, "svm_descriptor_csv", None)),
            "esm_only_pt": _p(getattr(args, "esm_only_pt", None)),
            "esm_geo_pt": _p(getattr(args, "esm_geo_pt", None)),
            "esm_qsar_pt": _p(getattr(args, "esm_qsar_pt", None)),
            "esm_combined_pt": _p(getattr(args, "esm_combined_pt", None)),
        },
        "models_run": names,
        "feature_columns_by_mode": mode_cols or {},
        "cli_args": {k: _p(v) if isinstance(v, (Path,)) else v for k, v in vars(args).items()},
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

FEATURE_SETS = [
    ('ESM-only', 'esm_only_pt'),
    ('ESM+Geo20', 'esm_geo_pt'),
    ('ESM+QSAR12', 'esm_qsar_pt'),
    ('ESM+Combined32', 'esm_combined_pt'),
]

# Checkpoint stem suffixes from run_gnn_train_final_models.py (FEATURE_CONFIGS keys)
_GNN_FEATURE_STEMS = ("ESM", "Geo", "QSAR", "Combined")

_FEATURE_SET_TO_ARG = {
    "ESM": "esm_only_pt",
    "Geo": "esm_geo_pt",
    "QSAR": "esm_qsar_pt",
    "Combined": "esm_combined_pt",
}

# Must match run_gnn_train_final_models.create_feature_cols / load_data_with_features.
GNN_GEO_COLS = [
    "radius_gyration",
    "end_to_end_distance",
    "max_pairwise_distance",
    "centroid_distance_mean",
    "centroid_distance_std",
    "fraction_helix",
    "fraction_sheet",
    "fraction_coil",
    "total_sasa",
    "hydrophobic_sasa",
    "fraction_hydrophobic_sasa",
    "length",
    "net_charge",
    "mean_hydrophobicity",
    "hydrophobic_moment",
    "curvature_mean",
    "curvature_std",
    "curvature_max",
    "torsion_mean",
    "torsion_std",
]
GNN_QSAR_COLS = [
    "netCharge",
    "FC",
    "LW",
    "DP",
    "NK",
    "AE",
    "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
]


def _build_gnn_feature_master(
    args: argparse.Namespace, ws: Path | None
) -> tuple[Path | None, dict[str, list[str]], bool]:
    """Merge geo + optional QSAR (+ optional pooled ESM2); write one CSV; column lists per GNN tabular mode."""
    if not getattr(args, "geo_csv", None) or not Path(args.geo_csv).is_file():
        return None, {}, bool(args.qsar_csv and Path(args.qsar_csv).is_file())

    geo_df = pd.read_csv(args.geo_csv)
    if "peptide_id" not in geo_df.columns:
        raise SystemExit("geo_csv must contain peptide_id for GNN tabular features")

    df = geo_df.copy()
    has_qsar = bool(args.qsar_csv and Path(args.qsar_csv).is_file())
    if has_qsar:
        qsar_df = pd.read_csv(args.qsar_csv)
        miss = [c for c in GNN_QSAR_COLS if c not in qsar_df.columns]
        if miss:
            raise SystemExit(f"QSAR CSV missing columns: {miss}")
        df = df.merge(qsar_df[["peptide_id"] + GNN_QSAR_COLS], on="peptide_id", how="left")

    # Optional: legacy checkpoints used pooled ESM2 (mean-pooled) as tabular inputs to the MLP.
    # When enabled, we merge the pooled embedding CSV into the same master file and include
    # those columns in the per-mode tabular column lists.
    esm_cols: list[str] = []
    use_pooled_esm = bool(getattr(args, "legacy_pooled_esm_tabular", False))
    if use_pooled_esm:
        esm_path = getattr(args, "esm2_csv", None)
        if not esm_path or not Path(esm_path).is_file():
            raise SystemExit(
                "--legacy-pooled-esm-tabular requires --esm2-csv (a pooled ESM2 embeddings CSV) to exist."
            )
        esm_df = pd.read_csv(esm_path)
        if "peptide_id" not in esm_df.columns:
            raise SystemExit(f"ESM2 CSV missing 'peptide_id' column: {esm_path}")

        def _sort_key(c: str) -> tuple[int, str]:
            # Prefer numeric suffix ordering (esm2_dim_0..), else lexical.
            try:
                return (int(c.rsplit("_", 1)[-1]), c)
            except Exception:
                return (10**9, c)

        candidates = [c for c in esm_df.columns if str(c).startswith("esm2_dim_")]
        if not candidates:
            raise SystemExit(
                f"--legacy-pooled-esm-tabular expects columns named esm2_dim_0.. in {esm_path} "
                "(from esm_sequence_processor.py --mode embeddings)."
            )
        esm_cols = sorted(candidates, key=_sort_key)

        # Some pooled embedding CSVs also include seqIndex or other metadata.
        keep = ["peptide_id"] + esm_cols
        df = df.merge(esm_df[keep], on="peptide_id", how="left")

    out_path = Path(args.geometric_qsar_combined_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    geo_present = [c for c in GNN_GEO_COLS if c in df.columns]
    qsar_present = [c for c in GNN_QSAR_COLS if c in df.columns]

    modes: dict[str, list[str]] = {
        "esm": esm_cols if use_pooled_esm else [],
        # Match training scaler column order: geometry/QSAR first, pooled ESM last.
        "geo20": (geo_present + esm_cols) if use_pooled_esm else geo_present,
        "qsar12": (qsar_present + esm_cols)
        if (use_pooled_esm and has_qsar)
        else (qsar_present if has_qsar else []),
        "combined32": (geo_present + qsar_present + esm_cols)
        if (use_pooled_esm and has_qsar)
        else ((geo_present + qsar_present) if has_qsar else []),
    }
    return out_path, modes, has_qsar


def _set_gnn_paths_from_dir(args: argparse.Namespace, gdir: Path) -> None:
    """Default .pt names from run_gnn_train_final_models.py."""
    arch = args.architecture
    args.esm_only_pt = str(gdir / f"{arch}_ready_{_GNN_FEATURE_STEMS[0]}.pt")
    args.esm_geo_pt = str(gdir / f"{arch}_ready_{_GNN_FEATURE_STEMS[1]}.pt")
    args.esm_qsar_pt = str(gdir / f"{arch}_ready_{_GNN_FEATURE_STEMS[2]}.pt")
    args.esm_combined_pt = str(gdir / f"{arch}_ready_{_GNN_FEATURE_STEMS[3]}.pt")


def _pick_gnn_dir_under_base(base: Path, arch: str) -> Path:
    """Prefer gnn/, gnn_ready_models/, ready_models/ with weights; else flat base/."""
    subdirs = [base / "gnn", base / "gnn_ready_models", base / "ready_models"]
    for cand in subdirs:
        if cand.is_dir() and list(cand.glob(f"{arch}_ready_*.pt")):
            return cand
    for cand in subdirs:
        if cand.is_dir() and (cand / "ready_models_summary.json").is_file():
            return cand
    if list(base.glob(f"{arch}_ready_*.pt")):
        return base
    for cand in subdirs:
        if cand.is_dir():
            return cand
    return base


def apply_checkpoints_base(args: argparse.Namespace) -> Path:
    """
    Single root for GNN .pt tree and QSAR-SVM bundle.

    GNN: ``<base>/gnn/``, ``<base>/gnn_ready_models/``, or ``<base>/*.pt`` (standard names).

    SVM: first existing of
    ``svm_qsar12_model.pkl``, ``svm/svm_qsar12_model.pkl``, ``svm_model.pkl``;
    ``svm_qsar12_zscores.txt``, ``svm/svm_qsar12_zscores.txt``.
    Descriptors: ``--qsar_csv`` if that file exists, else ``<base>/qsar12_descriptors.csv``
    or ``<base>/svm/qsar12_descriptors.csv``.
    """
    base = Path(getattr(args, "checkpoints_base")).expanduser().resolve()
    if not base.is_dir():
        raise SystemExit(f"--checkpoints-base is not a directory: {base}")

    arch = args.architecture
    gdir = _pick_gnn_dir_under_base(base, arch)
    _set_gnn_paths_from_dir(args, gdir)
    print(f"Checkpoints base: {base}  (GNN: {gdir})", flush=True)

    svm_pkl_cands = [
        base / "svm_qsar12_model.pkl",
        base / "svm" / "svm_qsar12_model.pkl",
        base / "svm_model.pkl",
    ]
    svm_z_cands = [
        base / "svm_qsar12_zscores.txt",
        base / "svm" / "svm_qsar12_zscores.txt",
    ]
    pkl = next((p for p in svm_pkl_cands if p.is_file()), None)
    zf = next((p for p in svm_z_cands if p.is_file()), None)
    if pkl and zf:
        desc_path: Path | None = None
        if Path(args.qsar_csv).is_file():
            desc_path = Path(args.qsar_csv).resolve()
        else:
            desc_cands = [
                base / "qsar12_descriptors.csv",
                base / "svm" / "qsar12_descriptors.csv",
            ]
            desc_path = next((p.resolve() for p in desc_cands if p.is_file()), None)
        if desc_path is not None:
            args.svm_pkl = str(pkl.resolve())
            args.svm_z_file = str(zf.resolve())
            args.svm_descriptor_csv = str(desc_path)
            print(
                f"SVM from base: {args.svm_pkl} + {args.svm_z_file} "
                f"(descriptors: {args.svm_descriptor_csv})",
                flush=True,
            )
        else:
            print(
                "Checkpoints base: found SVM pkl + z-scores but no descriptor CSV "
                "(--qsar_csv missing or not a file, and no qsar12_descriptors.csv under base); "
                "SVM skipped.",
                flush=True,
            )
    else:
        print(
            "Checkpoints base: no svm_qsar12_model.pkl + svm_qsar12_zscores.txt found; SVM step skipped.",
            flush=True,
        )

    return base


def _try_apply_ready_models_summary(arch: str, args: argparse.Namespace, *roots: Path) -> Path | None:
    """
    First usable ready_models_summary.json under any root wins (earlier roots preferred).
    Sets checkpoint paths when entries exist on disk.
    """
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None or not root.is_dir():
            continue
        for p in (
            root / "gnn_ready_models" / "ready_models_summary.json",
            root / "ready_models_summary.json",
        ):
            if p.is_file():
                k = str(p.resolve())
                if k not in seen:
                    seen.add(k)
                    candidates.append(p)
        for p in sorted(root.glob("**/ready_models_summary.json"), key=lambda x: len(str(x))):
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                candidates.append(p)

    for summary_path in candidates:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        models = data.get("models") or []
        arch_l = arch.strip().lower()
        n = 0
        for m in models:
            if str(m.get("architecture", "")).lower() != arch_l:
                continue
            fs = m.get("feature_set")
            ck = m.get("checkpoint")
            attr = _FEATURE_SET_TO_ARG.get(fs)
            if attr and ck and Path(ck).is_file():
                setattr(args, attr, str(Path(ck).resolve()))
                n += 1
        if n > 0:
            print(f"Using {n} checkpoint path(s) from {summary_path}", flush=True)
            return summary_path
    return None


def apply_pipeline_generated_workspace(
    args: argparse.Namespace,
    *,
    skip_svm_clear: bool,
) -> tuple[Path | None, bool]:
    """
    If GENERATED (positional) is set, fill paths from pipeline_manifest.json.

    Default GNN checkpoints are NOT changed (e.g. ``checkpoints/latest/*.pt``). Use
    ``--gnn-checkpoints-dir`` or ``--checkpoints-base`` to point at a workspace-trained
    ``gnn_ready_models/`` folder.

    Unless skip_svm_clear, clears this script's QSAR-SVM paths (use --svm_pkl etc. to keep).

    Returns ``(resolved_workspace_or_none, skip_workspace_checkpoint_roots)``. The second
    value is kept for API compatibility and is always False when a workspace path is given.
    """
    from peptide_pipeline.manifest_paths import load_pipeline_manifest, resolve_generated_workspace

    def _normalize_manifest_path(p: str) -> Path:
        """
        Convert WSL-style /mnt/<drive>/... paths into Windows paths when running on win32.
        Keeps native paths unchanged.
        """
        s = str(p)
        if sys.platform.startswith("win"):
            # /mnt/c/Users/...  ->  C:\Users\...
            if s.startswith("/mnt/") and len(s) >= 6 and s[5].isalpha() and s[6:7] == "/":
                drive = s[5].upper()
                rest = s[7:].replace("/", "\\")
                return Path(f"{drive}:\\{rest}")
            # \\wsl$\<distro>\mnt\c\... sometimes appears in copied paths; leave as-is (Path handles it)
        return Path(s)

    chosen = getattr(args, "generated", None)
    if not chosen:
        return None, False

    skip_workspace_checkpoint_roots = False

    ws = resolve_generated_workspace(chosen)
    m = load_pipeline_manifest(ws)
    for key in ("geometric_features", "structures_dir", "qsar12_descriptors", "esm2_embeddings"):
        if not m.get(key):
            raise SystemExit(
                f"Manifest missing {key!r}; run the full pipeline (QSAR + ESM2 required for GNN checkpoints)."
            )

    args.geo_csv = str(_normalize_manifest_path(m["geometric_features"]).resolve())
    args.pdb_dir = str(_normalize_manifest_path(m["structures_dir"]).resolve())
    args.qsar_csv = str(_normalize_manifest_path(m["qsar12_descriptors"]).resolve())
    args.geometric_qsar_combined_csv = str(ws / "compare_geo_qsar_merged.csv")
    if getattr(args, "esm2_csv", None) is None:
        args.esm2_csv = str(_normalize_manifest_path(m["esm2_embeddings"]).resolve())
    if getattr(args, "gnn_esm2_residue_dir", None) in (None, ""):
        pr = m.get("esm2_per_residue")
        if pr:
            args.gnn_esm2_residue_dir = str(_normalize_manifest_path(pr).resolve())
        else:
            emb = _normalize_manifest_path(m["esm2_embeddings"])
            args.gnn_esm2_residue_dir = str((emb.parent / "esm2_per_residue").resolve())

    if not skip_workspace_checkpoint_roots:
        # Do not override checkpoint defaults when a workspace is provided.
        # Users can opt-in to workspace-trained checkpoints via --gnn-checkpoints-dir
        # (or --checkpoints-base / ready_models_summary.json).
        pass

    if not skip_svm_clear:
        args.svm_descriptor_csv = ""
        args.svm_z_file = ""
        args.svm_pkl = ""

    print(f"Pipeline workspace: {ws}", flush=True)
    return ws, skip_workspace_checkpoint_roots


def _load_svm_predictions(descriptor_csv: str, z_file: str, pkl_path: str):
    """Load SVM, Z-score descriptors, return (ids, preds, prob_amp, distance)."""
    try:
        import joblib
    except ImportError:
        from sklearn.externals import joblib as joblib

    df = pd.read_csv(descriptor_csv)
    id_col = 'peptide_id' if 'peptide_id' in df.columns else ('name' if 'name' in df.columns else df.columns[0])
    ids = df[id_col].astype(str).tolist()

    with open(z_file, 'r') as f:
        desc_names = [x.strip() for x in f.readline().strip().split(',')]
        means = np.array([float(x) for x in f.readline().strip().split(',')])
        stds = np.array([float(x) for x in f.readline().strip().split(',')])

    mask = []
    for name in desc_names:
        if name not in df.columns:
            raise ValueError(f"Descriptor '{name}' from Z file not in CSV columns")
        mask.append(df.columns.tolist().index(name))
    X = df.iloc[:, mask].values.astype(np.float64)
    X = (X - means) / np.where(stds > 0, stds, 1.0)

    clf = joblib.load(pkl_path)
    raw_preds = np.asarray(clf.predict(X)).ravel()
    classes = getattr(clf, 'classes_', np.array([0, 1]))
    pos_class = int(1 if 1 in classes else classes[-1])
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)
        pos_idx = int(np.where(classes == pos_class)[0][0])
        prob_amp = proba[:, pos_idx].ravel()
    else:
        prob_amp = np.where(raw_preds == pos_class, 1.0, 0.0)
    if hasattr(clf, 'decision_function'):
        distance = np.asarray(clf.decision_function(X)).ravel()
    else:
        distance = np.full_like(prob_amp, np.nan, dtype=np.float64)

    preds = (raw_preds == pos_class).astype(int)
    return ids, preds, prob_amp, distance


def _default_tabular_scaler_path(model_path: str) -> str | None:
    p = Path(model_path).with_name(Path(model_path).stem + "_tabular_scaler.joblib")
    return str(p) if p.is_file() else None


def _pool_dim_for_gnn_classifier(hidden_channels: int, pooling: str) -> int:
    """Matches GCN/GAT/EGNN pooling branch in gnn/models.py (mean_max → 2× hidden)."""
    return int(hidden_channels * 2) if pooling == "mean_max" else int(hidden_channels)


def _classifier_input_dim_from_state_dict(sd: dict) -> int:
    for key in ("model.classifier.0.weight", "classifier.0.weight"):
        t = sd.get(key)
        if t is not None:
            return int(t.shape[1])
    keys = list(sd.keys())[:12]
    raise SystemExit(
        "Cannot infer tabular width from checkpoint (missing model.classifier.0.weight). "
        f"Sample keys: {keys}"
    )


def _run_gnn_predictions(csv_path: str,
                         pdb_dir: str,
                         model_path: str,
                         architecture: str,
                         hidden: int,
                         num_layers: int,
                         pooling: str,
                         batch_size: int,
                         geometric_feature_cols: list[str] | None = None,
                         tabular_scaler_path: str | None = None,
                         esm2_residue_dir: str | None = None,
                         canonical_seqs_path: str | None = None,
                         *,
                         node_feature_groups=None,
                         use_gnn_platt: bool = True):
    """Run one GNN checkpoint and return ids/preds/prob/logits/margin."""
    import torch
    from torch_geometric.loader import DataLoader

    from gnn.data_utils import PeptideGraphDataset
    from gnn.models import PeptideGNN, esm2_raw_dim_from_state_dict, esm2_hidden_dim_from_state_dict
    from gnn.platt import default_platt_path, load_platt_json, platt_prob_amp, softmax_prob_amp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd0 = torch.load(model_path, map_location="cpu", weights_only=True)
    esm2_raw_ckpt = esm2_raw_dim_from_state_dict(sd0)
    esm2_h_ckpt = esm2_hidden_dim_from_state_dict(sd0)
    ng_infer, in_base_ckpt, layout_notes = resolve_node_layout_for_checkpoint(
        model_path,
        sd0,
        architecture,
        user_node_groups=node_feature_groups,
    )
    for msg in layout_notes:
        print(f"  {msg}", flush=True)
    cin = _classifier_input_dim_from_state_dict(sd0)
    pool = _pool_dim_for_gnn_classifier(hidden, pooling)
    geo_dim_ckpt = cin - pool
    if geo_dim_ckpt < 0:
        raise SystemExit(
            f"Checkpoint {model_path!r} has classifier input {cin} but pool_dim is {pool} "
            f"for --gnn_hidden {hidden} and --gnn_pooling {pooling!r}."
        )
    cols = list(geometric_feature_cols or [])
    use_geo = geo_dim_ckpt > 0
    if not use_geo and cols:
        print(
            f"  Note: {Path(model_path).name} is graph-only; "
            f"not using {len(cols)} merged tabular column(s) for this checkpoint.",
            flush=True,
        )
    if use_geo and len(cols) != geo_dim_ckpt:
        raise SystemExit(
            f"Checkpoint {model_path!r} was trained with {geo_dim_ckpt} tabular columns "
            f"after graph pooling, but this run supplies {len(cols)}. "
            "Use the .pt that matches your merged CSV feature mode (ESM / Geo+ESM / …), "
            "or fix --esm_*_pt / --checkpoints-base."
        )
    if esm2_raw_ckpt > 0:
        if not esm2_residue_dir or not Path(esm2_residue_dir).is_dir():
            raise SystemExit(
                f"Checkpoint {model_path!r} expects per-residue ESM2 (encoder in={esm2_raw_ckpt}), "
                f"but --gnn-esm2-residue-dir is missing or not a directory ({esm2_residue_dir!r})."
            )
    platt_json = default_platt_path(model_path)
    platt = load_platt_json(platt_json) if use_gnn_platt else None
    if not use_gnn_platt:
        print("  GNN probabilities: softmax (--no-gnn-platt)", flush=True)
    elif platt is not None:
        print(f"  GNN probabilities: Platt scaling ({platt_json.name})", flush=True)
    else:
        print(
            f"  GNN probabilities: softmax (no {platt_json.name}; retrain with "
            "run_gnn_train_final_models.py for Platt)",
            flush=True,
        )
    scaler_path = tabular_scaler_path
    if scaler_path is None and use_geo:
        scaler_path = _default_tabular_scaler_path(model_path)
    canon_path = canonical_seqs_path
    if canon_path is None:
        # Default pipeline layout: <work_dir>/inputs/canonical_seqs.txt next to geometric_features.csv
        guess = Path(csv_path).resolve().parent / "inputs" / "canonical_seqs.txt"
        if guess.is_file():
            canon_path = str(guess)
    dataset = PeptideGraphDataset(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        use_geometric_features=use_geo,
        geometric_feature_cols=cols if use_geo else None,
        tabular_scaler_path=scaler_path,
        esm2_residue_dir=esm2_residue_dir if esm2_raw_ckpt > 0 else None,
        canonical_seqs_path=canon_path,
        node_feature_groups=ng_infer,
    )

    if use_geo and len(dataset) > 0 and hasattr(dataset[0], "geo_features"):
        geo_dim_data = int(dataset[0].geo_features.shape[1])
        if geo_dim_data != geo_dim_ckpt:
            raise SystemExit(
                f"Tabular width mismatch: checkpoint expects {geo_dim_ckpt}, "
                f"dataset rows produce {geo_dim_data}."
            )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if len(dataset) > 0:
        xw = int(dataset[0].x.shape[1])
        if xw != in_base_ckpt:
            raise SystemExit(
                f"Node feature width mismatch: checkpoint implies data.x width {in_base_ckpt}, "
                f"dataset produced {xw}."
            )
    model = PeptideGNN(
        architecture=architecture,
        in_channels=in_base_ckpt,
        hidden_channels=hidden,
        num_layers=num_layers,
        num_classes=2,
        pooling=pooling,
        geo_feature_dim=geo_dim_ckpt,
        esm2_raw_dim=esm2_raw_ckpt,
        esm2_hidden_dim=esm2_h_ckpt,
    )
    model.load_state_dict(sd0)
    model = model.to(device)
    model.eval()

    all_probs = []
    all_logit_amp = []
    all_logit_nonamp = []
    all_logit_margin = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            logit_nonamp = out[:, 0].cpu().numpy()
            logit_amp = out[:, 1].cpu().numpy()
            logit_margin = (out[:, 1] - out[:, 0]).cpu().numpy()
            if platt is not None:
                probs = platt_prob_amp(logit_margin, platt["coef"], platt["intercept"])
            else:
                probs = softmax_prob_amp(out)
            all_probs.extend(probs)
            all_logit_amp.extend(logit_amp)
            all_logit_nonamp.extend(logit_nonamp)
            all_logit_margin.extend(logit_margin)
    probs = np.array(all_probs)
    logit_amp = np.array(all_logit_amp)
    logit_nonamp = np.array(all_logit_nonamp)
    logit_margin = np.array(all_logit_margin)
    preds = (probs >= 0.5).astype(int)

    df = dataset.df
    id_col = 'peptide_id' if 'peptide_id' in df.columns else ('name' if 'name' in df.columns else None)
    ids = df[id_col].astype(str).tolist() if id_col else [str(i) for i in range(len(df))]
    return ids, preds, probs, logit_amp, logit_nonamp, logit_margin


def _confidence(prob_amp: np.ndarray) -> np.ndarray:
    return np.maximum(prob_amp, 1.0 - prob_amp)


def _raw_for_benchmark_z(model_name: str, result: dict) -> np.ndarray:
    """Raw logit-line score for per-run z-scoring: SVM decision_function; GNN logit margin."""
    if model_name == "SVM":
        return np.asarray(result["distance"], dtype=np.float64)
    return np.asarray(result["logit_margin"], dtype=np.float64)


def _zscore_aligned_to_ids(
    canonical_ids: list[str],
    ids: list[str],
    raw: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """Align raw scores to canonical_ids; z-score over finite values; return z, mu, sigma, n."""
    idx_map = {str(i): k for k, i in enumerate(ids)}
    aligned = np.full(len(canonical_ids), np.nan, dtype=np.float64)
    for j, pid in enumerate(canonical_ids):
        k = idx_map.get(str(pid))
        if k is not None:
            aligned[j] = float(raw[k])
    finite = np.isfinite(aligned)
    n_fin = int(finite.sum())
    z = np.full_like(aligned, np.nan, dtype=np.float64)
    if n_fin == 0:
        return z, float("nan"), float("nan"), 0
    v = aligned[finite]
    mu = float(v.mean())
    sig = float(v.std(ddof=0))
    if sig == 0.0 or not np.isfinite(sig):
        z[finite] = 0.0
        return z, mu, 0.0, n_fin
    z[finite] = (v - mu) / sig
    return z, mu, sig, n_fin


def _agreement_matrix(names: list, pred_dfs: list, ids_common: list):
    """Pairwise agreement counts (both predict AMP) and total overlap."""
    n = len(names)
    agree = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                agree[i, j] = len(ids_common)
                continue
            pi = pred_dfs[i].loc[ids_common, 'pred'].values
            pj = pred_dfs[j].loc[ids_common, 'pred'].values
            agree[i, j] = int((pi == pj).sum())
    return agree


def _print_agreement(names: list, agree: np.ndarray, n: int):
    print("\nPairwise agreement (same classification)")
    print("-" * (12 * (len(names) + 1)))
    header = "Model".ljust(12) + "".join(m[:10].ljust(12) for m in names)
    print(header)
    for i, name in enumerate(names):
        row = name[:10].ljust(12) + "".join(f"{agree[i, j]:>10} " for j in range(len(names)))
        print(row)
    print(f"(Total samples: {n})")


def _print_cli_report(architecture: str,
                      results: dict,
                      canonical_ids: list,
                      ids_common: list,
                      pred_frames: dict) -> None:
    """Pretty CLI report summarizing model behavior and agreement."""
    names = list(results.keys())
    n_samples = len(canonical_ids)

    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON REPORT  –  architecture = {architecture.upper()}")
    print("=" * 80)
    print(f"Total peptides in input CSV     : {n_samples}")
    print(f"Models run                      : {', '.join(names)}")

    if ids_common:
        print(f"Samples with all model outputs  : {len(ids_common)} / {n_samples}")

    print("\nPer-model summary (AMP predictions and confidence)")
    header = (
        f"{'Model':<18}"
        f"{'AMP count':>12}"
        f"{'% AMP':>10}"
        f"{'Mean conf':>14}"
        f"{'Mean conf (AMP)':>18}"
        f"{'Mean conf (non-AMP)':>22}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for m in names:
        r = results[m]
        pred = np.asarray(r["pred"])
        conf = np.asarray(r["confidence"])
        n = len(pred)
        n_amp = int(pred.sum())
        frac_amp = n_amp / n if n > 0 else 0.0

        if n_amp > 0:
            mean_conf_amp = float(conf[pred == 1].mean())
        else:
            mean_conf_amp = float("nan")

        if n_amp < n:
            mean_conf_non = float(conf[pred == 0].mean())
        else:
            mean_conf_non = float("nan")

        print(
            f"{m:<18}"
            f"{n_amp:>12d}"
            f"{frac_amp * 100:>9.1f}%"
            f"{float(conf.mean()):>14.4f}"
            f"{mean_conf_amp:>18.4f}"
            f"{mean_conf_non:>22.4f}"
        )

    if len(ids_common) >= 1 and len(names) >= 2:
        agree = _agreement_matrix(names, [pred_frames[m] for m in names], ids_common)
        _print_agreement(names, agree, len(ids_common))

def _print_benchmark_z_summary(names: list[str], results: dict, canonical_ids: list[str]) -> None:
    print("\nBenchmark z-scores (per model: mean=0, std=1 over finite scores in this run)")
    print("- Raw metric: SVM decision_function; GNN logit_AMP − logit_nonAMP")
    hdr = f"{'Model':<22}{'n':>6}{'raw μ':>12}{'raw σ':>12}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for m in names:
        raw = _raw_for_benchmark_z(m, results[m])
        _, mu, sig, n_fin = _zscore_aligned_to_ids(canonical_ids, results[m]["ids"], raw)
        print(f"{m:<22}{n_fin:>6}{mu:>12.4f}{sig:>12.4f}")
    print("-" * len(hdr))
    print("Same z-scores are written per peptide as <model>_score_z in the CSV.")


def main():
    cfg_path, argv_rest = argv_without_config_flags(sys.argv[1:])
    cfg = load_compare_models_config(cfg_path)

    ap = argparse.ArgumentParser(
        description='Compare SVM and GNN predictions on unlabeled test data',
        epilog=(
            "GENERATED (positional): manifest inputs only (checkpoint defaults unchanged, e.g. checkpoints/latest). "
            "Use --checkpoints-base / --gnn-checkpoints-dir / explicit .pt paths for workspace checkpoints. "
            "--checkpoints-base sets one tree for GNN + QSAR-SVM; --gnn-checkpoints-dir overrides GNN only. "
            "JSON defaults: pass --config PATH anywhere (default: configs/compare_models.json when omitted)."
        ),
    )
    ap.add_argument(
        "generated",
        nargs="?",
        default=None,
        metavar="GENERATED",
        help="Pipeline generated/ directory or parent containing generated/; reads manifest inputs (does not change checkpoint defaults).",
    )
    ap.add_argument(
        "--checkpoints-base",
        "--models-base",
        type=str,
        default=None,
        dest="checkpoints_base",
        help=(
            "One directory for GNN weights (subfolder gnn/, gnn_ready_models/, or flat *.pt) "
            "and QSAR-SVM (svm_qsar12_model.pkl + svm_qsar12_zscores.txt at base or base/svm/). "
            "Descriptor CSV defaults to --qsar_csv when that file exists. "
            "Runs after GENERATED; use --gnn-checkpoints-dir to override only the GNN folder."
        ),
    )
    ap.add_argument(
        "--gnn-checkpoints-dir",
        type=str,
        default=None,
        help=(
            "Folder with GNN .pt files ({arch}_ready_ESM.pt, …_Geo.pt, …_QSAR.pt, …_Combined.pt) "
            "and optional ready_models_summary.json. Use with GENERATED to override "
            "<generated>/gnn_ready_models/, or without GENERATED if you pass --geo_csv / --pdb_dir / --qsar_csv. "
            "Overrides the GNN location chosen by --checkpoints-base."
        ),
    )
    ap.add_argument('--geo_csv', type=str, default=cfg['geo_csv'], help='Test geometric_features.csv')
    ap.add_argument('--pdb_dir', type=str, default=cfg['pdb_dir'], help='Directory containing test PDB files')
    ap.add_argument('--qsar_csv', type=str, default=cfg['qsar_csv'], help='Optional QSAR-12 descriptors CSV for Combined32')
    ap.add_argument(
        '--esm2-csv',
        type=str,
        default=None,
        dest='esm2_csv',
        help=(
            'ESM2 mean-pooled CSV (optional for SVM / legacy); GNN uses per-residue tensors under '
            '--gnn-esm2-residue-dir or manifest esm2_per_residue.'
        ),
    )
    ap.add_argument(
        '--gnn-esm2-residue-dir',
        type=str,
        default=None,
        dest='gnn_esm2_residue_dir',
        help=(
            'Directory of {peptide_id}.pt per-residue ESM2 (defaults from GENERATED manifest '
            'esm2_per_residue or sibling esm2_per_residue/).'
        ),
    )
    ap.add_argument(
        '--geometric_qsar_combined_csv',
        type=str,
        default=cfg['geometric_qsar_combined_csv'],
        help='Merged geo+QSAR+ESM2 CSV written for all GNN tabular modes (matches training merges)',
    )
    ap.add_argument(
        '--legacy-pooled-esm-tabular',
        action='store_true',
        help=(
            'Compatibility mode for older GNN checkpoints that used pooled ESM2 embeddings as tabular MLP inputs '
            '(expects --esm2-csv with columns esm2_dim_0..esm2_dim_1279 + peptide_id).'
        ),
    )
    ap.add_argument(
        '--svm_descriptor_csv',
        type=str,
        default=argparse.SUPPRESS,
        help='Descriptor CSV for QSAR SVM (default: configs/compare_models.json; cleared when using GENERATED unless --svm_pkl is set)',
    )
    ap.add_argument(
        '--svm_z_file',
        type=str,
        default=argparse.SUPPRESS,
        help='Z-score file: names, means, stds',
    )
    ap.add_argument(
        '--svm_pkl',
        type=str,
        default=argparse.SUPPRESS,
        help='Trained SVM pickle',
    )
    ap.add_argument('--architecture', type=str, default=cfg['architecture'],
                    choices=['gcn', 'gat', 'egnn'],
                    help='GNN architecture to compare across feature sets')
    ap.add_argument('--esm_only_pt', type=str, default=cfg['esm_only_pt'],
                    help='Checkpoint for ESM-only model')
    ap.add_argument('--esm_geo_pt', type=str, default=cfg['esm_geo_pt'],
                    help='Checkpoint for ESM+Geo20 model')
    ap.add_argument('--esm_qsar_pt', type=str, default=cfg['esm_qsar_pt'],
                    help='Checkpoint for ESM+QSAR12 model')
    ap.add_argument('--esm_combined_pt', type=str, default=cfg['esm_combined_pt'],
                    help='Checkpoint for ESM+Combined32 model')
    ap.add_argument('--gnn_hidden', type=int, default=cfg['gnn_hidden'])
    ap.add_argument('--gnn_layers', type=int, default=cfg['gnn_layers'])
    ap.add_argument('--gnn_pooling', type=str, default=cfg['gnn_pooling'])
    ap.add_argument('--batch_size', type=int, default=cfg['batch_size'])
    ap.add_argument(
        '--output_csv',
        type=str,
        default=argparse.SUPPRESS,
        help=(
            'Primary CSV path (default: timestamped under GENERATED or results/comparisons/). '
            'Also writes model_comparison_latest.csv under GENERATED or results/comparisons/.'
        ),
    )
    ap.add_argument(
        '--no-save',
        action='store_true',
        help='Do not write CSV files (stdout report only)',
    )
    ap.add_argument('--only_amp', action='store_true',
                    help='If set, save only peptides predicted as AMP (1) by at least one model')
    ap.add_argument(
        '--no-gnn-platt',
        action='store_true',
        help='Use softmax on GNN logits for P(AMP) even when *_platt.json exists next to checkpoints',
    )
    ap.add_argument(
        '--node-groups',
        type=str,
        default=None,
        dest='node_groups',
        help=(
            'GNN node blocks: comma tokens no_vae, no_onehot, no_pdb, no_esm2 '
            '(overrides configs/compare_models.json node_feature_groups).'
        ),
    )
    args = ap.parse_args(argv_rest)

    node_feature_groups = (
        node_feature_groups_from_cli(args.node_groups)
        if args.node_groups
        else node_feature_groups_from_config_value(cfg.get("node_feature_groups"))
    )
    _nfg = node_feature_groups if node_feature_groups is not None else NodeFeatureGroups()
    print(
        f"GNN node feature groups (config hint): onehot={_nfg.onehot} pdb_continuous={_nfg.pdb_continuous} "
        f"vae_table={_nfg.vae_table} esm2_residue={_nfg.esm2_residue} "
        f"(graph x width {node_input_dim(node_feature_groups)}); each checkpoint uses *_gnn_meta.json or "
        f"inferred weights.",
        flush=True,
    )

    user_passed_svm_pkl = hasattr(args, "svm_pkl")
    if not hasattr(args, "svm_descriptor_csv"):
        args.svm_descriptor_csv = cfg["svm_descriptor_csv"]
    if not hasattr(args, "svm_z_file"):
        args.svm_z_file = cfg["svm_z_file"]
    if not hasattr(args, "svm_pkl"):
        args.svm_pkl = cfg["svm_pkl"]

    ws, skip_workspace_checkpoint_roots = apply_pipeline_generated_workspace(
        args, skip_svm_clear=user_passed_svm_pkl
    )

    base_for_summary: Path | None = None
    if getattr(args, "checkpoints_base", None):
        base_for_summary = apply_checkpoints_base(args)

    ckpt_root = getattr(args, "gnn_checkpoints_dir", None)
    summary_roots: list[Path] = []
    if base_for_summary is not None:
        summary_roots.append(base_for_summary)
    if ckpt_root:
        gdir = Path(ckpt_root).expanduser().resolve()
        if not gdir.is_dir():
            raise SystemExit(f"--gnn-checkpoints-dir is not a directory: {gdir}")
        _set_gnn_paths_from_dir(args, gdir)
        summary_roots.append(gdir)
        print(f"GNN checkpoints directory (override): {gdir}", flush=True)
    if ws is not None and not skip_workspace_checkpoint_roots:
        summary_roots.append(ws)
    if summary_roots:
        _try_apply_ready_models_summary(args.architecture, args, *summary_roots)

    if getattr(args, "no_save", False):
        args.output_csv = ""
    elif not hasattr(args, "output_csv"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ws is not None:
            args.output_csv = str(ws / f"model_comparison_{ts}.csv")
        else:
            comp_dir = _ROOT / "results" / "comparisons"
            comp_dir.mkdir(parents=True, exist_ok=True)
            args.output_csv = str(comp_dir / f"model_comparison_{ts}.csv")

    geo_df = pd.read_csv(args.geo_csv)
    id_col = 'peptide_id' if 'peptide_id' in geo_df.columns else ('name' if 'name' in geo_df.columns else geo_df.columns[0])
    canonical_ids = geo_df[id_col].astype(str).tolist()
    n_samples = len(canonical_ids)
    has_sequence = "sequence" in geo_df.columns
    seq_by_id: dict[str, object] = {}
    if has_sequence:
        g = geo_df[[id_col, "sequence"]].drop_duplicates(subset=[id_col], keep="first")
        seq_by_id = dict(zip(g[id_col].astype(str).str.strip(), g["sequence"]))

    results = {}
    pred_frames = {}

    if args.svm_pkl and args.svm_descriptor_csv and args.svm_z_file:
        print("Running SVM...")
        ids, preds, prob_amp, distance = _load_svm_predictions(
            args.svm_descriptor_csv, args.svm_z_file, args.svm_pkl
        )
        results['SVM'] = {
            'ids': ids,
            'pred': preds,
            'prob_amp': prob_amp,
            'confidence': _confidence(prob_amp),
            'distance': distance,
        }
        pred_frames['SVM'] = pd.DataFrame({'pred': preds, 'prob_amp': prob_amp, 'confidence': results['SVM']['confidence']}, index=ids)

    feature_models = [
        # (display_name, checkpoint_path, feature_mode) — modes match run_gnn_train_final_models.FEATURE_CONFIGS
        ('ESM-only', args.esm_only_pt, 'esm'),
        ('ESM+Geo20', args.esm_geo_pt, 'geo20'),
        ('ESM+QSAR12', args.esm_qsar_pt, 'qsar12'),
        ('ESM+Combined32', args.esm_combined_pt, 'combined32'),
    ]

    gnn_master_path, mode_cols, _has_qsar_merge = _build_gnn_feature_master(args, ws)
    if gnn_master_path is None:
        print(
            "GNN: missing --geo_csv (or GENERATED manifest geometric_features). Skipping GNN models.",
            flush=True,
        )

    for name, path, feat_mode in feature_models:
        if not path or not Path(path).exists():
            continue
        if gnn_master_path is None:
            continue
        geom_cols = mode_cols.get(feat_mode, [])
        if feat_mode != "esm" and not geom_cols:
            if feat_mode in ('qsar12', 'combined32'):
                print(f"Skipping {name}: need merged QSAR-12 (check --qsar_csv)", flush=True)
            continue
        print(f"Running {args.architecture.upper()} ({name})...")
        ids, preds, prob_amp, logit_amp, logit_nonamp, logit_margin = _run_gnn_predictions(
            str(gnn_master_path), args.pdb_dir, path, args.architecture,
            args.gnn_hidden, args.gnn_layers, args.gnn_pooling,
            args.batch_size,
            geometric_feature_cols=geom_cols,
            esm2_residue_dir=getattr(args, "gnn_esm2_residue_dir", None),
            node_feature_groups=node_feature_groups,
            use_gnn_platt=not args.no_gnn_platt,
        )
        results[name] = {
            'ids': ids,
            'pred': preds,
            'prob_amp': prob_amp,
            'confidence': _confidence(prob_amp),
            'logit_amp': logit_amp,
            'logit_nonamp': logit_nonamp,
            'logit_margin': logit_margin,
        }
        pred_frames[name] = pd.DataFrame({'pred': preds, 'prob_amp': prob_amp, 'confidence': results[name]['confidence']}, index=ids)

    if not results:
        print(
            "No models run: no SVM (expected with GENERATED unless you pass --svm_pkl / --svm_z_file / "
            "--svm_descriptor_csv) and no GNN checkpoints found on disk.",
            flush=True,
        )
        print(
            f"Expected GNN weights for --architecture {args.architecture} (default gat), e.g.:",
            flush=True,
        )
        for label, path in (
            ("ESM-only", args.esm_only_pt),
            ("ESM+Geo20", args.esm_geo_pt),
            ("ESM+QSAR12", args.esm_qsar_pt),
            ("ESM+Combined32", args.esm_combined_pt),
        ):
            ex = Path(path).is_file()
            print(f"  {label}: {path}  {'OK' if ex else 'missing'}", flush=True)
        print(
            "Train with: python scripts/run_gnn_train_final_models.py <GENERATED> "
            "(writes gnn_ready_models/ and ready_models_summary.json), or pass "
            "--esm_only_pt / --esm_geo_pt / ... pointing to your .pt files. "
            "Match --architecture to the backbone you trained (gat vs egnn vs gcn).",
            flush=True,
        )
        return 1

    names = list(results.keys())
    ids_set = set(canonical_ids)
    for m in names:
        ids_set &= set(results[m]['ids'])
    ids_common = [i for i in canonical_ids if i in ids_set]
    _print_cli_report(args.architecture, results, canonical_ids, ids_common, pred_frames)

    score_z_by_model: dict[str, np.ndarray] = {}
    for m in names:
        raw = _raw_for_benchmark_z(m, results[m])
        z, _, _, _ = _zscore_aligned_to_ids(canonical_ids, results[m]["ids"], raw)
        score_z_by_model[m] = z
    _print_benchmark_z_summary(names, results, canonical_ids)

    out_rows = []
    for idx, pid in enumerate(canonical_ids):
        row = {'peptide_id': pid}
        if has_sequence:
            row["sequence"] = seq_by_id.get(str(pid).strip(), pd.NA)
        for m in names:
            r = results[m]
            if pid in r['ids']:
                i = r['ids'].index(pid)
                row[f'{m}_pred'] = int(r['pred'][i])
                row[f'{m}_confidence'] = float(r['confidence'][i])
                row[f'{m}_prob_AMP'] = float(r['prob_amp'][i])
                zv = score_z_by_model[m][idx]
                row[f'{m}_score_z'] = float(zv) if np.isfinite(zv) else None
                if m == 'SVM':
                    d = float(r['distance'][i]) if np.isfinite(r['distance'][i]) else None
                    row[f'{m}_hyperplane_distance'] = d
                    row[f'{m}_distance'] = d
                else:
                    row[f'{m}_logit_AMP'] = float(r['logit_amp'][i])
                    row[f'{m}_logit_nonAMP'] = float(r['logit_nonamp'][i])
                    row[f'{m}_logit_margin'] = float(r['logit_margin'][i])
            else:
                row[f'{m}_pred'] = None
                row[f'{m}_confidence'] = None
                row[f'{m}_prob_AMP'] = None
                row[f'{m}_score_z'] = None
                if m == 'SVM':
                    row[f'{m}_hyperplane_distance'] = None
                    row[f'{m}_distance'] = None
                else:
                    row[f'{m}_logit_AMP'] = None
                    row[f'{m}_logit_nonAMP'] = None
                    row[f'{m}_logit_margin'] = None
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    preferred = ["peptide_id"] + (["sequence"] if has_sequence else []) + _ordered_result_column_names(names)
    ordered = [c for c in preferred if c in out_df.columns]
    trailing = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + trailing]

    # Optional: keep only peptides predicted as AMP (1) by at least one model.
    if args.only_amp and not out_df.empty:
        amp_mask = False
        for m in names:
            col = f'{m}_pred'
            if col in out_df.columns:
                amp_mask = amp_mask | (out_df[col] == 1)
        out_df = out_df[amp_mask].reset_index(drop=True)
    if args.output_csv:
        primary = Path(args.output_csv)
        primary.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(primary, index=False)
        snap = _comparison_snapshot_csv_path(ws)
        snap.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(snap, index=False)
        # Metadata JSON alongside each output.
        _write_comparison_metadata_json(
            json_path=primary.with_suffix("").with_name(primary.stem + "_meta.json"),
            args=args,
            ws=ws,
            results=results,
            mode_cols=mode_cols,
            gnn_master_path=gnn_master_path,
            canonical_count=len(canonical_ids),
        )
        _write_comparison_metadata_json(
            json_path=snap.with_suffix("").with_name(snap.stem + "_meta.json"),
            args=args,
            ws=ws,
            results=results,
            mode_cols=mode_cols,
            gnn_master_path=gnn_master_path,
            canonical_count=len(canonical_ids),
        )
        print(f"\nSaved comparison CSV: {primary.resolve()}", flush=True)
        print(f"Latest snapshot CSV:   {snap.resolve()}", flush=True)
    elif getattr(args, "no_save", False):
        print("\nNo CSV written (--no-save).", flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())

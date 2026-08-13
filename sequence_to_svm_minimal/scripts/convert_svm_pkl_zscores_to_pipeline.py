#!/usr/bin/env python3
"""
Convert a legacy SVM checkpoint (SVC .pkl + zscores.txt) into a single sklearn
Pipeline(StandardScaler, SVC) joblib — no retraining.

Legacy layout (from run_svm_training.py):
  line 1: comma-separated feature names
  line 2: means
  line 3: stds (zeros already replaced with 1.0 at train time)

Usage (from sequence_to_svm_minimal/):
  python scripts/convert_svm_pkl_zscores_to_pipeline.py \\
    --svm_pkl results/checkpoints/svm_alpha_beta_combined/svm_qsar12_model.pkl \\
    --svm_z_file results/checkpoints/svm_alpha_beta_combined/svm_qsar12_zscores.txt

Defaults match configs/compare_models.json when those files exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import joblib
except ImportError:  # pragma: no cover
    from sklearn.externals import joblib  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _defaults_from_compare_config() -> tuple[Path | None, Path | None, Path | None]:
    """Return (pkl, z_file, out) from compare_models.json when present."""
    try:
        from configs.load_config import load_compare_models_config
    except Exception:
        return None, None, None
    cfg = load_compare_models_config()
    pkl = Path(cfg["svm_pkl"]) if cfg.get("svm_pkl") else None
    zf = Path(cfg["svm_z_file"]) if cfg.get("svm_z_file") else None
    out = None
    if pkl is not None:
        out = pkl.with_name(pkl.stem.replace("_model", "") + "_pipeline.joblib")
        if out == pkl.with_suffix(".joblib"):
            out = pkl.with_name("svm_qsar12_pipeline.joblib")
    return pkl, zf, out


def load_zscores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        names = [x.strip() for x in f.readline().strip().split(",") if x.strip()]
        means = np.array([float(x) for x in f.readline().strip().split(",")], dtype=np.float64)
        stds = np.array([float(x) for x in f.readline().strip().split(",")], dtype=np.float64)
    if not names:
        raise ValueError(f"No feature names in {path}")
    if means.shape[0] != len(names) or stds.shape[0] != len(names):
        raise ValueError(
            f"Length mismatch in {path}: {len(names)} names, "
            f"{means.shape[0]} means, {stds.shape[0]} stds"
        )
    stds = np.where(stds > 0, stds, 1.0)
    return names, means, stds


def build_scaler(names: list[str], means: np.ndarray, stds: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds ** 2
    scaler.n_features_in_ = int(len(names))
    scaler.n_samples_seen_ = np.int64(1)
    scaler.feature_names_in_ = np.asarray(names, dtype=object)
    return scaler


def convert(pkl_path: Path, z_path: Path, out_path: Path) -> Path:
    svm = joblib.load(pkl_path)
    if not isinstance(svm, SVC):
        raise TypeError(f"Expected sklearn.svm.SVC in {pkl_path}, got {type(svm)!r}")

    names, means, stds = load_zscores(z_path)
    n_in = getattr(svm, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(names):
        raise ValueError(
            f"SVM expects {n_in} features but z-score file lists {len(names)} "
            f"({z_path})"
        )

    pipe = Pipeline(
        [
            ("scaler", build_scaler(names, means, stds)),
            ("svm", svm),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return out_path


def main() -> None:
    cfg_pkl, cfg_z, cfg_out = _defaults_from_compare_config()
    root = _repo_root()

    ap = argparse.ArgumentParser(
        description="Convert SVM .pkl + zscores.txt into a single Pipeline joblib (no retrain)."
    )
    ap.add_argument(
        "--svm_pkl",
        type=str,
        default=str(cfg_pkl) if cfg_pkl else None,
        help="Legacy SVC pickle (default: configs/compare_models.json svm_pkl)",
    )
    ap.add_argument(
        "--svm_z_file",
        type=str,
        default=str(cfg_z) if cfg_z else None,
        help="Z-score text file (default: configs/compare_models.json svm_z_file)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(cfg_out) if cfg_out else str(root / "results" / "svm" / "svm_qsar12_pipeline.joblib"),
        help="Output Pipeline joblib path",
    )
    args = ap.parse_args()

    if not args.svm_pkl or not args.svm_z_file:
        raise SystemExit("Provide --svm_pkl and --svm_z_file (or set them in configs/compare_models.json).")

    pkl_path = Path(args.svm_pkl).expanduser()
    z_path = Path(args.svm_z_file).expanduser()
    out_path = Path(args.out).expanduser()

    if not pkl_path.is_file():
        raise SystemExit(f"SVM pickle not found: {pkl_path}")
    if not z_path.is_file():
        raise SystemExit(f"Z-score file not found: {z_path}")

    saved = convert(pkl_path, z_path, out_path)
    names, _, _ = load_zscores(z_path)
    print(f"Wrote Pipeline (StandardScaler + SVC): {saved.resolve()}")
    print(f"  features ({len(names)}): {', '.join(names[:8])}{'...' if len(names) > 8 else ''}")
    print("Inference: pipe.predict(X_raw) / predict_proba / decision_function on unscaled columns.")
    print("Note: compare_model_predictions.py still expects the legacy .pkl + zscores pair;")
    print("      update loaders separately if you want to consume this pipeline file.")


if __name__ == "__main__":
    main()

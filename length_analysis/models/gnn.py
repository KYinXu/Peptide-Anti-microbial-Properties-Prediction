"""GNN model adapter for length analysis (flat checkpoint layout)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..variants.common import VariantRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
SVM_MINIMAL_ROOT = REPO_ROOT / "sequence_to_svm_minimal"
GNN_MINIMAL_ROOT = REPO_ROOT / "gnn_minimal"
DEFAULT_CHECKPOINT_DIR = SVM_MINIMAL_ROOT / "checkpoints" / "latest"
DEFAULT_CHECKPOINT_NAME = "gat_ready_QSAR.pt"
PT_SUFFIXES = {".pt"}


@dataclass(frozen=True)
class GnnPrediction:
    pred: int
    prob_AMP: float
    confidence: float
    logit_AMP: float
    logit_nonAMP: float
    logit_margin: float
    score_z: float


class GnnScorerFactory:
    def __init__(
        self,
        checkpoint: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        skip_if_exists: bool = True,
        force_process: bool = False,
    ) -> None:
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.device = device
        self.skip_if_exists = skip_if_exists
        self.force_process = force_process

    def __call__(self) -> "GnnScorer":
        return GnnScorer(
            self.checkpoint,
            batch_size=self.batch_size,
            device=self.device,
            skip_if_exists=self.skip_if_exists,
            force_process=self.force_process,
        )


class GnnScorer:
    def __init__(
        self,
        checkpoint: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        skip_if_exists: bool = True,
        force_process: bool = False,
    ) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.batch_size = batch_size
        self.device = device
        self.skip_if_exists = skip_if_exists
        self.force_process = force_process

    def score_variants(
        self,
        variants: Iterable[VariantRecord],
        workspace: Path,
    ) -> dict[str, GnnPrediction]:
        variant_list = list(variants)
        if not variant_list:
            return {}

        _ensure_import_paths()
        from gnn.checkpoint_meta import load_peptide_gnn_meta
        from gnn.models import esm2_raw_dim_from_state_dict
        import torch

        input_csv = workspace / "variants_input.csv"
        gen_dir = workspace / "generated"
        _write_variants_csv(variant_list, input_csv)

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        need_esm2 = esm2_raw_dim_from_state_dict(state) > 0
        feature_cols = _resolve_feature_columns(self.checkpoint)

        from process_data import ensure_processed_data

        process_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        force = self.force_process or not self.skip_if_exists
        ensure_processed_data(
            input_csv,
            gen_dir,
            need_esm2=need_esm2,
            require_labels=False,
            device=process_device,
            force=force,
        )

        master_csv = _build_master_csv(gen_dir, feature_cols)
        pdb_dir = gen_dir / "structures"
        esm2_dir = gen_dir / "esm2_per_residue" if need_esm2 else None

        meta = load_peptide_gnn_meta(self.checkpoint) or {}
        architecture = str(meta.get("architecture", "gat"))
        hidden = int(meta.get("hidden_channels", 64))
        num_layers = int(meta.get("num_layers", 3))
        pooling = str(meta.get("pooling", "mean_max"))

        sys.path.insert(0, str(SVM_MINIMAL_ROOT / "scripts" / "data_evaluation"))
        import compare_model_predictions as inference  # noqa: WPS433

        ids, pred, probability, logit_amp, logit_nonamp, margin = inference._run_gnn_predictions(
            str(master_csv),
            str(pdb_dir),
            str(self.checkpoint),
            architecture,
            hidden,
            num_layers,
            pooling,
            self.batch_size,
            geometric_feature_cols=feature_cols,
            esm2_residue_dir=str(esm2_dir) if esm2_dir else None,
            use_gnn_platt=True,
            progress_csv=None,
            resume_progress=False,
            checkpoint_every=10_000,
            num_workers=0,
        )

        margins = np.asarray(margin, dtype=float)
        finite = margins[np.isfinite(margins)]
        if finite.size == 0:
            z_scores = np.full(len(margins), np.nan)
        else:
            std = float(finite.std(ddof=0))
            mean = float(finite.mean())
            z_scores = (
                (margins - mean) / std
                if std > 0
                else np.where(np.isfinite(margins), 0.0, np.nan)
            )

        results: dict[str, GnnPrediction] = {}
        for index, peptide_id in enumerate(ids):
            prob = float(probability[index])
            results[str(peptide_id).strip()] = GnnPrediction(
                pred=int(pred[index]),
                prob_AMP=prob,
                confidence=float(max(prob, 1.0 - prob)),
                logit_AMP=float(logit_amp[index]),
                logit_nonAMP=float(logit_nonamp[index]),
                logit_margin=float(margin[index]),
                score_z=float(z_scores[index]) if np.isfinite(z_scores[index]) else float("nan"),
            )
        missing = [variant.variant_id for variant in variant_list if variant.variant_id not in results]
        if missing:
            raise RuntimeError(
                f"GNN inference missing predictions for {len(missing)} variant(s); "
                f"first missing: {missing[0]}"
            )
        return results


def resolve_gnn_checkpoint(
    *,
    checkpoint: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> Path:
    if checkpoint is not None:
        path = checkpoint.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GNN checkpoint not found: {path}")
        return path.resolve()

    directory = (checkpoint_dir or DEFAULT_CHECKPOINT_DIR).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {directory}")

    preferred = directory / DEFAULT_CHECKPOINT_NAME
    if preferred.is_file():
        return preferred.resolve()

    candidates = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in PT_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(f"No GNN .pt checkpoint found in {directory}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple GNN checkpoints in {directory}: {names}. "
            f"Pass --gnn-checkpoint (default preference: {DEFAULT_CHECKPOINT_NAME})."
        )
    return candidates[0].resolve()


def _ensure_import_paths() -> None:
    for root in (GNN_MINIMAL_ROOT, SVM_MINIMAL_ROOT):
        text = str(root)
        if text not in sys.path:
            sys.path.insert(0, text)


def _write_variants_csv(variants: list[VariantRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "id": [variant.variant_id for variant in variants],
            "sequence": [variant.sequence for variant in variants],
        }
    )
    frame.to_csv(path, index=False)


def _resolve_feature_columns(checkpoint: Path) -> list[str]:
    _ensure_import_paths()
    from gnn.extra_feature_scaler import load_extra_feature_scaler

    scaler_path = checkpoint.with_name(checkpoint.stem + "_tabular_scaler.joblib")
    if scaler_path.is_file():
        return list(load_extra_feature_scaler(str(scaler_path)).feature_cols)

    meta_path = checkpoint.with_name(checkpoint.stem + "_gnn_meta.json")
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("geo_feature_dim", 0)) == 0:
            return []
    return []


def _build_master_csv(gen_dir: Path, feature_cols: list[str]) -> Path:
    geo_csv = gen_dir / "geometric_features.csv"
    if not geo_csv.is_file():
        raise FileNotFoundError(f"Missing geometric features: {geo_csv}")
    frame = pd.read_csv(geo_csv)
    if "peptide_id" not in frame.columns:
        raise ValueError(f"{geo_csv} missing peptide_id column")

    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        qsar_csv = gen_dir / "qsar12_descriptors.csv"
        if not qsar_csv.is_file():
            raise FileNotFoundError(
                f"Checkpoint needs tabular columns {missing} but QSAR file is missing: {qsar_csv}"
            )
        qsar = pd.read_csv(qsar_csv)
        keep = ["peptide_id"] + [column for column in missing if column in qsar.columns]
        frame = frame.merge(qsar[keep], on="peptide_id", how="left")
    remaining = [column for column in feature_cols if column not in frame.columns]
    if remaining:
        raise ValueError(f"Processed features missing tabular columns: {remaining}")

    master_csv = gen_dir / "compare_geo_qsar_merged.csv"
    frame.to_csv(master_csv, index=False)
    return master_csv

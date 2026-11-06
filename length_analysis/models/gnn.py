"""GNN model adapter for length analysis, backed by gnn_minimal (not the legacy tree).

Each variant has its own sequence, so each is folded and scored against its own PDB.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_MINIMAL_ROOT = REPO_ROOT / "gnn_minimal"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "gnn" / "gnn_vae_qsar"
DEFAULT_CHECKPOINT_NAME = "gnn_model.pt"
PT_SUFFIXES = {".pt"}


def ensure_gnn_minimal_on_path() -> None:
    path = str(GNN_MINIMAL_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


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
        fitting: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.device = device
        self.skip_if_exists = skip_if_exists
        self.force_process = force_process
        self.fitting = fitting

    def __call__(self) -> "GnnScorer":
        return GnnScorer(
            self.checkpoint,
            batch_size=self.batch_size,
            device=self.device,
            skip_if_exists=self.skip_if_exists,
            force_process=self.force_process,
            fitting=self.fitting,
        )


class GnnScorer:
    """Loads the checkpoint once; folds variants and scores them against their PDBs."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        skip_if_exists: bool = True,
        force_process: bool = False,
        fitting: str | None = None,
    ) -> None:
        ensure_gnn_minimal_on_path()
        import torch
        from core.calibration import describe_calibration, resolve_calibration
        from core.checkpoint_meta import load_peptide_gnn_meta, resolve_node_layout_for_checkpoint
        from core.determinism import enable_deterministic_inference
        from core.models import (
            PeptideGNN,
            esm2_hidden_dim_from_state_dict,
            esm2_raw_dim_from_state_dict,
        )
        from inference import resolve_tabular_contract

        enable_deterministic_inference()
        self.checkpoint = checkpoint
        model_dir = str(checkpoint.parent)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)

        esm2_raw = esm2_raw_dim_from_state_dict(state)
        if esm2_raw > 0:
            raise ValueError(
                "Length analysis supports checkpoints without per-residue ESM2 features; "
                f"{checkpoint} expects {esm2_raw}-dim ESM2."
            )

        meta = load_peptide_gnn_meta(model_dir) or {}
        self.meta = meta
        self.scaler_path, self.feature_cols = resolve_tabular_contract(model_dir, meta)
        self.node_groups, self.in_channels, _notes = resolve_node_layout_for_checkpoint(
            model_dir, state, meta.get("architecture", "gat"), user_node_groups=None
        )
        self.geo_dim = int(meta.get("geo_feature_dim", len(self.feature_cols)))

        self.model = PeptideGNN(
            architecture=meta.get("architecture", "gat"),
            in_channels=self.in_channels,
            hidden_channels=meta.get("hidden_channels", 64),
            num_layers=meta.get("num_layers", 3),
            num_classes=2,
            pooling=meta.get("pooling", "mean_max"),
            geo_feature_dim=self.geo_dim,
            esm2_raw_dim=0,
            esm2_hidden_dim=esm2_hidden_dim_from_state_dict(state),
        )
        self.model.load_state_dict(state)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device).eval()
        self.scaler = None
        if self.scaler_path:
            from core.extra_feature_scaler import load_extra_feature_scaler

            self.scaler = load_extra_feature_scaler(str(self.scaler_path))
        self.fitting, self.calibration = resolve_calibration(model_dir, fitting)
        self.calibration_description = describe_calibration(self.fitting, self.calibration)
        self.batch_size = batch_size
        self.process_device = device
        self.skip_if_exists = skip_if_exists
        self.force_process = force_process

    def score_variants(
        self, variants: Iterable, workspace: Path, progress=None
    ) -> dict[str, GnnPrediction]:
        variant_list = list(variants)
        if not variant_list:
            return {}

        ensure_gnn_minimal_on_path()
        import torch
        from torch_geometric.loader import DataLoader
        from core.data_utils import PeptideGraphDataset
        from pipeline.qsar import compute_qsar12
        from process_data import ensure_processed_data

        ids = [variant.variant_id for variant in variant_list]
        sequences = [variant.sequence for variant in variant_list]

        input_csv = workspace / "variants_input.csv"
        gen_dir = workspace / "generated"
        input_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": ids, "sequence": sequences}).to_csv(input_csv, index=False)

        process_device = self.process_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if progress is not None:
            progress("folding", 0, len(ids))
        ensure_processed_data(
            input_csv,
            gen_dir,
            need_esm2=False,
            require_labels=False,
            device=process_device,
            force=self.force_process or not self.skip_if_exists,
        )

        if progress is not None:
            progress("descriptors", 0, len(ids))
        frame = compute_qsar12(sequences, ids)
        frame["pdb_file"] = [f"{identifier}.pdb" for identifier in ids]
        missing = [column for column in self.feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"QSAR output is missing checkpoint features: {missing}")

        dataset = PeptideGraphDataset(
            csv_path="",
            pdb_dir=str(gen_dir / "structures"),
            use_geometric_features=bool(self.feature_cols),
            geometric_feature_cols=self.feature_cols,
            tabular_scaler_path=str(self.scaler_path) if self.scaler_path else None,
            node_feature_groups=self.node_groups,
            dataframe=frame,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return self._predict(ids, loader, progress=progress)

    def _predict(self, ids: list[str], loader, progress=None) -> dict[str, GnnPrediction]:
        import torch
        from core.calibration import prob_amp

        scored = 0
        logits = []
        with torch.no_grad():
            for batch in loader:
                logits.append(self.model(batch.to(self.device)).cpu().numpy())
                if progress is not None:
                    scored += int(batch.num_graphs)
                    progress("scoring", scored, len(ids))
        logits = np.concatenate(logits, axis=0)
        if len(logits) != len(ids):
            raise RuntimeError(
                f"GNN inference returned {len(logits)} prediction(s) for {len(ids)} variant(s)."
            )

        margins = logits[:, 1] - logits[:, 0]
        probabilities = prob_amp(margins, self.fitting, self.calibration)

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
        for index, identifier in enumerate(ids):
            probability = float(probabilities[index])
            results[str(identifier).strip()] = GnnPrediction(
                pred=int(probability >= 0.5),
                prob_AMP=probability,
                confidence=float(max(probability, 1.0 - probability)),
                logit_AMP=float(logits[index][1]),
                logit_nonAMP=float(logits[index][0]),
                logit_margin=float(margins[index]),
                score_z=float(z_scores[index]) if np.isfinite(z_scores[index]) else float("nan"),
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

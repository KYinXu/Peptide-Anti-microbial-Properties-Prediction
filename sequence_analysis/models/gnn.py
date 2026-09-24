"""GNN sequence inference using a gnn_minimal model directory."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.data_loader import SequenceRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_MINIMAL_ROOT = REPO_ROOT / "gnn_minimal"
DEFAULT_GNN_ROOT = REPO_ROOT / "checkpoints" / "gnn" / "gnn_vae_qsar"


@dataclass(frozen=True)
class GnnSequencePrediction:
    id: str
    sequence: str
    prediction: int
    sigma: float
    p_amp: float


class GnnSequenceScorer:
    """Score full sequences with a trained peptide GNN.

    Structures, QSAR-12, and graph features are built beside ``source_csv``
    in ``generated/``, using the same non-nulled QSAR path as ``gnn_minimal``.
    ``sigma`` is the logit margin (AMP minus non-AMP). ``prediction`` is +1
    when calibrated P(AMP) is at least 0.5.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        force_process: bool = False,
    ) -> None:
        self.model_dir = model_dir.resolve()
        self.batch_size = batch_size
        self.device = device
        self.force_process = force_process
        self.meta, self.scaler_path, self.feature_cols = load_model_contract(self.model_dir)

    @classmethod
    def from_paths(
        cls,
        model_dir: Path,
        *,
        batch_size: int = 32,
        device: str | None = None,
        force_process: bool = False,
    ) -> "GnnSequenceScorer":
        return cls(
            model_dir,
            batch_size=batch_size,
            device=device,
            force_process=force_process,
        )

    def score(self, records: list[SequenceRecord], source_csv: Path) -> list[GnnSequencePrediction]:
        predictions, _ = self.score_with_descriptors(records, source_csv)
        return predictions

    def score_with_descriptors(
        self,
        records: list[SequenceRecord],
        source_csv: Path,
    ) -> tuple[list[GnnSequencePrediction], np.ndarray]:
        if not records:
            return [], np.empty((0, len(self.feature_cols)), dtype=np.float64)
        frame = processed_feature_frame(source_csv, self.model_dir, self.device, self.force_process)
        scored = score_feature_frame(
            frame,
            self.model_dir,
            self.meta,
            self.scaler_path,
            self.feature_cols,
            batch_size=self.batch_size,
            device=self.device,
        )
        return align_predictions(records, scored, self.feature_cols)


def resolve_gnn_model_dir(model_path: Path | None = None, checkpoint_dir: Path | None = None) -> Path:
    """Accept a model directory, ``gnn_model.pt``, or a folder that contains one checkpoint."""
    _ensure_gnn_import_path()
    from core.artifacts import resolve_model_dir

    if model_path is not None:
        return resolve_model_dir(model_path)
    directory = (checkpoint_dir or DEFAULT_GNN_ROOT).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"GNN checkpoint directory not found: {directory}. Pass --gnn-model."
        )
    if (directory / "gnn_model.pt").is_file() or (directory / "model_summary.json").is_file():
        return resolve_model_dir(directory)
    matches = sorted(path.parent for path in directory.rglob("gnn_model.pt") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No gnn_model.pt checkpoint found under {directory}")
    if len(matches) > 1:
        listed = ", ".join(str(path) for path in matches)
        raise ValueError(f"Multiple GNN checkpoints under {directory}: {listed}. Pass --gnn-model.")
    return resolve_model_dir(matches[0])


def load_model_contract(model_dir: Path) -> tuple[dict, Path | None, list[str]]:
    _ensure_gnn_import_path()
    from core.artifacts import model_checkpoint_path, model_summary_path
    from core.checkpoint_meta import load_peptide_gnn_meta
    from inference import resolve_tabular_contract

    checkpoint = model_checkpoint_path(model_dir)
    summary = model_summary_path(model_dir)
    if not checkpoint.is_file() or not summary.is_file():
        raise FileNotFoundError(
            f"GNN model directory must contain gnn_model.pt and model_summary.json: {model_dir}"
        )
    meta = load_peptide_gnn_meta(model_dir) or {}
    scaler_path, feature_cols = resolve_tabular_contract(model_dir, meta)
    return meta, scaler_path, list(feature_cols)


def processed_feature_frame(source_csv: Path, model_dir: Path, device: str | None, force: bool):
    _ensure_gnn_import_path()
    import torch
    from core.artifacts import model_checkpoint_path
    from core.models import esm2_raw_dim_from_state_dict
    from inference import load_processed_features
    from process_data import ensure_processed_data

    checkpoint = model_checkpoint_path(model_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    process_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generated = ensure_processed_data(
        source_csv,
        source_csv.parent / "generated",
        need_esm2=esm2_raw_dim_from_state_dict(state) > 0,
        require_labels=False,
        device=process_device,
        force=force,
    )
    _, _, feature_cols = load_model_contract(model_dir)
    frame = load_processed_features(
        generated / "geometric_features.csv",
        generated / "qsar12_descriptors.csv",
        feature_cols,
    )
    frame.attrs["generated_dir"] = str(generated)
    return frame


def score_feature_frame(
    frame,
    model_dir: Path,
    meta: dict,
    scaler_path: Path | None,
    feature_cols: list[str],
    *,
    batch_size: int,
    device: str | None,
):
    _ensure_gnn_import_path()
    import torch
    from torch_geometric.loader import DataLoader

    from core.artifacts import model_checkpoint_path, platt_scaling_path
    from core.checkpoint_meta import resolve_node_layout_for_checkpoint
    from core.data_utils import PeptideGraphDataset
    from core.models import esm2_raw_dim_from_state_dict
    from core.platt import load_platt_json

    generated = Path(frame.attrs["generated_dir"])
    checkpoint = model_checkpoint_path(model_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(state)
    architecture = str(meta.get("architecture", "gat"))
    node_groups, in_channels, _notes = resolve_node_layout_for_checkpoint(
        model_dir, state, architecture, user_node_groups=None
    )
    dataset = PeptideGraphDataset(
        csv_path=str(generated / "geometric_features.csv"),
        pdb_dir=str(generated / "structures"),
        use_geometric_features=bool(feature_cols),
        geometric_feature_cols=feature_cols,
        tabular_scaler_path=str(scaler_path) if scaler_path else None,
        esm2_residue_dir=str(generated / "esm2_per_residue") if esm2_raw > 0 else None,
        node_feature_groups=node_groups,
        dataframe=frame,
    )
    _check_widths(dataset, meta, in_channels)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(meta, state, architecture, in_channels, esm2_raw, feature_cols)
    model.load_state_dict(state)
    model = model.to(torch_device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities, margins = _collect_scores(
        model, loader, torch_device, load_platt_json(platt_scaling_path(model_dir))
    )
    scored = frame.copy()
    scored["peptide_id"] = scored["peptide_id"].astype(str)
    scored["_p_amp"] = probabilities
    scored["_sigma"] = margins
    return scored


def _load_model(meta, state, architecture: str, in_channels: int, esm2_raw: int, feature_cols: list[str]):
    from core.models import PeptideGNN, esm2_hidden_dim_from_state_dict

    return PeptideGNN(
        architecture=architecture,
        in_channels=in_channels,
        hidden_channels=int(meta.get("hidden_channels", 64)),
        num_layers=int(meta.get("num_layers", 3)),
        num_classes=int(meta.get("num_classes", 2)),
        pooling=str(meta.get("pooling", "mean_max")),
        geo_feature_dim=int(meta.get("geo_feature_dim", len(feature_cols))),
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=esm2_hidden_dim_from_state_dict(state),
    )


def align_predictions(
    records: list[SequenceRecord],
    scored,
    feature_cols: list[str],
) -> tuple[list[GnnSequencePrediction], np.ndarray]:
    indexed = scored.set_index(scored["peptide_id"].astype(str), drop=False)
    missing = [record.id for record in records if record.id not in indexed.index]
    if missing:
        raise RuntimeError(
            f"GNN inference missing predictions for {len(missing)} sequence(s); first missing: {missing[0]}"
        )
    predictions = []
    descriptor_rows = []
    for record in records:
        row = indexed.loc[record.id]
        probability = float(row["_p_amp"])
        predictions.append(
            GnnSequencePrediction(
                id=record.id,
                sequence=record.sequence,
                prediction=1 if probability >= 0.5 else -1,
                sigma=float(row["_sigma"]),
                p_amp=probability,
            )
        )
        descriptor_rows.append([float(row[column]) for column in feature_cols])
    return predictions, np.asarray(descriptor_rows, dtype=np.float64)


def _collect_scores(model, loader, device, platt_payload) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from core.platt import platt_prob_amp

    model.eval()
    probabilities = []
    margins = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            margin = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
            if platt_payload is not None:
                probability = platt_prob_amp(margin, platt_payload["coef"], platt_payload["intercept"])
            else:
                probability = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            margins.append(margin)
            probabilities.append(np.asarray(probability, dtype=np.float64))
    if not probabilities:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(probabilities), np.concatenate(margins)


def _check_widths(dataset, meta: dict, in_channels: int) -> None:
    if len(dataset) == 0:
        return
    graph = dataset[0]
    geo_dim = int(graph.geo_features.shape[1]) if hasattr(graph, "geo_features") else 0
    expected_geo = int(meta.get("geo_feature_dim", geo_dim))
    if geo_dim != expected_geo:
        raise ValueError(f"Tabular width mismatch: checkpoint expects {expected_geo}, dataset produced {geo_dim}")
    if int(graph.x.shape[1]) != in_channels:
        raise ValueError(
            f"Node width mismatch: checkpoint expects {in_channels}, dataset produced {int(graph.x.shape[1])}"
        )


def _ensure_gnn_import_path() -> None:
    root = str(GNN_MINIMAL_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

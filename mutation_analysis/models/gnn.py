"""GNN scorer for mutation variants, backed by gnn_minimal (not the legacy tree)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GNN_MINIMAL_ROOT = REPO_ROOT / "gnn_minimal"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "gnn" / "gnn_vae_qsar"


def ensure_gnn_minimal_on_path() -> None:
    path = str(GNN_MINIMAL_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass(frozen=True)
class Backbone:
    """A parent's fixed Ca frame, with its contact graph computed once.

    ``edges`` depends only on ``ca_coords``, so all mutants of a parent reuse it.
    """

    ca_coords: "np.ndarray"
    plddt: "np.ndarray"
    edges: tuple
    sequence: str


@dataclass(frozen=True)
class Prediction:
    pred: int
    prob_AMP: float
    logit_AMP: float
    logit_nonAMP: float
    logit_margin: float


class MutationScorer:
    """Loads the checkpoint once; scores batches of (id, sequence) against PDBs."""

    def __init__(
        self,
        checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
        *,
        batch_size: int = 32,
        device: str | None = None,
        fitting: str | None = None,
    ) -> None:
        ensure_gnn_minimal_on_path()
        import torch
        from core.artifacts import model_checkpoint_path, resolve_model_dir
        from core.calibration import describe_calibration, resolve_calibration
        from core.checkpoint_meta import load_peptide_gnn_meta, resolve_node_layout_for_checkpoint
        from core.determinism import enable_deterministic_inference
        from core.models import (
            PeptideGNN,
            esm2_hidden_dim_from_state_dict,
            esm2_raw_dim_from_state_dict,
        )

        sys.path.insert(0, str(GNN_MINIMAL_ROOT))
        from inference import resolve_tabular_contract

        enable_deterministic_inference()
        model_dir = resolve_model_dir(str(checkpoint_dir))
        state = torch.load(model_checkpoint_path(model_dir), map_location="cpu", weights_only=True)

        esm2_raw = esm2_raw_dim_from_state_dict(state)
        if esm2_raw > 0:
            raise ValueError(
                "Mutation analysis supports checkpoints without per-residue ESM2 features; "
                f"{model_dir} expects {esm2_raw}-dim ESM2."
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

    def score(
        self, ids: list[str], sequences: list[str], pdb_dir: Path, progress=None
    ) -> list[Prediction]:
        """Score sequences each against their own PDB in ``pdb_dir`` (refold controls)."""
        from torch_geometric.loader import DataLoader
        from core.data_utils import PeptideGraphDataset
        from pipeline.qsar import compute_qsar12

        if progress is not None:
            progress("descriptors", 0, len(ids))
        frame = compute_qsar12(sequences, ids)
        frame["pdb_file"] = [f"{identifier}.pdb" for identifier in ids]
        missing = [column for column in self.feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"QSAR output is missing checkpoint features: {missing}")

        dataset = PeptideGraphDataset(
            csv_path="",
            pdb_dir=str(pdb_dir),
            use_geometric_features=bool(self.feature_cols),
            geometric_feature_cols=self.feature_cols,
            tabular_scaler_path=str(self.scaler_path) if self.scaler_path else None,
            node_feature_groups=self.node_groups,
            dataframe=frame,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return self._predict(loader, progress=progress)

    def load_backbone(self, pdb_path: Path) -> Backbone:
        """Parse a parent PDB once and precompute its contact graph."""
        from core.data_utils import compute_edges, parse_pdb

        aa_sequence, ca_coords, plddt = parse_pdb(str(pdb_path))
        return Backbone(
            ca_coords=ca_coords,
            plddt=plddt,
            edges=compute_edges(ca_coords, 8.0),
            sequence="".join(aa_sequence),
        )

    def score_on_backbone(
        self, ids: list[str], sequences: list[str], backbone: Backbone, progress=None
    ) -> list[Prediction]:
        """Score sequences threaded onto a shared fixed backbone, with no PDB files."""
        from torch_geometric.loader import DataLoader
        from core.data_utils import graph_from_backbone
        from pipeline.qsar import compute_qsar12

        if progress is not None:
            progress("descriptors", 0, len(ids))
        frame = compute_qsar12(sequences, ids)
        missing = [column for column in self.feature_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"QSAR output is missing checkpoint features: {missing}")

        descriptors = None
        if self.feature_cols:
            descriptors = np.nan_to_num(
                frame[self.feature_cols].to_numpy(dtype=np.float64), nan=0.0
            )

        graphs = []
        if progress is not None:
            progress("graphs", 0, len(ids))
        for index, (identifier, sequence) in enumerate(zip(ids, sequences)):
            geometric = None
            if descriptors is not None:
                row = descriptors[index]
                if self.scaler is not None:
                    row = self.scaler.transform_vector(row)
                geometric = row.astype(np.float32)
            graphs.append(
                graph_from_backbone(
                    sequence,
                    backbone.ca_coords,
                    backbone.plddt,
                    peptide_id=identifier,
                    geometric_features=geometric,
                    node_feature_groups=self.node_groups,
                    edges=backbone.edges,
                )
            )
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        return self._predict(loader, progress=progress)

    def _predict(self, loader, progress=None) -> list[Prediction]:
        import torch
        from core.calibration import prob_amp

        total = len(loader.dataset)
        scored = 0
        logits = []
        with torch.no_grad():
            for batch in loader:
                logits.append(self.model(batch.to(self.device)).cpu().numpy())
                if progress is not None:
                    scored += int(batch.num_graphs)
                    progress("scoring", scored, total)
        logits = np.concatenate(logits, axis=0)

        margins = logits[:, 1] - logits[:, 0]
        probs = prob_amp(margins, self.fitting, self.calibration)

        return [
            Prediction(
                pred=int(prob >= 0.5),
                prob_AMP=float(prob),
                logit_AMP=float(row[1]),
                logit_nonAMP=float(row[0]),
                logit_margin=float(margin),
            )
            for prob, margin, row in zip(probs, margins, logits)
        ]


def fold_parents(parents: pd.DataFrame, generated_dir: Path, device: str | None) -> Path:
    """Run the standard pipeline on parent sequences only, producing parent structures."""
    ensure_gnn_minimal_on_path()
    import torch
    from process_data import ensure_processed_data

    generated_dir.parent.mkdir(parents=True, exist_ok=True)
    parents_csv = generated_dir.parent / "parents.csv"
    parents.to_csv(parents_csv, index=False)
    resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ensure_processed_data(
        parents_csv,
        generated_dir,
        need_esm2=False,
        require_labels=False,
        device=resolved,
        force=False,
    )
    return generated_dir / "structures"


def structure_problems(directory: Path, parents) -> list[str]:
    """Report parents whose PDB is missing, unreadable, or holds a different sequence."""
    if not directory.is_dir():
        return [f"{directory} is not a directory"]
    ensure_gnn_minimal_on_path()
    from core.data_utils import parse_pdb

    problems = []
    for parent in parents:
        path = directory / f"{parent.id}.pdb"
        if not path.is_file():
            problems.append(f"{parent.id}: no PDB")
            continue
        try:
            aa_sequence, _coords, _plddt = parse_pdb(str(path))
        except Exception as error:
            problems.append(f"{parent.id}: unreadable ({error})")
            continue
        if "".join(aa_sequence) != parent.sequence:
            problems.append(f"{parent.id}: sequence mismatch")
    return problems


def resolve_parent_structures(
    parents,
    *,
    workspace: Path,
    device: str | None,
    explicit: Path | None = None,
    input_dir: Path | None = None,
) -> Path:
    """Reuse parent structures when they match the input, otherwise fold them.

    Folding loads ESMFold, which dominates peak memory; reusing a prepared
    ``generated/structures`` directory avoids that entirely. A candidate is accepted
    only when every parent has a PDB whose residues equal its input sequence, so a
    stale structure can never be scored against a different peptide.
    """
    candidate = explicit
    if candidate is None and input_dir is not None:
        candidate = input_dir / "generated" / "structures"

    if candidate is not None:
        problems = structure_problems(candidate, parents)
        if not problems:
            print(f"Parent structures: reusing {len(parents)} from {candidate}", flush=True)
            return candidate
        if explicit is not None:
            raise ValueError(
                f"--parent-structures {candidate} is unusable for "
                f"{len(problems)} of {len(parents)} parent(s): {problems[:3]}"
            )
        print(
            f"Parent structures: not reusing {candidate} "
            f"({len(problems)} of {len(parents)} parent(s) missing or mismatched, "
            f"e.g. {problems[0]}); folding instead.",
            flush=True,
        )

    return fold_parents(
        pd.DataFrame({"id": [p.id for p in parents], "sequence": [p.sequence for p in parents]}),
        workspace / "parents_generated",
        device,
    )

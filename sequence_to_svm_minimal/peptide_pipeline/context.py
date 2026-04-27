"""Resolved paths and manifest for one pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from peptide_pipeline.constants import REPO_ROOT
from peptide_pipeline.config import RunConfig, default_work_dir


@dataclass
class RunContext:
    root: Path
    input_path: Path
    mode: str
    amp_input_path: Path | None
    decoy_input_path: Path | None
    work_dir: Path
    inputs_dir: Path
    canonical: Path
    canonical_amp: Path | None
    canonical_decoy: Path | None
    structures_dir: Path
    geo_csv: Path
    geo_clustered: Path
    qsar_csv: Path
    esm2_csv: Path
    py: str
    esmfold_script: Path
    build_geo_script: Path
    gen_qsar_script: Path
    esm2_script: Path
    prepare_clusters_script: Path
    run_svm_script: Path
    legacy_train_script: Path
    final_train_script: Path
    compare_script: Path
    manifest: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: RunConfig, *, py_executable: str) -> RunContext:
        inp = cfg.input_path.resolve()
        work = cfg.work_dir if cfg.work_dir is not None else default_work_dir(inp)
        work = Path(work).resolve()
        inputs_dir = work / "inputs"
        canonical = inputs_dir / "canonical_seqs.txt"
        canonical_amp = inputs_dir / "canonical_amp_seqs.txt" if cfg.is_train_mode() else None
        canonical_decoy = inputs_dir / "canonical_decoy_seqs.txt" if cfg.is_train_mode() else None
        root = REPO_ROOT
        manifest: dict = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": cfg.mode,
            "input": str(inp),
            "amp_input": str(cfg.amp_input_path.resolve()) if cfg.amp_input_path else None,
            "decoy_input": str(cfg.decoy_input_path.resolve()) if cfg.decoy_input_path else None,
            "work_dir": str(work),
            "steps": [],
        }
        return cls(
            root=root,
            input_path=inp,
            mode=cfg.mode,
            amp_input_path=cfg.amp_input_path.resolve() if cfg.amp_input_path else None,
            decoy_input_path=cfg.decoy_input_path.resolve() if cfg.decoy_input_path else None,
            work_dir=work,
            inputs_dir=inputs_dir,
            canonical=canonical,
            canonical_amp=canonical_amp,
            canonical_decoy=canonical_decoy,
            structures_dir=work / "structures",
            geo_csv=work / "geometric_features.csv",
            geo_clustered=work / "geometric_features_clustered.csv",
            qsar_csv=work / "qsar12_descriptors.csv",
            esm2_csv=work / "esm2_embeddings.csv",
            py=py_executable,
            esmfold_script=root / "models" / "run_esmfold_peptides.py",
            build_geo_script=root / "scripts" / "data_generation" / "build_geometric_features.py",
            gen_qsar_script=root / "scripts" / "data_generation" / "generate_qsar_features.py",
            esm2_script=root / "models" / "esm_sequence_processor.py",
            prepare_clusters_script=root / "nn_pipeline" / "prepare_clusters.py",
            run_svm_script=root / "scripts" / "data_generation" / "run_sequence_svm.py",
            legacy_train_script=root / "scripts" / "run_gnn_training.py",
            final_train_script=root / "scripts" / "run_gnn_train_final_models.py",
            compare_script=root / "scripts" / "data_evaluation" / "compare_model_predictions.py",
            manifest=manifest,
        )

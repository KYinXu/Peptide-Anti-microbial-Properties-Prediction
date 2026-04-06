"""Single configuration object for the sequence-to-features pipeline."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


def default_work_dir(input_path: Path) -> Path:
    return Path(input_path).resolve().parent / "generated"


@dataclass
class RunConfig:
    input_path: Path
    work_dir: Path | None = None
    dry_run: bool = False
    skip_if_exists: bool = False
    min_len: int | None = None
    max_len: int | None = None
    reset_esmfold: bool = False
    esmfold_max_length: int = 200
    esmfold_device: str | None = None
    skip_qsar: bool = False
    skip_esm2: bool = False
    with_cluster: bool = False
    cluster_simple_identity: float = 0.80
    cluster_run_cdhit: bool = False
    cdhit_path: str = "cd-hit"
    cdhit_identity: float = 0.40
    with_svm: bool = False
    svm_aaindex: str | None = None
    svm_model_pkl: str | None = None
    svm_scaler_csv: str | None = None
    svm_output_dir: str | None = None
    esm2_device: str | None = None
    esm2_max_length: int = 400
    train_legacy_gnn: bool = False
    legacy_gnn_architecture: str = "gcn"
    legacy_gnn_epochs: int = 100
    train_final_gnn: bool = False
    final_gnn_output_dir: str | None = None
    final_gnn_epochs: int | None = None

    @classmethod
    def from_args(cls, args: Namespace) -> RunConfig:
        return cls(
            input_path=Path(args.input),
            work_dir=Path(args.work_dir).resolve() if args.work_dir else None,
            dry_run=args.dry_run,
            skip_if_exists=args.skip_if_exists,
            min_len=args.min_len,
            max_len=args.max_len,
            reset_esmfold=args.reset_esmfold,
            esmfold_max_length=args.esmfold_max_length,
            esmfold_device=args.esmfold_device,
            skip_qsar=args.skip_qsar,
            skip_esm2=args.skip_esm2,
            with_cluster=args.with_cluster,
            cluster_simple_identity=args.cluster_simple_identity,
            cluster_run_cdhit=args.cluster_run_cdhit,
            cdhit_path=args.cdhit_path,
            cdhit_identity=args.cdhit_identity,
            with_svm=args.with_svm,
            svm_aaindex=args.svm_aaindex,
            svm_model_pkl=args.svm_model_pkl,
            svm_scaler_csv=args.svm_scaler_csv,
            svm_output_dir=args.svm_output_dir,
            esm2_device=args.esm2_device,
            esm2_max_length=args.esm2_max_length,
            train_legacy_gnn=args.train_legacy_gnn,
            legacy_gnn_architecture=args.legacy_gnn_architecture,
            legacy_gnn_epochs=args.legacy_gnn_epochs,
            train_final_gnn=args.train_final_gnn,
            final_gnn_output_dir=args.final_gnn_output_dir,
            final_gnn_epochs=args.final_gnn_epochs,
        )

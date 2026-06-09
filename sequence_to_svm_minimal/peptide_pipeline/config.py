"""Single configuration object for the sequence-to-features pipeline."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


def default_work_dir(input_path: Path) -> Path:
    return Path(input_path).resolve().parent / "generated"


@dataclass
class RunConfig:
    mode: str  # "blind" (unlabeled) or "train" (labeled AMP/DECOY)
    input_path: Path
    amp_input_path: Path | None = None
    decoy_input_path: Path | None = None
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
    skip_model_comparison: bool = False
    no_gnn_platt: bool = False
    compare_gnn_architecture: str = "gat"
    window_min_len: int | None = None
    window_max_len: int | None = None
    window_stride: int = 1
    # When False (default): keep parent sequences in canonical_seqs.txt for ESMFold/ESM2;
    # still write window_map.csv and canonical_windows_expanded.txt for SVM / joins.
    # When True: legacy behavior — canonical_seqs.txt lists every window (ESMFold per window).
    window_expand_canonical: bool = False

    def uses_windowing(self) -> bool:
        return self.window_min_len is not None and self.window_max_len is not None

    def is_train_mode(self) -> bool:
        return self.mode == "train"

    def is_blind_mode(self) -> bool:
        return self.mode == "blind"

    @classmethod
    def from_args(cls, args: Namespace) -> RunConfig:
        # Mode inference / backwards compatibility:
        # - Old usage: --input <file> (no --mode) => blind
        # - New usage: --mode train --amp-input --decoy-input
        # - Convenience: if amp/decoy provided and --mode omitted => train
        raw_mode = getattr(args, "mode", None)
        amp_in = getattr(args, "amp_input", None)
        decoy_in = getattr(args, "decoy_input", None)
        inp = getattr(args, "input", None)

        if raw_mode is None:
            if amp_in or decoy_in:
                mode = "train"
            else:
                mode = "blind"
        else:
            mode = str(raw_mode).strip().lower()

        if mode not in ("blind", "train"):
            raise ValueError("--mode must be one of: blind, train")

        if mode == "blind":
            if not inp:
                raise ValueError("--mode blind requires --input")
            if amp_in or decoy_in:
                raise ValueError("--mode blind cannot be used with --amp-input/--decoy-input")
            input_path = Path(inp)
            amp_input_path = None
            decoy_input_path = None
        else:
            if inp:
                raise ValueError("--mode train cannot be used with --input (use --amp-input/--decoy-input)")
            if not amp_in or not decoy_in:
                raise ValueError("--mode train requires both --amp-input and --decoy-input")
            amp_input_path = Path(amp_in)
            decoy_input_path = Path(decoy_in)
            # Use AMP input as the "primary" path for default work_dir behavior.
            input_path = amp_input_path

        # Default clustering behavior:
        # - train mode: clustering ON by default (prevents leakage in splits)
        # - blind mode: clustering OFF by default (not needed)
        # CLI can override via --with-cluster / --no-cluster (args.with_cluster is tri-state).
        raw_with_cluster = getattr(args, "with_cluster", None)
        with_cluster = bool(raw_with_cluster) if raw_with_cluster is not None else (mode == "train")

        return cls(
            mode=mode,
            input_path=input_path,
            amp_input_path=amp_input_path,
            decoy_input_path=decoy_input_path,
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
            with_cluster=with_cluster,
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
            skip_model_comparison=args.skip_model_comparison,
            no_gnn_platt=args.no_gnn_platt,
            compare_gnn_architecture=args.compare_gnn_architecture,
            window_min_len=getattr(args, "window_min_len", None),
            window_max_len=getattr(args, "window_max_len", None),
            window_stride=getattr(args, "window_stride", 1),
            window_expand_canonical=bool(getattr(args, "window_expand_canonical", False)),
        )

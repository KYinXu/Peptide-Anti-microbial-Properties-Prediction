"""Shared CLI helpers and validation for length analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .variants.deletion import DEFAULT_P_LEVELS, TERMINI


SUPPORTED_MODELS = ("gnn",)
SUPPORTED_MODES = ("extension", "deletion")
DEFAULT_MODE = "extension"
DEFAULT_MAX_FLANK_FRACTION = 0.5
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_PROTEOME_FASTA = (
    Path(__file__).resolve().parent
    / "uniprotkb_UP000005640_AND_model_organis_2026_06_27.fasta"
)


@dataclass(frozen=True)
class LengthConfig:
    max_flank_fraction: float = DEFAULT_MAX_FLANK_FRACTION
    stride: int = DEFAULT_STRIDE


@dataclass(frozen=True)
class DeletionConfig:
    p_levels: tuple[float, ...] = DEFAULT_P_LEVELS
    termini: tuple[str, ...] = TERMINI


def add_deletion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--p-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_LEVELS),
        help="Deletion fractions; k = round(p * L) per terminus (default: 0.05 0.1 0.2).",
    )
    parser.add_argument(
        "--termini",
        nargs="+",
        choices=TERMINI,
        default=list(TERMINI),
        help="Termini to truncate (default: both N and C).",
    )


def deletion_config_from_args(args: argparse.Namespace) -> DeletionConfig:
    return DeletionConfig(
        p_levels=tuple(args.p_levels),
        termini=tuple(args.termini),
    )


def validate_deletion_config(config: DeletionConfig) -> None:
    if not config.p_levels:
        raise ValueError("--p-levels must list at least one value.")
    for p in config.p_levels:
        if not 0.0 < p < 1.0:
            raise ValueError(f"--p-levels must be strictly between 0 and 1, got {p}.")
    if not config.termini:
        raise ValueError("--termini must list at least one terminus.")


def add_length_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-flank-fraction",
        type=float,
        default=DEFAULT_MAX_FLANK_FRACTION,
        help="Maximum N/C extension as a fraction of core length (default: 0.5).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Step size when enumerating left/right extensions (default: 1).",
    )


def length_config_from_args(args: argparse.Namespace) -> LengthConfig:
    return LengthConfig(
        max_flank_fraction=args.max_flank_fraction,
        stride=args.stride,
    )


def validate_length_config(config: LengthConfig) -> None:
    if config.max_flank_fraction < 0:
        raise ValueError("--max-flank-fraction must be >= 0.")
    if config.stride < 1:
        raise ValueError("--stride must be at least 1.")


def add_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidates-table",
        type=Path,
        help=(
            "final_candidates.csv/parquet with peptide_id, sequence, source_protein_id, start, end. "
            "Required for --mode extension; unused by --mode deletion."
        ),
    )
    parser.add_argument(
        "--proteome-fasta",
        type=Path,
        default=DEFAULT_PROTEOME_FASTA,
        help=(
            "Proteome FASTA whose headers match source_protein_id "
            f"(default: {DEFAULT_PROTEOME_FASTA.name})."
        ),
    )

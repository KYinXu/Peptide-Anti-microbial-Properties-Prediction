"""Shared CLI helpers and validation for length analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_MODELS = ("gnn",)
DEFAULT_MAX_FLANK_FRACTION = 0.5
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class LengthConfig:
    max_flank_fraction: float = DEFAULT_MAX_FLANK_FRACTION
    stride: int = DEFAULT_STRIDE


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
        required=True,
        help="final_candidates.csv/parquet with peptide_id, sequence, source_protein_id, start, end.",
    )
    parser.add_argument(
        "--proteome-fasta",
        type=Path,
        required=True,
        help="Proteome FASTA whose headers match source_protein_id.",
    )

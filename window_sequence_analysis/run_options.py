"""Shared CLI and validation helpers for window-analysis run types."""

from __future__ import annotations

import argparse
from typing import cast

from .sliding_windows.common import (
    PROFILE_AGGREGATIONS,
    ProfileAggregation,
    ProfileConfig,
    WindowConfig,
)


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-min-len", type=int, default=10, help="Minimum window length (default: 10).")
    parser.add_argument("--window-max-len", type=int, default=35, help="Maximum window length (default: 35).")
    parser.add_argument("--stride", type=int, default=1, help="Window start stride (default: 1).")
    parser.add_argument(
        "--batch-starts",
        type=int,
        default=64,
        help="Number of start positions to score per bounded in-memory batch (default: 64).",
    )


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-aggregation",
        choices=PROFILE_AGGREGATIONS,
        default="mean",
        help="Aggregate scores from covering windows by mean or maximum (default: mean).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Significant digits for serialized float profiles (default: 6).",
    )


def window_config_from_args(args: argparse.Namespace) -> WindowConfig:
    return WindowConfig(
        min_len=args.window_min_len,
        max_len=args.window_max_len,
        stride=args.stride,
        batch_starts=args.batch_starts,
    )


def profile_config_from_args(args: argparse.Namespace) -> ProfileConfig:
    window = window_config_from_args(args)
    return ProfileConfig(
        min_len=window.min_len,
        max_len=window.max_len,
        stride=window.stride,
        batch_starts=window.batch_starts,
        precision=args.precision,
        aggregation=cast(ProfileAggregation, args.profile_aggregation),
    )


def validate_window_config(config: WindowConfig) -> None:
    if config.min_len < 1:
        raise ValueError("--window-min-len must be at least 1.")
    if config.max_len < config.min_len:
        raise ValueError("--window-max-len must be greater than or equal to --window-min-len.")
    if config.stride < 1:
        raise ValueError("--stride must be at least 1.")
    if config.batch_starts < 1:
        raise ValueError("--batch-starts must be at least 1.")


def validate_profile_config(config: ProfileConfig) -> None:
    validate_window_config(config)
    if config.precision < 1:
        raise ValueError("--precision must be at least 1.")
    if config.aggregation not in PROFILE_AGGREGATIONS:
        raise ValueError(f"Unsupported profile aggregation: {config.aggregation}")

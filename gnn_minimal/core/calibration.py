"""Resolve and apply the P(AMP) calibration for a trained model directory.

P(AMP) = sigmoid(sigma / T) under temperature scaling, sigmoid(a*sigma + b) under
Platt, and sigmoid(sigma) uncalibrated, where sigma is the logit margin. Only Platt
carries an intercept, so it is the only method whose 0.5 crossing moves off
sigma = 0 -- switching methods changes the hard call, not just the probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import platt_scaling_path, temperature_scaling_path
from .platt import load_platt_json, platt_prob_amp
from .temperature import load_temperature_json, temperature_prob_amp


METHODS = ("temperature", "platt", "softmax")
DEFAULT_PRIORITY = ("temperature", "platt")

_FIT_HINTS = {
    "temperature": "Fit it with gnn_minimal/fit_temperature.py.",
    "platt": "Platt scaling is written by gnn_minimal/train.py.",
}
_LOADERS = {
    "temperature": (temperature_scaling_path, load_temperature_json),
    "platt": (platt_scaling_path, load_platt_json),
}


def resolve_calibration(model_dir: str | Path, fitting: str | None = None) -> tuple[str, dict | None]:
    """Return ``(method, payload)``; ``fitting=None`` walks the default cascade.

    The default prefers temperature scaling, falls back to Platt, then to the
    uncalibrated softmax. An explicitly requested method with no fitted file on
    disk is an error rather than a silent downgrade.
    """
    if fitting is not None and fitting not in METHODS:
        raise ValueError(f"fitting must be one of {METHODS} or None; got {fitting!r}")
    model_dir = Path(model_dir)

    if fitting == "softmax":
        return "softmax", None
    if fitting is not None:
        path_of, load = _LOADERS[fitting]
        path = path_of(model_dir)
        payload = load(path)
        if payload is None:
            raise FileNotFoundError(
                f"{fitting} scaling file not found or unreadable: {path}. {_FIT_HINTS[fitting]}"
            )
        return fitting, payload

    for method in DEFAULT_PRIORITY:
        path_of, load = _LOADERS[method]
        payload = load(path_of(model_dir))
        if payload is not None:
            return method, payload
    return "softmax", None


def prob_amp(margins, method: str, payload: dict | None) -> np.ndarray:
    """Map logit margins to P(AMP) under ``method``."""
    margins = np.asarray(margins, dtype=np.float64)
    if method == "temperature":
        return temperature_prob_amp(margins, payload["temperature"])
    if method == "platt":
        return platt_prob_amp(margins, payload["coef"], payload["intercept"])
    if method == "softmax":
        return 1.0 / (1.0 + np.exp(-np.clip(margins, -500.0, 500.0)))
    raise ValueError(f"Unknown probability fitting: {method!r}")


def decision_margin(method: str, payload: dict | None) -> float:
    """The logit margin at which P(AMP) = 0.5."""
    if method == "platt":
        return -float(payload["intercept"]) / float(payload["coef"])
    return 0.0


def describe_calibration(method: str, payload: dict | None) -> str:
    """One-line description naming the method and where it puts the 0.5 crossing."""
    threshold = f"P=0.5 at margin {decision_margin(method, payload):.3f}"
    if method == "temperature":
        return f"temperature-scaled P(AMP) (T={float(payload['temperature']):.3f}, {threshold})"
    if method == "platt":
        return (
            f"Platt-scaled P(AMP) (coef={float(payload['coef']):.3f}, "
            f"intercept={float(payload['intercept']):.3f}, {threshold})"
        )
    return f"softmax P(AMP), uncalibrated ({threshold})"

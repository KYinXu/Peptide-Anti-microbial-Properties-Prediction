"""Temperature scaling on the GNN logit margin for P(AMP)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_temperature(margins: np.ndarray, labels: np.ndarray) -> dict | None:
    """Maximum-likelihood temperature for P(AMP) = sigmoid(sigma / T)."""
    if margins.size == 0 or len(np.unique(labels)) < 2:
        return None
    from scipy.optimize import minimize_scalar

    scores = margins.astype(np.float64)
    targets = labels.astype(np.float64)

    def negative_log_likelihood(log_temperature: float) -> float:
        z = np.clip(scores / np.exp(log_temperature), -500.0, 500.0)
        log_positive = -np.logaddexp(0.0, -z)
        log_negative = -np.logaddexp(0.0, z)
        return float(-np.mean(targets * log_positive + (1.0 - targets) * log_negative))

    fitted = minimize_scalar(negative_log_likelihood, bounds=(-5.0, 5.0), method="bounded")
    if not fitted.success:
        raise RuntimeError(f"Temperature fit failed: {fitted.message}")
    return {
        "method": "temperature",
        "score": "logit_margin",
        "temperature": float(np.exp(fitted.x)),
        "n_calib": int(len(targets)),
    }


def temperature_prob_amp(margins: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    z = np.clip(margins.astype(np.float64) / float(temperature), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def save_temperature_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_temperature_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    temperature = data.get("temperature")
    if data.get("method") != "temperature" or not isinstance(temperature, (int, float)):
        return None
    if float(temperature) <= 0:
        return None
    return data

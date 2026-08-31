"""SVM model adapter for window scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from window_sequence_analysis.sliding_windows.common import WindowRecord, WindowScores


CHARGE = {
    "A": 0,
    "C": 0,
    "D": -1,
    "E": -1,
    "F": 0,
    "G": 0,
    "H": 1,
    "I": 0,
    "K": 1,
    "L": 0,
    "M": 0,
    "N": 0,
    "P": 0,
    "Q": 0,
    "R": 1,
    "S": 0,
    "T": 0,
    "V": 0,
    "W": 0,
    "Y": 0,
}
STANDARD_AA = set(CHARGE)
INVALID_RESIDUE_SUBSTITUTIONS = {
    "U": "C",
}


class SvmWindowScorer:
    """Scores sequence windows with a pickled SVM and external z-score file."""

    def __init__(self, svm: Any, descriptor_names: list[str], means: np.ndarray, stds: np.ndarray) -> None:
        self.svm = svm
        self.descriptor_names = descriptor_names
        self.means = means
        self.stds = stds

    @classmethod
    def from_paths(cls, svm_pkl: Path, zscores: Path) -> "SvmWindowScorer":
        descriptor_names, means, stds = read_zscores(zscores)
        return cls(load_svm(svm_pkl), descriptor_names, means, stds)

    def score(self, windows: list[WindowRecord]) -> WindowScores:
        if not windows:
            empty = np.asarray([], dtype=np.float64)
            return WindowScores(p_amp=empty, hyperplane_distance=empty)
        x_raw = descriptor_matrix(windows, self.descriptor_names)
        x_scaled = (x_raw - self.means) / self.stds
        return score_scaled_matrix(self.svm, x_scaled)


def read_zscores(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Z-score file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Z-score file must contain descriptor names, means, and standard deviations.")
    names = [name.strip() for name in lines[0].split(",") if name.strip()]
    means = np.asarray([float(value) for value in lines[1].split(",")], dtype=np.float64)
    stds = np.asarray([float(value) for value in lines[2].split(",")], dtype=np.float64)
    if len(names) != len(means) or len(names) != len(stds):
        raise ValueError("Z-score descriptor, mean, and standard deviation counts do not match.")
    return names, means, np.where(stds > 0, stds, 1.0)


def load_svm(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"SVM pickle not found: {path}")
    try:
        import joblib
    except ImportError:
        from sklearn.externals import joblib  # type: ignore
    return joblib.load(path)


def descriptor_matrix(windows: list[WindowRecord], names: list[str]) -> np.ndarray:
    rows = []
    for window in windows:
        values = qsar_descriptors(normalize_svm_sequence(window.sequence))
        missing = [name for name in names if name not in values]
        if missing:
            raise ValueError(f"Unsupported descriptor(s) in z-score file: {missing}")
        rows.append([values[name] for name in names])
    return np.asarray(rows, dtype=np.float64)


def normalize_svm_sequence(sequence: str) -> str:
    substituted = "".join(INVALID_RESIDUE_SUBSTITUTIONS.get(residue, residue) for residue in sequence.upper())
    invalid = sorted(set(substituted) - STANDARD_AA)
    if invalid:
        raise ValueError(f"SVM sequence contains unsupported residue(s): {''.join(invalid)!r}")
    return substituted


def qsar_descriptors(sequence: str) -> dict[str, float]:
    from propy import ProCheck
    from propy.PyPro import GetProDes

    if ProCheck.ProteinCheck(sequence) == 0:
        raise ValueError("ProPy rejected the sequence.")
    descriptor = GetProDes(sequence)
    dpc = descriptor.GetDPComp()
    ctd = descriptor.GetCTD()
    socn = safe_descriptor_call(lambda: descriptor.GetSOCN(maxlag=30))
    qso = safe_descriptor_call(lambda: descriptor.GetQSO(maxlag=30, weight=0.05))
    length = len(sequence)
    methionine = sequence.count("M")
    lysine = sequence.count("K")
    return {
        "netCharge": float(sum(CHARGE[residue] for residue in sequence)),
        "FC": round(dpc.get("FC", 0), 2),
        "LW": round(dpc.get("LW", 0), 2),
        "DP": round(dpc.get("DP", 0), 2),
        "NK": round(dpc.get("NK", 0), 2),
        "AE": round(dpc.get("AE", 0), 2),
        "pcMK": 0.0 if methionine == 0 else methionine / (methionine + lysine),
        "_SolventAccessibilityD1025": float(ctd.get("_SolventAccessibilityD1025", 0)),
        "tau2_GRAR740104": float(socn.get("tau2", 0) / (length - 2) if length > 2 else 0),
        "tau4_GRAR740104": float(socn.get("tau4", 0) / (length - 4) if length > 4 else 0),
        "QSO50_GRAR740104": float(qso.get("QSO50", 0)),
        "QSO29_GRAR740104": float(qso.get("QSO29", 0)),
    }


def safe_descriptor_call(call: Callable[[], dict[str, float]]) -> dict[str, float]:
    try:
        return call()
    except Exception:
        return {}


def score_scaled_matrix(svm: Any, x_scaled: np.ndarray) -> WindowScores:
    if not hasattr(svm, "predict_proba"):
        raise TypeError("SVM model must expose predict_proba to compute P(AMP).")
    if not hasattr(svm, "decision_function"):
        raise TypeError("SVM model must expose decision_function to compute hyperplane distance.")
    classes = np.asarray(getattr(svm, "classes_", [0, 1]))
    positive_class = 1 if 1 in classes else classes[-1]
    positive_index = int(np.where(classes == positive_class)[0][0])
    p_amp = np.asarray(svm.predict_proba(x_scaled))[:, positive_index].ravel()
    distance = np.asarray(svm.decision_function(x_scaled)).ravel()
    return WindowScores(
        p_amp=p_amp.astype(np.float64),
        hyperplane_distance=distance.astype(np.float64),
    )

"""SVM model adapter for window scoring."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sequence_analysis.utils.descriptor_ablation import null_descriptor_values, parse_null_descriptor_names

from ..sliding_windows.common import WindowRecord, WindowScores
from .sklearn_pickle_compat import alias_legacy_sklearn_modules, prepare_legacy_svm


REPO_ROOT = Path(__file__).resolve().parents[2]
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
SEQUENCE_ORDER_QSAR_COLUMNS = (
    "tau2_GRAR740104",
    "tau4_GRAR740104",
    "QSO50_GRAR740104",
    "QSO29_GRAR740104",
)


class SvmWindowScorer:
    """Scores sequence windows with a pickled SVM and external z-score file."""

    def __init__(
        self,
        svm: Any,
        descriptor_names: list[str],
        means: np.ndarray,
        stds: np.ndarray,
        null_descriptors: Iterable[str] = (),
    ) -> None:
        self.svm = svm
        self.descriptor_names = descriptor_names
        self.means = means
        self.stds = stds
        self.null_descriptors = tuple(null_descriptors)
        self.grar740104_matrix = load_grar740104_matrix() if needs_sequence_order(self.null_descriptors) else None

    @classmethod
    def from_paths(
        cls,
        svm_pkl: Path,
        zscores: Path,
        null_descriptors: Iterable[str] = (),
    ) -> "SvmWindowScorer":
        descriptor_names, means, stds = read_zscores(zscores)
        resolved_nulls = parse_null_descriptor_names(null_descriptors, descriptor_names)
        return cls(load_svm(svm_pkl), descriptor_names, means, stds, resolved_nulls)

    def score(self, windows: list[WindowRecord]) -> WindowScores:
        if not windows:
            empty = np.asarray([], dtype=np.float64)
            return WindowScores(p_amp=empty, hyperplane_distance=empty)
        x_raw = descriptor_matrix(
            windows,
            self.descriptor_names,
            self.grar740104_matrix,
            self.null_descriptors,
        )
        x_scaled = (x_raw - self.means) / self.stds
        return score_scaled_matrix(self.svm, x_scaled)


@dataclass(frozen=True)
class SvmScorerFactory:
    """Picklable constructor so worker processes can load their own scorer."""

    svm_pkl: Path
    zscores: Path
    null_descriptors: tuple[str, ...] = ()

    def __call__(self) -> SvmWindowScorer:
        return SvmWindowScorer.from_paths(self.svm_pkl, self.zscores, self.null_descriptors)


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
    alias_legacy_sklearn_modules()
    return prepare_legacy_svm(joblib.load(path), path.parent)


def load_grar740104_matrix() -> dict[str, dict[str, float]]:
    try:
        from propy import AAIndex
    except Exception as error:
        raise RuntimeError("Could not import ProPy. Install propy3 before running SVM inference.") from error
    aaindex_dir = REPO_ROOT / "descriptors" / "aaindex"
    return AAIndex.GetAAIndex23("GRAR740104", path=str(aaindex_dir))


def needs_sequence_order(null_descriptors: Iterable[str]) -> bool:
    return bool(set(SEQUENCE_ORDER_QSAR_COLUMNS) - set(null_descriptors))


def descriptor_matrix(
    windows: list[WindowRecord],
    names: list[str],
    grar740104_matrix: dict[str, dict[str, float]] | None,
    null_descriptors: Iterable[str] = (),
) -> np.ndarray:
    rows = []
    for window in windows:
        values = qsar_descriptors(normalize_svm_sequence(window.sequence), grar740104_matrix)
        null_descriptor_values(values, null_descriptors)
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


def qsar_descriptors(
    sequence: str,
    grar740104_matrix: dict[str, dict[str, float]] | None,
) -> dict[str, float]:
    from propy import ProCheck
    from propy.PyPro import GetProDes

    if ProCheck.ProteinCheck(sequence) == 0:
        raise ValueError("ProPy rejected the sequence.")
    descriptor = GetProDes(sequence)
    dpc = descriptor.GetDPComp()
    ctd = descriptor.GetCTD()
    if grar740104_matrix is None:
        socn = {}
        qso = {}
    else:
        socn = safe_descriptor_call(lambda: descriptor.GetSOCNp(maxlag=30, distancematrix=grar740104_matrix))
        qso = safe_descriptor_call(
            lambda: descriptor.GetQSOp(maxlag=30, weight=0.05, distancematrix=grar740104_matrix)
        )
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

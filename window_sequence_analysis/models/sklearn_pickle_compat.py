"""Compatibility helpers for unpickling SVMs saved with older scikit-learn."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def alias_legacy_sklearn_modules() -> None:
    _ignore_inconsistent_version_warning()
    try:
        import sklearn.svm._classes as svm_classes
    except ImportError:
        return
    sys.modules.setdefault("sklearn.svm.classes", svm_classes)
    try:
        import joblib
    except ImportError:
        return
    sys.modules.setdefault("sklearn.externals.joblib", joblib)


def prepare_legacy_svm(svm: Any, pickle_dir: Path) -> Any:
    unwrap_legacy_ndarray_attributes(svm, pickle_dir)
    fill_legacy_svc_attributes(svm)
    return svm


def unwrap_legacy_ndarray_attributes(obj: Any, pickle_dir: Path) -> None:
    values = getattr(obj, "__dict__", None)
    if not values:
        return
    missing: list[str] = []
    resolved: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not _is_ndarray_wrapper(value):
            continue
        sidecar = pickle_dir / Path(value.filename).name
        if not sidecar.is_file():
            missing.append(str(sidecar))
            continue
        resolved[key] = np.load(sidecar, allow_pickle=True)
    if missing:
        listed = "\n  ".join(missing)
        raise FileNotFoundError(
            "Legacy SVM pickle is split across sidecar .npy files that were not found:\n  "
            + listed
        )
    values.update(resolved)


def fill_legacy_svc_attributes(svm: Any) -> None:
    if not hasattr(svm, "break_ties"):
        svm.break_ties = False
    if not hasattr(svm, "decision_function_shape"):
        svm.decision_function_shape = "ovr"
    values = getattr(svm, "__dict__", {})
    if "_n_support" not in values and "n_support_" in values:
        values["_n_support"] = np.asarray(values["n_support_"], dtype=np.int32)
    if "_probA" not in values and "probA_" in values:
        values["_probA"] = np.asarray(values["probA_"])
    if "_probB" not in values and "probB_" in values:
        values["_probB"] = np.asarray(values["probB_"])
    if "n_features_in_" not in values and "support_vectors_" in values:
        support_vectors = np.asarray(values["support_vectors_"])
        if support_vectors.ndim == 2:
            values["n_features_in_"] = int(support_vectors.shape[1])


def _is_ndarray_wrapper(value: Any) -> bool:
    return type(value).__name__ == "NDArrayWrapper" and hasattr(value, "filename")


def _ignore_inconsistent_version_warning() -> None:
    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError:
        return
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

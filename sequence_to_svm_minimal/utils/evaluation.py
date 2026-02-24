#!/usr/bin/env python3
"""
Model evaluation and visualization utilities.

All plot functions tolerate missing data: they skip plots when inputs are
missing and return False (or an empty list) without raising.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _safe_read_json(path: Path) -> Optional[dict]:
    if path is None or not Path(path).exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _safe_read_csv(path: Path) -> Optional[Dict[str, List[float]]]:
    if path is None or not Path(path).exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {c: df[c].tolist() for c in df.columns}
    except (OSError, KeyError):
        return None


def load_history_from_fold_csv(fold_csv: Path) -> Optional[Dict[str, List[float]]]:
    """Load per-epoch history from a GNN-style fold CSV. Returns None if missing/invalid."""
    data = _safe_read_csv(fold_csv)
    if data is None:
        return None
    for key in ('epoch', 'train_loss', 'val_loss', 'val_auc_roc'):
        if key not in data:
            return None
    return data


def load_history_from_json(history_json: Path) -> Optional[Dict[str, List[float]]]:
    """Load history from a JSON with a 'history' key (e.g. NN/fusion). Returns None if missing."""
    raw = _safe_read_json(history_json)
    if raw is None:
        return None
    history = raw.get('history') if isinstance(raw.get('history'), dict) else raw
    if not history or 'train_loss' not in history and 'val_auc_roc' not in history:
        return None
    return {k: v for k, v in history.items() if isinstance(v, list)}


def load_probs_from_json(probs_json: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load y_true, y_prob from a JSON. Returns None if missing/invalid."""
    raw = _safe_read_json(probs_json)
    if raw is None:
        return None
    y_true = raw.get('y_true')
    y_prob = raw.get('y_prob')
    if y_true is None or y_prob is None:
        return None
    try:
        return np.array(y_true), np.array(y_prob)
    except (TypeError, ValueError):
        return None


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Learning curves",
) -> bool:
    """Plot train/val loss and val metric. Returns False if history missing or plot failed."""
    if not history or 'train_loss' not in history:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history['train_loss'], label='Train loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], label='Val loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(title)
    if 'val_auc_roc' in history and history['val_auc_roc']:
        axes[1].plot(epochs, history['val_auc_roc'], label='Val AUC-ROC')
    if 'val_f1' in history and history['val_f1']:
        axes[1].plot(epochs, history['val_f1'], label='Val F1')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].legend()
    axes[1].set_title(title)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    title_prefix: str = "",
) -> bool:
    """Plot ROC and PR curves. Returns False if data missing or plot failed."""
    if y_true is None or y_prob is None or len(y_true) == 0:
        return False
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC (AUC={auc(fpr, tpr):.3f})")
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend()
    axes[0].set_title(f"{title_prefix} ROC".strip())
    axes[1].plot(rec, prec)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f"{title_prefix} PR curve".strip())
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def plot_cv_summary_bars(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    metric: str = 'auc_roc_mean',
    title: str = "CV summary",
) -> bool:
    """Plot bar chart of metric by model. results[key] should have metric (e.g. auc_roc_mean). Returns False if empty."""
    if not results:
        return False
    labels = []
    values = []
    for name, r in results.items():
        if metric in r and r[metric] is not None:
            labels.append(name[:25] + '..' if len(name) > 25 else name)
            values.append(float(r[metric]))
    if not values:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def load_gnn_curves_from_run(
    curves_run_dir: Union[str, Path],
) -> List[Tuple[str, Dict[str, List[float]]]]:
    """Load (label, history) for each model in a GNN run. Uses fold_1.csv per model subdir."""
    curves_run_dir = Path(curves_run_dir)
    if not curves_run_dir.is_dir():
        return []
    out = []
    for model_dir in sorted(curves_run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        fold_csv = model_dir / "fold_1.csv"
        history = load_history_from_fold_csv(fold_csv)
        if history is not None:
            out.append((model_dir.name, history))
    return out


def load_mlp_history_from_json(path: Union[str, Path]) -> Optional[Dict[str, List[float]]]:
    """Load MLP history from PNAS or fusion JSON. Returns None if missing/invalid."""
    return load_history_from_json(Path(path))


def plot_learning_curves_comparison(
    series_list: List[Tuple[str, Dict[str, List[float]]]],
    save_path: Union[str, Path],
    title: str = "MLP vs GNN",
) -> bool:
    """Overlay multiple learning curves (e.g. MLP + GNN). Two panels: Loss, Metrics. Returns False if no valid series."""
    valid = [(label, h) for label, h in series_list if h and isinstance(h.get('train_loss'), list) and len(h['train_loss']) > 0]
    if not valid:
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    styles = ['-', '--', '-.', ':'] * ((len(valid) // 4) + 1)
    for idx, (label, history) in enumerate(valid):
        epochs = list(range(1, len(history['train_loss']) + 1))
        sty = styles[idx % len(styles)]
        axes[0].plot(epochs, history['train_loss'], label=label, linestyle=sty)
        if history.get('val_loss') and len(history['val_loss']) == len(history['train_loss']):
            axes[0].plot(epochs, history['val_loss'], label=f"{label} (val)", linestyle=sty, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].set_title('Train / Val loss')
    for idx, (label, history) in enumerate(valid):
        epochs = list(range(1, len(history['train_loss']) + 1))
        sty = styles[idx % len(styles)]
        if history.get('val_auc_roc') and len(history['val_auc_roc']) == len(history['train_loss']):
            axes[1].plot(epochs, history['val_auc_roc'], label=label, linestyle=sty)
        elif history.get('val_f1') and len(history['val_f1']) == len(history['train_loss']):
            axes[1].plot(epochs, history['val_f1'], label=label, linestyle=sty)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].set_title('Val AUC-ROC / F1')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return True


def generate_plots_from_gnn_run(
    curves_base_dir: Union[str, Path],
    comparison_json_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Generate all plots that have data in a GNN run. Skips missing data.
    Returns list of generated plot paths.
    """
    curves_base_dir = Path(curves_base_dir)
    output_dir = Path(output_dir) if output_dir else curves_base_dir.parent / "evaluation_plots"
    output_dir = output_dir / curves_base_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for model_dir in curves_base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        fold_csvs = sorted(model_dir.glob("fold_*.csv"))
        if fold_csvs:
            history = load_history_from_fold_csv(fold_csvs[0])
            if history and plot_learning_curves(
                history,
                output_dir / f"{name}_learning_curves.png",
                title=name,
            ):
                generated.append(str(output_dir / f"{name}_learning_curves.png"))
        probs_jsons = sorted(model_dir.glob("fold_*_val_probs.json"))
        if probs_jsons:
            probs = load_probs_from_json(probs_jsons[0])
            if probs is not None and plot_roc_pr(
                probs[0], probs[1],
                output_dir / f"{name}_roc_pr.png",
                title_prefix=name,
            ):
                generated.append(str(output_dir / f"{name}_roc_pr.png"))
    if comparison_json_path:
        comp = _safe_read_json(Path(comparison_json_path))
        if comp and 'results' in comp and plot_cv_summary_bars(
            comp['results'],
            output_dir / "cv_auc_by_model.png",
            metric='auc_roc_mean',
            title="GNN CV AUC-ROC",
        ):
            generated.append(str(output_dir / "cv_auc_by_model.png"))
    return generated


def generate_plots_from_probs_only(
    probs_json: Union[str, Path],
    save_path: Union[str, Path],
    title: str = "",
) -> bool:
    """Load y_true/y_prob from JSON and plot ROC/PR. Skips if file missing. Returns True if plotted."""
    probs = load_probs_from_json(Path(probs_json))
    if probs is None:
        return False
    return plot_roc_pr(probs[0], probs[1], Path(save_path), title_prefix=title)

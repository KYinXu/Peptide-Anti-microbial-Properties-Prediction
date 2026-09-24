"""Model-specific sequence inference adapters."""

from .gnn import DEFAULT_GNN_ROOT, GnnSequencePrediction, GnnSequenceScorer, resolve_gnn_model_dir
from .svm import SvmSequencePrediction, SvmSequenceScorer

__all__ = [
    "DEFAULT_GNN_ROOT",
    "GnnSequencePrediction",
    "GnnSequenceScorer",
    "SvmSequencePrediction",
    "SvmSequenceScorer",
    "resolve_gnn_model_dir",
]


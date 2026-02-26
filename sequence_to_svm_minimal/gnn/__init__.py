"""
GNN-based Peptide MIC Classification

This module implements Graph Neural Networks for antimicrobial peptide (AMP)
classification using ESMFold-predicted 3D structures.

Architecture:
- Each peptide is represented as a graph
- Nodes = amino acid residues
- Edges = sequential bonds + spatial contacts (Cα-Cα < threshold)
- Node features = AA properties + pLDDT + structural features
- Edge features = distances + edge types

Models available:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network  
- EGNN: E(n) Equivariant Graph Neural Network
"""

import warnings
warnings.filterwarnings('ignore', message='.*Disabling its usage.*', category=UserWarning)
warnings.filterwarnings('ignore', message=".*torch-scatter.*was not found", category=UserWarning)

from .data_utils import PeptideGraphDataset, pdb_to_graph, create_dataloaders
from .models import GCN, GAT, EGNN, PeptideGNN
from .train import train_epoch, evaluate, run_training

__all__ = [
    'PeptideGraphDataset',
    'pdb_to_graph', 
    'create_dataloaders',
    'GCN',
    'GAT', 
    'EGNN',
    'PeptideGNN',
    'train_epoch',
    'evaluate',
    'run_training',
]

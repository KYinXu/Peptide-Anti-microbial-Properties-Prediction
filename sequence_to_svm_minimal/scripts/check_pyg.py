#!/usr/bin/env python3
"""Check if PyTorch Geometric is installed."""
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    import torch_geometric
    print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    print("All PyG imports successful!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTo install PyTorch Geometric, run:")
    print("pip install torch_geometric")
    print("pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html")

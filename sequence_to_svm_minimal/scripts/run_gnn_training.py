#!/usr/bin/env python3
"""
GNN Training Script for Peptide MIC Classification

Usage:
    python run_gnn_training.py --architecture gcn --epochs 100
    python run_gnn_training.py --architecture gat --use_geo_features
    python run_gnn_training.py --architecture egnn --hidden_channels 128

This script trains GNN models on ESMFold-predicted peptide structures.
Supports cluster-based cross-validation and PNAS-style evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from gnn.data_utils import PeptideGraphDataset, create_dataloaders
from gnn.models import PeptideGNN
from gnn.train import run_training, evaluate, cross_validate, print_cv_summary

_LEGACY_DEFAULT_CSV = "data/training_dataset/geometric_features_clustered.csv"
_LEGACY_DEFAULT_PDB = "data/training_dataset"


def resolve_legacy_train_paths(args: argparse.Namespace) -> None:
    from peptide_pipeline.manifest_paths import gnn_legacy_training_paths_from_work_dir

    bundle = None
    if getattr(args, "pipeline_work_dir", None):
        bundle = gnn_legacy_training_paths_from_work_dir(Path(args.pipeline_work_dir))
    if getattr(args, "csv_path", None) is None:
        args.csv_path = bundle["csv_path"] if bundle else _LEGACY_DEFAULT_CSV
    if getattr(args, "pdb_dir", None) is None:
        args.pdb_dir = bundle["pdb_dir"] if bundle else _LEGACY_DEFAULT_PDB


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN for peptide classification')
    
    # Data arguments
    parser.add_argument(
        "--pipeline-work-dir",
        type=str,
        default=None,
        help="Pipeline workspace with pipeline_manifest.json; sets csv_path and pdb_dir unless overridden.",
    )
    parser.add_argument('--csv_path', type=str, default=argparse.SUPPRESS,
                        help='Path to CSV with peptide data')
    parser.add_argument('--pdb_dir', type=str,
                        default=argparse.SUPPRESS,
                        help='Directory containing PDB files')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='gcn',
                        choices=['gcn', 'gat', 'egnn'],
                        help='GNN architecture')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean_max',
                        choices=['mean', 'max', 'sum', 'mean_max'],
                        help='Global pooling method')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularization')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Graph arguments
    parser.add_argument('--distance_threshold', type=float, default=8.0,
                        help='Max Cα-Cα distance for spatial edges (Å)')
    parser.add_argument('--use_geo_features', action='store_true',
                        help='Include pre-computed geometric features')
    
    # Evaluation arguments
    parser.add_argument('--eval_protocol', type=str, default='cluster_cv',
                        choices=['cluster_cv', 'pnas_style'],
                        help='Evaluation protocol')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Test set size for PNAS-style evaluation')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--output_dir', type=str, default='results/gnn',
                        help='Output directory')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def load_dataset(args) -> tuple:
    """Load dataset and return dataset + metadata."""
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(args.csv_path)
    print(f"📂 Loaded: {args.csv_path}")
    print(f"   Samples: {len(df)}")
    
    # Check for cluster column
    has_clusters = 'cluster_id' in df.columns
    if has_clusters:
        n_clusters = df['cluster_id'].nunique()
        print(f"   Clusters: {n_clusters}")
    else:
        print("   ⚠️ No cluster_id column found")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    print(f"   Class distribution: {dict(class_counts)}")
    
    # Create dataset
    dataset = PeptideGraphDataset(
        csv_path=args.csv_path,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
        use_geometric_features=args.use_geo_features
    )
    
    # Get labels and clusters (convert -1/1 to 0/1)
    raw_labels = df['label'].values
    labels = np.where(raw_labels == 1, 1, 0)  # Convert -1 to 0, keep 1 as 1
    clusters = df['cluster_id'].values if has_clusters else None
    
    # Compute inverse-frequency class weights for imbalanced data
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    class_weights = torch.tensor(
        [n_samples / (n_classes * np.sum(labels == c)) for c in range(n_classes)],
        dtype=torch.float32
    )
    print(f"   Class weights: {dict(enumerate(class_weights.tolist()))}")
    
    return dataset, df, labels, clusters, class_weights


def run_cluster_cv(args, dataset, labels, clusters, device, class_weights):
    """Run cluster-based GroupKFold cross-validation."""
    print("\n" + "="*60)
    print(f"Protocol: Cluster-Based GroupKFold ({args.n_folds}-fold)")
    print("="*60)
    
    if clusters is None:
        print("⚠️ No clusters available, falling back to stratified CV")
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(cv.split(np.arange(len(labels)), labels))
    else:
        cv = GroupKFold(n_splits=args.n_folds)
        splits = list(cv.split(np.arange(len(labels)), labels, groups=clusters))
    
    # Determine geometric feature dimension dynamically from dataset
    geo_dim = 0
    if args.use_geo_features:
        for data in dataset:
            if hasattr(data, 'geo_features'):
                geo_dim = int(data.geo_features.shape[1])
                break
        print(f"Using geometric feature dimension: {geo_dim}")
    
    # Model factory
    def model_fn():
        return PeptideGNN(
            architecture=args.architecture,
            in_channels=26,  # Node feature dimension
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            pooling=args.pooling,
            geo_feature_dim=geo_dim
        )
    
    # Load all data into list
    print("Loading graphs...")
    all_data = [dataset[i] for i in range(len(dataset))]
    print(f"Loaded {len(all_data)} graphs")
    
    # Cross-validation
    cv_results = cross_validate(
        model_fn=model_fn,
        dataset=all_data,
        cv_splits=splits,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        class_weights=class_weights,
        verbose=True
    )
    
    print_cv_summary(cv_results)
    
    return cv_results


def run_pnas_evaluation(args, dataset, df, labels, device, class_weights):
    """Run PNAS-style evaluation with blind test set."""
    print("\n" + "="*60)
    print(f"Protocol: PNAS-Style (15-round CV + {args.test_size*100:.0f}% Blind Test)")
    print("="*60)
    
    # Determine geometric feature dimension dynamically from dataset
    geo_dim = 0
    if args.use_geo_features:
        for data in dataset:
            if hasattr(data, 'geo_features'):
                geo_dim = int(data.geo_features.shape[1])
                break
        print(f"Using geometric feature dimension: {geo_dim}")
    
    # Load all data
    print("Loading graphs...")
    all_data = [dataset[i] for i in range(len(dataset))]
    print(f"Loaded {len(all_data)} graphs")
    
    # Create blind test split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_pool_idx, test_idx = next(sss_test.split(np.arange(len(labels)), labels))
    
    print(f"Train pool: {len(train_pool_idx)}, Blind test: {len(test_idx)}")
    
    # 15-round CV on training pool
    n_rounds = 15
    cv_results = {
        'auc_roc': [], 'auc_pr': [], 'f1': [], 
        'accuracy': [], 'precision': [], 'recall': [], 'mcc': []
    }
    
    train_pool_labels = labels[train_pool_idx]
    
    for round_idx in range(n_rounds):
        print(f"\n--- Round {round_idx + 1}/{n_rounds} ---")
        
        # 80/20 split within training pool
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed + round_idx)
        train_idx_local, val_idx_local = next(sss.split(np.arange(len(train_pool_idx)), train_pool_labels))
        
        train_idx = train_pool_idx[train_idx_local]
        val_idx = train_pool_idx[val_idx_local]
        
        # Create dataloaders
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        
        # Fresh model
        model = PeptideGNN(
            architecture=args.architecture,
            in_channels=26,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            pooling=args.pooling,
            geo_feature_dim=geo_dim
        )
        
        # Train
        _, best_metrics = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            class_weights=class_weights,
            verbose=False
        )
        
        for metric in cv_results:
            cv_results[metric].append(best_metrics.get(metric, 0.0))
        
        print(f"Round {round_idx + 1}: AUC-ROC={best_metrics['auc_roc']:.4f}, F1={best_metrics['f1']:.4f}")
    
    # Print CV summary
    print("\n" + "-"*40)
    print("CV Summary (15 rounds)")
    print("-"*40)
    for metric, values in cv_results.items():
        print(f"{metric:15s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Train final model on full training pool
    print("\n" + "-"*40)
    print("Training final model on full training pool...")
    print("-"*40)
    
    # 90/10 split for early stopping
    sss_final = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=args.seed)
    final_train_idx_local, final_val_idx_local = next(sss_final.split(np.arange(len(train_pool_idx)), train_pool_labels))
    final_train_idx = train_pool_idx[final_train_idx_local]
    final_val_idx = train_pool_idx[final_val_idx_local]
    
    train_data = [all_data[i] for i in final_train_idx]
    val_data = [all_data[i] for i in final_val_idx]
    test_data = [all_data[i] for i in test_idx]
    
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    final_model = PeptideGNN(
        architecture=args.architecture,
        in_channels=26,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
        pooling=args.pooling,
        geo_feature_dim=geo_dim
    )
    
    _, _ = run_training(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        class_weights=class_weights,
        verbose=True
    )
    
    # Evaluate on blind test
    print("\n" + "-"*40)
    print("Evaluating on blind test set...")
    print("-"*40)
    
    test_metrics = evaluate(final_model, test_loader, device)
    
    print("\nBlind Test Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric:15s}: {value:.4f}")
    
    return cv_results, test_metrics, final_model


def main():
    args = parse_args()
    resolve_legacy_train_paths(args)

    # Setup
    set_seed(args.seed)
    device = get_device(args.device)
    
    print("="*60)
    print("GNN Training for Peptide MIC Classification")
    print("="*60)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n📊 Configuration:")
    print(f"   Architecture: {args.architecture.upper()}")
    print(f"   Hidden channels: {args.hidden_channels}")
    print(f"   Num layers: {args.num_layers}")
    print(f"   Pooling: {args.pooling}")
    print(f"   Use geometric features: {args.use_geo_features}")
    print(f"   Evaluation protocol: {args.eval_protocol}")
    
    # Load data
    dataset, df, labels, clusters, class_weights = load_dataset(args)
    class_weights = class_weights.to(device)
    
    # Run evaluation
    if args.eval_protocol == 'cluster_cv':
        cv_results = run_cluster_cv(args, dataset, labels, clusters, device, class_weights)
        results = {
            'protocol': 'cluster_cv',
            'cv_results': {k: [float(v) for v in vals] for k, vals in cv_results.items()},
            'cv_summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                          for k, v in cv_results.items()}
        }
    else:
        cv_results, test_metrics, final_model = run_pnas_evaluation(args, dataset, df, labels, device, class_weights)
        results = {
            'protocol': 'pnas_style',
            'cv_results': {k: [float(v) for v in vals] for k, vals in cv_results.items()},
            'cv_summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} 
                          for k, v in cv_results.items()},
            'blind_test': {k: float(v) for k, v in test_metrics.items()}
        }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Add config to results
    results['config'] = vars(args)
    results['timestamp'] = timestamp
    
    # Save JSON
    json_path = os.path.join(args.output_dir, f'gnn_{args.architecture}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Save model checkpoint if PNAS style
    if args.eval_protocol == 'pnas_style':
        ckpt_path = os.path.join(args.output_dir, f'gnn_{args.architecture}_{timestamp}.pt')
        torch.save(final_model.state_dict(), ckpt_path)
        print(f"💾 Model saved to: {ckpt_path}")
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()

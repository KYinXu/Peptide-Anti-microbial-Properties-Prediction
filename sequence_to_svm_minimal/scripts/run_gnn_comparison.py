#!/usr/bin/env python3
"""
GNN Architecture Comparison Script

Compares GCN, GAT, and EGNN architectures with different feature combinations:
- Graph-only (node features from structure)
- Graph + Geo-24 (adding geometric features)
- Graph + Combined-36 (adding geometric + QSAR features)

Prints results side-by-side for comparison with MLP/SVM baselines.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gnn.data_utils import PeptideGraphDataset
from gnn.models import PeptideGNN
from gnn.train import run_training, cross_validate, evaluate, evaluate_probs
from torch_geometric.loader import DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'csv_path': 'data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/geometric_features_clustered.csv',
    'pdb_dir': 'data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced',
    'qsar_csv': 'data/gnn_training_dataset/alpha_and_beta_combined/generated/spliced/qsar12_descriptors.csv',
    'seed': 42,
    'n_folds': 5,
    'epochs': 500,
    'batch_size': 32,
    'lr': 1e-3,
    'patience': 30,
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.2,
    'distance_threshold': 8.0,
}

# Feature configurations
# Note: geometric feature dims exclude pLDDT-derived confidence scores
FEATURE_CONFIGS = {
    'Graph-only': {'use_geo': False, 'use_qsar': False, 'geo_dim': 0},
    'Graph+Geo20': {'use_geo': True, 'use_qsar': False, 'geo_dim': 20},
    'Graph+QSAR12': {'use_geo': False, 'use_qsar': True, 'geo_dim': 12},
    'Graph+Combined32': {'use_geo': True, 'use_qsar': True, 'geo_dim': 32},
}

# GNN architectures
ARCHITECTURES = ['gcn', 'gat', 'egnn']


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_with_features(config):
    """Load dataset with optional QSAR features merged in."""
    # Load geometric features CSV
    geo_df = pd.read_csv(config['csv_path'])
    
    # Load QSAR features if needed
    qsar_df = pd.read_csv(config['qsar_csv'])
    
    # Merge QSAR features
    qsar_cols = ['netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK', 
                 '_SolventAccessibilityD1025', 'tau2_GRAR740104', 
                 'tau4_GRAR740104', 'QSO50_GRAR740104', 'QSO29_GRAR740104']
    
    # Merge on peptide_id
    merged_df = geo_df.merge(qsar_df[['peptide_id'] + qsar_cols], on='peptide_id', how='left')
    
    return merged_df, qsar_cols


def create_dataset_with_features(df, config, use_geo=True, use_qsar=False, qsar_cols=None):
    """Create a dataset with specified feature combination."""
    
    # Geometric feature columns (pLDDT excluded to avoid leakage)
    geo_cols = [
        'radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
        'centroid_distance_mean', 'centroid_distance_std',
        'fraction_helix', 'fraction_sheet', 'fraction_coil',
        'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa',
        'length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment',
        'curvature_mean', 'curvature_std', 'curvature_max',
        'torsion_mean', 'torsion_std'
    ]
    
    # Determine which columns to use
    feature_cols = []
    if use_geo:
        feature_cols.extend(geo_cols)
    if use_qsar and qsar_cols:
        feature_cols.extend(qsar_cols)
    
    return feature_cols


class CustomPeptideDataset:
    """Custom dataset that supports different feature combinations."""
    
    def __init__(self, df, pdb_dir, feature_cols, distance_threshold=8.0):
        self.df = df
        self.pdb_dir = Path(pdb_dir)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold
        
        # Import here to avoid circular imports
        from gnn.data_utils import pdb_to_graph, parse_pdb, compute_node_features, compute_edges
        self.pdb_to_graph = pdb_to_graph
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        from gnn.data_utils import parse_pdb, compute_node_features, compute_edges
        
        row = self.df.iloc[idx]
        
        # Find PDB file
        pdb_file = row.get('pdb_file', f"{row['peptide_id']}.pdb")
        pdb_path = None
        for subdir in ['structures/AMP', 'structures/DECOY', 'structures', '']:
            candidate = self.pdb_dir / subdir / pdb_file
            if candidate.exists():
                pdb_path = candidate
                break
        
        if pdb_path is None:
            raise FileNotFoundError(f"PDB not found: {pdb_file}")
        
        # Parse PDB
        aa_sequence, ca_coords, plddt_values = parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)
        
        # Compute node features
        x = compute_node_features(aa_sequence, plddt_values, n_residues)
        
        # Compute edges
        edge_index, edge_attr = compute_edges(ca_coords, self.distance_threshold)
        
        # Coordinates
        pos = torch.tensor(ca_coords, dtype=torch.float32)
        
        # Label (convert -1/1 to 0/1)
        raw_label = int(row['label'])
        label = 1 if raw_label == 1 else 0
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_residues
        )
        
        # Add extra features if specified
        if self.feature_cols:
            extra_feats = row[self.feature_cols].values.astype(np.float32)
            # Handle NaN values
            extra_feats = np.nan_to_num(extra_feats, nan=0.0)
            data.geo_features = torch.tensor(extra_feats, dtype=torch.float32).unsqueeze(0)
        
        return data


def run_single_experiment(arch, feature_name, feature_config, all_data, labels, clusters, device, config, curves_base_dir, save_checkpoints=False):
    """Run a single experiment configuration and save training curves."""
    
    geo_dim = feature_config['geo_dim']
    
    # Create folder for this model's curves
    model_name = f"{arch.upper()}_{feature_name.replace('+', '_plus_')}"
    curves_dir = Path(curves_base_dir) / model_name
    curves_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoints:
        (curves_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Model factory
    def model_fn():
        return PeptideGNN(
            architecture=arch,
            in_channels=26,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=2,
            pooling='mean_max',
            geo_feature_dim=geo_dim
        )
    
    # Create CV splits
    if clusters is not None:
        cv = GroupKFold(n_splits=config['n_folds'])
        splits = list(cv.split(np.arange(len(labels)), labels, groups=clusters))
    else:
        cv = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
        splits = list(cv.split(np.arange(len(labels)), labels))
    
    # Cross-validation
    cv_results = {
        'auc_roc': [], 'auc_pr': [], 'f1': [], 
        'accuracy': [], 'mcc': []
    }
    
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Create dataloaders
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
        
        # Fresh model
        model = model_fn()
        
        # Train and capture history
        history, best_metrics = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config['epochs'],
            lr=config['lr'],
            patience=config['patience'],
            verbose=False
        )
        
        # Save training curves for this fold
        fold_df = pd.DataFrame({
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_auc_roc': history['val_auc_roc'],
            'val_f1': history['val_f1'],
        })
        fold_csv_path = curves_dir / f"fold_{fold+1}.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        
        # Also save as JSON for more detailed info
        fold_json = {
            'fold': fold + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'best_epoch': int(np.argmax(history['val_auc_roc']) + 1),
            'best_metrics': {k: float(v) for k, v in best_metrics.items()},
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }
        fold_json_path = curves_dir / f"fold_{fold+1}.json"
        with open(fold_json_path, 'w') as f:
            json.dump(fold_json, f, indent=2)
        
        y_true, y_prob = evaluate_probs(model, val_loader, device)
        val_probs_path = curves_dir / f"fold_{fold+1}_val_probs.json"
        with open(val_probs_path, 'w') as f:
            json.dump({
                'y_true': y_true.tolist(),
                'y_prob': y_prob.tolist(),
            }, f)
        
        all_histories.append(history)
        
        # Record results
        for metric in cv_results:
            cv_results[metric].append(best_metrics.get(metric, 0.0))
        
        if save_checkpoints:
            ckpt_path = curves_dir / "checkpoints" / f"fold_{fold+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
        
        print(f"      Fold {fold+1}: AUC={best_metrics['auc_roc']:.4f}, F1={best_metrics['f1']:.4f}")
    
    # Save summary for this model
    summary = {
        'model': model_name,
        'architecture': arch,
        'feature_set': feature_name,
        'geo_dim': geo_dim,
        'config': {
            'epochs': config['epochs'],
            'patience': config['patience'],
            'lr': config['lr'],
            'hidden_channels': config['hidden_channels'],
            'num_layers': config['num_layers'],
        },
        'cv_results': {
            'auc_roc': {'mean': float(np.mean(cv_results['auc_roc'])), 'std': float(np.std(cv_results['auc_roc'])), 'values': [float(x) for x in cv_results['auc_roc']]},
            'f1': {'mean': float(np.mean(cv_results['f1'])), 'std': float(np.std(cv_results['f1'])), 'values': [float(x) for x in cv_results['f1']]},
            'mcc': {'mean': float(np.mean(cv_results['mcc'])), 'std': float(np.std(cv_results['mcc'])), 'values': [float(x) for x in cv_results['mcc']]},
        }
    }
    
    summary_path = curves_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"      📁 Curves saved to: {curves_dir}")
    
    return cv_results


def parse_args():
    parser = argparse.ArgumentParser(description='Compare GCN, GAT, and EGNN architectures')
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save a .pt checkpoint for each fold of each model (under curves/<model>/checkpoints/)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*80)
    print("GNN Architecture Comparison: GCN vs GAT vs EGNN")
    print("="*80)
    
    # Setup
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n📊 Configuration:")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Patience: {CONFIG['patience']}")
    print(f"   Hidden channels: {CONFIG['hidden_channels']}")
    print(f"   Num layers: {CONFIG['num_layers']}")
    print(f"   CV folds: {CONFIG['n_folds']}")
    print(f"   Save checkpoints: {args.save_checkpoints}")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading Data...")
    print("-"*80)
    
    merged_df, qsar_cols = load_data_with_features(CONFIG)
    print(f"Loaded {len(merged_df)} peptides")
    
    # Get labels and clusters
    raw_labels = merged_df['label'].values
    labels = np.where(raw_labels == 1, 1, 0)
    clusters = merged_df['cluster_id'].values if 'cluster_id' in merged_df.columns else None
    
    if clusters is not None:
        print(f"Clusters: {len(np.unique(clusters))}")
    
    # Results storage
    all_results = {}
    
    # Create base directory for training curves
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    curves_base_dir = Path('results/gnn/curves') / f'run_{timestamp}'
    curves_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Training curves will be saved to: {curves_base_dir}")
    
    # Run experiments
    print("\n" + "="*80)
    print("Running Experiments (Cluster-Based 5-Fold CV)")
    print("="*80)
    
    for feature_name, feature_config in FEATURE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Feature Set: {feature_name}")
        print(f"{'='*60}")
        
        # Create dataset for this feature configuration
        feature_cols = create_dataset_with_features(
            merged_df, CONFIG,
            use_geo=feature_config['use_geo'],
            use_qsar=feature_config['use_qsar'],
            qsar_cols=qsar_cols
        )
        
        print(f"Extra features: {len(feature_cols)} dims")
        
        # Create dataset
        dataset = CustomPeptideDataset(
            merged_df, 
            CONFIG['pdb_dir'], 
            feature_cols if feature_cols else None,
            CONFIG['distance_threshold']
        )
        
        # Load all data
        print("Loading graphs...")
        all_data = [dataset[i] for i in range(len(dataset))]
        print(f"Loaded {len(all_data)} graphs")
        
        for arch in ARCHITECTURES:
            print(f"\n   🔬 {arch.upper()}...")
            
            cv_results = run_single_experiment(
                arch, feature_name, feature_config,
                all_data, labels, clusters, device, CONFIG,
                curves_base_dir,
                save_checkpoints=args.save_checkpoints
            )
            
            # Store results
            key = f"{arch.upper()} ({feature_name})"
            all_results[key] = {
                'auc_roc_mean': np.mean(cv_results['auc_roc']),
                'auc_roc_std': np.std(cv_results['auc_roc']),
                'f1_mean': np.mean(cv_results['f1']),
                'f1_std': np.std(cv_results['f1']),
                'mcc_mean': np.mean(cv_results['mcc']),
                'mcc_std': np.std(cv_results['mcc']),
            }
            
            print(f"      → AUC-ROC: {all_results[key]['auc_roc_mean']:.4f} ± {all_results[key]['auc_roc_std']:.4f}")
            print(f"      → F1:      {all_results[key]['f1_mean']:.4f} ± {all_results[key]['f1_std']:.4f}")
            print(f"      → MCC:     {all_results[key]['mcc_mean']:.4f} ± {all_results[key]['mcc_std']:.4f}")
    
    # Print final comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY: GNN Architecture Comparison")
    print("="*80)
    
    print("\n📊 Cluster-Based CV (Mean ± Std)")
    print("-"*100)
    print(f"{'Model':<35} {'AUC-ROC':<20} {'F1':<20} {'MCC':<20}")
    print("-"*100)
    
    # Group by feature set for cleaner display
    for feature_name in FEATURE_CONFIGS.keys():
        for arch in ARCHITECTURES:
            key = f"{arch.upper()} ({feature_name})"
            if key in all_results:
                r = all_results[key]
                print(f"{key:<35} "
                      f"{r['auc_roc_mean']:.4f} ± {r['auc_roc_std']:.4f}   "
                      f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}   "
                      f"{r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}")
        print()
    
    # Print comparison with MLP baselines
    print("\n" + "="*80)
    print("COMPARISON WITH MLP/SVM BASELINES (from previous experiments)")
    print("="*80)
    print("""
MLP/SVM Results (for reference):
---------------------------------
SVM (QSAR-12)              0.9275 ± 0.0157    0.8468 ± 0.0531    0.7026 ± 0.0889
MLP (QSAR-12)              0.9343 ± 0.0119    0.8617 ± 0.0328    0.7229 ± 0.0589
SVM (Geo-24)               0.9733 ± 0.0169    0.9291 ± 0.0247    0.8626 ± 0.0444
MLP (Geo-24)               0.9806 ± 0.0165    0.9395 ± 0.0184    0.8836 ± 0.0334
SVM (Combined-36)          0.9850 ± 0.0064    0.9374 ± 0.0292    0.8794 ± 0.0536
MLP (Combined-36)          0.9908 ± 0.0040    0.9453 ± 0.0243    0.8946 ± 0.0448
""")
    
    # Save results
    os.makedirs('results/gnn', exist_ok=True)
    
    results_dict = {
        'config': CONFIG,
        'results': all_results,
        'timestamp': timestamp,
        'curves_dir': str(curves_base_dir)
    }
    
    json_path = f'results/gnn/gnn_comparison_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    print(f"📁 Training curves saved to: {curves_base_dir}")
    if args.save_checkpoints:
        print(f"📁 Model checkpoints saved under: {curves_base_dir}/<model_name>/checkpoints/")
    
    print("\n✅ Comparison complete!")


if __name__ == '__main__':
    main()

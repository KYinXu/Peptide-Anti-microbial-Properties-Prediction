#!/usr/bin/env python3
"""
GNN feature ablation: full control over geometric (CSV) and node (PDB-derived) features.

Single run:
    python run_gnn_feature_ablation.py --node_exclude plddt --geo_exclude fraction_helix,fraction_sheet,fraction_coil
    python run_gnn_feature_ablation.py --geo radius_gyration,length,net_charge

Multi-run comparison (presets):
    python run_gnn_feature_ablation.py --runs all no_plddt graph_only
    python run_gnn_feature_ablation.py --runs no_bfactor   # exclude PDB B-factor (pLDDT) from node features
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
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnn.data_utils import PeptideGraphDataset, node_feature_keep_indices_from_exclude
from gnn.models import PeptideGNN
from gnn.train import cross_validate, print_cv_summary

# Node features (26-dim): indices 0-19 = one-hot AA (always kept); 20-25 = these (excludable via --node_exclude)
NODE_FEATURE_NAMES = ['plddt', 'hydrophobicity', 'charge', 'mw', 'volume', 'rel_position']

DEFAULT_GEO_COLS = [
    'radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
    'centroid_distance_mean', 'centroid_distance_std',
    'fraction_helix', 'fraction_sheet', 'fraction_coil',
    'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa',
    'length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment',
    'curvature_mean', 'curvature_std', 'curvature_max',
    'torsion_mean', 'torsion_std',
]

RUN_PRESETS = {
    'all': {'node_exclude': [], 'use_geo': True, 'geo_exclude': []},
    'no_plddt': {'node_exclude': ['plddt'], 'use_geo': True, 'geo_exclude': []},
    'no_bfactor': {'node_exclude': ['plddt'], 'use_geo': True, 'geo_exclude': []},
    'graph_only': {'node_exclude': [], 'use_geo': False, 'geo_exclude': []},
    'no_charge': {'node_exclude': ['charge'], 'use_geo': True, 'geo_exclude': []},
    'no_hydrophobicity': {'node_exclude': ['hydrophobicity'], 'use_geo': True, 'geo_exclude': []},
    'none': {'node_exclude': NODE_FEATURE_NAMES, 'use_geo': False, 'geo_exclude': []},
}


def parse_args():
    p = argparse.ArgumentParser(description='GNN feature ablation: full control over geo and node features')
    p.add_argument('--csv_path', type=str, default='data/gnn_training_dataset/alpha-helix_only/geometric_features_clustered.csv')
    p.add_argument('--pdb_dir', type=str, default='data/gnn_training_dataset/alpha-helix_only')
    p.add_argument('--architecture', type=str, default='gcn', choices=['gcn', 'gat', 'egnn'])
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default='results/gnn/feature_ablation')
    p.add_argument('--geo', type=str, default='',
                   help='Comma-separated geometric column names to use (overrides --geo_exclude)')
    p.add_argument('--geo_exclude', type=str, default='',
                   help='Comma-separated names to remove from default geometric columns')
    p.add_argument('--node_exclude', type=str, default='',
                   help=f'Comma-separated node feature names to exclude. Options: {", ".join(NODE_FEATURE_NAMES)}')
    p.add_argument('--runs', type=str, nargs='*',
                   help='Preset run names to compare (e.g. all no_plddt graph_only). If not set, single run with --geo/--geo_exclude/--node_exclude')
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_geo_cols(geo_spec: str, geo_exclude_spec: str) -> list:
    if geo_spec:
        return [c.strip() for c in geo_spec.split(',') if c.strip()]
    exclude = [c.strip() for c in geo_exclude_spec.split(',') if c.strip()]
    return [c for c in DEFAULT_GEO_COLS if c not in exclude]


def run_one_run(geo_cols, use_geo, node_feature_keep_indices, run_name, args, splits, device):
    use_geo = use_geo and len(geo_cols) > 0
    dataset = PeptideGraphDataset(
        csv_path=args.csv_path,
        pdb_dir=args.pdb_dir,
        distance_threshold=8.0,
        use_geometric_features=use_geo,
        geometric_feature_cols=geo_cols if geo_cols else None,
        node_feature_keep_indices=node_feature_keep_indices,
    )
    all_data = [dataset[i] for i in range(len(dataset))]
    in_channels = len(node_feature_keep_indices)
    geo_dim = 0
    if use_geo and geo_cols:
        for d in all_data:
            if hasattr(d, 'geo_features'):
                geo_dim = int(d.geo_features.shape[1])
                break

    def model_fn():
        return PeptideGNN(
            architecture=args.architecture,
            in_channels=in_channels,
            hidden_channels=64,
            num_layers=3,
            dropout=0.2,
            num_classes=2,
            pooling='mean_max',
            geo_feature_dim=geo_dim,
        )

    return cross_validate(
        model_fn=model_fn,
        dataset=all_data,
        cv_splits=splits,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        verbose=False,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(args.csv_path)
    labels = np.where(df['label'].values == 1, 1, 0)
    clusters = df['cluster_id'].values if 'cluster_id' in df.columns else None

    if clusters is not None:
        cv = GroupKFold(n_splits=args.n_folds)
        splits = list(cv.split(np.arange(len(labels)), labels, groups=clusters))
    else:
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(cv.split(np.arange(len(labels)), labels))

    if args.runs:
        for name in args.runs:
            if name not in RUN_PRESETS:
                print('Unknown preset:', name, 'Available:', list(RUN_PRESETS.keys()))
                sys.exit(1)

    print('='*60)
    print('GNN feature ablation')
    print('='*60)
    print(f'Splits: {args.n_folds}-fold')
    print(f'Device: {device}\n')

    all_results = []
    if args.runs:
        for name in args.runs:
            p = RUN_PRESETS[name]
            use_geo = p['use_geo']
            geo_cols = resolve_geo_cols('', ','.join(p['geo_exclude'])) if use_geo else []
            node_keep = node_feature_keep_indices_from_exclude(p['node_exclude'])
            print(f"\n--- {name} ---")
            cv_results = run_one_run(geo_cols, use_geo, node_keep, name, args, splits, device)
            summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in cv_results.items()}
            all_results.append({'config': name, 'summary': summary, 'cv_results': cv_results})
            print_cv_summary(cv_results)
    else:
        geo_cols = resolve_geo_cols(args.geo, args.geo_exclude)
        use_geo = len(geo_cols) > 0
        node_exclude = [x.strip() for x in args.node_exclude.split(',') if x.strip()] if args.node_exclude else []
        node_keep = node_feature_keep_indices_from_exclude(node_exclude)
        print("\n--- single ---")
        cv_results = run_one_run(geo_cols, use_geo, node_keep, 'single', args, splits, device)
        summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in cv_results.items()}
        all_results.append({'config': 'single', 'summary': summary, 'cv_results': cv_results})
        print_cv_summary(cv_results)

    metrics = ['auc_roc', 'f1', 'accuracy', 'precision', 'recall', 'mcc']
    print('\n' + '='*60)
    print('Comparison (mean ± std)')
    print('='*60)
    rows = []
    for r in all_results:
        row = {'config': r['config']}
        for m in metrics:
            if m in r['summary']:
                row[m] = f"{r['summary'][m]['mean']:.4f} ± {r['summary'][m]['std']:.4f}"
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    comp_df.to_csv(os.path.join(args.output_dir, f'ablation_{ts}.csv'), index=False)
    out = {
        'timestamp': ts,
        'args': vars(args),
        'configs': [r['config'] for r in all_results],
        'comparison': [{'config': r['config'], 'summary': r['summary']} for r in all_results],
        'cv_results_per_config': {
            r['config']: {k: [float(x) for x in v] for k, v in r['cv_results'].items()}
            for r in all_results
        }
    }
    with open(os.path.join(args.output_dir, f'ablation_{ts}.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {args.output_dir}/ablation_{ts}.csv and .json')


if __name__ == '__main__':
    main()

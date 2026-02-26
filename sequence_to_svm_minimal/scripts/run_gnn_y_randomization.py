#!/usr/bin/env python3
"""
Y-randomization sanity check: run the same GNN training with labels randomly permuted.

Expected: AUC ~0.5 when labels are shuffled. If AUC >> 0.5, investigate leakage or bugs.

Usage:
    python run_gnn_y_randomization.py
    python run_gnn_y_randomization.py --compare   # also run with real labels and print comparison
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

from gnn.data_utils import PeptideGraphDataset
from gnn.models import PeptideGNN
from gnn.train import cross_validate, print_cv_summary


def parse_args():
    p = argparse.ArgumentParser(description='Y-randomization sanity check for GNN training')
    p.add_argument('--csv_path', type=str,
                   default='data/training_dataset/geometric_features_clustered.csv')
    p.add_argument('--pdb_dir', type=str, default='data/training_dataset')
    p.add_argument('--architecture', type=str, default='gcn', choices=['gcn', 'gat', 'egnn'])
    p.add_argument('--hidden_channels', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--pooling', type=str, default='mean_max', choices=['mean', 'max', 'sum', 'mean_max'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--distance_threshold', type=float, default=8.0)
    p.add_argument('--use_geo_features', action='store_true')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--output_dir', type=str, default='results/gnn')
    p.add_argument('--compare', action='store_true',
                   help='Also run CV with real labels and print comparison table')
    p.add_argument('--verbose', action='store_true', help='Print per-fold results')
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print('='*60)
    print('Y-randomization sanity check')
    print('='*60)
    print(f'Labels will be permuted (seed={args.seed}). Expected: AUC ~0.5.')
    print(f'Device: {device}\n')

    df = pd.read_csv(args.csv_path)
    labels = np.where(df['label'].values == 1, 1, 0)
    clusters = df['cluster_id'].values if 'cluster_id' in df.columns else None

    if clusters is not None:
        cv = GroupKFold(n_splits=args.n_folds)
        splits = list(cv.split(np.arange(len(labels)), labels, groups=clusters))
    else:
        cv = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        splits = list(cv.split(np.arange(len(labels)), labels))

    dataset = PeptideGraphDataset(
        csv_path=args.csv_path,
        pdb_dir=args.pdb_dir,
        distance_threshold=args.distance_threshold,
        use_geometric_features=args.use_geo_features,
    )
    print('Loading graphs...')
    all_data = [dataset[i] for i in range(len(dataset))]
    print(f'Loaded {len(all_data)} graphs\n')

    geo_dim = 0
    if args.use_geo_features:
        for d in all_data:
            if hasattr(d, 'geo_features'):
                geo_dim = int(d.geo_features.shape[1])
                break
        print(f'Using geometric feature dimension: {geo_dim}\n')

    def model_fn():
        return PeptideGNN(
            architecture=args.architecture,
            in_channels=26,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            pooling=args.pooling,
            geo_feature_dim=geo_dim,
        )

    rng = np.random.default_rng(args.seed)
    shuffled_labels = rng.permutation(labels)
    all_data_shuffled = []
    for i in range(len(all_data)):
        d = all_data[i].clone()
        d.y = torch.tensor([shuffled_labels[i]], dtype=torch.long)
        all_data_shuffled.append(d)

    print('--- CV with shuffled labels ---')
    shuffled_cv = cross_validate(
        model_fn=model_fn,
        dataset=all_data_shuffled,
        cv_splits=splits,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        verbose=args.verbose,
    )
    print_cv_summary(shuffled_cv)

    baseline_cv = None
    if args.compare:
        print('\n--- CV with real labels (baseline) ---')
        baseline_cv = cross_validate(
            model_fn=model_fn,
            dataset=all_data,
            cv_splits=splits,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            verbose=args.verbose,
        )
        print_cv_summary(baseline_cv)
        print('\n--- Comparison ---')
        metrics = ['auc_roc', 'f1', 'accuracy', 'precision', 'recall', 'mcc']
        print(f'{"metric":12s}  {"real (mean ± std)":25s}  {"shuffled (mean ± std)":25s}')
        print('-'*70)
        for m in metrics:
            real_m, real_s = np.mean(baseline_cv[m]), np.std(baseline_cv[m])
            shuf_m, shuf_s = np.mean(shuffled_cv[m]), np.std(shuffled_cv[m])
            print(f'{m:12s}  {real_m:.4f} ± {real_s:.4f}           {shuf_m:.4f} ± {shuf_s:.4f}')

    print('\nExpected: AUC ~0.5 for random labels. If AUC >> 0.5, check for leakage or bug.')

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = {
        'timestamp': ts,
        'args': vars(args),
        'permutation_seed': args.seed,
        'shuffled_cv_results': {k: [float(x) for x in v] for k, v in shuffled_cv.items()},
        'shuffled_cv_summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                               for k, v in shuffled_cv.items()},
    }
    if baseline_cv is not None:
        out['baseline_cv_results'] = {k: [float(x) for x in v] for k, v in baseline_cv.items()}
        out['baseline_cv_summary'] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                                      for k, v in baseline_cv.items()}
    path = os.path.join(args.output_dir, f'y_randomization_{ts}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nResults saved to {path}')


if __name__ == '__main__':
    main()

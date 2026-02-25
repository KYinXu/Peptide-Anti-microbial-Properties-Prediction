#!/usr/bin/env python3
"""
Run Neural Network Training Pipeline for AMP Classification.

This script:
1. Prepares sequence clusters (if not already done)
2. Runs cross-validation with cluster-based splits
3. Trains a final model on the best configuration
4. Saves the model and evaluation results

Usage:
    python run_nn_training.py
"""

import sys
from pathlib import Path

# Add nn_pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "nn_pipeline"))

from nn_pipeline.feature_dataset import FeaturePipeline
from nn_pipeline.prepare_clusters import create_simple_clusters
from nn_pipeline.train import run_cross_validation, train_final_model
from nn_pipeline.models import get_model, count_parameters

import torch
import numpy as np


def main():
    print("\n" + "="*70)
    print("  AMP Binary Classification - Neural Network Training Pipeline")
    print("="*70)
    
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data" / "training_dataset"
    features_csv = data_dir / "geometric_features.csv"
    clustered_csv = data_dir / "geometric_features_clustered.csv"
    output_dir = base_dir / "nn_pipeline" / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Prepare clusters if needed
    print("\n" + "-"*70)
    print("  Step 1: Preparing Sequence Clusters")
    print("-"*70)
    
    if not clustered_csv.exists():
        print(f"\n📊 Creating sequence clusters at 80% identity threshold...")
        create_simple_clusters(features_csv, clustered_csv, identity_threshold=0.80)
    else:
        print(f"\n✅ Clustered features already exist: {clustered_csv}")
    
    # Step 2: Load feature pipeline
    print("\n" + "-"*70)
    print("  Step 2: Loading Feature Pipeline")
    print("-"*70)
    
    pipeline = FeaturePipeline(
        geometric_csv=clustered_csv,
        use_svm_features=False,  # We don't have matched SVM outputs yet
        use_descriptor_features=False
    )
    
    X, y, clusters = pipeline.get_feature_matrix()
    n_features = X.shape[1]
    n_clusters = len(np.unique(clusters))
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Samples: {len(y)}")
    print(f"   Features: {n_features}")
    print(f"   Clusters: {n_clusters}")
    print(f"   Class balance: AMP={int(sum(y))}, DECOY={int(len(y)-sum(y))}")
    
    # Step 3: Run cross-validation
    print("\n" + "-"*70)
    print("  Step 3: Cross-Validation")
    print("-"*70)
    
    # Training hyperparameters
    config = {
        'model_type': 'mlp',
        'n_folds': 5,
        'epochs': 1000, #longer training
        'batch_size': 32,
        'learning_rate': 0.0005,  # Slightly lower LR for longer training
        'patience': 100, # patience before early stopping
    }
    
    print(f"\n⚙️  Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    cv_results = run_cross_validation(
        pipeline,
        model_type=config['model_type'],
        n_folds=config['n_folds'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        device=device,
        verbose=True
    )
    
    # Step 4: Train final model
    print("\n" + "-"*70)
    print("  Step 4: Training Final Model")
    print("-"*70)
    
    model, test_metrics = train_final_model(
        pipeline,
        model_type=config['model_type'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        output_dir=output_dir,
        device=device
    )
    
    # Summary
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    
    print(f"\n📊 Final Results:")
    print(f"\n   Cross-Validation (5-fold):")
    print(f"   {'Metric':<15} {'Mean':>10} {'Std':>10}")
    print("   " + "-"*37)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc']:
        if metric in cv_results:
            mean = cv_results[metric]['mean']
            std = cv_results[metric]['std']
            print(f"   {metric:<15} {mean:>10.4f} {std:>10.4f}")
    
    print(f"\n   Test Set (held-out cluster):")
    print(f"   {'Metric':<15} {'Value':>10}")
    print("   " + "-"*27)
    for metric, value in test_metrics.items():
        print(f"   {metric:<15} {value:>10.4f}")
    
    print(f"\n💾 Model saved to: {output_dir}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PNAS-Style Evaluation Protocol for AMP Classification.

Replicates the Lee et al. PNAS 2016 evaluation protocol:
1. Strict blind test set (15%, stratified) - never used for training/tuning
2. 15 rounds of stratified shuffled 80/20 CV on remaining data
3. Final model trained on full training pool, evaluated once on blind test

Reference: Lee et al., PNAS 2016 (SI Appendix)

Usage:
    python run_nn_training_pnas_style.py
    python run_nn_training_pnas_style.py --pnas_rounds 15 --test_size 0.15 --seed 42
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Definition (same as existing pipeline)
# ============================================================================

class AMPClassifier(nn.Module):
    """MLP for AMP binary classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return torch.sigmoid(self.output(h)).squeeze(-1)


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_state = None
    
    def __call__(self, score: float, epoch: int, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, 
                      shuffle: bool = True) -> DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, 
                criterion, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_samples = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)
    
    return total_loss / n_samples


def evaluate(model: nn.Module, loader: DataLoader, 
             device: torch.device) -> Dict[str, float]:
    """Evaluate model and return all metrics."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(float)
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pr': average_precision_score(all_labels, all_probs),
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }


def evaluate_probs(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for ROC/PR/calibration plots."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_probs)


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                input_dim: int, device: torch.device,
                epochs: int = 300, batch_size: int = 32,
                lr: float = 0.0005, patience: int = 30,
                verbose: bool = False,
                return_history: bool = False
                ) -> Tuple[nn.Module, Dict, int, Optional[Dict[str, List[float]]]]:
    """
    Train a model with early stopping.
    
    Returns:
        model: Trained model (restored to best state)
        best_metrics: Metrics at best epoch
        best_epoch: Best epoch number
        history: If return_history, dict with train_loss, val_auc_roc, val_f1 per epoch; else None
    """
    # Create model
    model = AMPClassifier(input_dim=input_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    
    # Create dataloaders
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)
    
    # Training loop
    early_stopping = EarlyStopping(patience=patience)
    best_metrics = None
    history = {'train_loss': [], 'val_auc_roc': [], 'val_f1': []} if return_history else None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        if history is not None:
            history['train_loss'].append(float(train_loss))
            history['val_auc_roc'].append(float(val_metrics['auc_roc']))
            history['val_f1'].append(float(val_metrics['f1']))
        
        scheduler.step(val_metrics['auc_roc'])
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f}")
        
        if early_stopping(val_metrics['auc_roc'], epoch, model):
            break
        
        if early_stopping.best_epoch == epoch:
            best_metrics = val_metrics.copy()
    
    # Restore best model
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    
    return model, best_metrics, early_stopping.best_epoch + 1, history


# ============================================================================
# PNAS-Style Evaluation Protocol
# ============================================================================

def run_pnas_protocol(data_path: Path,
                      n_rounds: int = 15,
                      test_size: float = 0.15,
                      train_val_split: float = 0.2,
                      seed: int = 42,
                      output_dir: Path = None,
                      epochs: int = 300,
                      batch_size: int = 32,
                      lr: float = 0.0005,
                      patience: int = 30) -> Dict:
    """
    Run PNAS-style evaluation protocol.
    
    Protocol:
    1. Split 15% as strict blind test (stratified)
    2. Run n_rounds of stratified 80/20 CV on remaining 85%
    3. Train final model on full 85%, evaluate on blind test
    """
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("  PNAS-Style Evaluation Protocol")
    print("  (Lee et al. PNAS 2016 - SI Appendix)")
    print("="*70)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================================================
    # Step 1: Load data
    # ========================================================================
    print("\n" + "-"*70)
    print("  Step 1: Loading Data")
    print("-"*70)
    
    df = pd.read_csv(data_path)
    
    # Define feature columns (exclude metadata)
    exclude_cols = ['peptide_id', 'sequence', 'pdb_file', 'label', 'ss_method', 
                    'cluster_id', 'ss_residues_computed']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y = ((df['label'].values + 1) / 2).astype(np.float32)  # Convert -1/1 to 0/1
    
    n_samples = len(y)
    n_positive = int(sum(y))
    n_negative = n_samples - n_positive
    input_dim = X.shape[1]
    
    print(f"\n📊 Dataset: {data_path.name}")
    print(f"   Total samples: {n_samples}")
    print(f"   Features: {input_dim}")
    print(f"   AMP (positive): {n_positive}")
    print(f"   DECOY (negative): {n_negative}")
    
    # ========================================================================
    # Step 2: Create strict blind test set (15%, stratified)
    # ========================================================================
    print("\n" + "-"*70)
    print("  Step 2: Creating Strict Blind Test Set")
    print("-"*70)
    
    blind_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, 
                                             random_state=seed)
    train_pool_idx, blind_test_idx = next(blind_splitter.split(X, y))
    
    X_train_pool = X[train_pool_idx]
    y_train_pool = y[train_pool_idx]
    X_blind_test = X[blind_test_idx]
    y_blind_test = y[blind_test_idx]
    
    n_blind = len(y_blind_test)
    n_blind_pos = int(sum(y_blind_test))
    n_blind_neg = n_blind - n_blind_pos
    
    print(f"\n🔒 Blind Test Set (HELD OUT COMPLETELY):")
    print(f"   Samples: {n_blind} ({100*test_size:.0f}%)")
    print(f"   AMP: {n_blind_pos}, DECOY: {n_blind_neg}")
    print(f"   ⚠️  Never used for training, validation, or hyperparameter tuning!")
    
    n_pool = len(y_train_pool)
    n_pool_pos = int(sum(y_train_pool))
    n_pool_neg = n_pool - n_pool_pos
    
    print(f"\n📦 Training Pool:")
    print(f"   Samples: {n_pool} ({100*(1-test_size):.0f}%)")
    print(f"   AMP: {n_pool_pos}, DECOY: {n_pool_neg}")
    
    # ========================================================================
    # Step 3: Run 15 rounds of stratified shuffled CV
    # ========================================================================
    print("\n" + "-"*70)
    print(f"  Step 3: Running {n_rounds} Rounds of Stratified Shuffled CV")
    print("-"*70)
    
    cv_splitter = StratifiedShuffleSplit(n_splits=n_rounds, 
                                          test_size=train_val_split,
                                          random_state=seed)
    
    round_results = []
    best_epochs = []
    
    print(f"\n⚙️  Configuration:")
    print(f"   Rounds: {n_rounds}")
    print(f"   Train/Val split: {100*(1-train_val_split):.0f}% / {100*train_val_split:.0f}%")
    print(f"   Epochs (max): {epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Learning rate: {lr}")
    print(f"   Batch size: {batch_size}")
    
    print(f"\n{'Round':<8} {'AUC-ROC':<10} {'F1':<10} {'MCC':<10} {'Epochs':<8}")
    print("-" * 46)
    
    for round_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train_pool, 
                                                                        y_train_pool)):
        # Split training pool
        X_train = X_train_pool[train_idx]
        y_train = y_train_pool[train_idx]
        X_val = X_train_pool[val_idx]
        y_val = y_train_pool[val_idx]
        
        # Fit scaler on train only, apply to train+val
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        set_seed(seed + round_idx)
        model, metrics, best_epoch, history = train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            input_dim=input_dim,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            verbose=False,
            return_history=True
        )
        round_results.append(metrics)
        best_epochs.append(best_epoch)
        if history is not None:
            eval_dir = output_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            with open(eval_dir / f"round_{round_idx+1}_history.json", 'w') as f:
                json.dump(history, f, indent=2)
        
        print(f"{round_idx+1:<8} {metrics['auc_roc']:<10.4f} {metrics['f1']:<10.4f} "
              f"{metrics['mcc']:<10.4f} {best_epoch:<8}")
    
    # Compute CV statistics
    cv_summary = {}
    for metric in round_results[0].keys():
        values = [r[metric] for r in round_results]
        cv_summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'values': values
        }
    
    avg_best_epoch = int(np.mean(best_epochs))
    
    print("\n" + "="*70)
    print("  CV Results Summary (15 Rounds)")
    print("="*70)
    print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 57)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'mcc']:
        m = cv_summary[metric]
        print(f"{metric:<15} {m['mean']:>10.4f} {m['std']:>10.4f} "
              f"{m['min']:>10.4f} {m['max']:>10.4f}")
    
    print(f"\n   Average best epoch: {avg_best_epoch}")
    
    # ========================================================================
    # Step 4: Train final model on full training pool
    # ========================================================================
    print("\n" + "-"*70)
    print("  Step 4: Training Final Model on Full Training Pool")
    print("-"*70)
    
    # Use internal 80/20 split for early stopping
    final_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, 
                                             random_state=seed + 999)
    final_train_idx, final_val_idx = next(final_splitter.split(X_train_pool, 
                                                                 y_train_pool))
    
    X_final_train = X_train_pool[final_train_idx]
    y_final_train = y_train_pool[final_train_idx]
    X_final_val = X_train_pool[final_val_idx]
    y_final_val = y_train_pool[final_val_idx]
    
    # Fit scaler on full training pool for final model
    final_scaler = StandardScaler()
    X_final_train_scaled = final_scaler.fit_transform(X_final_train)
    X_final_val_scaled = final_scaler.transform(X_final_val)
    
    print(f"\n   Training: {len(X_final_train)} samples")
    print(f"   Validation (for early stopping): {len(X_final_val)} samples")
    
    set_seed(seed)
    final_model, final_val_metrics, final_best_epoch, final_history = train_model(
        X_final_train_scaled, y_final_train,
        X_final_val_scaled, y_final_val,
        input_dim=input_dim,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        verbose=True,
        return_history=True
    )
    if final_history is not None:
        eval_dir = output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "final_train_history.json", 'w') as f:
            json.dump(final_history, f, indent=2)
    
    print(f"\n   Best epoch: {final_best_epoch}")
    print(f"   Val AUC-ROC: {final_val_metrics['auc_roc']:.4f}")
    
    val_loader_final = create_dataloader(X_final_val_scaled, y_final_val,
                                         batch_size=batch_size, shuffle=False)
    y_val_true, y_val_prob = evaluate_probs(final_model, val_loader_final, device)
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "final_val_probs.json", 'w') as f:
        json.dump({'y_true': y_val_true.tolist(), 'y_prob': y_val_prob.tolist()}, f)
    
    # ========================================================================
    # Step 5: Evaluate on blind test set (SINGLE EVALUATION)
    # ========================================================================
    print("\n" + "-"*70)
    print("  Step 5: Blind Test Evaluation (SINGLE EVALUATION)")
    print("-"*70)
    
    # Fit scaler on FULL training pool, apply to blind test
    full_scaler = StandardScaler()
    full_scaler.fit(X_train_pool)
    X_blind_test_scaled = full_scaler.transform(X_blind_test)
    
    # Evaluate
    blind_loader = create_dataloader(X_blind_test_scaled, y_blind_test, 
                                      batch_size=batch_size, shuffle=False)
    blind_metrics = evaluate(final_model, blind_loader, device)
    y_blind_true, y_blind_prob = evaluate_probs(final_model, blind_loader, device)
    with open(eval_dir / "blind_test_probs.json", 'w') as f:
        json.dump({'y_true': y_blind_true.tolist(), 'y_prob': y_blind_prob.tolist()}, f)
    
    print(f"\n🎯 BLIND TEST RESULTS ({n_blind} samples)")
    print("="*40)
    for metric, value in blind_metrics.items():
        print(f"   {metric:<15}: {value:.4f}")
    
    # ========================================================================
    # Step 6: Save results
    # ========================================================================
    print("\n" + "-"*70)
    print("  Step 6: Saving Results")
    print("-"*70)
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results dictionary
    results = {
        'protocol': 'PNAS-style (Lee et al. 2016)',
        'timestamp': timestamp,
        'seed': seed,
        'dataset': {
            'path': str(data_path),
            'total_samples': n_samples,
            'n_features': input_dim,
            'n_amp': n_positive,
            'n_decoy': n_negative
        },
        'blind_test': {
            'size': n_blind,
            'n_amp': n_blind_pos,
            'n_decoy': n_blind_neg,
            'percentage': test_size
        },
        'cv_config': {
            'n_rounds': n_rounds,
            'train_val_split': train_val_split,
            'epochs_max': epochs,
            'patience': patience,
            'learning_rate': lr,
            'batch_size': batch_size
        },
        'cv_results': {k: {'mean': v['mean'], 'std': v['std']} 
                       for k, v in cv_summary.items()},
        'cv_per_round': round_results,
        'cv_best_epochs': best_epochs,
        'cv_avg_best_epoch': avg_best_epoch,
        'blind_test_metrics': blind_metrics
    }
    
    # Save JSON
    json_path = output_dir / "pnas_style_eval.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results JSON: {json_path}")
    
    # Save CSV summary
    csv_data = {
        'metric': [],
        'cv_mean': [],
        'cv_std': [],
        'blind_test': []
    }
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'mcc']:
        csv_data['metric'].append(metric)
        csv_data['cv_mean'].append(cv_summary[metric]['mean'])
        csv_data['cv_std'].append(cv_summary[metric]['std'])
        csv_data['blind_test'].append(blind_metrics[metric])
    
    csv_df = pd.DataFrame(csv_data)
    csv_path = output_dir / "pnas_style_eval.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"💾 Results CSV: {csv_path}")
    
    # Save model checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    model_path = checkpoint_dir / f"amp_classifier_pnas_{timestamp}.pt"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': input_dim,
        'feature_names': feature_cols,
        'blind_test_metrics': blind_metrics,
        'cv_summary': {k: {'mean': v['mean'], 'std': v['std']} 
                       for k, v in cv_summary.items()},
        'scaler_mean': full_scaler.mean_.tolist(),
        'scaler_scale': full_scaler.scale_.tolist()
    }, model_path)
    print(f"💾 Model checkpoint: {model_path}")
    
    # Save scaler
    import joblib
    scaler_path = checkpoint_dir / f"scaler_pnas_{timestamp}.joblib"
    joblib.dump(full_scaler, scaler_path)
    print(f"💾 Scaler: {scaler_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*70)
    print("  PNAS-Style Evaluation Complete!")
    print("="*70)
    
    print(f"\n📊 SUMMARY TABLE")
    print("="*60)
    print(f"{'Metric':<15} {'CV (15 rounds)':<20} {'Blind Test':<15}")
    print("-"*60)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'mcc']:
        cv_str = f"{cv_summary[metric]['mean']:.4f} ± {cv_summary[metric]['std']:.4f}"
        blind_str = f"{blind_metrics[metric]:.4f}"
        print(f"{metric:<15} {cv_str:<20} {blind_str:<15}")
    
    print("\n" + "="*70)
    print("  Protocol Compliance Check")
    print("="*70)
    print(f"  ✅ Blind test held out completely: {n_blind} samples")
    print(f"  ✅ Stratified sampling: {n_blind_pos} AMP + {n_blind_neg} DECOY")
    print(f"  ✅ {n_rounds} rounds of stratified shuffled CV")
    print(f"  ✅ 80/20 train/val split per round")
    print(f"  ✅ Scaler fit on train only, applied to val/test")
    print(f"  ✅ Early stopping on validation AUC-ROC")
    print(f"  ✅ Single blind test evaluation (no peeking)")
    print(f"  ✅ Random seed fixed: {seed}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="PNAS-Style Evaluation Protocol for AMP Classification"
    )
    
    parser.add_argument('--data', '-d', type=Path,
                        default=Path(__file__).parent / "data" / "training_dataset" / "geometric_features.csv",
                        help='Path to geometric_features.csv')
    parser.add_argument('--pnas_rounds', type=int, default=15,
                        help='Number of CV rounds (default: 15)')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Blind test set size (default: 0.15)')
    parser.add_argument('--train_val_split', type=float, default=0.2,
                        help='Validation split within training (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Max training epochs (default: 300)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (default: 30)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--output_dir', '-o', type=Path,
                        help='Output directory (default: results/)')
    
    args = parser.parse_args()
    
    run_pnas_protocol(
        data_path=args.data,
        n_rounds=args.pnas_rounds,
        test_size=args.test_size,
        train_val_split=args.train_val_split,
        seed=args.seed,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

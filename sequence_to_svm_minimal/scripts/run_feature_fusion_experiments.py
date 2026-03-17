#!/usr/bin/env python3
"""
Feature Fusion Experiments for AMP Classification.

Compares different feature sets and models:
1. SVM on QSAR-12 only
2. MLP on QSAR-12 only
3. SVM on QSAR-12 + Geometric-24
4. MLP on QSAR-12 + Geometric-24
5. Two-Tower Fusion MLP (optional)

Evaluation Protocols:
A) Cluster-based GroupKFold (rigorous)
B) PNAS-style blind test + 15-round shuffled CV

Usage:
    python run_feature_fusion_experiments.py
    python run_feature_fusion_experiments.py --geo-csv path/to/geo.csv --qsar-cache path/to/qsar.csv --results-dir path/to/out
"""

import argparse
import json
import random
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
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
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

# ============================================================================
# QSAR-12 Descriptor Computation (reuse existing pipeline)
# ============================================================================

def compute_qsar12_descriptors(sequences: List[str], peptide_ids: List[str]) -> pd.DataFrame:
    """
    Compute QSAR-12 descriptors for peptide sequences.
    
    Features:
    1. netCharge - Net electrical charge
    2-6. Dipeptide compositions (FC, LW, DP, NK, AE)
    7. pcMK - M/(M+K) ratio
    8. _SolventAccessibilityD1025 - Solvent accessibility
    9-10. tau2, tau4 - Sequence-order coupling
    11-12. QSO50, QSO29 - Quasi-sequence-order descriptors
    """
    from propy.PyPro import GetProDes
    from propy import ProCheck
    
    print(f"   Computing QSAR-12 descriptors for {len(sequences)} sequences...")
    
    # Charge dictionary
    charge_dict = {"A":0, "C":0, "D":-1, "E":-1, "F":0, "G":0, "H":1, "I":0, 
                   "K":1, "L":0, "M":0, "N":0, "P":0, "Q":0, "R":1, "S":0, 
                   "T":0, "V":0, "W":0, "Y":0}
    
    results = []
    for i, (seq, pid) in enumerate(zip(sequences, peptide_ids)):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(sequences)}...")
        
        try:
            # Check valid protein sequence
            if ProCheck.ProteinCheck(seq) == 0:
                raise ValueError(f"Invalid sequence: {seq}")
            
            Des = GetProDes(seq)
            row = {'peptide_id': pid, 'sequence': seq}
            
            # 1. Net charge
            row['netCharge'] = sum([charge_dict.get(x, 0) for x in seq])
            
            # 2-6. Dipeptide compositions
            dpc = Des.GetDPComp()
            for handle in ['FC', 'LW', 'DP', 'NK', 'AE']:
                row[handle] = float('%.2f' % dpc.get(handle, 0))
            
            # 7. pcMK (M/K ratio)
            n_m = sum(1 for x in seq if x == 'M')
            n_k = sum(1 for x in seq if x == 'K')
            row['pcMK'] = 0 if n_m == 0 else float(n_m) / float(n_m + n_k) if (n_m + n_k) > 0 else 0
            
            # 8. Solvent accessibility
            ctd = Des.GetCTD()
            row['_SolventAccessibilityD1025'] = ctd.get('_SolventAccessibilityD1025', 0)
            
            # 9-12. Sequence-order features (tau2, tau4, QSO50, QSO29)
            # Use simplified version without custom AAIndex
            try:
                socn = Des.GetSOCN(maxlag=30)
                seq_len = len(seq)
                row['tau2_GRAR740104'] = socn.get('tau2', 0) / float(seq_len - 2) if seq_len > 2 else 0
                row['tau4_GRAR740104'] = socn.get('tau4', 0) / float(seq_len - 4) if seq_len > 4 else 0
            except:
                row['tau2_GRAR740104'] = 0
                row['tau4_GRAR740104'] = 0
            
            try:
                qso = Des.GetQSO(maxlag=30, weight=0.05)
                row['QSO50_GRAR740104'] = qso.get('QSO50', 0)
                row['QSO29_GRAR740104'] = qso.get('QSO29', 0)
            except:
                row['QSO50_GRAR740104'] = 0
                row['QSO29_GRAR740104'] = 0
            
            results.append(row)
            
        except Exception as e:
            # Fill with zeros on error
            row = {'peptide_id': pid, 'sequence': seq}
            for name in ['netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK',
                        '_SolventAccessibilityD1025', 'tau2_GRAR740104', 
                        'tau4_GRAR740104', 'QSO50_GRAR740104', 'QSO29_GRAR740104']:
                row[name] = 0
            results.append(row)
    
    df = pd.DataFrame(results)
    print(f"   ✅ Computed QSAR-12 for {len(df)} peptides")
    return df


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Definitions
# ============================================================================

class AMPClassifier(nn.Module):
    """Standard MLP classifier."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
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
    
    def forward(self, x):
        return torch.sigmoid(self.output(self.hidden(x))).squeeze(-1)


class TwoTowerFusionMLP(nn.Module):
    """
    Two-tower architecture with residual fusion.
    Separate encoders for QSAR and geometric features, then fusion.
    """
    
    def __init__(self, qsar_dim: int, geo_dim: int, 
                 tower_hidden: int = 64, fusion_hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        
        # QSAR tower
        self.qsar_tower = nn.Sequential(
            nn.Linear(qsar_dim, tower_hidden),
            nn.BatchNorm1d(tower_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden, tower_hidden // 2),
            nn.ReLU()
        )
        
        # Geometric tower
        self.geo_tower = nn.Sequential(
            nn.Linear(geo_dim, tower_hidden),
            nn.BatchNorm1d(tower_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden, tower_hidden // 2),
            nn.ReLU()
        )
        
        # Fusion head
        fusion_input_dim = tower_hidden  # 2 * (tower_hidden // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )
        
        self.qsar_dim = qsar_dim
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Split input into QSAR and geometric
        x_qsar = x[:, :self.qsar_dim]
        x_geo = x[:, self.qsar_dim:]
        
        # Process through towers
        h_qsar = self.qsar_tower(x_qsar)
        h_geo = self.geo_tower(x_geo)
        
        # Concatenate and fuse
        h_fused = torch.cat([h_qsar, h_geo], dim=1)
        return torch.sigmoid(self.fusion(h_fused)).squeeze(-1)


# ============================================================================
# Training Utilities
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(float)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              kernel: str = 'rbf') -> Tuple[SVC, Dict[str, float]]:
    """Train SVM classifier."""
    svm = SVC(kernel=kernel, probability=True, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    y_prob = svm.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_prob)
    return svm, metrics


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              model: nn.Module, device: torch.device,
              epochs: int = 200, batch_size: int = 32,
              lr: float = 0.001, patience: int = 20,
              evaluation_save_path: Optional[Path] = None
              ) -> Tuple[nn.Module, Dict[str, float], int]:
    """Train MLP classifier with early stopping. Optionally save history and val probs."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    
    train_tensor = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_tensor = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    
    best_score = 0
    best_epoch = 0
    best_state = None
    best_metrics = None
    patience_counter = 0
    history = {'train_loss': [], 'val_auc_roc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
            n += len(y_batch)
        train_loss = train_loss / n if n else 0.0
        
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        metrics = compute_metrics(all_labels, all_probs)
        history['train_loss'].append(float(train_loss))
        history['val_auc_roc'].append(float(metrics['auc_roc']))
        history['val_f1'].append(float(metrics['f1']))
        scheduler.step(metrics['auc_roc'])
        
        if metrics['auc_roc'] > best_score:
            best_score = metrics['auc_roc']
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    if evaluation_save_path is not None:
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        evaluation_save_path = Path(evaluation_save_path)
        evaluation_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(evaluation_save_path, 'w') as f:
            json.dump({
                'history': history,
                'y_true': np.array(all_labels).tolist(),
                'y_prob': np.array(all_probs).tolist(),
            }, f)
    
    return model, best_metrics, best_epoch


# ============================================================================
# Leakage Checks
# ============================================================================

def check_cluster_leakage(train_idx: np.ndarray, test_idx: np.ndarray, 
                          cluster_ids: np.ndarray) -> bool:
    """Check that no cluster appears in both train and test."""
    train_clusters = set(cluster_ids[train_idx])
    test_clusters = set(cluster_ids[test_idx])
    overlap = train_clusters & test_clusters
    if overlap:
        print(f"   ⚠️ LEAKAGE: {len(overlap)} clusters in both train and test!")
        return False
    return True


def check_blind_test_leakage(cv_indices: List[Tuple], blind_idx: np.ndarray) -> bool:
    """Check that blind test indices never appear in CV."""
    blind_set = set(blind_idx)
    for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
        train_overlap = set(train_idx) & blind_set
        val_overlap = set(val_idx) & blind_set
        if train_overlap or val_overlap:
            print(f"   ⚠️ LEAKAGE: Fold {fold_idx+1} contains blind test samples!")
            return False
    return True


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(X: np.ndarray, y: np.ndarray, 
                   model_name: str, feature_set: str,
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   qsar_dim: int, device: torch.device,
                   seed: int = 42,
                   evaluation_dir: Optional[Path] = None,
                   experiment_id: Optional[str] = None) -> Dict[str, float]:
    """Run a single experiment configuration. Optionally save history/val_probs for MLPs."""
    set_seed(seed)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    input_dim = X.shape[1]
    eval_path = None
    if evaluation_dir is not None and experiment_id is not None:
        eval_path = Path(evaluation_dir) / f"{experiment_id}.json"
    
    if model_name == 'svm_linear':
        _, metrics = train_svm(X_train_scaled, y_train, X_val_scaled, y_val, kernel='linear')
    elif model_name == 'svm_rbf':
        _, metrics = train_svm(X_train_scaled, y_train, X_val_scaled, y_val, kernel='rbf')
    elif model_name == 'mlp':
        model = AMPClassifier(input_dim=input_dim)
        _, metrics, _ = train_mlp(X_train_scaled, y_train, X_val_scaled, y_val, 
                                   model, device, evaluation_save_path=eval_path)
    elif model_name == 'two_tower':
        geo_dim = input_dim - qsar_dim
        model = TwoTowerFusionMLP(qsar_dim=qsar_dim, geo_dim=geo_dim)
        _, metrics, _ = train_mlp(X_train_scaled, y_train, X_val_scaled, y_val,
                                  model, device, evaluation_save_path=eval_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return metrics


def run_cluster_cv(X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray,
                   model_name: str, feature_set: str, qsar_dim: int,
                   device: torch.device, n_folds: int = 5,
                   seed: int = 42,
                   evaluation_dir: Optional[Path] = None) -> Dict:
    """Run cluster-based GroupKFold cross-validation."""
    gkf = GroupKFold(n_splits=n_folds)
    
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=cluster_ids)):
        assert check_cluster_leakage(train_idx, val_idx, cluster_ids), "Cluster leakage detected!"
        exp_id = f"{model_name}_{feature_set}_fold_{fold+1}" if evaluation_dir else None
        metrics = run_experiment(X, y, model_name, feature_set, 
                                 train_idx, val_idx, qsar_dim, device, seed + fold,
                                 evaluation_dir=evaluation_dir, experiment_id=exp_id)
        fold_results.append(metrics)
    
    # Aggregate
    summary = {}
    for metric in fold_results[0].keys():
        values = [r[metric] for r in fold_results]
        summary[metric] = {'mean': np.mean(values), 'std': np.std(values)}
    
    return summary


def run_pnas_cv(X: np.ndarray, y: np.ndarray,
                model_name: str, feature_set: str, qsar_dim: int,
                device: torch.device, n_rounds: int = 15,
                test_size: float = 0.15, seed: int = 42,
                evaluation_dir: Optional[Path] = None) -> Tuple[Dict, Dict]:
    """Run PNAS-style evaluation with blind test."""
    set_seed(seed)
    
    blind_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    pool_idx, blind_idx = next(blind_splitter.split(X, y))
    
    X_pool, y_pool = X[pool_idx], y[pool_idx]
    X_blind, y_blind = X[blind_idx], y[blind_idx]
    
    cv_splitter = StratifiedShuffleSplit(n_splits=n_rounds, test_size=0.2, random_state=seed)
    cv_indices = list(cv_splitter.split(X_pool, y_pool))
    
    # Check leakage (ensure blind test not in CV)
    # Convert pool indices to global indices for comparison
    # (blind_idx are already global)
    
    round_results = []
    for round_idx, (train_idx, val_idx) in enumerate(cv_indices):
        exp_id = f"{model_name}_{feature_set}_round_{round_idx+1}" if evaluation_dir else None
        metrics = run_experiment(X_pool, y_pool, model_name, feature_set,
                                 train_idx, val_idx, qsar_dim, device, seed + round_idx,
                                 evaluation_dir=evaluation_dir, experiment_id=exp_id)
        round_results.append(metrics)
    
    # CV summary
    cv_summary = {}
    for metric in round_results[0].keys():
        values = [r[metric] for r in round_results]
        cv_summary[metric] = {'mean': np.mean(values), 'std': np.std(values)}
    
    # Train final model on pool, evaluate on blind
    set_seed(seed)
    
    # Use 80/20 split within pool for early stopping
    final_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + 999)
    final_train_idx, final_val_idx = next(final_splitter.split(X_pool, y_pool))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_pool[final_train_idx])
    X_val_scaled = scaler.transform(X_pool[final_val_idx])
    X_blind_scaled = scaler.transform(X_blind)
    
    input_dim = X.shape[1]
    
    if model_name == 'svm_linear':
        model, _ = train_svm(X_train_scaled, y_pool[final_train_idx],
                             X_val_scaled, y_pool[final_val_idx], kernel='linear')
        y_prob_blind = model.predict_proba(X_blind_scaled)[:, 1]
    elif model_name == 'svm_rbf':
        model, _ = train_svm(X_train_scaled, y_pool[final_train_idx],
                             X_val_scaled, y_pool[final_val_idx], kernel='rbf')
        y_prob_blind = model.predict_proba(X_blind_scaled)[:, 1]
    elif model_name == 'mlp':
        model = AMPClassifier(input_dim=input_dim)
        model, _, _ = train_mlp(X_train_scaled, y_pool[final_train_idx],
                                 X_val_scaled, y_pool[final_val_idx], model, device)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blind_scaled, dtype=torch.float32).to(device)
            y_prob_blind = model(X_t).cpu().numpy()
    elif model_name == 'two_tower':
        geo_dim = input_dim - qsar_dim
        model = TwoTowerFusionMLP(qsar_dim=qsar_dim, geo_dim=geo_dim)
        model, _, _ = train_mlp(X_train_scaled, y_pool[final_train_idx],
                                 X_val_scaled, y_pool[final_val_idx], model, device)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blind_scaled, dtype=torch.float32).to(device)
            y_prob_blind = model(X_t).cpu().numpy()
    
    blind_metrics = compute_metrics(y_blind, y_prob_blind)
    if evaluation_dir is not None and model_name in ('mlp', 'two_tower'):
        path = Path(evaluation_dir) / f"{model_name}_{feature_set}_blind_test_probs.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'y_true': y_blind.tolist(), 'y_prob': y_prob_blind.tolist()}, f)
    
    return cv_summary, blind_metrics


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    base_dir = Path(__file__).parent
    default_geo = base_dir / "data" / "training_dataset" / "geometric_features_clustered.csv"
    default_qsar = base_dir / "data" / "training_dataset" / "qsar12_descriptors.csv"
    default_results = base_dir / "results"

    parser = argparse.ArgumentParser(description="Feature Fusion Experiments")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_folds', type=int, default=5, help='Folds for cluster CV')
    parser.add_argument('--pnas_rounds', type=int, default=15)
    parser.add_argument('--skip_qsar_compute', action='store_true',
                        help='Skip QSAR computation if already cached')
    parser.add_argument('--geo-csv', type=str, default=None,
                        help=f'Path to geometric features CSV (default: {default_geo})')
    parser.add_argument('--qsar-cache', type=str, default=None,
                        help=f'Path to QSAR-12 cache CSV; computed and saved here if missing (default: {default_qsar})')
    parser.add_argument('--results-dir', type=str, default=None,
                        help=f'Directory for results JSON/CSV and evaluation outputs (default: {default_results})')
    args = parser.parse_args()

    geo_csv = Path(args.geo_csv) if args.geo_csv else default_geo
    qsar_cache = Path(args.qsar_cache) if args.qsar_cache else default_qsar
    results_dir = Path(args.results_dir) if args.results_dir else default_results

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("  Feature Fusion Experiments: AMP vs DECOY Classification")
    print("="*80)
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n📂 Paths: geo={geo_csv}, qsar_cache={qsar_cache}, results={results_dir}")

    # ========================================================================
    # Load Data
    # ========================================================================
    print("\n" + "-"*80)
    print("  Loading Data")
    print("-"*80)

    # Load geometric features
    geo_df = pd.read_csv(geo_csv)
    print(f"\n📂 Geometric features: {geo_csv.name}")
    print(f"   Samples: {len(geo_df)}")
    
    # Get sequences and IDs
    peptide_ids = geo_df['peptide_id'].tolist()
    sequences = geo_df['sequence'].tolist()
    
    # Load or compute QSAR-12
    if qsar_cache.exists() and not args.skip_qsar_compute:
        print(f"\n📂 Loading cached QSAR-12: {qsar_cache.name}")
        qsar_df = pd.read_csv(qsar_cache)
    else:
        print(f"\n🔧 Computing QSAR-12 descriptors...")
        qsar_df = compute_qsar12_descriptors(sequences, peptide_ids)
        qsar_cache.parent.mkdir(parents=True, exist_ok=True)
        qsar_df.to_csv(qsar_cache, index=False)
        print(f"   Cached to: {qsar_cache}")
    
    # Verify alignment
    assert list(geo_df['peptide_id']) == list(qsar_df['peptide_id']), \
        "Peptide ID mismatch between geometric and QSAR features!"
    print(f"\n✅ Alignment verified: {len(geo_df)} peptides matched")
    
    # Define feature columns
    geo_exclude = ['peptide_id', 'sequence', 'pdb_file', 'label', 'ss_method', 
                   'cluster_id', 'ss_residues_computed']
    geo_cols = [c for c in geo_df.columns if c not in geo_exclude]
    
    qsar_cols = ['netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK',
                 '_SolventAccessibilityD1025', 'tau2_GRAR740104', 
                 'tau4_GRAR740104', 'QSO50_GRAR740104', 'QSO29_GRAR740104']
    
    print(f"\n📊 Feature Sets:")
    print(f"   QSAR-12: {len(qsar_cols)} features")
    print(f"   Geometric-24: {len(geo_cols)} features")
    print(f"   Combined: {len(qsar_cols) + len(geo_cols)} features")
    
    # Prepare feature matrices
    X_qsar = qsar_df[qsar_cols].values.astype(np.float32)
    X_geo = geo_df[geo_cols].values.astype(np.float32)
    X_combined = np.concatenate([X_qsar, X_geo], axis=1)
    
    y = ((geo_df['label'].values + 1) / 2).astype(np.float32)
    cluster_ids = geo_df['cluster_id'].values
    
    qsar_dim = len(qsar_cols)
    
    # Check for NaN
    if np.isnan(X_qsar).any():
        print("   ⚠️ NaN in QSAR features - filling with column mean")
        X_qsar = np.nan_to_num(X_qsar, nan=np.nanmean(X_qsar, axis=0))
        X_combined = np.concatenate([X_qsar, X_geo], axis=1)
    
    print(f"\n📊 Dataset:")
    print(f"   Samples: {len(y)}")
    print(f"   AMP: {int(sum(y))}, DECOY: {int(len(y) - sum(y))}")
    print(f"   Clusters: {len(np.unique(cluster_ids))}")
    
    # ========================================================================
    # Define Experiments
    # ========================================================================
    
    experiments = [
        ('SVM (QSAR-12)', 'svm_rbf', 'qsar', X_qsar),
        ('MLP (QSAR-12)', 'mlp', 'qsar', X_qsar),
        ('SVM (Geo-24)', 'svm_rbf', 'geo', X_geo),
        ('MLP (Geo-24)', 'mlp', 'geo', X_geo),
        ('SVM (Combined-36)', 'svm_rbf', 'combined', X_combined),
        ('MLP (Combined-36)', 'mlp', 'combined', X_combined),
    ]
    
    results = {
        'cluster_cv': {},
        'pnas_cv': {},
        'pnas_blind': {}
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = results_dir / "evaluation"
    
    # ========================================================================
    # Protocol A: Cluster-based GroupKFold
    # ========================================================================
    print("\n" + "="*80)
    print("  Protocol A: Cluster-Based GroupKFold (5-fold)")
    print("="*80)
    
    for exp_name, model_name, feat_set, X in experiments:
        print(f"\n🔬 {exp_name}...")
        summary = run_cluster_cv(X, y, cluster_ids, model_name, feat_set, 
                                 qsar_dim, device, n_folds=args.n_folds, seed=args.seed,
                                 evaluation_dir=evaluation_dir)
        results['cluster_cv'][exp_name] = summary
        print(f"   AUC-ROC: {summary['auc_roc']['mean']:.4f} ± {summary['auc_roc']['std']:.4f}")
        print(f"   F1:      {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
    
    # ========================================================================
    # Protocol B: PNAS-style
    # ========================================================================
    print("\n" + "="*80)
    print("  Protocol B: PNAS-Style (15-round CV + Blind Test)")
    print("="*80)
    
    for exp_name, model_name, feat_set, X in experiments:
        print(f"\n🔬 {exp_name}...")
        cv_summary, blind_metrics = run_pnas_cv(X, y, model_name, feat_set,
                                                 qsar_dim, device, 
                                                 n_rounds=args.pnas_rounds,
                                                 seed=args.seed,
                                                 evaluation_dir=evaluation_dir)
        results['pnas_cv'][exp_name] = cv_summary
        results['pnas_blind'][exp_name] = blind_metrics
        print(f"   CV AUC-ROC:    {cv_summary['auc_roc']['mean']:.4f} ± {cv_summary['auc_roc']['std']:.4f}")
        print(f"   Blind AUC-ROC: {blind_metrics['auc_roc']:.4f}")
    
    # ========================================================================
    # Summary Tables
    # ========================================================================
    print("\n" + "="*80)
    print("  RESULTS SUMMARY")
    print("="*80)
    
    # Protocol A table
    print("\n📊 Protocol A: Cluster-Based CV (Mean ± Std)")
    print("-"*90)
    print(f"{'Model':<30} {'AUC-ROC':<18} {'F1':<18} {'MCC':<18}")
    print("-"*90)
    for exp_name in results['cluster_cv']:
        r = results['cluster_cv'][exp_name]
        auc = f"{r['auc_roc']['mean']:.4f} ± {r['auc_roc']['std']:.4f}"
        f1 = f"{r['f1']['mean']:.4f} ± {r['f1']['std']:.4f}"
        mcc = f"{r['mcc']['mean']:.4f} ± {r['mcc']['std']:.4f}"
        print(f"{exp_name:<30} {auc:<18} {f1:<18} {mcc:<18}")
    
    # Protocol B table
    print("\n📊 Protocol B: PNAS-Style (CV / Blind Test)")
    print("-"*100)
    print(f"{'Model':<30} {'CV AUC-ROC':<18} {'Blind AUC-ROC':<15} {'Blind F1':<12} {'Blind MCC':<12}")
    print("-"*100)
    for exp_name in results['pnas_cv']:
        cv = results['pnas_cv'][exp_name]
        blind = results['pnas_blind'][exp_name]
        cv_auc = f"{cv['auc_roc']['mean']:.4f} ± {cv['auc_roc']['std']:.4f}"
        print(f"{exp_name:<30} {cv_auc:<18} {blind['auc_roc']:<15.4f} "
              f"{blind['f1']:<12.4f} {blind['mcc']:<12.4f}")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n" + "-"*80)
    print("  Saving Results")
    print("-"*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    results_clean = convert_numpy(results)
    
    json_path = results_dir / f"fusion_experiments_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'seed': args.seed,
            'n_folds': args.n_folds,
            'pnas_rounds': args.pnas_rounds,
            'results': results_clean
        }, f, indent=2)
    print(f"\n💾 JSON: {json_path}")
    
    # CSV summary
    csv_rows = []
    for exp_name in results['cluster_cv']:
        cluster = results['cluster_cv'][exp_name]
        pnas_cv = results['pnas_cv'][exp_name]
        pnas_blind = results['pnas_blind'][exp_name]
        csv_rows.append({
            'model': exp_name,
            'cluster_auc_mean': cluster['auc_roc']['mean'],
            'cluster_auc_std': cluster['auc_roc']['std'],
            'cluster_f1_mean': cluster['f1']['mean'],
            'cluster_mcc_mean': cluster['mcc']['mean'],
            'pnas_cv_auc_mean': pnas_cv['auc_roc']['mean'],
            'pnas_cv_auc_std': pnas_cv['auc_roc']['std'],
            'pnas_blind_auc': pnas_blind['auc_roc'],
            'pnas_blind_f1': pnas_blind['f1'],
            'pnas_blind_mcc': pnas_blind['mcc']
        })
    
    csv_df = pd.DataFrame(csv_rows)
    csv_path = results_dir / f"fusion_experiments_{timestamp}.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"💾 CSV: {csv_path}")
    
    print("\n✅ Feature fusion experiments complete!")


if __name__ == "__main__":
    main()

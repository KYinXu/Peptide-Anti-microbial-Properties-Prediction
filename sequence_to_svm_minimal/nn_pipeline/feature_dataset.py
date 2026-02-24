#!/usr/bin/env python3
"""
Feature Dataset for AMP Classification.

Combines:
- Geometric features from ESMFold structures (25 features)
- SVM outputs: σ (distToMargin) and P(+1) (2 features)
- Optional: 12 original biochemical descriptors

Total: 27-39 features depending on configuration.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import json


# Feature column groups
# Note: PLDDT_COLS are kept for reference but are intentionally
# excluded from GEOMETRIC_FEATURE_COLS to avoid data leakage.
PLDDT_COLS = ['plddt_mean', 'plddt_std', 'plddt_min', 'plddt_max']
COMPACTNESS_COLS = ['radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
                    'centroid_distance_mean', 'centroid_distance_std']
SECONDARY_STRUCTURE_COLS = ['fraction_helix', 'fraction_sheet', 'fraction_coil']
SASA_COLS = ['total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa']
SEQUENCE_COLS = ['length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment']
CURVATURE_COLS = ['curvature_mean', 'curvature_std', 'curvature_max', 
                  'torsion_mean', 'torsion_std']

# Geometric features used for training (pLDDT excluded)
GEOMETRIC_FEATURE_COLS = (COMPACTNESS_COLS + SECONDARY_STRUCTURE_COLS +
                          SASA_COLS + SEQUENCE_COLS + CURVATURE_COLS)

# SVM output features
SVM_COLS = ['svm_sigma', 'svm_prob_positive']

# Metadata columns (not used as features)
META_COLS = ['peptide_id', 'sequence', 'pdb_file', 'label', 'ss_method', 
             'cluster_id', 'ss_residues_computed']


class AMPDataset(Dataset):
    """PyTorch Dataset for AMP classification."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 peptide_ids: Optional[List[str]] = None):
        """
        Args:
            features: Feature matrix (N, D)
            labels: Binary labels (N,) with values 0 or 1
            peptide_ids: Optional list of peptide IDs for tracking
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.peptide_ids = peptide_ids
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class FeaturePipeline:
    """
    Complete feature pipeline for AMP classification.
    
    Handles:
    - Loading geometric features from CSV
    - Loading and merging SVM outputs
    - Feature standardization
    - Cluster-based train/test splitting
    """
    
    def __init__(self, 
                 geometric_csv: Union[str, Path],
                 svm_csv: Optional[Union[str, Path]] = None,
                 descriptors_csv: Optional[Union[str, Path]] = None,
                 use_svm_features: bool = True,
                 use_descriptor_features: bool = False):
        """
        Args:
            geometric_csv: Path to geometric_features.csv
            svm_csv: Path to SVM predictions CSV (optional)
            descriptors_csv: Path to 12-descriptor CSV (optional)
            use_svm_features: Whether to include SVM outputs
            use_descriptor_features: Whether to include 12 biochemical descriptors
        """
        self.geometric_csv = Path(geometric_csv)
        self.svm_csv = Path(svm_csv) if svm_csv else None
        self.descriptors_csv = Path(descriptors_csv) if descriptors_csv else None
        
        self.use_svm_features = use_svm_features
        self.use_descriptor_features = use_descriptor_features
        
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load and merge all data sources."""
        # Load geometric features
        self.df = pd.read_csv(self.geometric_csv)
        print(f"📂 Loaded geometric features: {len(self.df)} samples")
        
        # Determine which feature columns exist
        self.feature_cols = [c for c in GEOMETRIC_FEATURE_COLS if c in self.df.columns]
        print(f"   Geometric features: {len(self.feature_cols)}")
        
        # Load SVM outputs if available
        if self.use_svm_features and self.svm_csv and self.svm_csv.exists():
            self._merge_svm_features()
        elif self.use_svm_features:
            print("   ⚠️  SVM features requested but not available")
            self.use_svm_features = False
            
        # Load descriptor features if available
        if self.use_descriptor_features and self.descriptors_csv and self.descriptors_csv.exists():
            self._merge_descriptor_features()
        elif self.use_descriptor_features:
            print("   ⚠️  Descriptor features requested but not available")
            self.use_descriptor_features = False
        
        # Store final feature names
        self.feature_names = self.feature_cols.copy()
        
        print(f"\n📊 Total features: {len(self.feature_names)}")
        print(f"   Classes: AMP={sum(self.df['label']==1)}, "
              f"DECOY={sum(self.df['label']==-1)}")
        
    def _merge_svm_features(self):
        """Merge SVM prediction outputs."""
        svm_df = pd.read_csv(self.svm_csv)
        
        # Rename columns to be clear
        svm_df = svm_df.rename(columns={
            'distToMargin': 'svm_sigma',
            'P(+1)': 'svm_prob_positive'
        })
        
        # Try to match by index or sequence
        if 'seqIndex' in svm_df.columns:
            # Need to figure out mapping - for now, assume direct index match
            # This needs to be adjusted based on your actual data structure
            print(f"   ⚠️  SVM merge by index - may need adjustment")
            
        # For now, just add as placeholder if sizes match
        if len(svm_df) == len(self.df):
            self.df['svm_sigma'] = svm_df['svm_sigma'].values
            self.df['svm_prob_positive'] = svm_df['svm_prob_positive'].values
            self.feature_cols.extend(['svm_sigma', 'svm_prob_positive'])
            print(f"   SVM features: 2 (sigma, P(+1))")
        else:
            print(f"   ⚠️  SVM data size mismatch: {len(svm_df)} vs {len(self.df)}")
            self.use_svm_features = False
            
    def _merge_descriptor_features(self):
        """Merge 12 biochemical descriptor features."""
        desc_df = pd.read_csv(self.descriptors_csv)
        
        # Get descriptor column names (everything except sequence/ID columns)
        desc_cols = [c for c in desc_df.columns 
                     if c not in ['sequence', 'seqIndex', 'peptide_id', 'Unnamed: 0']]
        
        if len(desc_df) == len(self.df):
            for col in desc_cols:
                self.df[col] = desc_df[col].values
            self.feature_cols.extend(desc_cols)
            print(f"   Descriptor features: {len(desc_cols)}")
        else:
            print(f"   ⚠️  Descriptor data size mismatch")
            self.use_descriptor_features = False
    
    def get_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the feature matrix, labels, and cluster IDs.
        
        Returns:
            X: Feature matrix (N, D)
            y: Labels (N,) converted to 0/1
            clusters: Cluster IDs (N,) or None if not available
        """
        X = self.df[self.feature_cols].values.astype(np.float32)
        
        # Convert labels from {-1, 1} to {0, 1}
        y = ((self.df['label'].values + 1) / 2).astype(np.float32)
        
        # Get cluster IDs if available
        if 'cluster_id' in self.df.columns:
            clusters = self.df['cluster_id'].values
        else:
            clusters = None
            
        return X, y, clusters
    
    def get_peptide_ids(self) -> List[str]:
        """Get list of peptide IDs."""
        return self.df['peptide_id'].tolist()
    
    def create_cluster_splits(self, n_splits: int = 5, 
                               random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cluster-based train/test splits.
        
        Args:
            n_splits: Number of folds
            random_state: Random seed
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        X, y, clusters = self.get_feature_matrix()
        
        if clusters is None:
            print("⚠️  No cluster IDs - creating random clusters based on peptide IDs")
            # Create pseudo-clusters based on first character of sequence
            clusters = np.array([hash(s) % (len(self.df) // 5) 
                                for s in self.df['sequence']])
        
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups=clusters))
        
        print(f"\n📊 Created {n_splits} cluster-based splits:")
        for i, (train_idx, test_idx) in enumerate(splits):
            train_clusters = len(set(clusters[train_idx]))
            test_clusters = len(set(clusters[test_idx]))
            print(f"   Fold {i+1}: train={len(train_idx)} ({train_clusters} clusters), "
                  f"test={len(test_idx)} ({test_clusters} clusters)")
        
        return splits
    
    def create_dataloaders(self, train_idx: np.ndarray, test_idx: np.ndarray,
                           batch_size: int = 32, 
                           fit_scaler: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training.
        
        Args:
            train_idx: Training indices
            test_idx: Test indices
            batch_size: Batch size
            fit_scaler: Whether to fit the scaler on training data
            
        Returns:
            train_loader, test_loader
        """
        X, y, _ = self.get_feature_matrix()
        peptide_ids = self.get_peptide_ids()
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize features
        if fit_scaler:
            X_train = self.scaler.fit_transform(X_train)
        else:
            X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create datasets
        train_ids = [peptide_ids[i] for i in train_idx]
        test_ids = [peptide_ids[i] for i in test_idx]
        
        train_dataset = AMPDataset(X_train, y_train, train_ids)
        test_dataset = AMPDataset(X_test, y_test, test_ids)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                 shuffle=False, drop_last=False)
        
        return train_loader, test_loader
    
    def save_scaler(self, path: Union[str, Path]):
        """Save the fitted scaler for inference."""
        import joblib
        joblib.dump(self.scaler, path)
        print(f"💾 Saved scaler to {path}")
        
    def load_scaler(self, path: Union[str, Path]):
        """Load a fitted scaler."""
        import joblib
        self.scaler = joblib.load(path)
        print(f"📂 Loaded scaler from {path}")


def main():
    """Demo of the feature pipeline."""
    base_dir = Path(__file__).parent.parent
    
    # Initialize pipeline
    pipeline = FeaturePipeline(
        geometric_csv=base_dir / "data" / "training_dataset" / "geometric_features.csv",
        use_svm_features=False,  # Set to True if you have SVM outputs
        use_descriptor_features=False
    )
    
    # Get feature matrix
    X, y, clusters = pipeline.get_feature_matrix()
    print(f"\n📊 Feature matrix shape: {X.shape}")
    print(f"   Label distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    
    # Create cluster splits
    if clusters is None:
        print("\n⚠️  No cluster IDs found. Run prepare_clusters.py first.")
        print("   For now, using random clusters as fallback...")
    
    splits = pipeline.create_cluster_splits(n_splits=5)
    
    # Create dataloaders for first fold
    train_idx, test_idx = splits[0]
    train_loader, test_loader = pipeline.create_dataloaders(
        train_idx, test_idx, batch_size=32
    )
    
    print(f"\n📦 DataLoader created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Sample batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\n📥 Sample batch:")
    print(f"   X shape: {X_batch.shape}")
    print(f"   y shape: {y_batch.shape}")


if __name__ == "__main__":
    main()

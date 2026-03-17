#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
# Add the parent directory to the path so we can import the gnn modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.data_utils import PeptideGraphDataset
from gnn.models import PeptideGNN
from gnn.train import evaluate, evaluate_probs
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np


def get_predictions_and_probs(model, loader, device):
    """Run inference and return probs (class 1) in dataset order."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)
def parse_args():
    parser = argparse.ArgumentParser(description='Test a saved GNN model on a new dataset')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the saved .pt checkpoint file')
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Path to the test set geometric_features.csv')
    parser.add_argument('--pdb_dir', type=str, required=True, 
                        help='Directory containing the test set PDB files')
    parser.add_argument('--architecture', type=str, required=True, choices=['gcn', 'gat', 'egnn'], 
                        help='Model architecture used during training')
    
    # These must match the parameters used during training
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--pooling', type=str, default='mean_max')
    parser.add_argument('--use_geo_features', action='store_true', 
                        help='Set this flag if the model was trained with geometric features')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Optional path to save a CSV of raw predictions (e.g., test_preds.csv)')
    
    return parser.parse_args()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading test dataset from {args.csv_path}...")
    dataset = PeptideGraphDataset(
        csv_path=args.csv_path,
        pdb_dir=args.pdb_dir,
        use_geometric_features=args.use_geo_features
    )
    
    # Dynamically determine geometric feature dimension just like in the training script
    geo_dim = 0
    if args.use_geo_features and len(dataset) > 0:
        if hasattr(dataset[0], 'geo_features'):
            geo_dim = int(dataset[0].geo_features.shape[1])
            print(f"Detected geometric feature dimension: {geo_dim}")
            
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nInitializing {args.architecture.upper()} model...")
    model = PeptideGNN(
        architecture=args.architecture,
        in_channels=26,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=2,
        pooling=args.pooling,
        geo_feature_dim=geo_dim
    )
    
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.to(device)

    probs = get_predictions_and_probs(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)
    confidence = np.maximum(probs, 1 - probs)
    df = dataset.df
    id_col = 'peptide_id' if 'peptide_id' in df.columns else ('name' if 'name' in df.columns else None)
    ids = df[id_col].astype(str).tolist() if id_col else [str(i) for i in range(len(df))]

    print("\n" + "=" * 60)
    print("PER-SAMPLE PREDICTIONS (confidence = certainty in predicted class)")
    print("=" * 60)
    print(f"{'Sample':<28} {'Prediction':<12} {'Confidence':>12}")
    print("-" * 60)
    for sid, pred, conf in zip(ids, preds, confidence):
        label_str = "AMP" if pred == 1 else "non-AMP"
        print(f"{sid:<28} {label_str:<12} {conf:>12.4f}")
    print("=" * 60)

    has_labels = 'label' in df.columns
    if has_labels:
        y_true = df['label'].values
        y_true = (y_true == 1).astype(int)
        if len(np.unique(y_true)) > 1:
            metrics = evaluate(model, test_loader, device)
            print("\nTEST SET METRICS (labeled data)")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"{metric:15s}: {value:.4f}")

    if args.save_predictions:
        out = {'peptide_id': ids, 'predicted_class': preds, 'confidence': confidence, 'prob_AMP': probs}
        if has_labels:
            out['true_label'] = df['label'].values
        pd.DataFrame(out).to_csv(args.save_predictions, index=False)
        print(f"\nSaved predictions to {args.save_predictions}")
if __name__ == '__main__':
    main()
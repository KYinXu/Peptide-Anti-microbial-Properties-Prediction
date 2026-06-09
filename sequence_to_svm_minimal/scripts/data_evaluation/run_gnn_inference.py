#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
# Add the parent directory to the path so we can import the gnn modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from gnn.checkpoint_meta import resolve_node_layout_for_checkpoint
from gnn.data_utils import PeptideGraphDataset
from gnn.models import PeptideGNN, esm2_raw_dim_from_state_dict, esm2_hidden_dim_from_state_dict
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
    parser.add_argument(
        '--geometric_feature_cols',
        type=str,
        default=None,
        help='Comma-separated column names (must match training order, e.g. Geo+QSAR+ESM2).',
    )
    parser.add_argument(
        '--tabular_scaler_path',
        type=str,
        default=None,
        help='Joblib scaler from training (default: <model_stem>_tabular_scaler.joblib next to checkpoint).',
    )
    parser.add_argument(
        '--esm2_residue_dir',
        type=str,
        default=None,
        help='Per-residue ESM2 .pt directory (required if checkpoint includes esm2_encoder).',
    )
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Optional path to save a CSV of raw predictions (e.g., test_preds.csv)')
    
    return parser.parse_args()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\nLoading test dataset from {args.csv_path}...")
    geo_cols = None
    if args.geometric_feature_cols:
        geo_cols = [c.strip() for c in args.geometric_feature_cols.split(",") if c.strip()]
    scaler_path = args.tabular_scaler_path
    if scaler_path is None and args.use_geo_features and args.model_path:
        cand = Path(args.model_path).with_name(Path(args.model_path).stem + "_tabular_scaler.joblib")
        if cand.is_file():
            scaler_path = str(cand)
            print(f"Using tabular scaler: {scaler_path}")
    sd0 = torch.load(args.model_path, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(sd0)
    esm2_h = esm2_hidden_dim_from_state_dict(sd0)
    ng_infer, in_base_ckpt, layout_notes = resolve_node_layout_for_checkpoint(
        args.model_path,
        sd0,
        args.architecture,
        user_node_groups=None,
    )
    for msg in layout_notes:
        print(msg, flush=True)
    if esm2_raw > 0 and not args.esm2_residue_dir:
        raise SystemExit(
            f"Checkpoint expects per-residue ESM2 (in_dim={esm2_raw}); pass --esm2_residue_dir."
        )

    dataset = PeptideGraphDataset(
        csv_path=args.csv_path,
        pdb_dir=args.pdb_dir,
        use_geometric_features=args.use_geo_features,
        geometric_feature_cols=geo_cols,
        tabular_scaler_path=scaler_path,
        esm2_residue_dir=args.esm2_residue_dir if esm2_raw > 0 else None,
        node_feature_groups=ng_infer,
    )
    
    # Dynamically determine geometric feature dimension just like in the training script
    geo_dim = 0
    if args.use_geo_features and len(dataset) > 0:
        if hasattr(dataset[0], 'geo_features'):
            geo_dim = int(dataset[0].geo_features.shape[1])
            print(f"Detected geometric feature dimension: {geo_dim}")
            
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if len(dataset) > 0:
        xw = int(dataset[0].x.shape[1])
        if xw != in_base_ckpt:
            raise SystemExit(
                f"Node feature width mismatch: checkpoint implies data.x width {in_base_ckpt}, "
                f"dataset produced {xw}."
            )
    print(f"\nInitializing {args.architecture.upper()} model (in_channels={in_base_ckpt})...")
    model = PeptideGNN(
        architecture=args.architecture,
        in_channels=in_base_ckpt,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_classes=2,
        pooling=args.pooling,
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=esm2_h,
    )
    
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(sd0)
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
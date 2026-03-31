#!/usr/bin/env python3
"""
Compare model predictions on unlabeled test data (SVM, GCN, GAT, EGNN).

Outputs: per-sample predictions and confidence per model, agreement statistics,
optional raw scores / logits, and per-model z-scores of a benchmark score (computed
on this run so SVM distance and GNN logit margins are on comparable scale). No
ground-truth metrics (data is unlabeled).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Default paths (relative to sequence_to_svm_minimal/). Override with CLI.
_ROOT = Path(__file__).resolve().parents[2]

# Default config: compare feature sets for a single architecture.
# Checkpoints should come from run_gnn_train_final_models.py.
CONFIG = {
    'geo_csv': str(_ROOT / 'data/test/geometric_features.csv'),
    'pdb_dir': str(_ROOT / 'data/test/structures/sequences'),
    'qsar_csv': str(_ROOT / 'data/test/qsar12_descriptors.csv'),
    'svm_descriptor_csv': str(_ROOT / 'data/test/qsar12_descriptors.csv'),
    'svm_z_file': str(_ROOT / 'results/checkpoints/svm_alpha_beta_combined/svm_qsar12_zscores.txt'),
    'svm_pkl': str(_ROOT / 'results/checkpoints/svm_alpha_beta_combined/svm_qsar12_model.pkl'),
    'architecture': 'gat',  # 'gcn', 'gat', or 'egnn'
    'esm_only_pt': str(_ROOT / 'results/gnn/alpha_and_beta_combined/ready_models/gat_ready_Graph-only.pt'),
    'esm_geo_pt': str(_ROOT / 'results/gnn/alpha_and_beta_combined/ready_models/gat_ready_Graph_plus_Geo20.pt'),
    'esm_qsar_pt': str(_ROOT / 'results/gnn/alpha_and_beta_combined/ready_models/gat_ready_Graph_plus_QSAR12.pt'),
    'esm_combined_pt': str(_ROOT / 'results/gnn/alpha_and_beta_combined/ready_models/gat_ready_Graph_plus_Combined32.pt'),
    'gnn_hidden': 64,
    'gnn_layers': 3,
    'gnn_pooling': 'mean_max',
    'batch_size': 32,
    'output_csv': str(_ROOT / 'results/test_model_comparison.csv'),
}

FEATURE_SETS = [
    ('ESM-only', 'esm_only_pt'),
    ('ESM+Geo20', 'esm_geo_pt'),
    ('ESM+QSAR12', 'esm_qsar_pt'),
    ('ESM+Combined32', 'esm_combined_pt'),
]


def _load_svm_predictions(descriptor_csv: str, z_file: str, pkl_path: str):
    """Load SVM, Z-score descriptors, return (ids, preds, prob_amp, distance)."""
    try:
        import joblib
    except ImportError:
        from sklearn.externals import joblib as joblib

    df = pd.read_csv(descriptor_csv)
    id_col = 'peptide_id' if 'peptide_id' in df.columns else ('name' if 'name' in df.columns else df.columns[0])
    ids = df[id_col].astype(str).tolist()

    with open(z_file, 'r') as f:
        desc_names = [x.strip() for x in f.readline().strip().split(',')]
        means = np.array([float(x) for x in f.readline().strip().split(',')])
        stds = np.array([float(x) for x in f.readline().strip().split(',')])

    mask = []
    for name in desc_names:
        if name not in df.columns:
            raise ValueError(f"Descriptor '{name}' from Z file not in CSV columns")
        mask.append(df.columns.tolist().index(name))
    X = df.iloc[:, mask].values.astype(np.float64)
    X = (X - means) / np.where(stds > 0, stds, 1.0)

    clf = joblib.load(pkl_path)
    raw_preds = np.asarray(clf.predict(X)).ravel()
    classes = getattr(clf, 'classes_', np.array([0, 1]))
    pos_class = int(1 if 1 in classes else classes[-1])
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X)
        pos_idx = int(np.where(classes == pos_class)[0][0])
        prob_amp = proba[:, pos_idx].ravel()
    else:
        prob_amp = np.where(raw_preds == pos_class, 1.0, 0.0)
    if hasattr(clf, 'decision_function'):
        distance = np.asarray(clf.decision_function(X)).ravel()
    else:
        distance = np.full_like(prob_amp, np.nan, dtype=np.float64)

    preds = (raw_preds == pos_class).astype(int)
    return ids, preds, prob_amp, distance


def _default_tabular_scaler_path(model_path: str) -> str | None:
    p = Path(model_path).with_name(Path(model_path).stem + "_tabular_scaler.joblib")
    return str(p) if p.is_file() else None


def _run_gnn_predictions(csv_path: str,
                         pdb_dir: str,
                         model_path: str,
                         architecture: str,
                         hidden: int,
                         num_layers: int,
                         pooling: str,
                         batch_size: int,
                         use_geometric_features: bool,
                         geometric_feature_cols=None,
                         tabular_scaler_path: str | None = None):
    """Run one GNN checkpoint and return ids/preds/prob/logits/margin."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader

    from gnn.data_utils import PeptideGraphDataset
    from gnn.models import PeptideGNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_path = tabular_scaler_path
    if scaler_path is None and use_geometric_features:
        scaler_path = _default_tabular_scaler_path(model_path)
    dataset = PeptideGraphDataset(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        use_geometric_features=use_geometric_features,
        geometric_feature_cols=geometric_feature_cols,
        tabular_scaler_path=scaler_path,
    )

    # Match geo_feature_dim to how the checkpoint was trained:
    # - Graph-only models: no geometric features (geo_dim = 0)
    # - Graph+Geo / Graph+Combined models: use whatever geo_features are present.
    if use_geometric_features and len(dataset) > 0 and hasattr(dataset[0], 'geo_features'):
        geo_dim = int(dataset[0].geo_features.shape[1])
    else:
        geo_dim = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = PeptideGNN(
        architecture=architecture,
        in_channels=26,
        hidden_channels=hidden,
        num_layers=num_layers,
        num_classes=2,
        pooling=pooling,
        geo_feature_dim=geo_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_probs = []
    all_logit_amp = []
    all_logit_nonamp = []
    all_logit_margin = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            logit_nonamp = out[:, 0].cpu().numpy()
            logit_amp = out[:, 1].cpu().numpy()
            logit_margin = (out[:, 1] - out[:, 0]).cpu().numpy()
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_logit_amp.extend(logit_amp)
            all_logit_nonamp.extend(logit_nonamp)
            all_logit_margin.extend(logit_margin)
    probs = np.array(all_probs)
    logit_amp = np.array(all_logit_amp)
    logit_nonamp = np.array(all_logit_nonamp)
    logit_margin = np.array(all_logit_margin)
    preds = (probs >= 0.5).astype(int)

    df = dataset.df
    id_col = 'peptide_id' if 'peptide_id' in df.columns else ('name' if 'name' in df.columns else None)
    ids = df[id_col].astype(str).tolist() if id_col else [str(i) for i in range(len(df))]
    return ids, preds, probs, logit_amp, logit_nonamp, logit_margin


def _confidence(prob_amp: np.ndarray) -> np.ndarray:
    return np.maximum(prob_amp, 1.0 - prob_amp)


def _raw_for_benchmark_z(model_name: str, result: dict, use_prob: bool) -> np.ndarray:
    """Raw score to z-score on this run: P(AMP), or SVM decision_function / GNN logit margin."""
    if use_prob:
        return np.asarray(result["prob_amp"], dtype=np.float64)
    if model_name == "SVM":
        return np.asarray(result["distance"], dtype=np.float64)
    return np.asarray(result["logit_margin"], dtype=np.float64)


def _zscore_aligned_to_ids(
    canonical_ids: list[str],
    ids: list[str],
    raw: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """Align raw scores to canonical_ids; z-score over finite values; return z, mu, sigma, n."""
    idx_map = {str(i): k for k, i in enumerate(ids)}
    aligned = np.full(len(canonical_ids), np.nan, dtype=np.float64)
    for j, pid in enumerate(canonical_ids):
        k = idx_map.get(str(pid))
        if k is not None:
            aligned[j] = float(raw[k])
    finite = np.isfinite(aligned)
    n_fin = int(finite.sum())
    z = np.full_like(aligned, np.nan, dtype=np.float64)
    if n_fin == 0:
        return z, float("nan"), float("nan"), 0
    v = aligned[finite]
    mu = float(v.mean())
    sig = float(v.std(ddof=0))
    if sig == 0.0 or not np.isfinite(sig):
        z[finite] = 0.0
        return z, mu, 0.0, n_fin
    z[finite] = (v - mu) / sig
    return z, mu, sig, n_fin


def _agreement_matrix(names: list, pred_dfs: list, ids_common: list):
    """Pairwise agreement counts (both predict AMP) and total overlap."""
    n = len(names)
    agree = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                agree[i, j] = len(ids_common)
                continue
            pi = pred_dfs[i].loc[ids_common, 'pred'].values
            pj = pred_dfs[j].loc[ids_common, 'pred'].values
            agree[i, j] = int((pi == pj).sum())
    return agree


def _print_agreement(names: list, agree: np.ndarray, n: int):
    print("\nPairwise agreement (same classification)")
    print("-" * (12 * (len(names) + 1)))
    header = "Model".ljust(12) + "".join(m[:10].ljust(12) for m in names)
    print(header)
    for i, name in enumerate(names):
        row = name[:10].ljust(12) + "".join(f"{agree[i, j]:>10} " for j in range(len(names)))
        print(row)
    print(f"(Total samples: {n})")


def _print_cli_report(architecture: str,
                      results: dict,
                      canonical_ids: list,
                      ids_common: list,
                      pred_frames: dict,
                      output_csv: str | None) -> None:
    """Pretty CLI report summarizing model behavior and agreement."""
    names = list(results.keys())
    n_samples = len(canonical_ids)

    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON REPORT  –  architecture = {architecture.upper()}")
    print("=" * 80)
    print(f"Total peptides in input CSV     : {n_samples}")
    print(f"Models run                      : {', '.join(names)}")

    if ids_common:
        print(f"Samples with all model outputs  : {len(ids_common)} / {n_samples}")

    print("\nPer-model summary (AMP predictions and confidence)")
    header = (
        f"{'Model':<18}"
        f"{'AMP count':>12}"
        f"{'% AMP':>10}"
        f"{'Mean conf':>14}"
        f"{'Mean conf (AMP)':>18}"
        f"{'Mean conf (non-AMP)':>22}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for m in names:
        r = results[m]
        pred = np.asarray(r["pred"])
        conf = np.asarray(r["confidence"])
        n = len(pred)
        n_amp = int(pred.sum())
        frac_amp = n_amp / n if n > 0 else 0.0

        if n_amp > 0:
            mean_conf_amp = float(conf[pred == 1].mean())
        else:
            mean_conf_amp = float("nan")

        if n_amp < n:
            mean_conf_non = float(conf[pred == 0].mean())
        else:
            mean_conf_non = float("nan")

        print(
            f"{m:<18}"
            f"{n_amp:>12d}"
            f"{frac_amp * 100:>9.1f}%"
            f"{float(conf.mean()):>14.4f}"
            f"{mean_conf_amp:>18.4f}"
            f"{mean_conf_non:>22.4f}"
        )

    if len(ids_common) >= 1 and len(names) >= 2:
        agree = _agreement_matrix(names, [pred_frames[m] for m in names], ids_common)
        _print_agreement(names, agree, len(ids_common))

    if output_csv:
        print(f"\nCombined per-peptide predictions saved to: {output_csv}")


def _print_benchmark_z_summary(
    names: list[str], results: dict, canonical_ids: list[str], use_prob: bool
) -> None:
    print("\nBenchmark z-scores (per model: mean=0, std=1 over finite scores in this run)")
    if use_prob:
        print("- Raw metric: P(AMP) for every model")
    else:
        print("- Raw metric: SVM decision_function; GNN logit_AMP − logit_nonAMP")
    hdr = f"{'Model':<22}{'n':>6}{'raw μ':>12}{'raw σ':>12}"
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for m in names:
        raw = _raw_for_benchmark_z(m, results[m], use_prob)
        _, mu, sig, n_fin = _zscore_aligned_to_ids(canonical_ids, results[m]["ids"], raw)
        print(f"{m:<22}{n_fin:>6}{mu:>12.4f}{sig:>12.4f}")
    print("-" * len(hdr))


def main():
    ap = argparse.ArgumentParser(description='Compare SVM and GNN predictions on unlabeled test data')
    ap.add_argument('--geo_csv', type=str, default=CONFIG['geo_csv'], help='Test geometric_features.csv')
    ap.add_argument('--pdb_dir', type=str, default=CONFIG['pdb_dir'], help='Directory containing test PDB files')
    ap.add_argument('--qsar_csv', type=str, default=CONFIG['qsar_csv'], help='Optional QSAR-12 descriptors CSV for Combined32')
    ap.add_argument('--svm_descriptor_csv', type=str, default=CONFIG['svm_descriptor_csv'], help='Descriptor CSV for SVM')
    ap.add_argument('--svm_z_file', type=str, default=CONFIG['svm_z_file'], help='Z-score file: names, means, stds')
    ap.add_argument('--svm_pkl', type=str, default=CONFIG['svm_pkl'], help='Trained SVM pickle')
    ap.add_argument('--architecture', type=str, default=CONFIG['architecture'],
                    choices=['gcn', 'gat', 'egnn'],
                    help='GNN architecture to compare across feature sets')
    ap.add_argument('--esm_only_pt', type=str, default=CONFIG['esm_only_pt'],
                    help='Checkpoint for ESM-only model')
    ap.add_argument('--esm_geo_pt', type=str, default=CONFIG['esm_geo_pt'],
                    help='Checkpoint for ESM+Geo20 model')
    ap.add_argument('--esm_qsar_pt', type=str, default=CONFIG['esm_qsar_pt'],
                    help='Checkpoint for ESM+QSAR12 model')
    ap.add_argument('--esm_combined_pt', type=str, default=CONFIG['esm_combined_pt'],
                    help='Checkpoint for ESM+Combined32 model')
    ap.add_argument('--gnn_hidden', type=int, default=CONFIG['gnn_hidden'])
    ap.add_argument('--gnn_layers', type=int, default=CONFIG['gnn_layers'])
    ap.add_argument('--gnn_pooling', type=str, default=CONFIG['gnn_pooling'])
    ap.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    ap.add_argument('--output_csv', type=str, default=CONFIG['output_csv'], help='Save combined predictions CSV')
    ap.add_argument('--only_amp', action='store_true',
                    help='If set, save only peptides predicted as AMP (1) by at least one model')
    ap.add_argument('--store_svm_distance', action='store_true',
                    help='If set, store SVM decision_function distance in output CSV')
    ap.add_argument('--store_gnn_logits', action='store_true',
                    help='If set, store GNN raw logits and logit margin in output CSV')
    ap.add_argument(
        '--score_z',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include per-model z-scores of benchmark score on this run (SVM: decision_function; GNN: logit margin). Use --no-score_z to omit.',
    )
    ap.add_argument(
        '--score_z_prob',
        action='store_true',
        help='If set with --score_z, z-score P(AMP) per model instead of distance / logit margin.',
    )
    args = ap.parse_args()

    geo_df = pd.read_csv(args.geo_csv)
    id_col = 'peptide_id' if 'peptide_id' in geo_df.columns else ('name' if 'name' in geo_df.columns else geo_df.columns[0])
    canonical_ids = geo_df[id_col].astype(str).tolist()
    n_samples = len(canonical_ids)

    results = {}
    pred_frames = {}

    if args.svm_pkl and args.svm_descriptor_csv and args.svm_z_file:
        print("Running SVM...")
        ids, preds, prob_amp, distance = _load_svm_predictions(
            args.svm_descriptor_csv, args.svm_z_file, args.svm_pkl
        )
        results['SVM'] = {
            'ids': ids,
            'pred': preds,
            'prob_amp': prob_amp,
            'confidence': _confidence(prob_amp),
            'distance': distance,
        }
        pred_frames['SVM'] = pd.DataFrame({'pred': preds, 'prob_amp': prob_amp, 'confidence': results['SVM']['confidence']}, index=ids)

    feature_models = [
        # (name, checkpoint_path, use_geometric_features_flag, feature_mode)
        # feature_mode: 'graph', 'geo20', 'qsar12', or 'combined32'
        ('ESM-only', args.esm_only_pt, False, 'esm'),
        ('ESM+Geo20', args.esm_geo_pt, True, 'geo20'),
        ('ESM+QSAR12', args.esm_qsar_pt, True, 'qsar12'),
        ('ESM+Combined32', args.esm_combined_pt, True, 'combined32'),
    ]

    # Precompute merged CSV for Combined32 if requested
    combined_csv_path = None
    combined_feature_cols = None
    if args.qsar_csv and Path(args.qsar_csv).exists():
        geo_df = pd.read_csv(args.geo_csv)
        qsar_df = pd.read_csv(args.qsar_csv)
        qsar_cols = [
            "netCharge",
            "FC",
            "LW",
            "DP",
            "NK",
            "AE",
            "pcMK",
            "_SolventAccessibilityD1025",
            "tau2_GRAR740104",
            "tau4_GRAR740104",
            "QSO50_GRAR740104",
            "QSO29_GRAR740104",
        ]
        geo_cols = [
            'radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
            'centroid_distance_mean', 'centroid_distance_std',
            'fraction_helix', 'fraction_sheet', 'fraction_coil',
            'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa',
            'length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment',
            'curvature_mean', 'curvature_std', 'curvature_max',
            'torsion_mean', 'torsion_std'
        ]
        merged_df = geo_df.merge(qsar_df[["peptide_id"] + qsar_cols], on="peptide_id", how="left")
        combined_feature_cols = geo_cols + qsar_cols
        combined_csv_path = _ROOT / "data/test/geometric_qsar_combined.csv"
        combined_csv_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(combined_csv_path, index=False)

    for name, path, use_geo, feat_mode in feature_models:
        if not path or not Path(path).exists():
            continue
        print(f"Running {args.architecture.upper()} ({name})...")
        # Select CSV and feature columns depending on feature set
        run_csv = args.geo_csv
        geom_cols = None
        if feat_mode == 'combined32' and combined_csv_path is not None and combined_feature_cols is not None:
            run_csv = str(combined_csv_path)
            geom_cols = combined_feature_cols
        elif feat_mode == 'qsar12' and combined_csv_path is not None:
            run_csv = str(combined_csv_path)
            geom_cols = [
                "netCharge",
                "FC",
                "LW",
                "DP",
                "NK",
                "AE",
                "pcMK",
                "_SolventAccessibilityD1025",
                "tau2_GRAR740104",
                "tau4_GRAR740104",
                "QSO50_GRAR740104",
                "QSO29_GRAR740104",
            ]
        elif feat_mode in ('combined32', 'qsar12'):
            print(f"Skipping {name}: requires --qsar_csv with QSAR-12 descriptors")
            continue

        ids, preds, prob_amp, logit_amp, logit_nonamp, logit_margin = _run_gnn_predictions(
            run_csv, args.pdb_dir, path, args.architecture,
            args.gnn_hidden, args.gnn_layers, args.gnn_pooling,
            args.batch_size, use_geometric_features=use_geo,
            geometric_feature_cols=geom_cols
        )
        results[name] = {
            'ids': ids,
            'pred': preds,
            'prob_amp': prob_amp,
            'confidence': _confidence(prob_amp),
            'logit_amp': logit_amp,
            'logit_nonamp': logit_nonamp,
            'logit_margin': logit_margin,
        }
        pred_frames[name] = pd.DataFrame({'pred': preds, 'prob_amp': prob_amp, 'confidence': results[name]['confidence']}, index=ids)

    if not results:
        print("No models run. Provide at least: SVM inputs and/or one or more feature-set checkpoints for the chosen architecture.")
        return 1

    names = list(results.keys())
    ids_set = set(canonical_ids)
    for m in names:
        ids_set &= set(results[m]['ids'])
    ids_common = [i for i in canonical_ids if i in ids_set]
    _print_cli_report(args.architecture, results, canonical_ids, ids_common, pred_frames, args.output_csv)

    score_z_by_model: dict[str, np.ndarray] = {}
    if args.score_z:
        for m in names:
            raw = _raw_for_benchmark_z(m, results[m], args.score_z_prob)
            z, _, _, _ = _zscore_aligned_to_ids(canonical_ids, results[m]["ids"], raw)
            score_z_by_model[m] = z
        _print_benchmark_z_summary(names, results, canonical_ids, args.score_z_prob)

    out_rows = []
    for idx, pid in enumerate(canonical_ids):
        row = {'peptide_id': pid}
        for m in names:
            r = results[m]
            if pid in r['ids']:
                i = r['ids'].index(pid)
                row[f'{m}_pred'] = int(r['pred'][i])
                row[f'{m}_confidence'] = float(r['confidence'][i])
                row[f'{m}_prob_AMP'] = float(r['prob_amp'][i])
                if args.score_z:
                    zv = score_z_by_model[m][idx]
                    row[f'{m}_score_z'] = float(zv) if np.isfinite(zv) else None
                if args.store_svm_distance and m == 'SVM':
                    row[f'{m}_distance'] = float(r['distance'][i]) if np.isfinite(r['distance'][i]) else None
                if args.store_gnn_logits and m != 'SVM':
                    row[f'{m}_logit_AMP'] = float(r['logit_amp'][i])
                    row[f'{m}_logit_nonAMP'] = float(r['logit_nonamp'][i])
                    row[f'{m}_logit_margin'] = float(r['logit_margin'][i])
            else:
                row[f'{m}_pred'] = None
                row[f'{m}_confidence'] = None
                row[f'{m}_prob_AMP'] = None
                if args.score_z:
                    row[f'{m}_score_z'] = None
                if args.store_svm_distance and m == 'SVM':
                    row[f'{m}_distance'] = None
                if args.store_gnn_logits and m != 'SVM':
                    row[f'{m}_logit_AMP'] = None
                    row[f'{m}_logit_nonAMP'] = None
                    row[f'{m}_logit_margin'] = None
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # Optional: keep only peptides predicted as AMP (1) by at least one model.
    if args.only_amp and not out_df.empty:
        amp_mask = False
        for m in names:
            col = f'{m}_pred'
            if col in out_df.columns:
                amp_mask = amp_mask | (out_df[col] == 1)
        out_df = out_df[amp_mask].reset_index(drop=True)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())

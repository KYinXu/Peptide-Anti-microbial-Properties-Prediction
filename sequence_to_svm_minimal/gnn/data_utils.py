"""
Data utilities for converting PDB files to PyTorch Geometric graphs.

Each peptide becomes a graph where:
- Nodes = amino acid residues
- Edges = sequential bonds (i, i+1) + spatial contacts (Cα-Cα < threshold)
- Node features = AA type, pLDDT, hydrophobicity, charge, etc.
- Edge features = distance, edge type
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from Bio.PDB import PDBParser
import warnings
import functools

warnings.filterwarnings('ignore', category=Warning, module='Bio')


# =============================================================================
# AMINO ACID PROPERTIES
# =============================================================================

# One-hot encoding order (alphabetical)
AA_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Kyte-Doolittle hydrophobicity scale (normalized to [-1, 1])
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
HYDRO_MIN, HYDRO_MAX = -4.5, 4.5

# Charge at pH 7
CHARGE_PH7 = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

# Molecular weight (normalized)
MOLWEIGHT = {
    'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
    'Q': 146, 'E': 147, 'G': 75, 'H': 155, 'I': 131,
    'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
    'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117
}
MW_MIN, MW_MAX = 75, 204

# Volume (Å³)
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}
VOL_MIN, VOL_MAX = 60.1, 227.8

# Node feature indices (0-19 one-hot AA; 20-25 continuous)
NODE_FEATURE_INDEX = {
    'plddt': 20, 'hydrophobicity': 21, 'charge': 22,
    'mw': 23, 'volume': 24, 'rel_position': 25
}


def node_feature_keep_indices_from_exclude(exclude_names: List[str]) -> List[int]:
    """Return list of node feature indices to keep. One-hot 0-19 always kept; continuous by name."""
    exclude = set(s.strip().lower() for s in exclude_names if s)
    keep = list(range(20))
    for name, idx in NODE_FEATURE_INDEX.items():
        if name not in exclude:
            keep.append(idx)
    return keep


# =============================================================================
# PDB PARSING
# =============================================================================

def parse_pdb(pdb_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Parse PDB file and extract residue information.
    
    Returns:
        aa_sequence: List of 1-letter amino acid codes
        ca_coords: Cα coordinates (N, 3)
        plddt_values: Per-residue pLDDT from B-factor (N,)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_path)
    
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    aa_sequence = []
    ca_coords = []
    plddt_values = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip heteroatoms
                    continue
                    
                resname = residue.get_resname()
                aa = aa_map.get(resname, 'X')
                
                if aa == 'X':
                    continue
                
                if 'CA' in residue:
                    aa_sequence.append(aa)
                    ca_coords.append(residue['CA'].get_coord())
                    plddt_values.append(residue['CA'].get_bfactor())
    
    return aa_sequence, np.array(ca_coords), np.array(plddt_values)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def compute_node_features(
    aa_sequence: List[str],
    plddt_values: np.ndarray,
    length: int
) -> torch.Tensor:
    """
    Compute node features for each residue.
    
    Features (per residue):
    - One-hot AA type (20)
    - pLDDT confidence (1)
    - Hydrophobicity normalized (1)
    - Charge at pH 7 (1)
    - Molecular weight normalized (1)
    - Volume normalized (1)
    - Relative position (1)
    
    Total: 26 features per node
    """
    n_residues = len(aa_sequence)
    n_features = 20 + 6  # one-hot + continuous
    
    features = np.zeros((n_residues, n_features), dtype=np.float32)
    
    for i, aa in enumerate(aa_sequence):
        # One-hot encoding (20 dims)
        if aa in AA_TO_IDX:
            features[i, AA_TO_IDX[aa]] = 1.0
        
        # pLDDT (normalized to [0, 1])
        features[i, 20] = plddt_values[i] / 100.0 if plddt_values[i] > 1 else plddt_values[i]
        
        # Hydrophobicity (normalized to [-1, 1])
        hydro = HYDROPHOBICITY.get(aa, 0)
        features[i, 21] = (hydro - HYDRO_MIN) / (HYDRO_MAX - HYDRO_MIN) * 2 - 1
        
        # Charge
        features[i, 22] = CHARGE_PH7.get(aa, 0)
        
        # Molecular weight (normalized to [0, 1])
        mw = MOLWEIGHT.get(aa, 100)
        features[i, 23] = (mw - MW_MIN) / (MW_MAX - MW_MIN)
        
        # Volume (normalized to [0, 1])
        vol = VOLUME.get(aa, 100)
        features[i, 24] = (vol - VOL_MIN) / (VOL_MAX - VOL_MIN)
        
        # Relative position in sequence [0, 1]
        features[i, 25] = i / (length - 1) if length > 1 else 0.5
    
    return torch.tensor(features, dtype=torch.float32)


def compute_edges(
    ca_coords: np.ndarray,
    distance_threshold: float = 8.0,
    include_sequential: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph edges based on spatial proximity and sequential connectivity.
    
    Args:
        ca_coords: Cα coordinates (N, 3)
        distance_threshold: Max Cα-Cα distance for spatial edges (Å)
        include_sequential: Whether to include i→i+1 edges
        
    Returns:
        edge_index: (2, E) tensor of edge indices
        edge_attr: (E, 3) tensor of edge features [distance, seq_dist, edge_type]
    """
    n_residues = len(ca_coords)
    
    edges = []
    edge_features = []
    
    for i in range(n_residues):
        for j in range(n_residues):
            if i == j:
                continue
            
            # Compute Euclidean distance
            dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
            seq_dist = abs(i - j)
            
            # Sequential edge (i, i+1)
            is_sequential = seq_dist == 1
            
            # Spatial edge (within threshold)
            is_spatial = dist < distance_threshold
            
            if include_sequential and is_sequential:
                edges.append([i, j])
                # [distance (normalized), seq_distance (normalized), edge_type (0=seq, 1=spatial)]
                edge_features.append([
                    dist / 20.0,  # Normalize distance (typical max ~20Å)
                    seq_dist / n_residues,  # Normalize by length
                    0.0  # Sequential edge type
                ])
            elif is_spatial and not is_sequential:
                edges.append([i, j])
                edge_features.append([
                    dist / 20.0,
                    seq_dist / n_residues,
                    1.0  # Spatial edge type
                ])
    
    if len(edges) == 0:
        # Fallback: at least include sequential edges
        for i in range(n_residues - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])
            edge_features.append([0.19, 1.0 / n_residues, 0.0])
            edge_features.append([0.19, 1.0 / n_residues, 0.0])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    return edge_index, edge_attr


def pdb_to_graph(
    pdb_path: str,
    label: int,
    peptide_id: str = None,
    distance_threshold: float = 8.0,
    geometric_features: Optional[np.ndarray] = None,
    node_feature_keep_indices: Optional[List[int]] = None
) -> Data:
    """
    Convert a PDB file to a PyTorch Geometric Data object.
    node_feature_keep_indices: if set, keep only these node feature dimensions (0-25).
    """
    aa_sequence, ca_coords, plddt_values = parse_pdb(pdb_path)
    n_residues = len(aa_sequence)
    if n_residues < 2:
        raise ValueError(f"Peptide too short: {n_residues} residues")
    x = compute_node_features(aa_sequence, plddt_values, n_residues)
    if node_feature_keep_indices is not None:
        x = x[:, node_feature_keep_indices]
    
    # Compute edges
    edge_index, edge_attr = compute_edges(ca_coords, distance_threshold)
    
    # Store coordinates for EGNN
    pos = torch.tensor(ca_coords, dtype=torch.float32)
    
    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=n_residues
    )
    
    # Add optional geometric features
    if geometric_features is not None:
        data.geo_features = torch.tensor(geometric_features, dtype=torch.float32).unsqueeze(0)
    
    # Add metadata
    if peptide_id:
        data.peptide_id = peptide_id
    seq_joined = ''.join(aa_sequence)
    if any(c not in AA_TO_IDX for c in seq_joined):
        raise ValueError(
            f"PDB {pdb_path!r} contains non-standard residue letters in parsed sequence"
        )
    data.sequence = seq_joined

    return data


def _is_missing_pdb_file_value(pdb_file: object) -> bool:
    if pdb_file is None:
        return True
    if isinstance(pdb_file, float) and pd.isna(pdb_file):
        return True
    s = str(pdb_file).strip()
    return not s or s.lower() == "nan"


def esm2_residue_file_stem(peptide_id: str | int | float) -> str:
    """Filesystem-safe stem matching esm_sequence_processor per-residue saves."""
    s = str(peptide_id).strip().replace("\\", "/")
    return s.replace("/", "_").replace(":", "_")


def load_esm2_per_residue_tensor(esm2_residue_dir: Path | str, peptide_id: str | int | float) -> torch.Tensor:
    """
    Load per-residue ESM-2 representations (L, D) from ``{stem}.pt`` under esm2_residue_dir.
    File format: dict with key ``embedding`` (float tensor) or a raw tensor.
    """
    d = Path(esm2_residue_dir)
    stem = esm2_residue_file_stem(peptide_id)
    candidates = [d / f"{stem}.pt"]
    # Backwards/interop: some pipelines generate per-residue tensors without AMP_/DECOY_ prefix
    # while geometric_features/peptide_id may include it.
    s = str(peptide_id).strip()
    for prefix in ("AMP_", "DECOY_"):
        if s.startswith(prefix):
            candidates.append(d / f"{esm2_residue_file_stem(s[len(prefix):])}.pt")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Per-residue ESM2 file not found: {candidates[0]}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        t = payload["embedding"]
    else:
        t = payload
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.to(dtype=torch.float32)


def read_canonical_sequences(canonical_path: Path | str) -> Dict[str, str]:
    """
    Read canonical sequences file produced by the pipeline (space-separated: "<id> <sequence>").
    Returns dict mapping peptide_id -> sequence (uppercase, no whitespace).
    """
    p = Path(canonical_path)
    if not p.is_file():
        raise FileNotFoundError(f"Canonical sequences file not found: {p}")
    out: Dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        pid = parts[0].strip()
        seq = "".join(parts[1:]).strip().upper()
        if pid and seq:
            out[pid] = seq
    if not out:
        raise ValueError(f"No sequences parsed from canonical file: {p}")
    return out


class ESM2WindowResolver:
    """
    Resolve a window-level sequence to a parent per-residue ESM2 tensor slice.

    This is a compatibility path for pipelines that:
    - generate many window PDBs/feature rows (e.g. IDs like SEQ_1234), but
    - only store per-residue ESM2 tensors for the original parent sequences (canonical ids).
    """

    def __init__(self, esm2_residue_dir: Path | str, canonical_seqs_path: Path | str):
        self.esm2_residue_dir = Path(esm2_residue_dir)
        self.parents = read_canonical_sequences(canonical_seqs_path)
        # Precompute search order (longest parents first helps ambiguous substring matches).
        self._parent_items = sorted(self.parents.items(), key=lambda kv: len(kv[1]), reverse=True)
        self._parent_tensor_cache: Dict[str, torch.Tensor] = {}

    def _find_parent_and_start(self, window_seq: str) -> Tuple[str, int]:
        w = str(window_seq).strip().upper()
        if not w:
            raise ValueError("Empty window sequence; cannot resolve ESM2 slice.")
        for pid, parent_seq in self._parent_items:
            start = parent_seq.find(w)
            if start >= 0:
                return pid, start
        raise KeyError(
            "Could not map window sequence to any canonical parent sequence. "
            "Provide per-window ESM2 tensors, or ensure canonical_seqs.txt matches your windows."
        )

    def slice_window_tensor(self, window_seq: str) -> torch.Tensor:
        pid, start = self._find_parent_and_start(window_seq)
        parent_t = self._parent_tensor_cache.get(pid)
        if parent_t is None:
            parent_t = load_esm2_per_residue_tensor(self.esm2_residue_dir, pid)
            self._parent_tensor_cache[pid] = parent_t
        wlen = len(str(window_seq).strip())
        end = start + wlen
        if end > int(parent_t.shape[0]):
            raise ValueError(
                f"Window slice [{start}:{end}] out of range for parent {pid!r} "
                f"(parent_len={int(parent_t.shape[0])}, window_len={wlen})."
            )
        return parent_t[start:end].clone()


def resolve_peptide_pdb_path(
    pdb_dir: Path | str,
    pdb_file: str | float | None,
    peptide_id: str | int | float,
) -> Optional[Path]:
    """Locate a peptide PDB under ``pdb_dir``.

    Layouts supported:

    - **Pipeline / unlabeled ESMFold** (``run_esmfold_peptides --unlabeled``): PDBs under
      ``<structures_dir>/sequences/<id>.pdb`` while ``geometric_features.csv`` often stores
      only ``<id>.pdb``.
    - **Labeled ESMFold**: ``<structures_dir>/AMP/`` and ``.../DECOY/``.
    - **Flat**: PDBs directly in ``pdb_dir`` (e.g. ``pdb_dir`` already ``.../structures/sequences``).
    - **Legacy**: ``pdb_dir/structures/{AMP,DECOY,sequences}/`` when ``pdb_dir`` is a parent
      of a ``structures/`` tree.

    If ``pdb_file`` is a relative path (e.g. from ``results_log``), it is resolved under
    ``pdb_dir`` first.
    """
    pdb_dir = Path(pdb_dir)
    pid = str(peptide_id).strip()

    name: str
    if not _is_missing_pdb_file_value(pdb_file):
        pf = str(pdb_file).strip().replace("\\", "/")
        direct = pdb_dir / pf
        if direct.is_file():
            return direct
        name = Path(pf).name
    else:
        name = pid if pid.endswith(".pdb") else f"{pid}.pdb"

    candidates: List[Path] = [
        pdb_dir / name,
        pdb_dir / "sequences" / name,
        pdb_dir / "AMP" / name,
        pdb_dir / "DECOY" / name,
        pdb_dir / "structures" / "sequences" / name,
        pdb_dir / "structures" / "AMP" / name,
        pdb_dir / "structures" / "DECOY" / name,
        pdb_dir / "structures" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# =============================================================================
# DATASET CLASS
# =============================================================================

class PeptideGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for peptide graphs.
    
    Loads PDB files and converts them to graphs on-the-fly or from cache.
    """
    
    def __init__(
        self,
        csv_path: str,
        pdb_dir: str,
        distance_threshold: float = 8.0,
        use_geometric_features: bool = False,
        geometric_feature_cols: Optional[List[str]] = None,
        tabular_scaler_path: Optional[str] = None,
        node_feature_keep_indices: Optional[List[int]] = None,
        esm2_residue_dir: Optional[str] = None,
        canonical_seqs_path: Optional[str] = None,
        transform=None,
        pre_transform=None
    ):
        self.csv_path = csv_path
        self.pdb_dir = Path(pdb_dir)
        self.distance_threshold = distance_threshold
        self.use_geometric_features = use_geometric_features
        self.node_feature_keep_indices = node_feature_keep_indices
        self.esm2_residue_dir = Path(esm2_residue_dir).resolve() if esm2_residue_dir else None
        self._esm2_window_resolver: Optional[ESM2WindowResolver] = None
        if self.esm2_residue_dir is not None and canonical_seqs_path:
            try:
                self._esm2_window_resolver = ESM2WindowResolver(self.esm2_residue_dir, canonical_seqs_path)
            except Exception:
                # Best-effort: if this fails, we still try direct per-window loads.
                self._esm2_window_resolver = None
        self.tabular_scaler = None
        if tabular_scaler_path:
            from gnn.extra_feature_scaler import load_extra_feature_scaler

            self.tabular_scaler = load_extra_feature_scaler(str(tabular_scaler_path))
        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        
        # Default geometric feature columns (pLDDT excluded to avoid leakage)
        if geometric_feature_cols is None:
            self.geo_cols = [
                'radius_gyration', 'end_to_end_distance', 'max_pairwise_distance',
                'centroid_distance_mean', 'centroid_distance_std',
                'fraction_helix', 'fraction_sheet', 'fraction_coil',
                'total_sasa', 'hydrophobic_sasa', 'fraction_hydrophobic_sasa',
                'length', 'net_charge', 'mean_hydrophobicity', 'hydrophobic_moment',
                'curvature_mean', 'curvature_std', 'curvature_max',
                'torsion_mean', 'torsion_std'
            ]
        else:
            self.geo_cols = geometric_feature_cols
        
        # Validate columns exist
        if self.use_geometric_features:
            missing = [c for c in self.geo_cols if c not in self.df.columns]
            if missing:
                print(f"Warning: Missing geometric feature columns: {missing}")
                self.geo_cols = [c for c in self.geo_cols if c in self.df.columns]
            if self.tabular_scaler is not None and self.geo_cols != self.tabular_scaler.feature_cols:
                raise ValueError(
                    "tabular_scaler feature column order/names must match geometric_feature_cols "
                    f"(scaler: {len(self.tabular_scaler.feature_cols)} cols, "
                    f"dataset: {len(self.geo_cols)} cols)"
                )
        
        super().__init__(None, transform, pre_transform)
    
    def len(self) -> int:
        return len(self.df)
    
    def get(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        
        pdb_file = row.get("pdb_file", None)
        pdb_path = resolve_peptide_pdb_path(self.pdb_dir, pdb_file, row["peptide_id"])
        if pdb_path is None:
            hint = row.get("pdb_file", f"{row['peptide_id']}.pdb")
            raise FileNotFoundError(
                f"PDB not found for peptide_id={row['peptide_id']!r} pdb_file={hint!r} under {self.pdb_dir}"
            )
        
        # Get label and convert to 0/1 (handle -1/1 or 0/1 formats)
        raw_label = int(row['label'])
        label = 1 if raw_label == 1 else 0  # Convert -1 to 0, keep 1 as 1
        
        # Get geometric features if requested
        geo_feats = None
        if self.use_geometric_features and self.geo_cols:
            raw = row[self.geo_cols].values.astype(np.float64)
            raw = np.nan_to_num(raw, nan=0.0)
            if self.tabular_scaler is not None:
                raw = self.tabular_scaler.transform_vector(raw)
            geo_feats = raw.astype(np.float32)
        
        data = pdb_to_graph(
            str(pdb_path),
            label,
            peptide_id=row['peptide_id'],
            distance_threshold=self.distance_threshold,
            geometric_features=geo_feats,
            node_feature_keep_indices=self.node_feature_keep_indices
        )
        if self.esm2_residue_dir is not None:
            try:
                esm = load_esm2_per_residue_tensor(self.esm2_residue_dir, row["peptide_id"])
            except FileNotFoundError:
                if self._esm2_window_resolver is None:
                    raise
                esm = self._esm2_window_resolver.slice_window_tensor(row.get("sequence", ""))
            if int(esm.shape[0]) != int(data.num_nodes):
                raise ValueError(
                    f"ESM2 length {esm.shape[0]} != graph nodes {data.num_nodes} "
                    f"for peptide_id={row['peptide_id']!r}"
                )
            data.esm2_node = esm
        return data


def create_dataloaders(
    csv_path: str,
    pdb_dir: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: Optional[np.ndarray] = None,
    batch_size: int = 32,
    distance_threshold: float = 8.0,
    use_geometric_features: bool = False,
    num_workers: int = 0
) -> Tuple:
    """
    Create train/val/test dataloaders.
    
    Args:
        csv_path: Path to CSV file
        pdb_dir: Directory with PDB files
        train_idx, val_idx, test_idx: Sample indices for each split
        batch_size: Batch size
        distance_threshold: Edge distance threshold
        use_geometric_features: Include pre-computed geometric features
        num_workers: DataLoader workers
        
    Returns:
        train_loader, val_loader, (test_loader if test_idx provided)
    """
    from torch_geometric.loader import DataLoader
    
    # Load full dataset
    full_dataset = PeptideGraphDataset(
        csv_path=csv_path,
        pdb_dir=pdb_dir,
        distance_threshold=distance_threshold,
        use_geometric_features=use_geometric_features
    )
    
    # Create subset datasets
    train_dataset = [full_dataset[i] for i in train_idx]
    val_dataset = [full_dataset[i] for i in val_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if test_idx is not None:
        test_dataset = [full_dataset[i] for i in test_idx]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader

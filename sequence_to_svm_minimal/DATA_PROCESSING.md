# Data processing: TXT/FASTA to GNN-ready inputs

Pipeline from sequence files to `geometric_features_clustered.csv` and PDBs consumed by the GNN.

---

## 1. Sequences → ESMFold input format

**Input:** FASTA files (e.g. `amps.fasta`, `decoys.fasta`) or plain TXT (one sequence per line or `id\tsequence`).

**Script:** `scripts/data_generation/convert_fasta_to_svm.py`

- Reads AMP and decoy FASTA; supports same-line header+sequence.
- Filters by length (`--min-length`, `--max-length`), optional decoy subsample (`--max-decoys`), valid IDs.
- Writes two files: `seqs_AMP.txt`, `seqs_decoy.txt` in **SVM format**: one line per sequence, `id\tsequence` (tab-separated). The first column is the sequence ID (used as prefix for PDB names).

**Example:**

```bash
python scripts/data_generation/convert_fasta_to_svm.py --amp-fasta amps.fasta --decoy-fasta decoys.fasta --output-dir data/gnn_training_dataset
# → data/gnn_training_dataset/seqs_AMP.txt, seqs_decoy.txt
```

**Alternative:** If you already have a TXT with one `index sequence` or `id\tsequence` per line, you can use that directly as the ESMFold input (next step). The first column becomes the identifier; the second is the sequence.

---

## 2. ESMFold: sequences → PDBs and results log

**Input:** `seqs_AMP.txt`, `seqs_decoy.txt` (or equivalent: first column = identifier, second = sequence).

**Script:** `models/run_esmfold_peptides.py`

- Parses each file; builds `unique_id = {prefix}_{first_column}` (e.g. `AMP_1`, `DECOY_42`).
- Runs ESMFold per sequence; writes one PDB per peptide as `{unique_id}.pdb`.
- PDBs go under `{output_dir}/structures/AMP/` and `{output_dir}/structures/DECOY/`.
- Appends one row per sequence to `{output_dir}/results_log.csv`: `unique_id`, `original_idx`, `sequence`, `length`, `label` (1 AMP, -1 decoy), `status`, `pdb_file`, timings.

**Example:**

```bash
python models/run_esmfold_peptides.py --amp-file data/gnn_training_dataset/seqs_AMP.txt --decoy-file data/gnn_training_dataset/seqs_decoy.txt --output data/gnn_training_dataset
# → data/gnn_training_dataset/structures/AMP/*.pdb, structures/DECOY/*.pdb, results_log.csv
```

---

## 3. PDBs + results log → geometric features CSV

**Input:** Directory containing PDBs (recursively found). For **label** and **sequence** columns (required for GNN training and sequence clustering), a results log is required.

**Script:** `scripts/data_generation/build_geometric_features.py`

- Finds all `*.pdb` under `--pdb-dir`. `peptide_id` = PDB filename stem (e.g. `AMP_1`).
- **Including labels (and sequence):** The CSV gets `label` and `sequence` only when `results_log.csv` is available. That file is produced by `run_esmfold_peptides.py` (step 2) and contains `unique_id`, `sequence`, `label` (1 = AMP, -1 = decoy), etc. The script will:
  - **Auto-load** `results_log.csv` from `pdb_dir`, or from `pdb_dir.parent`, if that file exists (so if PDBs are in `.../structures/AMP/`, passing `--pdb-dir .../structures` still finds `.../results_log.csv`).
  - Or you pass it explicitly: `--results-log path/to/results_log.csv`.
  - PDB filename stem (e.g. `AMP_1` from `AMP_1.pdb`) must match the results log’s `unique_id` (or `peptide_id`) so each row can get a label.
- For each PDB: extracts structure-based features; if results log was loaded, adds `sequence` and `label` to the row. Always adds `peptide_id`, `pdb_file`.
- Writes one row per peptide to `geometric_features.csv` (or `.parquet`).

**Example (labels included – use ESMFold output dir so results_log.csv is present):**

```bash
# PDBs and results_log.csv live under data/gnn_training_dataset (from step 2)
python scripts/data_generation/build_geometric_features.py --pdb-dir data/gnn_training_dataset --output data/gnn_training_dataset/geometric_features.csv
```

**Example (labels when PDBs and results log are in different places):**

```bash
python scripts/data_generation/build_geometric_features.py --pdb-dir path/to/pdbs --results-log path/to/results_log.csv --output path/to/geometric_features.csv
```

**Without a results log:** The output CSV will have no `label` or `sequence` column. Clustering and GNN training will not work until you add labels (and sequence if you need sequence-based clustering).

---

## 4. Geometric features CSV → clustered CSV (for GNN CV)

**Input:** `geometric_features.csv` (must have `peptide_id`, `sequence`; `label` and feature columns as produced by step 3).

**Option A – CD-HIT (external):**  
`nn_pipeline/prepare_clusters.py` can export a FASTA for CD-HIT, then you parse the cluster file back:

```bash
python nn_pipeline/prepare_clusters.py --generate-fasta --input data/gnn_training_dataset/geometric_features.csv --output data/gnn_training_dataset/sequences.fasta
# Run CD-HIT: cd-hit -i sequences.fasta -o clusters -c 0.40 -n 2 -M 16000
python nn_pipeline/prepare_clusters.py --parse-clusters --clstr-file data/gnn_training_dataset/clusters.clstr --features-csv data/gnn_training_dataset/geometric_features.csv --output data/gnn_training_dataset/geometric_features_clustered.csv
```

**Option B – Simple sequence-identity clustering (no CD-HIT):**  
Use `create_simple_clusters()` (e.g. from `nn_pipeline/prepare_clusters.py` or `scripts/run_nn_training.py`):

```bash
python nn_pipeline/prepare_clusters.py --simple-clusters --input data/gnn_training_dataset/geometric_features.csv --output data/gnn_training_dataset/geometric_features_clustered.csv
```

Output: same columns as the input CSV plus `cluster_id` for group-based splits.

---

## 5. GNN consumption (CSV + PDBs → graphs)

**Inputs:** `geometric_features_clustered.csv`, PDB directory (e.g. `data/gnn_training_dataset` with `structures/AMP/`, `structures/DECOY/`).

**Code:** `gnn/data_utils.py` — `PeptideGraphDataset` and `pdb_to_graph`.

- Dataset reads the CSV; for each row resolves PDB path from `pdb_file` or `{peptide_id}.pdb` under `pdb_dir` (tries `structures/AMP`, `structures/DECOY`, `structures`, then root).
- For each PDB: parse Cα + B-factor → node features (26-dim: one-hot AA, pLDDT, hydrophobicity, charge, mw, volume, rel_position); build edges (sequential + spatial < 8 Å); `pos` = Cα coords.
- Optional: attach selected CSV columns as graph-level `geo_features`.
- Output: one PyG `Data` per row (`x`, `edge_index`, `edge_attr`, `pos`, `y`, optional `geo_features`).

See the second half of this doc (below) for the exact graph construction steps if needed.

---

## Summary flow

```
FASTA/TXT  →  convert_fasta_to_svm  →  seqs_AMP.txt, seqs_decoy.txt
     →  run_esmfold_peptides  →  structures/AMP|DECOY/*.pdb, results_log.csv
     →  build_geometric_features  →  geometric_features.csv
     →  prepare_clusters (CD-HIT or simple)  →  geometric_features_clustered.csv
     →  PeptideGraphDataset(csv, pdb_dir)  →  PyG Data (graphs)
```

---

## Graph construction (from CSV + PDB)

- **CSV:** Required columns `peptide_id`, `label`. Optional `pdb_file`, `cluster_id`, and geometric feature columns.
- **PDB lookup:** First existing of `{pdb_dir}/structures/AMP/{pdb_file}`, `structures/DECOY/`, `structures/`, or `{pdb_dir}/`.
- **Parse PDB:** Cα residue names → 1-letter AA; Cα (x,y,z); Cα B-factor → `aa_sequence`, `ca_coords`, `plddt_values`.
- **Node features:** 26-dim per residue (one-hot 0–19; 20 pLDDT; 21–25 hydrophobicity, charge, mw, volume, rel_position). Optional ablation drops dimensions.
- **Edges:** Sequential (i, i+1) plus spatial (Cα–Cα < threshold); `edge_attr` (E, 3).
- **pos:** Cα coordinates (N, 3).
- **geo_features:** Optional vector from CSV columns, one per graph.


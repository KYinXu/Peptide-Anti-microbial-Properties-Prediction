# Data processing: FASTA/TXT → trainable inputs

Compact pipeline from sequence files to GNN/NN trainable data. All paths below are placeholders; replace with your dirs.

---

## Input formats

- **AMP + decoy:** Two files (one sequence per line).
- **Accepted:** (1) Plain — one sequence per line. (2) Indexed — `index sequence` or `id\tsequence` per line; `#` ignored.
- **ESMFold** assigns IDs from line order if only one column: `AMP_1`, `AMP_2`, … and `DECOY_1`, … (labels: 1 / -1).

---

## 1. Optional: FASTA → seqs

**Script:** `scripts/data_generation/convert_fasta_to_svm.py`

- Input: `--amp-fasta`, `--decoy-fasta`. Output: `--output-dir`/`seqs_AMP.txt`, `seqs_decoy.txt` (id, sequence).
- Optional: `--max-decoys`, `--min-length`, `--max-length`, `--seed`.

---

## 2. ESMFold: sequences → PDBs + results log

**Script:** `models/run_esmfold_peptides.py`

- Args: `--amp-file`, `--decoy-file`, `--output` (directory).
- **Output:** `{output}/AMP/*.pdb`, `{output}/DECOY/*.pdb`, `{output}/results_log.csv`. Use `--output` = a dir named e.g. `structures` so that GNN can use its parent as `pdb_dir` (see below).

---

## 3. PDBs + results log → geometric features

**Script:** `scripts/data_generation/build_geometric_features.py`

- Args: `--pdb-dir` (finds PDBs recursively), `--output` (CSV/parquet). For labels/sequence: `--results-log` or leave `results_log.csv` under `pdb-dir` (auto-loaded).
- Optional: `--svm-predictions`, `--qsar-descriptors` to merge SVM/12-descriptor columns (index must align with peptide_id suffix).
- **Output:** `geometric_features.csv` (peptide_id, sequence, pdb_file, label, features). Main trainable table for NN and GNN.

---

## 4. Optional: Clustering (for cluster-based CV)

**Script:** `nn_pipeline/prepare_clusters.py`

- **CD-HIT:** `--generate-fasta` (from geometric_features.csv) → run `cd-hit -i … -o clusters -c 0.40 -n 2 -M 16000` → `--parse-clusters` with `--clstr-file`, `--features-csv`, `--output` → `geometric_features_clustered.csv`.
- **Simple:** `--simple-clusters --input … --output …` (no CD-HIT). Adds `cluster_id`.

---

## 5. SVM path (sequences → descriptors → predictions)

- **FASTA → seqs:** Step 1 (convert_fasta_to_svm) if starting from FASTA.
- **Seqs → descriptors + SVM:** `scripts/data_generation/run_sequence_svm.py` — `--seqs` (2-col: index, sequence), `--aaindex`, `--output-dir`, `--model-pkl`, `--scaler-csv`. Produces `descriptors.csv`, `descriptors_PREDICTIONS.csv` in `--output-dir`. Optional `--start`, `--stop`.
- Use these CSVs in step 3 via `--svm-predictions` / `--qsar-descriptors` so geometric CSV has SVM/QSAR columns for NN pipeline.

---

## Consumers

| Consumer | CSV | PDB dir |
|----------|-----|--------|
| GNN (`scripts/run_gnn_training.py`) | geometric_features.csv or _clustered | Parent of dir containing `structures/AMP/` and `structures/DECOY/` |
| NN / FeaturePipeline | geometric_features.csv (or _clustered) | — |

SVM/descriptor columns are optional; training works with geometric_features.csv only.

---

## Flow

```
FASTA/TXT → [convert_fasta_to_svm] → seqs_AMP.txt, seqs_decoy.txt
  → run_esmfold_peptides → {output}/AMP|DECOY/*.pdb, results_log.csv
  → build_geometric_features → geometric_features.csv
  → [prepare_clusters] → geometric_features_clustered.csv
  → GNN: PeptideGraphDataset(csv, pdb_dir); NN: FeaturePipeline(geometric_csv=…)
```

---

## Graph build (GNN)

- **CSV:** `peptide_id`, `label`; optional `pdb_file`, `cluster_id`, geometric columns.
- **PDB resolve:** `pdb_dir`/`structures/AMP`, `structures/DECOY`, `structures`, or `pdb_dir` + `pdb_file` or `{peptide_id}.pdb`.
- **Per PDB:** Cα + B-factor → nodes (26-dim); sequential + spatial edges (< 8 Å); `pos` = Cα; optional graph-level `geo_features` from CSV.

# Data processing: FASTA/TXT → trainable inputs

Compact pipeline from sequence files to GNN/NN trainable data. All paths below are placeholders; replace with your dirs.

---

## Single entry: `scripts/run_data_pipeline.py`

Orchestrates the default **unlabeled** path in one command: normalize input → ESMFold → geometric features → **QSAR-12** → **ESM-2 embeddings**, writing outputs under **`generated/`** next to the input file (same directory as `--input`), unless you override with `--work-dir`, plus `pipeline_manifest.json` there.

- **Input:** `--input` path to a **txt** (one sequence per line, or `index sequence` / `id sequence` per line; `#` lines skipped) or **FASTA** (`.fa`/`.fasta`).
- **Defaults:** clustering is **off** (use `--with-cluster` for `prepare_clusters`). QSAR and ESM2 run unless `--skip-qsar` / `--skip-esm2`.
- **Outputs (typical):** `inputs/canonical_seqs.txt`, `structures/` (PDBs + `results_log.csv`), `geometric_features.csv`, `qsar12_descriptors.csv`, `esm2_embeddings.csv` (with `peptide_id` added for merges). With `--with-cluster`: `geometric_features_clustered.csv` and QSAR built from that file.
- **Optional:** `--with-svm` (+ `--svm-aaindex`, `--svm-model-pkl`, `--svm-scaler-csv`), `--train-legacy-gnn`, `--train-final-gnn` (passes `--csv_path`, `--pdb_dir`, `--qsar_csv`, `--esm2_csv` into `run_gnn_train_final_models.py`). See `python scripts/run_data_pipeline.py --help`.
- **Run from** the `sequence_to_svm_minimal` directory.

```bash
python scripts/run_data_pipeline.py --input data/test/Figshare/test_seqs.txt
# Writes to data/test/Figshare/generated/ by default
python scripts/run_data_pipeline.py --input seqs.fasta --work-dir data/my_run --dry-run
```

The sections below document each underlying script if you need to run steps individually.

---

## Input formats

- **AMP + decoy:** Two files (one sequence per line).
- **Accepted:** (1) Plain — one sequence per line. (2) Indexed — `index sequence` or `id\tsequence` per line; `#` ignored.
- **ESMFold** assigns IDs from line order if only one column: `AMP_1`, `AMP_2`, … and `DECOY_1`, … (labels: 1 / -1).

---

## 0. Optional: Clean FASTA data

**Script:** `scripts/data_generation/clean_fasta_file.py`

- Removes duplicate sequences from raw FASTA.
- Optional filters: `--min-len`, `--max-len`, `--drop-empty-id`.
- Controls duplicate retention: `--keep first` (default) or `last`.
- Input: `--input raw_amps.fasta`. Output: `--output amps.fasta`.

**Example:**

```bash
python scripts/data_generation/clean_fasta_file.py --input raw_amps.fasta --output amps.fasta --min-len 5 --max-len 200
```

---

## 1. Optional: FASTA → seqs

**Script:** `scripts/data_generation/convert_fasta_to_svm.py`

- Input: `--amp-fasta`, `--decoy-fasta`. Output: `--output-dir`/`seqs_AMP.txt`, `seqs_decoy.txt` (id, sequence).
- Optional: `--max-decoys`, `--min-length`, `--max-length`, `--seed`.

**Example:**

```bash
python scripts/data_generation/convert_fasta_to_svm.py --amp-fasta amps.fasta --decoy-fasta decoys.fasta --output-dir data/seqs --max-decoys 1000
```

---

## 2. ESMFold: sequences → PDBs + results log

**Script:** `models/run_esmfold_peptides.py`

- Args: `--amp-file`, `--decoy-file`, `--output` (directory).
- **Output:** `{output}/AMP/*.pdb`, `{output}/DECOY/*.pdb`, `{output}/results_log.csv`. Use `--output` = a dir named e.g. `structures` so that GNN can use its parent as `pdb_dir` (see below).

**Example:**

```bash
python models/run_esmfold_peptides.py --amp-file data/seqs/seqs_AMP.txt --decoy-file data/seqs/seqs_decoy.txt --output data
```

---

## 3. PDBs + results log → geometric features

**Script:** `scripts/data_generation/build_geometric_features.py`

- Args: `--pdb-dir` (finds PDBs recursively), `--output` (CSV/parquet). For labels/sequence: `--results-log` or leave `results_log.csv` under `pdb-dir` (auto-loaded).
- Optional: `--svm-predictions`, `--qsar-descriptors` to merge SVM/12-descriptor columns (index must align with peptide_id suffix).
- **Output:** `geometric_features.csv` (peptide_id, sequence, pdb_file, label, features). Main trainable table for NN and GNN.

**Example:**

```bash
python scripts/data_generation/build_geometric_features.py --pdb-dir data/structures --output data/geometric_features.csv
```

---

## 4. Optional: Clustering (for cluster-based CV)

**Script:** `nn_pipeline/prepare_clusters.py`

- **CD-HIT:** `--generate-fasta` (from geometric_features.csv) → run `cd-hit -i … -o clusters -c 0.40 -n 2 -M 16000` → `--parse-clusters` with `--clstr-file`, `--input`, `--output` → `geometric_features_clustered.csv`.
- **Simple:** `--simple-clusters --input … --output …` (no CD-HIT). Adds `cluster_id`.

**Example:**

```bash
# Using simple clustering (built-in fallback)
python nn_pipeline/prepare_clusters.py --simple-clusters --input data/geometric_features.csv --output data/geometric_features_clustered.csv

# Or using CD-HIT directly (if installed)
python nn_pipeline/prepare_clusters.py --run-cdhit --input data/geometric_features.csv --output data/geometric_features_clustered.csv
```

---

## 5. SVM path (sequences → 12 descriptors → SVM predictions)

This step is **optional**. It feeds the **trained legacy SVM** (margin / class probabilities), not the standalone QSAR-12 table used by `run_gnn_train_final_models.py` (that is step 6).

**What it does**

1. Reads a sequence list and runs `descriptors/descripGen_12_py3.py` to compute the **same 12 sequence descriptors** used historically with AAindex (written as `descriptors.csv`).
2. Runs `predictionsParameters/predictSVC.py` with your `--model-pkl` and `--scaler-csv` to produce **SVM outputs** (`descriptors_PREDICTIONS_unsorted.csv`, `descriptors_PREDICTIONS.csv`).
3. Those files can be merged into the geometric table in **step 3** via `build_geometric_features.py --svm-predictions` and/or `--qsar-descriptors` so the NN / legacy GNN path sees SVM margin and descriptor columns (index alignment with `peptide_id` as documented for that script).

**Script:** `scripts/data_generation/run_sequence_svm.py`

- **FASTA → seqs:** step 1 if you start from FASTA.
- **Args:** `--seqs` (each line: `index sequence`, or **one sequence per line** with auto 1…*n* index; `#` lines skipped), `--aaindex` (directory with `aaindex1` / `aaindex2` / `aaindex3`), `--output-dir`, `--model-pkl`, `--scaler-csv`. Optional `--start`, `--stop` (1-based row indices **after** skipping blanks/comments).

**Example:**

```bash
python scripts/data_generation/run_sequence_svm.py --seqs data/seqs/seqs_AMP.txt --aaindex descriptors/aaindex --output-dir data/svm_out --model-pkl models/svm_model.pkl --scaler-csv models/scaler.csv
```

---

## 6. QSAR-12 features (for final GNN / `run_gnn_train_final_models.py`)

**Script:** `scripts/data_generation/generate_qsar_features.py`

- Computes the **twelve QSAR-style columns** (`netCharge`, `FC`, … — see script) with `peptide_id` + `sequence` for each row.
- **Input (CSV, recommended):** e.g. `geometric_features.csv` from step 3 — must include `sequence` and an id column (`peptide_id`, or `name` / `id` / first column). That keeps IDs aligned with the GNN merge.
- **Input (plain text):** one sequence per line (IDs default to `seq_1`, `seq_2`, …), **or** `id<TAB>sequence` per line. Space-separated `id sequence` text files (like `seqs_AMP.txt`) are **not** parsed as two columns; convert to TSV or use CSV.
- **Output:** e.g. `qsar12_descriptors.csv`. **`peptide_id` in this file must match** `peptide_id` in `geometric_features.csv` so **step 8b** can merge.

**Example:**

```bash
python scripts/data_generation/generate_qsar_features.py --input data/geometric_features.csv --output data/qsar12_descriptors.csv
python scripts/data_generation/generate_qsar_features.py -i raw_sequences_one_per_line.txt -o data/qsar12_descriptors.csv
```

---

## 7. ESM-2 embeddings (for final GNN)

**Script:** `models/esm_sequence_processor.py`

- **Mode:** `--mode embeddings` (add `--mode fold` or `both` only if you also want structures from this entry point; structures for the main pipeline usually come from step 2).
- **Input:** SVM-style text (`index sequence` or one sequence per line), or FASTA (see script help).
- **Output:** CSV with `seqIndex` (the index from the input file) and columns `esm2_dim_0`, `esm2_dim_1`, …  
- **Alignment:** `run_gnn_train_final_models.py` can load **one merged CSV** (`peptide_id` or `seqIndex`) or **AMP + DECOY CSVs** whose `seqIndex` matches the first column of `seqs_AMP.txt` / `seqs_decoy.txt`. Use the **same** seq files (and order) as for ESMFold so rows line up with `results_log.csv` / geometric features.

**Example:**

```bash
python models/esm_sequence_processor.py --input data/seqs/seqs_AMP.txt --output data/esm2_amp.csv --mode embeddings --device cuda
python models/esm_sequence_processor.py --input data/seqs/seqs_decoy.txt --output data/esm2_decoy.csv --mode embeddings
```

More detail: `models/README.md`.

---

## 8. GNN Training

### 8a. Legacy / CV script (graph + optional Geo + QSAR only)

**Script:** `scripts/run_gnn_training.py`

- Uses `geometric_features*.csv` merged with QSAR (see script/config). No ESM2 embeddings in this path.
- Requires the CSV and the PDB directory.

**Example:**

```bash
python scripts/run_gnn_training.py --csv_path data/geometric_features_clustered.csv --pdb_dir data/ --architecture gcn --epochs 100
```

### 8b. Final models with ESM2 + Geo + QSAR (recommended for compare/inference)

**Script:** `scripts/run_gnn_train_final_models.py`

- **Inputs (unchanged upstream):** same `geometric_features_clustered.csv` (or `geometric_features.csv`) from step 3, PDBs from step 2, plus:
  - **QSAR-12:** from **step 6** (`generate_qsar_features.py`), e.g. `qsar12_descriptors.csv` (merged on `peptide_id` by the script).
  - **ESM2:** from **step 7** (`esm_sequence_processor.py --mode embeddings`); embedding tables with columns `esm2_dim_*` — AMP/DECOY split CSVs or one merged CSV as in `run_gnn_train_final_models.py` `CONFIG`.
- **Tabular scaling:** By default, Geo / QSAR / ESM2 blocks are **RobustScaler**-normalized on the **training split only** and a sidecar file is saved next to each checkpoint: `{checkpoint_stem}_tabular_scaler.joblib`. Raw CSVs are **not** modified. Use `--no_tabular_robust_scaler` to match older checkpoints trained on raw values.
- **Inference:** `scripts/data_evaluation/compare_model_predictions.py` and `scripts/data_evaluation/run_gnn_inference.py` load that sidecar automatically when it sits beside the `.pt` file; test CSV columns and order must match training.

**Optional diagnostics:** `scripts/data_evaluation/plot_feature_value_scales.py` — plots raw value scales across Geo / QSAR / ESM2 (no effect on training).

---

## Consumers


| Consumer | CSV | PDB dir |
| -------- | --- | ------- |
| GNN (`scripts/run_gnn_training.py`) | geometric_features.csv or _clustered (+ QSAR merge per script) | Parent of dir containing `structures/AMP/` and `structures/DECOY/` (or `structures/` as used by your layout) |
| GNN (`scripts/run_gnn_train_final_models.py`) | Same geometric CSV + `qsar12_descriptors.csv` + ESM2 CSV(s) with `esm2_dim_*` | Same as above (script `pdb_dir` must resolve PDBs) |
| `compare_model_predictions.py` / `run_gnn_inference.py` | Test geometric CSV (and merged columns matching the checkpoint) | Same PDB layout |
| NN / FeaturePipeline | geometric_features.csv (or _clustered) | — |

SVM/descriptor columns are optional for the legacy GNN script; the final-models script expects ESM2 columns as configured in `CONFIG`.

---

## Generated Data Reference


| Generated File                     | Created By Script             | What It Is / Purpose                                                                                           |
| ---------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `seqs_AMP.txt`, `seqs_decoy.txt`   | `convert_fasta_to_svm.py`     | Simple two-column text files (ID and Sequence) required by ESMFold and the SVM.                                |
| `AMP/*.pdb`, `DECOY/*.pdb`         | `run_esmfold_peptides.py`     | 3D structure files generated by ESMFold. Used by the GNN for node features and spatial edges.                  |
| `results_log.csv`                  | `run_esmfold_peptides.py`     | Log of ESMFold processing, mapping `peptide_id` to its `sequence`, `label` (AMP/DECOY), and `pdb_file`.        |
| `geometric_features.csv`           | `build_geometric_features.py` | Master tabular dataset. Contains global geometric properties (radius of gyration, SASA), labels, and sequence. |
| `geometric_features_clustered.csv` | `prepare_clusters.py`         | Same as `geometric_features.csv`, but adds a `cluster_id` column for fair cross-validation splitting.          |
| `descriptors.csv`                  | `run_sequence_svm.py`         | 12 QSAR-like sequence descriptors derived from the AAindex database.                                           |
| `descriptors_PREDICTIONS.csv`      | `run_sequence_svm.py`         | Predictions (like probability scores) from a baseline Support Vector Machine (SVM) model.                      |
| `qsar12_descriptors.csv` (typical) | `generate_qsar_features.py` | Twelve QSAR-style columns + `peptide_id` / `sequence`; merged in **step 8b** on `peptide_id`.        |
| `esm2_*.csv` (AMP/DECOY or merged) | `models/esm_sequence_processor.py` (`--mode embeddings`; see `models/README.md`) | Per-sequence embedding columns `esm2_dim_0`, … for GNN concatenation. |
| `*_tabular_scaler.joblib`         | `run_gnn_train_final_models.py` (default) | Fitted RobustScaler state for Geo/QSAR/ESM2 blocks; keep next to matching `.pt` for inference. |


---

## Flow

```
FASTA → [clean_fasta_file] → Clean FASTA
  → [convert_fasta_to_svm] → seqs_AMP.txt, seqs_decoy.txt
  → run_esmfold_peptides → {output}/AMP|DECOY/*.pdb, results_log.csv
  → build_geometric_features → geometric_features.csv
  → [prepare_clusters] → geometric_features_clustered.csv
  → [run_sequence_svm] → descriptors.csv, descriptors_PREDICTIONS.csv (optional; SVM path for step 3 merge)
  → [generate_qsar_features] → qsar12_descriptors.csv (optional; for final GNN, step 8b)
  → [esm_sequence_processor --mode embeddings] → esm2_amp.csv / esm2_decoy.csv (optional; for final GNN, step 8b)
  → GNN (final): run_gnn_train_final_models.py merges geometric + QSAR12 + ESM2; saves .pt + optional _tabular_scaler.joblib
  → GNN (legacy) / PeptideGraphDataset: run_gnn_training.py; NN: FeaturePipeline(geometric_csv=…)
```

---

## Graph build (GNN)

- **CSV:** `peptide_id`, `label`; optional `pdb_file`, `cluster_id`, geometric columns.
- **PDB resolve:** `pdb_dir`/`structures/AMP`, `structures/DECOY`, `structures`, or `pdb_dir` + `pdb_file` or `{peptide_id}.pdb`.
- **Per PDB:** Cα + B-factor → nodes (26-dim); sequential + spatial edges (< 8 Å); `pos` = Cα; optional graph-level `geo_features` from CSV.


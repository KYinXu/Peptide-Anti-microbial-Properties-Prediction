# Data, checkpoints, results

## `data/` layout

Every dataset directory follows the same shape:

```
data/<dataset>/[<experiment>/]
    ├── <raw source files>        # CSV exports, .fasta, .txt
    ├── sequences.csv             # canonical output of data_normalizer
    ├── generated/                # derived features (gitignored)
    │   ├── qsar12_descriptors.csv
    │   ├── geometric_features.csv
    │   ├── structures/           # one .pdb per sequence (gitignored)
    │   └── esm2_per_residue/     # per-residue embeddings
    └── results/runs/<timestamp>/ # analysis_outputs run dirs
```

`.gitignore` excludes `**/data/**/generated/` and `**/data/**/structures/`, so
`generated/` content is local-only and must be regenerated on a fresh clone.

| Directory | Contents | Has `generated/` |
|---|---|---|
| `Michael_sequences/` | Per-experiment subdirs (`10-21-wheat-mutations`, `10-27-DNA-fib`, `sanity_test`); sheet-export CSVs plus `sequences.csv` and run dirs | yes, all three |
| `Sabrina_sanity_test/` | `b3ppsexp_sequences.csv` | yes |
| `Tiff_sequences/` | Eight dated experiment subdirs (H2A variants, Buforin/LL37, Siryaporn, a2-a3, canonical H2A) | no |
| `proteomes/candidate_list/` | `sequences.csv`/`.txt`, `final_candidates.csv` | yes |
| `sanity_test/` | `sequences.csv`, `test_seqs.txt`, plus two stale loose `predictions*.csv` | yes |
| `skin_samples/` | UniProt `idmapping_*.fasta`, `ms_tissue.tsv`, `sequences.csv`, and `top_fragments/` with its own run dirs | no |
| `training/` | `Michael_training_sets/` (positive FASTA + negative CSV) and `expanded_amps/` | yes, both |
| `venom_AMPs/` | `sequences.csv` and run dirs | no |

Note that `.pdb` filenames in `structures/` are derived from sequence ids,
which can contain spaces and non-ASCII characters. Quote paths.

## `checkpoints/` — committed models

One directory per trained variant. SVM variants hold a pickle, a z-score table,
and an optional `summary/` of training-time CSVs.

| Directory | Contents |
|---|---|
| `svm/`, `svm_no_class_weight/`, `svm_null/`, `svm_null_QSO/`, `svm_null_QSO50/` | `svm_qsar12_model.pkl`, `svm_qsar12_zscores.txt`, `summary/{…training_descriptors.csv,…training_scores.csv}` |
| `original_dataset/original_svm/` | `svc.pkl` with **sharded** `svc.pkl_01.npy` … `_11.npy`, plus `Z_score_mean_std__intersect_noflip.csv` — the original-paper model |
| `original_dataset/null_svm/` | same layout as the `svm_*` variants |
| `original_dataset/broken_svm/` | name flags it as unusable; do not load |
| `gnn/gnn_vae_qsar/` | `gnn_model.pt`, `gnn_platt_scaling`, `gnn_temperature_scaling`, `model_summary.json`, `tabular_scaler.joblib` |
| `gnn/Michael_gnn_1/` | as above, without temperature scaling |
| `gnn/gnn_vae_qsar_original/` | older naming: `gat_ready_QSAR.pt`, `…_gnn_meta.json`, `…_platt.json` |

**Constraint:** `sequence_analysis` resolves SVM checkpoints by suffix search
within `--checkpoint-dir`, so a variant directory must contain **exactly one**
pickle and one z-score file, or resolution fails.

## `results/` — gitignored outputs

- `results/gnn_models/run_<YYYYMMDD_HHMMSS>/` — GNN training runs. Artifacts
  are `<arch>_ready_<featureset>.pt` with sibling `_gnn_meta.json`,
  `_platt.json`, sometimes `_tabular_scaler.joblib`, plus
  `ready_models_summary.json`. The two newest runs instead use a `gat_QSAR/`
  subdirectory, matching the newer `gnn_minimal` layout.
- `results/per_cluster_plots/<cluster id>/` — `fragment_heatmap.png`,
  `metric_histograms.png`, `sequence_heatmap.png`.
- `results/svm/` — empty.

### Two timestamp conventions coexist

| Producer | Format | Resolution |
|---|---|---|
| `analysis_outputs/runs.py` | `runs/%Y-%m-%d_%H-%M` | UTC, minute, `_NN` collision suffix |
| GNN training | `run_%Y%m%d_%H%M%S` | local time, seconds |

Only the first is produced by `analysis_outputs`. Don't assume one when
globbing for the other.

## `descriptors/`

`descripGen_12_py3.py` is a Python-3 port of a Python-2 original, deliberately
kept bit-compatible with it (including forced 2-decimal formatting to avoid
float drift). It computes 12 QSAR descriptors: `netCharge`, five dipeptide
compositions (`FC`, `LW`, `DP`, `NK`, `AE`), `pcMK`,
`_SolventAccessibilityD1025`, `tau2`/`tau4_GRAR740104`, and
`QSO50`/`QSO29_GRAR740104`. The worker is
`descripGen_bespoke(proseq, aap_dict_grar740104) -> (names, values)`
(`descripGen_12_py3.py:74`), with internal `v_lambda=7`, `v_nlag=30`,
`v_weight=0.05`.

```bash
python descriptors/descripGen_12_py3.py <aaindexDir> <input> <outDir> <start> <stop>
```

`start`/`stop` are 1-based inclusive; output is `<outDir>/descriptors.csv`.

**This script cannot run as checked out.** It requires a `propy` package from
one of three places: a vendored `descriptors/propy3_vendor/propy/`, a fallback
`descriptors/propy3_src/`, or an installed `propy`. Neither vendor directory
exists in the repo, and `import propy` fails in the `gnn` environment. Treat it
as reference for the descriptor definitions, not as a working entry point —
`gnn_minimal/pipeline/qsar.py` is the live QSAR path.

`descriptors/aaindex/` holds `aaindex1`, `aaindex2`, `aaindex3`. Only the
`GRAR740104` property is used, loaded once up front because repeated parsing is
expensive. A directory passed as `<aaindexDir>` must contain all three files.

## Strays at the repo root

| File | What it is |
|---|---|
| `test_overlap_speed.py` | Throwaway script despite the `test_` prefix — top-level code, no test functions, hardcoded legacy paths. A bare `pytest` from the root would collect and error on it. |
| `predictions.csv` | Stale 6-row dump (`id,predicted_class,prob_AMP`); matches no current runner schema. Leftover. |
| `notes.txt` | Gitignored one-line scratch note. |
| `output.png` | Gitignored orphan plot. |
| `env.yml` | Conda env `gnn`: python 3.10, `cd-hit`, `uv`. Notably does **not** list numpy/pandas/torch — those are installed separately. |
| `pyrightconfig.json` | Pyright with `typeCheckingMode: "off"`; effectively just a scan-scope limiter excluding the large generated trees. |
| `.cursor/` | Empty, gitignored. |
| `.vscode/` | Gitignored; only a `files.watcherExclude` for `esm2_per_residue/` and `structures/`. |

## `tests/`

Per the research-code rule, this is a remnant, not a standard to extend — do not add to it
unless asked. Two plain `unittest` files, no pytest config, no `conftest.py`,
no root `pyproject.toml`.

```bash
python -m unittest discover -s tests -t .   # from the repo root
```

`test_analysis_outputs.py` pins the run-directory and manifest behavior with a
fixed clock, including the same-minute collision path and the
failure-still-writes-a-manifest guarantee. `test_length_analysis.py` covers
variant generation, provenance recovery, and the length-prediction row shape;
its real-proteome smoke test self-skips unless large legacy artifacts exist.

`data_normalizer/`, `descriptors/` and `gnn_minimal/` have no tests.

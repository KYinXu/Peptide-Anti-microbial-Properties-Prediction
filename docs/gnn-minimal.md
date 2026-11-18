# `gnn_minimal/`

Self-contained GNN pipeline: labeled CSV → structures and features → trained
model → predictions. No Python import crosses out of this folder.

```
gnn_minimal/
├── train.py             CLI: labeled CSV -> model directory
├── train_premade_split.py  CLI: train/val/test CSVs -> model directory
├── inference.py         CLI: model dir + CSV -> predictions.csv
├── fit_temperature.py   CLI: post-hoc temperature calibration
├── process_data.py      library (not a CLI) — CSV -> generated/ artifacts
├── core/                model architectures, training loop, calibration, graph building
├── pipeline/            ESMFold/ESM2, geometric features, QSAR, resume
├── configs/             gnn_final_train.json (live), pipeline_defaults.json (dead)
└── descriptors/         descriptor_table_vae.csv, aaindex/
```

## Entry points

Each script does `sys.path.insert(0, Path(__file__).resolve().parent)` and then
imports `core.*` / `pipeline.*` as **top-level** modules. So run them as files
from anywhere; `python -m gnn_minimal.train` will **not** resolve the same way
(there is no `gnn_minimal/__init__.py`).

### `train.py`

```bash
python train.py path/to/dataset.csv
python train.py path/to/dataset.csv --presets Graph-only QSAR
python train.py path/to/generated/      # reuse processed artifacts
```

| Flag | Default | Notes |
|---|---|---|
| `input` (positional) | required | labeled CSV `id,label,sequence`, or a `generated/` dir |
| `--config` | `configs/gnn_final_train.json` | resolved module-relative |
| `--output_dir` | `results/gnn_models` | **cwd**-relative |
| `--val_size` | `0.2` | |
| `--weight_decay` | `1e-4` | |
| `--esm2_hidden_dim` | `64` | |
| `--presets` | all from config | names from `feature_sets` |
| `--force-process` | off | regenerate artifacts |
| `--device` | auto | **processing only** — training device is always auto |

There are no `--epochs`/`--lr`/`--arch` flags; those live in the config.

### `train_premade_split.py`

```bash
python train_premade_split.py --train train.csv --val val.csv --test test.csv
python train_premade_split.py --train train.csv --val val.csv --test test.csv --presets Combined
```

Same config, model layout and artifacts as `train.py`; the only difference is
where the split comes from. `--train`/`--val`/`--test` are required and each
takes a labeled CSV (or an already-processed `generated*/` dir). `--val_size`
is absent — the val set is given, not drawn. Other flags match `train.py`:
`--config`, `--output_dir`, `--weight_decay`, `--esm2_hidden_dim`,
`--presets`, `--force-process`, `--device`.

The scaler is fit on train only, early stopping and Platt use val, and test is
scored once at the end; its numbers land in `model_summary.json` under
`metrics` with a `test_` prefix beside the unprefixed val metrics. Overlapping
`peptide_id` between any two splits is an error. Like `train.py` it writes
Platt but never temperature; for temperature, pass the same val file to
`fit_temperature.py --val`.

### `inference.py`

```bash
python inference.py --model_path results/gnn_models/run_.../gat_QSAR \
                    --input path/to/new.csv
```

`--model_path` (required) may be the model dir or the `.pt`. `--input`
(required) takes `id,sequence` with `label` optional. `--batch_size` 32,
`--save_predictions` `predictions.csv`, `--force-process`, `--device`
(processing only).

### `fit_temperature.py`

```bash
python fit_temperature.py --model results/.../gat_QSAR --input labeled.csv  # re-derive the split
python fit_temperature.py --model results/.../gat_QSAR --val val.csv        # premade split
```

`--model` required, plus exactly one of `--input` or `--val`. `--input` is the
whole labeled table and the val rows are re-derived from it by `--seed` (42) /
`--val-size` (0.2), which must match the training run; `--val` is the val CSV
(or `generated*/` dir) of a premade split and is used as given, ignoring
`--seed`/`--val-size`. Also `--batch-size` 32, `--device`, `--force-process`.
Writes `gnn_temperature_scaling` into the model dir — **neither training
script ever writes it.** The JSON records `seed`/`val_size` in the first mode
and `val_source` in the second.

`--input` requires the CSV to cover the rows in its `generated/` dir; a CSV
labeling only a subset of a shared `generated/` fails with "Training data is
missing labels" (true of `train.py` too).

## Data flow

### Input

Training needs `id,label,sequence`. **`label == 1` is AMP; everything else —
including `0` and `-1` — maps to 0** (`train.py:55`). Extra columns are ignored.
`id` must be unique after stripping. Sequences must be the standard 20 amino
acids; anything else is dropped at folding time.

### `generated/` artifacts — always `<csv_parent>/generated/`

(Exception: `train_premade_split.py` uses `<csv_parent>/generated/generated_<csv_stem>/`
so split CSVs sitting in one directory cannot overwrite each other's tables.
Cached structures and feature rows are still reused from any `generated/` dir
above the CSV, so the per-split dirs are cheap.)

| Path | Contents |
|---|---|
| `structures/<id>.pdb` | ESMFold structures; pLDDT in the B-factor column |
| `esm2_per_residue/<id>.pt` | `{"seqIndex": str, "embedding": (L,1280)}` |
| `esm2_embeddings.csv` | pooled `peptide_id, seqIndex, esm2_dim_0…1279`; only when ESM2 runs |
| `qsar12_descriptors.csv` | `peptide_id, sequence,` + the 12 QSAR columns |
| `geometric_features.csv` | `peptide_id, sequence, pdb_file[, label],` + ~28 structural columns |

`process_dataset(..., fold_batch_size=N)` folds N sequences per ESMFold call,
grouped by exact length, and `ensure_processed_data` forwards it. The default 1
is one-at-a-time and bitwise reproducible; `length_analysis` passes 16. Mixed
lengths in one batch raise `IndexError` from `compute_tm()` under fp16, which is
why grouping is by exact length (`docs/analysis-modules.md` has the measured
speed and the structural effect).

`label` appears in `geometric_features.csv` only when the source CSV was
labeled. The canonical id key in every generated table is `peptide_id`, mapped
from the input `id`.

### Model directory — `<output_dir>/run_<YYYYmmdd_HHMMSS>/<arch>_<preset>/`

(`+` in a preset name becomes `_plus_`.) Filenames are defined once in
`core/artifacts.py:7`:

| File | Notes |
|---|---|
| `gnn_model.pt` | bare `state_dict`, keys prefixed `model.` |
| `model_summary.json` | `schema`, architecture, `node_feature_groups`, dims, `edge_feature_dim`, `normalize_geo`, `tabular_feature_cols`, metrics |
| `gnn_platt_scaling` | JSON **without an extension**; only if the fit succeeded |
| `tabular_scaler.joblib` | Geo / QSAR / Combined presets only |
| `gnn_temperature_scaling` | written only by `fit_temperature.py` |

### `model_summary.json` schema 3

Schema 3 fixed three input-scaling defects. Because none of them changes
`data.x` width, a checkpoint cannot be identified from its weights alone, so the
summary carries the answer and `peptide_gnn_arch_kwargs` (`core/checkpoint_meta.py`)
turns it back into `PeptideGNN` arguments:

| Key | Schema 3 | Pre-schema-3 default |
|---|---|---|
| `node_feature_groups.vae_zscore` | `true` — VAE table z-scored per descriptor | `false`, raw table |
| `edge_feature_dim` | `3` — `GATConv(edge_dim=3)`, `edge_attr` consumed | `0`, `edge_attr` discarded |
| `normalize_geo` | `true` — `BatchNorm1d` on `geo_features` before the concat | `false`, raw concat |

Every load site passes `**peptide_gnn_arch_kwargs(meta)`, so the five
`checkpoints/gnn/` models keep scoring exactly as before (verified: max drift
1.2e-6 over the 15 `proteome_sanity_test` peptides). The defaults are the *old*
behavior precisely because that is what a summary-less checkpoint is; forgetting
the call cannot mis-score a model, it raises in `load_state_dict` on the missing
`geo_bn` / `lin_edge` keys.

### Output

`predictions.csv`: `id, predicted_class, prob_AMP`, plus `true_label` when the
input was labeled. `predicted_class = prob_AMP >= 0.5`.

## Architectures and the two feature axes

`gcn`, `gat`, `egnn` (`core/models.py:446`), selected by `config.architectures`
— every listed architecture is trained for every selected preset. Shared trunk:
node features → optional ESM2 projection → message passing → `mean_max`
pooling (hardcoded) → concat tabular features (`BatchNorm1d` first, schema 3)
→ 2-layer MLP → 2 logits. `gat` passes `edge_attr` into every `GATConv` from
schema 3 on; `gcn` cannot (`GCNConv` carries a scalar `edge_weight` only) and
`egnn` always did.

**Axis 1 — node features, pre-message-passing** (`node_feature_groups`):

| Group | Width |
|---|---|
| `onehot` | 20 |
| `pdb_continuous` | 6 (pLDDT, hydrophobicity, charge, MW, volume, rel. position) |
| `vae_table` | 13 (from `descriptors/descriptor_table_vae.csv`, z-scored per row from schema 3) |
| `esm2_residue` | 1280 → `esm2_hidden_dim` via a learned linear layer |

All-on base width is `NODE_INPUT_DIM = 39`. The shipped config uses
`onehot + vae_table` = 33.

With `onehot` on, `vae_table` adds **no information** — both blocks are
functions of residue identity, so the 33-column node matrix has rank 20 over the
20 amino acids, same as the one-hot alone. It only changes conditioning, and the
raw table was bad at it: `MW` spans 75–204 and `vol` 3.4–21.7 against ±1 for
`F1`–`F6`, giving a condition number of 631 (10.3 z-scored) and making `MW` the
feature the first `GATConv`'s attention keyed on hardest. Hence `vae_zscore`.
Note that the *disabled* `pdb_continuous` block min-max normalizes its own MW and
volume columns; the VAE block never did.

**Axis 2 — graph-level tabular, post-pooling** (`feature_sets` presets):
`Graph-only` (none), `Geo` (20 cols), `QSAR` (12 cols), `Combined` (32 cols).

## Config

`configs/gnn_final_train.json` is the only live config. Keys that drive
behavior: `seed`, `epochs`, `batch_size`, `lr`, `patience`,
`hidden_channels`, `num_layers`, `dropout`, `distance_threshold`,
`label_smoothing`, `logit_penalty`, `node_feature_groups`, `architectures`,
`feature_sets_default`, `feature_sets`.

`node_feature_groups` accepts `vae_zscore` alongside the four block toggles
(`false` reproduces pre-schema-3 node features). Unlike the others it does not
change `data.x` width, so `node_feature_groups_for_base_dim` cannot recover it.

**Config keys that are silently inert:** `val_size`, `weight_decay`,
`esm2_hidden_dim`. The merge is `if not hasattr(args, k)` (`train.py:373`), so
any key that is also an argparse flag never applies. The shipped JSON repeats
the argparse defaults, which hides this — editing those three in the JSON does
nothing. Change them on the command line.

`configs/pipeline_defaults.json` contains only a `_documentation` string
pointing at a script that does not exist. Dead file.

## Graphs without a PDB

`core/data_utils.py` normally goes PDB → `parse_pdb` → `compute_edges` → `Data`.
`graph_from_backbone(sequence, ca_coords, plddt_values, ..., edges=None)` skips the
file: it threads a sequence onto Cα coordinates already in memory, and `edges`
accepts a previous `compute_edges` result. The contact graph depends only on
`ca_coords`, so every sequence sharing a fixed backbone can share one edge tensor —
`compute_edges` is an O(N²) Python loop, so reusing it matters. `_build_peptide_graph_data`
takes the same optional `edges`; `pdb_to_graph` and `pdb_to_graph_window` are unchanged.

Valid only for checkpoints that read nothing from a PDB but residue identity and Cα
coordinates (`pdb_continuous=False`, `esm2_residue=False`). `mutation_analysis` uses
it for its fixed-backbone sweep and was verified bit-identical to the file-based path.

## Calibration

Both methods operate on the logit margin σ = logit₁ − logit₀.

- **Platt** (`core/platt.py`) — P(AMP) = sigmoid(aσ + b), logistic regression
  on σ. Written by `train.py` to `gnn_platt_scaling`. Note it is fit on the
  **same validation split used for early stopping**, so it is in-sample with
  respect to model selection.
- **Temperature** (`core/temperature.py`) — P(AMP) = sigmoid(σ/T). Written only
  by `fit_temperature.py`.

`core/calibration.py` is the single resolver every runner uses
(`gnn_minimal/inference.py`, `sequence_analysis`, `length_analysis`,
`mutation_analysis`): `resolve_calibration(model_dir, fitting)` returns
`(method, payload)`, `prob_amp` applies it, `describe_calibration` names it.

**The default cascade is temperature → platt → softmax.** Every runner takes
`--fitting`/`-f` to pin one of the three; an explicitly requested method with
no fitted file is a `FileNotFoundError` rather than a silent downgrade, and
every runner prints the method it resolved.

**Only Platt has an intercept**, so it is the only method whose P = 0.5
crossing moves off σ = 0 — for `gnn_vae_qsar` it sits at σ = 1.340
(`coef` 1.284, `intercept` −1.720) against σ = 0 under T = 0.766. Switching
method therefore changes the **hard call**, not just the probability, and any
metric derived from `pred` moves with it. Temperature and softmax are
monotone in σ with the same threshold, so they produce **identical `pred`**
and differ only in `prob_AMP`. `describe_calibration` prints the crossing for
exactly this reason.

`load_platt_json` and `load_temperature_json` return `None` on a malformed
file, so under the default cascade a corrupt calibration file degrades to the
next method rather than erroring — the printed method is the check.

Reproducing the training split for a temperature fit requires the same `--seed`
and `--val-size`, **and** an unchanged row order in `geometric_features.csv` —
there is no stored split manifest.

## Gotchas

**Stale artifacts are not detected.** `artifacts_ready` (`process_data.py:38`)
is existence-only. Add rows to a CSV whose `generated/` is already populated and
the new peptides are **silently absent** from training and inference. Use
`--force-process` or delete the artifact CSVs. Likewise `need_esm2` only checks
that `esm2_per_residue/` exists as a directory, and that directory is created
unconditionally — so an earlier ESM2-less run leaves an empty dir that passes
the check and fails later with `FileNotFoundError`.

**Inference scores `geometric_features.csv`, not `--input`.** Row ids come from
the generated table's order. Point `--input` at a CSV whose sibling
`generated/` was built for a superset and you silently predict the superset.

**Two dataset classes with different invariants.** `CustomPeptideDataset`
(training, `train.py:141`) does not validate that the ESM2 tensor length
matches the node count; `PeptideGraphDataset` (inference) does. A model can
train on mismatched tensors and fail only later.

**Weights beat metadata.** `resolve_node_layout_for_checkpoint` prefers the
state dict over `model_summary.json` on disagreement, printing a note. If
`model_summary.json` is missing entirely, inference assumes
`gat`/64/3/`mean_max` — wrong for a `gcn` or `egnn` checkpoint.

**Column names are hardcoded in three places.** The 20 geo and 12 QSAR names
live in `train.py:72`/`:109`, `core/extra_feature_scaler.py:18`/`:41`, and
(geo only) `core/data_utils.py:749`. Editing one without the others breaks the
scaler's block partitioning.

**Histidine charge is inconsistent:** `0.1` in `core/data_utils.py:46` and
`pipeline/features.py:45`, but `1` in `pipeline/qsar.py:34` — and `0` / `1`
(pH 7.4 / pH 5) in the two charge rows of `descriptor_table_vae.csv`. Of those,
the QSAR `netCharge` and the VAE rows are live model inputs under the shipped
config, and they disagree by a full charge unit.

**Six of the twelve QSAR columns have a zero training IQR.** `FC`, `LW`, `DP`,
`NK` and `AE` are single-dipeptide percentages from `GetDPComp` and are 0 in
94–98% of rows; `QSO50` is 0 in ~81%. sklearn's `RobustScaler` forces
`scale_ = 1.0` when the IQR is 0, which left those columns in raw units — `LW`
reached `classifier[0]` at 9.09 for a 12-mer while every scaled column sat near
1, and `QSO50` arrived at ~0.005 with a compensating weight norm of 3.5.
`_repair_degenerate_scales` (`core/extra_feature_scaler.py`) now rescales them by
max absolute deviation from the centre, printing the list at fit time. The
percentages are length-coupled (one `LW` is 9.09 in a 12-mer, 2.56 in a 40-mer)
and `length` is not a QSAR column, so the model cannot calibrate for it.

**Pre-schema-3 checkpoints are rebuilt with the old wiring** and print
`Checkpoint predates schema 3; using the raw VAE descriptor table.` That is
correct — they were trained that way — but it means scores from a schema-2 and a
schema-3 checkpoint are not comparable channel by channel, only end to end.

Smaller ones:

- `_load_vae_descriptor_matrix` runs at **import time**; a missing or malformed
  `descriptor_table_vae.csv` breaks `import core.data_utils` outright.
- `tabular_scaler.joblib` is a pickled custom class — loading needs
  `core.extra_feature_scaler` importable at the same module path.
- A degenerate run whose val AUC never exceeds `1e-4` leaves `best_state=None`,
  so last-epoch weights are saved and metrics print as `nan`.
- `compute_edges` is an O(N²) pure-Python loop — the dominant cost on long
  peptides. Its output is otherwise sound: symmetric, duplicate-free, self-loop
  free, and `edge_attr` consistent with the coordinates, verified against an
  independent implementation over 321 structures.
- `edge_attr[:, 1]` is `|i-j| / SEQ_SEPARATION_SCALE` (10.0). It used to divide
  by `n_residues`, which made the same local relationship a different number at
  every length (i,i+2 was 0.22 in a 9-mer and 0.034 in a 59-mer) — harmless
  while only `egnn` read it, a length leak once `gat` does.
- `train.py` materializes every graph into memory before training.
- Filename sanitization maps `/` and `:` to `_`, so ids `a/b` and `a_b` collide.
- `esmfold_v1_local/` is absent; the first run downloads from HuggingFace.
- Dead or unused: `configs/pipeline_defaults.json`,
  `pipeline/esm2_processor.py:main()`, `core/train.py:cross_validate`,
  `core/data_utils.py:create_dataloaders`.

## Self-containment

True for imports — all non-stdlib dependencies are third-party (`torch`,
`torch_geometric`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`,
`biopython` with an external `mkdssp` binary for DSSP, `tqdm`, `transformers`,
`fair-esm`, `propy3`). Data assets are local.

Two soft couplings: `pipeline/resume.py:11` treats `parents[2]` as the repo
root when scanning for sibling `generated/` dirs, so moving this folder changes
which caches are visible; and default output paths are cwd-relative, assuming
the repo root.

Coupling is **one-way inward**: `sequence_analysis` and `length_analysis` both
inject this folder on `sys.path` and import `core.artifacts`, `core.temperature`
and others. Renaming anything in `core/artifacts.py` or `core/temperature.py`
breaks those modules.

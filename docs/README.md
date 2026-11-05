# Documentation index

Navigation docs for this repository. Read `../CLAUDE.md` first — it holds the
rules that govern how code here is written. The one that matters most: this is
research code for running experiments, not production code. Don't add tests or
production scaffolding unless asked.

## Where to start

| If you need to… | Read |
|---|---|
| Understand how the repo fits together | `architecture.md` |
| Add a module, or change an input/output shape | `contracts.md` |
| Train or run the GNN | `gnn-minimal.md` |
| Score sequences, windows, or length variants | `analysis-modules.md` |
| Find a dataset, checkpoint or result | `data-and-artifacts.md` |
| Work in a notebook | `notebooks.md` |
| Recover something from the old codebase | `legacy.md` |

## The pages

- **`architecture.md`** — module map, the data-flow diagram, the verified
  import graph, and the two known coupling defects.
- **`contracts.md`** — the two shared contracts: the canonical sequence CSV
  from `data_normalizer/`, and the run directory + manifest from
  `analysis_outputs/`. The most important page if you are writing new code.
- **`gnn-minimal.md`** — the self-contained GNN pipeline: entry points,
  artifacts, model directory layout, calibration, and its gotchas.
- **`analysis-modules.md`** — `sequence_analysis/`,
  `window_sequence_analysis/`, `length_analysis/`: invocation, flags, output
  schemas, where their conventions diverge, and the duplication inventory.
- **`data-and-artifacts.md`** — `data/`, `checkpoints/`, `results/`,
  `descriptors/`, `tests/`, and the stray root files.
- **`notebooks.md`** — every notebook, one line each, and which still depend on
  legacy code.
- **`legacy.md`** — `sequence_to_svm_minimal/`: what lived there, what has no
  root replacement yet, and which of its docs are still worth reading.

## Shape of the pipeline

```
raw input (csv / fasta / txt)
        │  data_normalizer/
        ▼
canonical CSV:  id, sequence, [extras]
        │
        ├─► gnn_minimal/              → generated/ → model dir → predictions.csv
        ├─► sequence_analysis/        ┐
        ├─► window_sequence_analysis/ ├─► analysis_outputs → results/runs/<timestamp>/
        └─► length_analysis/          ┘        ├── <result>.csv
                                               └── run_manifest.json
```

Each root folder is an independent module. They compose through these file
shapes, not through Python imports — `analysis_outputs/` and
`data_normalizer/shared/` being the sanctioned exceptions, because they *are*
the contract.

## Things that will bite you

Collected from the module pages; each is documented in detail there.

- **`gnn_minimal` does not detect stale artifacts.** Adding rows to a CSV with
  an existing `generated/` dir silently omits the new peptides. Use
  `--force-process`.
- **GNN inference scores `geometric_features.csv`, not `--input`.** A
  `generated/` dir built for a superset silently predicts the superset.
- **Prediction encoding is not uniform:** `+1/-1` with `sigma`/`p_amp` in
  `sequence_analysis` and `window_sequence_analysis`, but `0/1` with
  `logit_margin`/`prob_AMP` in `length_analysis`.
- **`length_analysis` GNN scoring depends on deprecated code** — a private
  function in a legacy CLI script, loaded via `sys.path` injection.
- **Three config keys in `gnn_final_train.json` are silently inert**
  (`val_size`, `weight_decay`, `esm2_hidden_dim`) because argparse shadows them.
- **`--profile-aggregation max` makes `p_amp_mean` report the max**, and
  `--workers > 1` makes profile row order non-deterministic.
- **Two timestamp conventions coexist:** `runs/%Y-%m-%d_%H-%M` (UTC, from
  `analysis_outputs`) and `run_%Y%m%d_%H%M%S` (local, GNN training).
- **`descriptors/descripGen_12_py3.py` cannot run as checked out** — `propy`
  is missing. `gnn_minimal/pipeline/qsar.py` is the live QSAR path.
- **Four QSAR descriptors fail silently to 0.0** if ProPy throws, because
  `safe_descriptor_call` swallows every exception.

## Keeping these docs honest

These pages describe real line numbers and schemas, so they drift. If you
change an output schema, a CLI flag, or a module boundary, update the
corresponding page in the same change — that is in scope, not a drive-by edit.
If you find a statement here that is wrong, fix it rather than working around
it; a doc that has lied once stops being read.

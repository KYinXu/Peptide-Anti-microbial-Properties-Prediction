# Architecture

## Shape of the repository

Each root-level folder is an independent module. Modules are composed by the
files they read and write, not by importing each other. Two modules are shared
contracts rather than pipeline stages:

- `data_normalizer/` — turns any raw input format into the canonical
  sequence CSV that every downstream module accepts.
- `analysis_outputs/` — lays out timestamped run directories and writes the
  provenance manifest that accompanies every result CSV.

Everything else is a consumer of those two contracts:

| Module | Role |
|---|---|
| `data_normalizer/` | raw CSV / FASTA / TXT → canonical `id,sequence[,extras]` CSV |
| `analysis_outputs/` | run directory + manifest + run-bound CSV writer |
| `descriptors/` | QSAR descriptor generation and the AAindex data tables |
| `gnn_minimal/` | self-contained graph-neural-network train / infer pipeline |
| `sequence_analysis/` | whole-sequence prediction (SVM and GNN) |
| `window_sequence_analysis/` | sliding-window prediction and per-residue profiles |
| `length_analysis/` | length-variant generation and scoring |
| `tests/` | legacy `unittest` files over the shared contracts; not a standard to extend |
| `notebooks/` | exploratory and figure-producing analysis |
| `sequence_to_svm_minimal/` | **legacy**, reference only |

## Data flow

```
raw input (csv / fasta / txt)
        │
        ▼
data_normalizer/            →  canonical CSV:  id, sequence, [extra columns]
        │
        ├──────────────► gnn_minimal/            → generated/ artifacts → model dir → predictions.csv
        ├──────────────► sequence_analysis/      ┐
        ├──────────────► window_sequence_analysis/├─ via analysis_outputs → runs/<timestamp>/
        └──────────────► length_analysis/        ┘        ├── <result>.csv   (run_id-stamped rows)
                                                          └── run_manifest.json
```

## Import graph

Verified by grep over all non-legacy `.py` files. Only these cross-module
imports exist:

| From | Imports | Legitimacy |
|---|---|---|
| `length_analysis/data_loader.py`, `length_analysis/provenance.py`, `window_sequence_analysis/data_loader.py` | `data_normalizer.shared.records.normalize_sequence` | contract module — expected |
| `length_analysis/run_length_analysis.py`, `length_analysis/scoring/results_io.py`, `sequence_analysis/record_predictions.py`, `window_sequence_analysis/run_svm_prediction.py`, `window_sequence_analysis/run_raw_prediction.py`, `window_sequence_analysis/sliding_windows/{results_io,raw_results_io}.py` | `analysis_outputs` | contract module — expected |
| `tests/*` | all analysis modules | tests may reach anywhere |
| **`window_sequence_analysis/models/svm.py:12`** | **`sequence_analysis.utils.descriptor_ablation`** | **known violation** |

Nothing outside `gnn_minimal/` imports `gnn_minimal`, and nothing imports
`descriptors/` as a package.

No root module imports `sequence_to_svm_minimal`, statically or at runtime.

### The former live dependency on legacy code (resolved)

`length_analysis/models/gnn.py` used to inject both `gnn_minimal/` and
`sequence_to_svm_minimal/` onto `sys.path` with the legacy folder first, take
most of its GNN stack from the legacy tree, and score through
`compare_model_predictions._run_gnn_predictions` — a private function in a
deprecated CLI script. It now loads the checkpoint and scores through
`gnn_minimal` only, the same path `mutation_analysis` and `sequence_analysis`
use, and puts only `gnn_minimal/` on `sys.path`.

What forced the change: the legacy loader
(`sequence_to_svm_minimal/gnn/extra_feature_scaler.py:173`) rejects the
`tabular_scaler.joblib` shipped with `checkpoints/gnn/gnn_vae_qsar`, because
that file is pickled as `core.extra_feature_scaler.ExtraFeatureRobustScaler`
from `gnn_minimal`. The legacy tree reloads the scaler a second time inside
`gnn/data_utils.py:730`, so no import-level patch in `length_analysis` could
have worked — and editing legacy is out of bounds. Scoring was broken for
**both** length modes until the port.

Parent probabilities from the ported scorer match the `mutation_analysis` run
on the same checkpoint to 7 significant figures (differences ~1e-7, from
re-folding structures), which is the check to repeat if this path is touched
again.

One import-time side effect still worth knowing:
`sequence_analysis/models/svm.py:17` adds the repo root and the legacy folder to
`sys.path` on import, for no current consumer. Importing that adapter changes
global import resolution.

### The one known violation

`window_sequence_analysis/models/svm.py` imports
`null_descriptor_values` and `parse_null_descriptor_names` from
`sequence_analysis.utils.descriptor_ablation`. This couples two sibling
analysis modules. Do not cite it as precedent. When it next needs touching, the
descriptor-ablation helpers should move to a shared location (alongside the
other contracts) or be duplicated into each module, whichever keeps the modules
independently runnable.

## The two shared contracts

### Canonical sequence CSV — `data_normalizer/shared/records.py`

Required columns are exactly `id` and `sequence`
(`records.py:9`). A record may carry arbitrary string `extras`, which
`write_normalized_csv` emits as additional columns sorted by name after the two
required ones (`shared/writers.py:14`). Sequences are whitespace-stripped and
upper-cased, and an empty sequence is an error (`records.py:32`).

Any module that reads sequences should accept this shape, and should accept the
`extras` columns being present or absent.

### Run directories — `analysis_outputs/`

`execute_csv_run(spec, write_csv)` (`runs.py:62`) is the entry point. It

1. creates `<output_root>/runs/<YYYY-MM-DD_HH-MM>/`, suffixing `_02`, `_03`, …
   on collision (`runs.py:93`),
2. writes `run_manifest.json` with status `running`,
3. calls your `write_csv(run) -> row_count`,
4. marks the manifest `completed` with the output file's sha256, size and row
   count — or `failed` with the exception type and message, then re-raises.

The manifest records the runner name, model name and model file hashes, the
config plus a `config_sha256`, every input file's hash, the exact `sys.argv`,
and the git commit with a dirty flag (`runs.py:109`). `RunConfiguration`
(`config.py:11`) is the standard config payload: an `output_mode` string, the
`effective` settings, and the raw `arguments`.

`RunCsvWriter` (`csv_writer.py:17`) is a context manager that prepends a
`run_id` column to every row, so a result row can always be traced back to its
manifest.

Follow this for any new analysis that produces a CSV. Do not invent a second
output layout.

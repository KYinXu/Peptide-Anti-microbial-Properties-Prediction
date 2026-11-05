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

No root module has a static `import sequence_to_svm_minimal`, but one reaches
the legacy tree at runtime — see below.

### The live dependency on legacy code

`length_analysis/models/gnn.py` injects both `gnn_minimal/` and
`sequence_to_svm_minimal/` onto `sys.path` — with **the legacy folder first**
(`gnn.py:213`). It then takes most of its GNN stack from the legacy tree
(`gnn.models`, `gnn.checkpoint_meta`, `gnn.extra_feature_scaler`; calibration
now comes from `gnn_minimal/core/calibration.py` instead of `gnn.platt`), borrows only `process_data.ensure_processed_data`
from `gnn_minimal`, and finally does:

```python
sys.path.insert(0, str(SVM_MINIMAL_ROOT / "scripts" / "data_evaluation"))
import compare_model_predictions as inference
... inference._run_gnn_predictions(...)
```

(`gnn.py:124-128`.) So `length_analysis` GNN scoring depends on a **private
function in a deprecated CLI script**, plus four legacy modules. This is the
most fragile edge in the repository: it violates module independence, depends
on legacy code the rules say not to maintain, reaches past a `_`-private boundary,
and the `sys.path` ordering means any module name present in both trees
resolves to the legacy copy.

`sequence_analysis` does this properly for comparison — it uses
`gnn_minimal/core` directly. So if the legacy folder were deleted,
`length_analysis` GNN scoring would break and `sequence_analysis` would not.

Treat it as the known debt it is. If GNN scoring in `length_analysis` breaks,
this is why. The fix is to route it through `gnn_minimal`, the supported
self-contained path — not to patch the legacy script.

One more import-time side effect worth knowing:
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

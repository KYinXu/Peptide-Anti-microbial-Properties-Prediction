# `sequence_to_svm_minimal/` — legacy

Earlier generation of this project, kept for reference. Per `CLAUDE.md`
the legacy-reference rule: **do not extend it, fix it, or import from it.** When something
here is still needed, reimplement it as a root-level module.

It is still git-tracked and large (roughly 7,000 tracked files, most of them
data). Read from it to understand history or to recover an algorithm; do not
add to it.

## What lived where

| Directory | What it was | Status at root |
|---|---|---|
| `peptide_pipeline/` | Step/runner orchestration framework (normalize, esmfold, geometric, qsar, esm2, cluster, window_aggregate, train, comparison) with manifest/path contracts | Stages reimplemented in `gnn_minimal/process_data.py`; the orchestration framework, cluster step and comparison step are **not** |
| `proteome_candidate_generator/` | Proteome FASTA → pepsickle cleavage → candidate enumeration → Pane/PDDP scoring | **Not reimplemented** |
| `nn_pipeline/` | MLP classifier on geometric features, cluster-based k-fold CV | **Not reimplemented** (root has GNN + SVM only) |
| `gnn/` | GNN core: models, data utils, train, platt, checkpoint meta, scaler | Reimplemented as `gnn_minimal/core/*`, which adds `temperature.py` |
| `features/` | `geometric_features.py` | Carried over near-verbatim as `gnn_minimal/pipeline/features.py` |
| `models/` | ESMFold/ESM2 runners, HF cache fixes; MIC regression stubs | ESMFold/ESM2 reimplemented in `gnn_minimal/pipeline/esm2_processor.py`; MIC was never implemented |
| `scripts/` | Driver layer: training, comparison, ablation, y-randomization, SVM predict, `data_evaluation/` | Inference reimplemented across the root analysis modules; **training and experiment harnesses are not** |
| `configs/` | JSON presets + `load_config.py`, incl. windowed stride presets | No root preset system; root modules use per-module `run_options.py` |
| `predictionsParameters/` | Original-paper artifacts: `predictSVC.py`, `svc.pkl`, z-score table, descriptor CSVs | Superseded by `checkpoints/svm*`, but this is the **provenance** of the current SVM |
| `structure/`, `cli/` | TODO stubs only, never implemented | Nothing to port |
| `experiments/` | Two ad-hoc run snapshots — data, not code | Nothing to port |
| `utils/`, `tests/` | Caching/eval helpers; 4 tests (geometric features, windowing, pipeline, candidate generator) | Mostly folded into root modules |
| `data/`, `checkpoints/`, `results/` | The historical corpus and run outputs | Root equivalents are separate and newer, but several notebooks still read legacy `data/` |

## Capabilities with no root replacement

Ranked by how likely you are to need them:

1. **Proteome candidate generation** — pepsickle cleavage, candidate
   enumeration/filtering, Pane PDDP scoring. Still used by
   `notebooks/candidate_list_comparison.ipynb`.
2. **Training anything except a GNN** — SVM training (which produced the root
   `checkpoints/svm_null*`) and the NN/MLP pipeline. Root code only *loads*
   SVMs; only `gnn_minimal/train.py` trains.
3. **Experiment harnesses** — GNN feature ablation, y-randomization, feature
   fusion, GNN-vs-MLP comparison.
4. **Pipeline orchestration and sequence clustering.**
5. **Two-model comparison output** (`model_comparison_latest.csv` /
   `model_comparison_windowed_latest.csv`) — the schema most notebooks read.
6. **JSON config presets**, including windowed stride presets.

MIC regression appears in the legacy plan documents but only ever existed as
stubs. Treat it as unbuilt, not as lost.

## Legacy docs worth reading

Useful:

- `DATA_PROCESSING.md` — the FASTA/TXT → ESMFold → geometric → QSAR-12 → ESM-2
  `generated/` layout. The most useful legacy doc; the on-disk shapes root
  modules consume descend from it.
- `README.md` — accurate overview of the AMP-vs-decoy ESMFold+GNN system.
- `KEY_FINDINGS.txt` — factual characterization of the original SVM
  (484 samples, 12 descriptors). Cited in paper comparisons.
- `ENVIRONMENT_GUIDE.md` — the two-conda-env setup (a Python 3.7 /
  scikit-learn 0.19.2 environment for the legacy pickle, plus a modern one).
  Still load-bearing: the root `sklearn_pickle_compat.py` shims exist because
  of this.
- `proteome_candidate_generator/README.md` and `PEPSICKLE_PIPELINE.md` — the
  only documentation of a capability that exists nowhere else.

Stale or misleading — don't rely on these:

- `ESMFOLD_INTEGRATION_PROJECT_PLAN.md`, `QUICK_START_GUIDE.md`,
  `README_FOR_NEXT_AGENT.md` — plans and handoff notes for the MIC-prediction
  project that was never built. The stubs in `cli/`, `structure/` and
  `models/train_mic.py` are its remains.
- `CODEBASE_ANALYSIS_SUMMARY.md`, `EXECUTIVE_SUMMARY.md` — overlap
  `KEY_FINDINGS.txt`, describe an SVM-only system.
- `SETUP.md`, `STRUCTURE.md`, `WORKFLOW_EXAMPLE.md` — describe the legacy tree.

## Live references from root code

There is no static `import sequence_to_svm_minimal` anywhere in root code, but
two real references exist:

- **`length_analysis/models/gnn.py`** imports the private
  `_run_gnn_predictions` from the legacy
  `scripts/data_evaluation/compare_model_predictions.py` via `sys.path`
  injection. This is a genuine runtime dependency on deprecated code — see
  `architecture.md` for details and the intended fix.
- **`test_overlap_speed.py`** (loose root script) hardcodes paths to legacy
  proteome CSVs.

## `sprints/inference_module_refactor.md`

Gitignored but present. It plans to extract shared SVM/GNN inference out of the
legacy CLI scripts into a new `inference/` package, leaving the CLIs as thin
adapters. Its diagnosis is worth reading — it names the conflated
responsibilities, the cross-script private imports, the quadratic
`.index()`-based alignment, and the unbounded memory use.

**Caveat:** the plan targets paths *inside* `sequence_to_svm_minimal/`, which
the legacy-reference rule forbids editing. The plan either predates the deprecation or
needs re-targeting to a root-level module. Do not execute it as written; treat
it as a problem statement, and note that its heavy test matrix also conflicts
with the research-code rule.

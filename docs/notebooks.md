# `notebooks/`

Exploratory and figure-producing analysis. Notebooks consume the analysis
modules almost entirely **through CSV file contracts**, not Python imports —
which is why module output schemas are the thing to keep stable.

## How notebooks depend on modules

Only one root module is imported by any notebook:
`window_sequence_analysis.models.svm.load_svm`, in three files
(`windowed_DNGC_analysis`, `single_cluster_analysis`,
`final_presentation_single_cluster_display`).

No notebook imports `sequence_analysis`, `length_analysis`, `gnn_minimal`,
`analysis_outputs`, `data_normalizer` or `descriptors`.

**Five notebooks still import legacy Python** (`gnn.*`, `peptide_pipeline.*`,
`configs.*`, `proteome_candidate_generator.*`) and are the migration backlog:
`gnn_graph_connectivity_train_val`, `gnn_variation_test`,
`svm_pipeline_sanity`, `window_exploration`, `candidate_list_comparison`.

## Root

| Notebook | What it does |
|---|---|
| `H2A_analysis.ipynb` | Per-window P(AMP) and sigma along H2A variant parents |
| `candidate_list_comparison.ipynb` | Generated candidates vs the paper's PDDP Table 6; exact/near/substring matching. **Imports legacy `PaneScorer`** |
| `comparison_exploration.ipynb` | Ad-hoc score-distribution exploration over comparison CSVs |
| `gnn_graph_connectivity_train_val.ipynb` | Cα-proximity graph statistics for train vs val splits. **Legacy imports** |
| `gnn_variation_test.ipynb` | Runs GNN training 5× at identical seed to measure nondeterminism. **Legacy imports** |
| `graph_connectivity.ipynb` | **Empty (0 bytes)** — delete candidate |
| `overlapping_score_distributions.ipynb` | Overlays normalized SVM hyperplane-distance and GNN logit-margin histograms |
| `skin_amp_discovery.ipynb` | Skin sequences + SVM output; high-confidence subset, breakdown by expression cluster |
| `svm_descriptor_analysis.ipynb` | Standardized descriptor distributions; sigma vs the four GRAR740104 sequence-order descriptors |
| `svm_pipeline_sanity.ipynb` | Legacy `.pkl` vs `StandardScaler+SVC` equivalence check. **Legacy imports** |
| `svm_window_heatmaps.ipynb` | Per-window distances → per-residue scores → colored sigma activation bars |
| `training_data.ipynb` | Training-set composition stats (class balance, length distribution) |
| `window_exploration.ipynb` | Per-window P(AMP) along parent sequences. **Legacy imports** |
| `window_profile_analysis.ipynb` | Reads the latest `window_sequence_analysis` run; mean-sigma and P(AMP) profiles. The model consumer-by-file-contract |

## `pddps/`

| Notebook | What it does |
|---|---|
| `proteome_prediction_analysis.ipynb` | Proteome prediction vs labeled sequences; KDE score distributions |
| `final_figures_proteome_prediction.ipynb` | Publication figures over the 465k proteome, 50k MAPP and 1k confirmed sets, GNN+QSAR, with the 10 paper candidates marked |
| `robustness_analysis.ipynb` | Reads the latest `mutation_analysis` run; Δσ / ΔP(AMP) versus mutation level per scheme and AMP call retention curves |

## `skin_clusters/`

| Notebook | What it does |
|---|---|
| `windowed_DNGC_analysis.ipynb` | Main skin-sample windowed DNGC analysis. **Imports `load_svm`** |
| `single_cluster_analysis.ipynb` | One expression cluster: mean σNGC aggregation, DNGC = A/S, metric histograms. **Imports `load_svm`** |
| `final_presentation_single_cluster_display.ipynb` | Presentation-styled single-cluster display. **Imports `load_svm`** |
| `cluster_intensity_measurement.ipynb` | Window profiles grouped by expression cluster; DNGC, prevalence, S, potency, coverage; matches skin intensity from `ms_tissue.tsv` |
| `final_presentation_figures.ipynb` | Final presentation figure set |
| `fragment_windows.ipynb` | Top-20 fragment analysis over raw 15–40-residue windows; highest-sigma window per fragment with curve fits |
| `top_fragment_analysis.ipynb` | Exports fragments from the top 10 clusters with adjacent flanks (WIP) |
| `top_sequence_selection.ipynb` | Selects and ranks top sequences for downstream use |

## Conventions

Notebooks are research artifacts and fall squarely under the research-code rule — they are
not expected to be clean or tested. Two things are still worth holding to:

- **Read module output through its documented schema**, not by positional
  column index, so a schema addition doesn't break old notebooks.
- **When logic in a notebook turns out to be reusable**, move it into the
  relevant root module rather than copying the cell between notebooks. The
  `load_svm` import in the `skin_clusters` notebooks is the pattern to follow.

If you find yourself needing a legacy import in a new notebook, that capability
belongs in a root module — see `legacy.md` for what has not been reimplemented.

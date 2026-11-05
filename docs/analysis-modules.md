# Analysis modules

`sequence_analysis/`, `window_sequence_analysis/`, `length_analysis/` — three
runner packages that read the canonical CSV, score it, and write a run
directory via `analysis_outputs/`.

## Shared skeleton

All three have the same shape:

```
<module>/__main__.py       one-line dispatch into a runner's main()
<module>/data_loader.py    CSV reader (CSV only — no FASTA/TXT)
<module>/models/           model adapters (svm.py and/or gnn.py)
<module>/run_options.py    CLI arg groups + config dataclass (absent in sequence_analysis)
<module>/.../results_io.py RunCsvWriter subclass with a fixed column list
<module>/run_*.py          argparse entry, builds RunSpec, calls execute_csv_run
```

Every runner re-inserts the repo root on `sys.path` when `__package__` is
empty, so `python path/to/run_x.py` works as well as `python -m <module>`.

All three read CSV only, requiring `{id, sequence}`. Convert other formats with
`data_normalizer/` first.

## `sequence_analysis/` — whole-sequence scoring (SVM or GNN)

```bash
python -m sequence_analysis -i data/<set>/sequences.csv
python -m sequence_analysis -i … --model gnn
python -m sequence_analysis.print_predictions -i …   # stdout only, no file
```

`python -m sequence_analysis` runs `record_predictions.main()`.
`print_predictions` is a stdout-only variant, not reachable from `__main__`.

| Flag | Default |
|---|---|
| `--input`/`-i` | required |
| `--model` | `svm` (`svm`\|`gnn`) |
| `--gnn-model` | `checkpoints/gnn/gnn_vae_qsar` |
| `--checkpoint-dir` | **None** — resolved per model at runtime |
| `--svm-pkl` / `--zscores` | sole matching file in the checkpoint dir |
| `--fitting`/`-f` | auto: temperature → platt → softmax |
| `--batch-size` | 32 |
| `--device` | auto (free-form string) |
| `--output`/`-o` | `<input dir>/results/sequence_predictions.csv` |
| `--include-descriptors` | off |
| `--null-descriptors` | `[]` |
| `--force-process` | off |

`--fitting` is rejected for `svm`; `--null-descriptors` is rejected for `gnn`.

**Output** `<input dir>/results/runs/<ts>/sequence_predictions.csv`:
`run_id, id, sequence,` *input extras*, *descriptor columns if requested*,
`prediction, sigma, p_amp`. `prediction` is **+1 / -1**. Input extras that
collide with a reserved name are silently dropped.

With `--model gnn`, feature artifacts are written to `generated/` **next to the
input CSV**, so repeated runs reuse (and pollute) the input directory.

## `window_sequence_analysis/` — sliding windows (SVM only)

```bash
python -m window_sequence_analysis -i …                     # profile mode
python -m window_sequence_analysis.run_raw_prediction -i …  # raw per-window
```

**`python -m window_sequence_analysis` silently means profile mode.** Raw mode
has no `-m <module>` shortcut.

Shared flags: `--input`/`-i`, `--checkpoint-dir` (eager default
`checkpoints/svm_no_class_weight`), `--svm-pkl`, `--zscores`,
`--null-descriptors`, `--output`/`-o`, `--quiet`, plus the window group:
`--window-min-len` 10, `--window-max-len` 35, `--stride` 1,
`--batch-starts` 64.

Profile mode adds `--profile-aggregation {mean,max,both}` (default `both`),
`--precision` 6, and `--workers` (default `os.cpu_count()`). Raw mode adds
`--model {svm}` and is strictly serial.

**Profile output** `window_sequence_profiles.csv` — one row per *sequence*:
`run_id, id,` *label columns (prefixed `label_` on collision)*, window
parameters, `window_count`, scalar summaries, best-window columns, and four
`;`-delimited per-residue profile strings. `profile_index_base` is 1.

**Raw output** `raw_window_predictions.csv` — one row per *window*:
`run_id, id, window_start_0based, window_end_0based_exclusive, prediction,
sigma, p_amp`. **Raw mode drops all input extras.**

Gotchas specific to this module:

- With `--profile-aggregation max`, the `p_amp_mean` and
  `hyperplane_distance_mean` scalars are computed from the **max** profile, and
  the `*_mean_profile` columns come out empty. Always check
  `profile_aggregation` before reading `p_amp_mean`.
- With `--workers > 1`, rows arrive via `imap_unordered` — **output row order
  does not match input order**.
- Sequences shorter than `--window-min-len` produce no rows at all.
- `U` is substituted to `C` before scoring; other non-standard residues are
  rejected (`models/svm.py:164`). `sequence_analysis` has no such
  normalization and will raise on `X`/`B`/`Z`.

## `length_analysis/` — flank extension sweep (GNN only)

```bash
python -m length_analysis -i subset.csv --candidates-table candidates.csv
```

`--model` exists but only accepts `gnn`.

| Flag | Default |
|---|---|
| `--input`/`-i` | required |
| `--candidates-table` | **required** (`.csv` or `.parquet`) |
| `--proteome-fasta` | the bundled UniProt human proteome |
| `--checkpoint-dir` | `checkpoints/gnn/gnn_vae_qsar` |
| `--gnn-checkpoint` | `gnn_model.pt` in the checkpoint dir |
| `--fitting`/`-f` | auto: temperature → platt → softmax |
| `--max-flank-fraction` | 0.5 |
| `--stride` | 1 |
| `--batch-size` | 32 |
| `--device` | auto (`cuda`\|`cpu`, choices-restricted) |
| `--output`/`-o` | `<input dir>/results/length_predictions.csv` |

Three inputs must agree. The candidates table needs
`peptide_id, sequence, source_protein_id, start, end`; the subset sequence must
equal **both** the candidates-table sequence and `protein[start:end]`. Any
drift aborts the whole run rather than skipping the record
(`provenance.py:111`).

Variants are the cartesian product of left and right flank extensions, capped
at `floor(max_flank_fraction * core_length)`, ids `<parent>_L<n>_R<m>`. Both
loops include 0, so the unextended parent is always emitted as `_L0_R0`.

**Output** `length_predictions.csv`: `run_id, variant_id, parent_id,
source_protein_id, start, end, left_added, right_added, core_length,
variant_length, sequence, pred, prob_AMP, confidence, logit_AMP, logit_nonAMP,
logit_margin, score_z`, then subset extras. Intermediate artifacts go under
`runs/<ts>/workspace/`, clean per run — but structures are recomputed every
time.

`score_z` is z-scored over **the current run's variant batch only**, so it is
not comparable across runs.

## `mutation_analysis/` — substitution robustness sweep (GNN only)

```bash
python -m mutation_analysis -i subset.csv
python -m mutation_analysis -i subset.csv --refold-sample 25
```

Input is the canonical CSV (`id, sequence`) — no candidates table or proteome,
unlike `length_analysis`.

| Flag | Default |
|---|---|
| `--input`/`-i` | required |
| `--checkpoint-dir` | `checkpoints/gnn/gnn_vae_qsar` |
| `--p-levels` | `0.10 0.15 … 0.50` (9 levels) |
| `--replicates` | 500 |
| `--schemes` | all of `physicochemical blosum62 uniform` |
| `--seed` | 0 |
| `--blosum-temperature` | 1.0 |
| `--refold-sample` | 0 (off) |
| `--parent-structures` | auto: `<input dir>/generated/structures` if it matches, else fold |
| `--fitting`/`-f` | auto: temperature → platt → softmax |
| `--write-variants` | off (only the collapses are kept) |
| `--batch-size` | 32 |
| `--device` | auto (`cuda`\|`cpu`) |
| `--output`/`-o` | `<input dir>/results/mutation_robustness.csv` |

`k = round(p * L)` substitutions per variant, **half-up** — Python's `round()`
is half-to-even and would give k=2 for L=10, p=0.25. Defaults score
`9 × 3 × 500 + 1 = 13,501` variants per parent, but collapse them to **3**
stored rows (one per scheme).

Three schemes (`mutants/rules.py`): `physicochemical` uses the Figure 5 rule
tables (charge, aromatic, polar, aliphatic classes; `A → G`); `blosum62`
samples a softmax over the BLOSUM62 row **with the parent residue excluded and
renormalized**; `uniform` samples the other 19 evenly. Under `physicochemical`,
**P, C and G are never substituted** — P/C by rule 6, G because rule 5 lists
only `A → G`. `k` is capped at the eligible-position count, so `k_requested`
and `k_applied` can differ; the other two schemes can substitute anything.

**Backbone.** Each mutant is scored on its parent's Cα frame with the residue
identities replaced. This is **exact, not approximate**, for checkpoints with
`pdb_continuous=False` and `esm2_residue=False` — the only things read from a
PDB are residue identity and Cα coordinates, so side chains and pLDDT never
matter. Verified: the `_parent` row reproduces `gnn_minimal/inference.py` to
6+ decimal places. The scorer **rejects** a checkpoint that wants per-residue
ESM2, since that assumption would no longer hold.

Mutant graphs are built **in memory**, never through a PDB file
(`MutationScorer.load_backbone` / `score_on_backbone`, on
`core.data_utils.graph_from_backbone`). Because the backbone is fixed, the Cα
coordinates and the 8 Å contact graph are identical for every mutant of a
parent, so `compute_edges` runs **once per parent** and its result is shared by
all 13,501 graphs. Previously each mutant was written to a PDB, re-parsed with
Biopython and re-edged; that was ~57% of CPU time spent recomputing identical
values. Verified bit-identical to the old file-based path across 120 mutants on
all of `prob_AMP`, `logit_AMP`, `logit_nonAMP`, `logit_margin` and `pred`.

`mutants/backbone.py` (`read_ca_lines`, `write_mutant_pdb`) implemented the old
path and now has **no callers**; it is kept as the reference the in-memory path
was validated against.

The cost argument: ESMFold measures ~0.65 s/fold here, so refolding every
variant is ~2.4 h per peptide (~102 days for 1,000). Fixed backbone is one fold
per parent — or none, see below. `--refold-sample N` genuinely folds N mutants
per (parent, scheme, p) cell and scores each against **its own** structure (the
one remaining PDB-based path, `MutationScorer.score`), emitting them as extra
rows with `backbone=refold` to quantify what the assumption costs.

**Parent structures.** `resolve_parent_structures` reuses a prepared directory
of `<parent id>.pdb` instead of folding, which is what keeps ESMFold out of the
process: measured peak RSS drops from **10.7 GB to 2.0 GB**. A candidate is
accepted only when every parent has a PDB whose parsed residues equal its input
sequence, so a stale structure can never be scored against a different peptide.
`--parent-structures DIR` is checked strictly and **errors** on any mismatch;
the auto-detected `<input dir>/generated/structures` falls back to folding and
prints why. The directory actually used is recorded in the manifest as
`config.effective.parent_structures_used`.

**Measured cost** (checkpoint `gnn_vae_qsar`, L≈20, defaults, structures
reused): **~7.8 s per parent** marginal plus ~5 s startup, i.e. **~2.3 h for
1,074 peptides**. Before the in-memory path and structure reuse the same run
was ~14 h, most of it writing and re-parsing 14.5M mutant PDBs — on a Windows
`/mnt/c` mount that file churn alone cost ~1.7 ms per mutant versus 0.017 ms on
native ext4. With the PDB round-trip gone the run directory holds only the
output CSVs, so the output filesystem no longer affects runtime. Remaining
per-mutant CPU is dominated by `compute_qsar12`. Parents are independent, so
sharding the input CSV across processes scales nearly linearly; at ~2 GB per
worker that is now memory-feasible.

**Outputs.** Variants are collapsed per parent while that parent's batch is
still in memory, so nothing re-reads a large file and peak memory is one
parent's worth of predictions (`scoring/aggregate.py`).

`mutation_robustness.csv` (primary, one row per parent × scheme × backbone):
`run_id, parent_id, scheme, backbone, sequence, length, parent_pred,
parent_prob_AMP, parent_logit_margin, n_variants, n_p_levels,
mean_k_requested, mean_k_applied, retention_auc, retention_first_p,
retention_last_p, p50_retention, mean_prob_AMP, mean_delta_prob_AMP,
margin_auc, margin_slope`, then subset extras.

- `retention(p)` is the fraction of that cell's variants still called AMP
  (`pred == 1`, i.e. calibrated P(AMP) ≥ 0.5).
- `retention_auc` is the **trapezoidal integral of `retention(p)` over the
  tested p range, divided by its span** — 1.0 = every variant survives every
  p, 0.0 = none. This is the headline ranking metric.
- `p50_retention` is the lowest p where `retention` crosses 0.5, linearly
  interpolated, anchored at `(p=0, parent_pred)`. Blank when it never crosses
  within the tested range, and blank for a parent the model does not call AMP
  (the curve never starts above 0.5, so the quantity is undefined).
- `margin_slope` is the OLS slope of `logit_margin` on p over all variants of
  that scheme. Unlike retention it does not saturate, so it still separates
  peptides whose variants are all on one side of the threshold.
- `mean_k_requested` vs `mean_k_applied` exposes the eligibility confound: a
  peptide rich in P/C has few `physicochemical`-eligible positions, so `k` is
  capped and it looks resistant because it was barely mutated. Compare the two
  before trusting a high `retention_auc`.

`mutation_curves.csv` (one row per parent × scheme × backbone × p, ~27 per
parent): `run_id, parent_id, scheme, backbone, p, n_variants, k_requested,
k_applied, retention, mean_prob_AMP, sd_prob_AMP, mean_delta_prob_AMP,
mean_logit_margin, sd_logit_margin`. This is the degradation curve Figure 5
plots; join to the summary or the subset on `parent_id`. Standard deviations
are population (`ddof=0`).

`mutation_predictions.csv` (only with `--write-variants`): the full per-variant
dump described above — `run_id, mutant_id, parent_id, scheme, p, k_requested,
k_applied, replicate, mutations, sequence, length, backbone, pred, prob_AMP,
logit_AMP, logit_nonAMP, logit_margin, delta_prob_AMP, delta_logit_margin`,
then subset extras. `mutations` is `;`-delimited and **1-based** (`K7R;L12V`).
`pred` is 0/1 and probabilities use the resolved calibration (temperature by
default; the manifest records which under `config.effective.fitting`). At defaults this is ~13.5k rows per parent
(~13.5M for 1,000 peptides, several GB), which is why it is off by default;
`sequence` and `mutations` make every variant reconstructable from its row.

Each parent is still scored unmutated (`scheme=parent`, `p=0`) and every delta
is measured against it, but that row is now a reference used by the collapse
rather than a stored output row (it appears only under `--write-variants`).

`--refold-sample` rows aggregate **separately**, under `backbone=refold`, so
their much smaller per-cell n never mixes into the fixed-backbone statistics.

The run ends by printing the top 10 AMP-called parents, ranked by their
**weakest** scheme's `retention_auc` — resistant under all three scales, not
just the gentle one. The table shows **every** scheme's `retention_auc` beside
its `mean_k_applied`, plus the parent's length and scheme-independent
`mean_k_requested`, so an inflated score is visible in place; a cell whose
applied k fell below `k_req` is marked `*`. The same ranking in pandas:

```python
fixed = summary[(summary.backbone == "fixed") & (summary.parent_pred == 1)]
fixed.groupby("parent_id").retention_auc.min().sort_values(ascending=False)
```

Caveats:

- Fixed backbone freezes the contact graph, so all variation comes from the
  node features and the 12 QSAR descriptors. The two unconstrained schemes can
  introduce proline (~42% chance of ≥1 at p=0.5, L=19) and break a helix that
  a frozen backbone cannot see — weight `--refold-sample` toward those schemes
  at high p.
- The `physicochemical` scheme preserves almost exactly the quantities the
  model measures (charge class, polarity, aromaticity), so its near-flat curve
  is partly a restatement of the rule table, not an independent finding.
- Unlike `length_analysis`, no `score_z`: it would not be comparable across
  runs. Use `delta_prob_AMP` / `delta_logit_margin`.
- Backed by `gnn_minimal`, **not** the legacy tree — it does not share
  `length_analysis`'s dependency on `sequence_to_svm_minimal`.

## Divergences to know

**Prediction encoding is not uniform.** `sequence_analysis` and
`window_sequence_analysis` emit `prediction` as **+1/-1** with `sigma` and
`p_amp`. `length_analysis` emits `pred` as **0/1** with `logit_margin` and
`prob_AMP`. Don't write code that assumes one shape across modules.

| Concern | `sequence_analysis` | `window_sequence_analysis` | `length_analysis` |
|---|---|---|---|
| `run_options.py` | **absent** (flags inline in two files) | present | present |
| `--checkpoint-dir` default | lazy `None` | eager | eager |
| `--device` | free-form `str` | n/a | `choices=(cuda,cpu)` |
| GNN flag | `--gnn-model` (dir or `.pt`) | n/a | `--gnn-checkpoint` (`.pt`) |
| Input extras | kept, collisions dropped | profile: `label_` prefix; raw: dropped | appended |
| GNN backend | `gnn_minimal` | n/a | **legacy** (see below) |

`length_analysis` GNN scoring still runs mostly on the legacy tree. It puts
`sequence_to_svm_minimal` *ahead of* `gnn_minimal` on `sys.path`
(`models/gnn.py:214`), imports `gnn.models`, `gnn.checkpoint_meta`,
`gnn.extra_feature_scaler` from the legacy tree, and calls the private
`compare_model_predictions._run_gnn_predictions`. Calibration is the one piece
that now comes from `gnn_minimal/core/calibration.py`, so `--fitting` behaves
the same here as elsewhere; it additionally honours the legacy
`<stem>_platt.json` naming (`_stem_platt`) that `core/artifacts.py` does not
know, which `gnn_vae_qsar_original` uses. If the legacy folder were removed,
`length_analysis` GNN scoring breaks while `sequence_analysis` does not. See
`architecture.md`.

`window_sequence_analysis/models/svm.py:12` imports from `sequence_analysis` —
the one sibling-to-sibling import. It cannot run without `sequence_analysis`
importable.

`sequence_analysis/models/svm.py:17` mutates `sys.path` as an import side
effect (adding the repo root and the legacy folder) with no current consumer.

## Duplication inventory

Known duplication across the three modules. Per the minimal-diffs rule, don't refactor
these on sight — but know they exist so you update all copies when you must
touch one.

1. `sklearn_pickle_compat.py` — byte-identical in `sequence_analysis/utils/`
   and `window_sequence_analysis/models/`.
2. SVM scoring bodies — `read_zscores`, `load_svm`,
   `load_grar740104_matrix`, `qsar_descriptors`, `safe_descriptor_call`,
   `CHARGE` duplicated verbatim between the two SVM adapters.
3. `find_single_checkpoint_file` and the checkpoint-suffix constants — three
   copies.
4. `output_path_from_args` — four copies.
5. The CSV loader — three near-identical classes, with the extras field named
   `extra_columns` in one and `label_columns` in the other two.
6. `ProgressReporter` — two **incompatible** protocols:
   `update(count, label="")` in `length_analysis` vs
   `update(count, record_id)` in `window_sequence_analysis`.
7. `normalize_sequence` — `length_analysis` and
   `window_sequence_analysis` import it from `data_normalizer.shared.records`;
   `sequence_analysis` keeps its own identical copy.

## SVM specifics

Both SVM adapters compute the same 12 QSAR descriptors: `netCharge`, `FC`,
`LW`, `DP`, `NK`, `AE`, `pcMK`, `_SolventAccessibilityD1025`,
`tau2_GRAR740104`, `tau4_GRAR740104`, `QSO50_GRAR740104`, `QSO29_GRAR740104`.
Only the last four need the AAIndex matrix, read as a *data* directory from
`descriptors/aaindex` — `descriptors/descripGen_12_py3.py` is never imported.

**`safe_descriptor_call` swallows every exception and returns `{}`**, so those
four descriptors silently become 0.0 if ProPy throws. A scoring difference with
no warning.

Z-scores come from a 3-line text file: names, means, stds; a non-positive std
becomes 1.0.

`sklearn_pickle_compat.py` makes SVMs pickled by very old scikit-learn
loadable: it aliases moved modules, reconstitutes `NDArrayWrapper` sidecar
`.npy` shards (this is what makes
`checkpoints/original_dataset/original_svm/svc.pkl` usable), back-fills
attributes modern `SVC` requires, and suppresses the version warning.
`prepare_legacy_svm(svm, pickle_dir)` is the single entry point.

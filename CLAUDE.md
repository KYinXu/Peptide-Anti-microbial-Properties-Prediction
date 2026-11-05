# Repository Rules

- Research code: No unit tests, logging frameworks, defensive validation, or retry wrappers unless explicitly asked. Verify by running scripts on real data. Correct numbers matter most.
- Minimal diffs: Touch only lines/files requested. No refactoring, import reordering, or drive-by formatting.
- No speculative code: Functions over classes; abstract only when a second caller exists. Minimal comments (math/citations only).
- Defaults over flags: Reasonable defaults so the common call is just the script and its input. Add a CLI option only when a function is genuinely used more than one way.
- Architecture: Root directories are isolated modules communicating via disk artifacts, never sibling imports. `data_normalizer/` and `analysis_outputs/` define those artifact shapes and are the only importable shared modules.
- Legacy reference: `sequence_to_svm_minimal/` is strictly read-only for mathematical/pipeline reference. Never import from it, edit it, or duplicate its code structure.
- Reuse before adding: Match existing structure, naming, and output layout; extend a known helper rather than adding a parallel one.

## Reference

| Topic | Doc |
|---|---|
| Module map, import graph, known coupling defects | `docs/architecture.md` |
| Canonical sequence CSV + run directory/manifest contracts | `docs/contracts.md` |
| GNN train / inference / calibration | `docs/gnn-minimal.md` |
| Sequence, window, and length scoring runners | `docs/analysis-modules.md` |
| Datasets, checkpoints, results, descriptors | `docs/data-and-artifacts.md` |
| Notebook inventory | `docs/notebooks.md` |
| What legacy holds and what lacks a replacement | `docs/legacy.md` |

Index and known gotchas: `docs/README.md`.

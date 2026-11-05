# Shared contracts: `data_normalizer/` and `analysis_outputs/`

These two modules are the interface between every other module. Analysis
modules may import them; that is the one sanctioned cross-module import. Read
this before adding a module or changing an output shape.

## `data_normalizer/` — raw input → canonical CSV

Three sibling entry points, one per input format, with identical CLI shape:

```bash
python data_normalizer/normalize_csv.py   --input in.csv   [--output out.csv]
python data_normalizer/normalize_fasta.py --input in.fasta [--output out.csv]
python data_normalizer/normalize_txt.py   --input in.txt   [--output out.csv]
```

`-i`/`-o` short forms work, and `python -m data_normalizer.normalize_fasta` works
too — each script inserts the repo root on `sys.path` when run directly.

**Flags:** only two. `--input` is required; `--output` defaults to
`<input dir>/sequences.csv`. That is the whole surface — a good model for
the defaults-over-flags rule. `main()` returns 0, or prints `Error: …` to stderr and returns 1.

**Per-format rules:**

| Format | Accepts | Notes |
|---|---|---|
| CSV | `.csv` only; must have `id` and `sequence` columns | read as `utf-8-sig` (BOM-tolerant); all other columns pass through as extras; blank `id` errors with the 1-based row number |
| FASTA | `.fa`, `.fasta`, `.faa`, `.fna` | multi-line sequences concatenated; errors on an empty header or sequence data before the first `>` |
| TXT | any suffix | one `id sequence` pair per line, split on first whitespace; blank and `#` lines skipped |

### The canonical CSV

Required columns are exactly `id,sequence`
(`shared/records.py:9`). Extras follow, **sorted alphabetically** by name, not
in insertion order (`shared/writers.py:15`). Sequences are whitespace-stripped
and upper-cased; an empty sequence raises (`shared/records.py:31`).

FASTA-derived files typically look like `id,sequence,name,rawtext`, plus a
column per `key=value` field found in the header.

Any module reading sequences should accept this shape and tolerate extras being
present or absent.

### FASTA header parsing

`parse_fasta_label(header) -> FastaLabel(id, name, fields, rawtext)`
(`shared/label_parser.py:20`):

- `id` is the first whitespace token, verbatim — no `sp|…|…` splitting.
- A token matching `^([A-Za-z][A-Za-z0-9_.:-]*)=(.*)$` opens a keyed field;
  subsequent plain tokens append to it, so values may contain spaces
  (`OS=Homo sapiens` → `{"OS": "Homo sapiens"}`).
- Tokens before the first `key=` become `name` (`None` if empty).
- A duplicate key overwrites the earlier one.

### Python API

Re-exported from `data_normalizer/__init__.py`:

- `normalize_csv_to_csv(input_csv, output_csv) -> int`
- `normalize_fasta_to_csv(input_fasta, output_csv) -> int`
- `normalize_txt_to_csv(input_txt, output_csv) -> int`
- `NormalizedSequenceRecord(id, sequence, extras)` with `.to_csv_row()`
- `normalize_sequence(sequence, record_id) -> str`

Also importable: the `iter_normalized_*` generators and
`write_normalized_csv(path, records) -> int`. Note `write_normalized_csv`
materializes all records to a list before writing — it is not streaming.

## `analysis_outputs/` — run directories and provenance

Every analysis that produces a CSV goes through this. Do not invent a second
output layout.

### The one entry point

```python
execute_csv_run(spec: RunSpec, write_csv: Callable[[RunOutput], int],
                *, now: datetime | None = None) -> RunOutput
```

(`runs.py:63`.) It creates the run directory, writes a `running` manifest,
calls your `write_csv(run)` — which must return the row count — then marks the
manifest `completed`. On **any** `BaseException` it marks it `failed` with the
exception type and message, then re-raises. `now` exists for test injection.

### Directory convention

```
<output_root>/runs/<YYYY-MM-DD_HH-MM>/
    ├── <output_filename>.csv
    └── run_manifest.json
```

- Timestamp is **UTC, minute resolution** (`runs.py:20`).
- Collisions append `_02`, `_03`, … up to 9999; the directory is claimed with a
  bare `mkdir()`, so an existing run is never reused (`runs.py:95`).
- `run_id` **is** the directory name, and it is stamped into every CSV row.

In practice `output_root` is `<dataset dir>/results`, giving
`<dataset>/results/runs/<timestamp>/`. Each runner derives it as
`args.output or args.input.parent / "results" / OUTPUT_FILENAME`, then passes
`output_root=target.parent, output_filename=target.name`.

**Gotcha:** `--output` names a *file*, but the file lands one level deeper,
inside `<its parent>/runs/<timestamp>/`.

### `RunSpec`

```python
RunSpec(output_root, output_filename, runner, model,
        config={}, inputs={}, model_files={},
        command=None, repository_root=None)
```

`output_filename` must be a bare filename ending in `.csv`. `command=None`
records `sys.argv`; `repository_root=None` walks up from the cwd looking for
`.git`.

### `RunConfiguration`

A manifest payload shape only — no behavior, no defaults applied
(`config.py:11`). Fields: `output_mode` (required, non-empty), `effective` (the
resolved settings that actually governed the run), `arguments` (usually
`vars(args)`). `RunSpec.config` also accepts a plain mapping.

### `RunCsvWriter`

Context manager (`csv_writer.py:17`). Prepends `run_id` as the first column
when a `run_id` is given, and omits it entirely when `None`. Uses
`extrasaction="ignore"`, so **unknown keys in a row are silently dropped** —
a misspelled column name fails quietly. `write_row` outside the `with` raises.

### Manifest schema

`schema_version: 1`. Records `run_id`, `status`
(`running`/`completed`/`failed`), created/completed UTC timestamps, `runner`,
`model` (name + per-file sha256/size), `config` plus `config_sha256`, every
`inputs` file's hash, the exact `command`, `repository`
(`{root, commit, dirty}`, or `null` outside a git checkout), `runtime.python`,
and `output` (path relative to the run dir, sha256, size, `row_count`). On
failure it adds `error: {type, message}`. Written atomically via a `.tmp` file
plus `replace`.

This is what makes a result reproducible, and it is why the rules can drop
unit tests without losing rigor. Keep it populated honestly.

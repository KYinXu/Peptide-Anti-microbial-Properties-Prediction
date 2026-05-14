# Proteome Candidate Generator

Generate AMP-like peptide candidates from a proteome FASTA using pepsickle cleavage predictions.

This folder is intentionally separate from `peptide_pipeline/`. It creates a filtered sequence-only candidate set first, then hands the final subset to the existing feature/model pipeline.

## Requirements

Run commands from `sequence_to_svm_minimal`.

Install pepsickle in the active Python environment:

```bash
pip install pepsickle
```

Optional Parquet output requires:

```bash
pip install pyarrow
```

Without `pyarrow`, `--output-format auto` writes CSV.

## Quick Start

```bash
python -m proteome_candidate_generator \
  --input data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta \
  --output-dir data/proteomes/generated \
  --top-n 400000 \
  --resume
```

The default command is `all`, which runs:

1. FASTA parsing and standard amino-acid filtering.
2. Batch FASTA writing.
3. Pepsickle constitutive and immunoproteasome predictions.
4. Cleavage-site unioning at `P > threshold`.
5. Fragment expansion for peptides of length 8-30.
6. Charge and hydrophobicity filtering.
7. Hydrophobic-moment ranking and top-N selection.
8. Metadata and pipeline-compatible output writing.

## Stage Commands

Preprocess only:

```bash
python -m proteome_candidate_generator preprocess \
  --input data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta \
  --output-dir data/proteomes/generated
```

Run or resume pepsickle calls:

```bash
python -m proteome_candidate_generator run-pepsickle \
  --input data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta \
  --output-dir data/proteomes/generated \
  --workers 4 \
  --resume
```

Build candidates from existing pepsickle outputs:

```bash
python -m proteome_candidate_generator build-candidates \
  --input data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta \
  --output-dir data/proteomes/generated \
  --top-n 400000
```

Validate pepsickle on a tiny run:

```bash
python -m proteome_candidate_generator validate \
  --input data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta \
  --output-dir data/proteomes/generated/validate \
  --limit-proteins 1 \
  --force
```

## Important Options

- `--threshold 0.5`: cleavage probability cutoff used when parsing pepsickle TSV outputs.
- `--batch-size 1000`: number of proteins per pepsickle FASTA batch.
- `--workers 1`: number of pepsickle subprocesses to run in parallel.
- `--resume`: skip existing pepsickle output files.
- `--force`: rerun pepsickle even when output files already exist.
- `--limit-proteins N`: process only the first N FASTA records, useful for smoke tests.
- `--min-len 8 --max-len 30`: peptide length range.
- `--min-charge 2`: minimum `count(R,K) - count(D,E)`.
- `--min-hydrophobicity 0.30`: minimum fraction of `A,I,L,M,F,V,P,G`.
- `--top-n 400000`: keep the highest-ranked peptides after hard filters.
- `--no-terminal-boundaries`: do not add protein termini as fragment boundaries.
- `--output-format auto|csv|parquet`: choose metadata table format.
- `--no-progress`: disable progress bars/status messages for long-running stages.

Pepsickle constitutive and immunoproteasome runs are invoked as `-m in-vitro -p C` and `-m in-vitro -p I`. Output labels are kept as `constitutive` and `immunoproteasome`.

## Output Layout

Default output directory:

```text
data/proteomes/generated/
  inputs/batches/
    batch_00001.fasta
  pepsickle/
    batch_00001.constitutive.tsv
    batch_00001.immunoproteasome.tsv
  cleavage_sites.jsonl
  final_candidates.csv or final_candidates.parquet
  final_candidates.txt
  run_manifest.json
```

`final_candidates.txt` contains `peptide_id sequence` rows and is intended for the existing blind pipeline.

`final_candidates.csv` or `.parquet` contains:

- `peptide_id`
- `sequence`
- `source_protein_id`
- `start`
- `end`
- `length`
- `net_charge`
- `hydrophobicity`
- `hydrophobic_moment`
- `rank_score`
- `left_cleavage_probability`
- `right_cleavage_probability`
- `predicted_cleavage_probability`

Coordinates are zero-based Python slice boundaries in the final table. Pepsickle `position` values are interpreted as 1-based residue positions and converted to boundaries between residues.

## Downstream Pipeline

Run the current pipeline on the filtered subset:

```bash
python scripts/run_data_pipeline.py \
  --mode blind \
  --input data/proteomes/generated/final_candidates.txt \
  --work-dir data/proteomes/generated/pipeline \
  --skip-if-exists
```

This keeps expensive ESMFold, geometric, QSAR, ESM2, SVM, or GNN work limited to the filtered candidate set.

## Troubleshooting

If pepsickle is missing, install it or pass `--pepsickle-bin` with the full executable path.

If parsing fails with a missing column error, run `validate` and inspect `run_manifest.json`. The parser expects pepsickle-style TSV columns equivalent to `position`, `cleav_prob`, and `protein_id`.

If memory is tight, keep `--top-n` enabled and reduce `--workers`. Candidate expansion streams per protein and keeps only the dedupe set plus retained top-ranked rows.

If a run is interrupted, rerun the same command with `--resume`. Use `--force` only when you want to regenerate pepsickle outputs.

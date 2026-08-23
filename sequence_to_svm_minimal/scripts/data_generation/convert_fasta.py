#!/usr/bin/env python3
"""
Convert FASTA to CSV with id, gene, name, and sequence columns.

UniProt-style headers are split into gene (first token) and name (description before
KEY=value parameters such as OS=, GN=, PE=). Generated ids are seq_01, seq_02, …

Optionally join Human Protein Atlas annotation columns (e.g. Single cell expression
cluster) from an MS-style TSV keyed by UniProt accession.

Example header:
  >sp|P01023|A2MG_HUMAN Alpha-2-macroglobulin OS=Homo sapiens OX=9606 GN=A2M PE=1 SV=3

Example:
  python scripts/data_generation/convert_fasta.py \\
    --input data/test/proteins.fasta \\
    --output data/test/proteins.csv \\
    --annotation-tsv data/test/MS.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

# UniProt / Swiss-Prot trailer: space + KEY= (KEY is 2+ uppercase letters)
_PARAM_START = re.compile(r"\s[A-Z]{2}=")

ANNOTATION_COL_CLUSTER = "Single cell expression cluster"
OUTPUT_COL_CLUSTER = "single_cell_expression_cluster"
UNIPROT_COL_CANDIDATES = ("Uniprot", "UniProt", "uniprot", "Entry", "accession")


@dataclass(frozen=True)
class FastaRecord:
    gene: str
    name: str
    sequence: str


@dataclass(frozen=True)
class UHandlingStats:
    n_u_substituted: int
    n_records_dropped_for_u: int


@dataclass(frozen=True)
class AnnotationJoinStats:
    n_annotation_rows: int
    n_accessions_mapped: int
    n_records_matched: int
    n_records_unmatched: int


def parse_fasta_header(header: str) -> tuple[str, str]:
    """
    Return (gene, name) from a FASTA header line (without leading '>').

    gene: first space-separated token (e.g. sp|P01023|A2MG_HUMAN)
    name: remainder before KEY=value parameters (e.g. Alpha-2-macroglobulin)
    """
    text = header.strip()
    if not text:
        return "", ""

    cut = _PARAM_START.search(text)
    if cut:
        text = text[: cut.start()].strip()

    parts = text.split(None, 1)
    gene = parts[0]
    name = parts[1].strip() if len(parts) > 1 else ""
    return gene, name


def uniprot_accession(gene_token: str) -> str:
    """Extract UniProt accession from a FASTA id token (sp|P01023|NAME or bare)."""
    text = (gene_token or "").strip()
    if not text:
        return ""
    parts = text.split("|")
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip()
    return text


def iter_fasta_records(path: Path):
    """Yield FastaRecord(gene, name, sequence) from a FASTA file."""
    current_gene = ""
    current_name = ""
    seq_parts: list[str] = []
    in_record = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_record:
                    seq = "".join(seq_parts).replace(" ", "").upper()
                    yield FastaRecord(current_gene, current_name, seq)
                current_gene, current_name = parse_fasta_header(line[1:])
                seq_parts = []
                in_record = True
            elif in_record:
                seq_parts.append(line)

    if in_record:
        seq = "".join(seq_parts).replace(" ", "").upper()
        yield FastaRecord(current_gene, current_name, seq)


def _handle_u(seq: str, mode: str, replacement: str) -> tuple[str | None, bool]:
    has_u = "U" in seq
    if not has_u:
        return seq, False
    if mode == "drop":
        return None, False
    if mode == "keep":
        return seq, False
    return seq.replace("U", replacement), True


def _split_accessions(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = re.split(r"[,;\s]+", text)
    return [p.strip() for p in parts if p.strip()]


def load_cluster_by_uniprot(annotation_tsv: Path) -> tuple[dict[str, str], int]:
    """
    Map UniProt accession → Single cell expression cluster from an HPA MS TSV.

    Comma/semicolon-separated UniProt cells are expanded to one entry per accession.
    If an accession appears more than once, the first non-empty cluster wins.
    """
    with open(annotation_tsv, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"No header in annotation TSV: {annotation_tsv}")

        uniprot_col = next(
            (c for c in UNIPROT_COL_CANDIDATES if c in reader.fieldnames), None
        )
        if uniprot_col is None:
            raise ValueError(
                f"No UniProt column in {annotation_tsv}. "
                f"Tried {UNIPROT_COL_CANDIDATES}. Available: {list(reader.fieldnames)}"
            )
        if ANNOTATION_COL_CLUSTER not in reader.fieldnames:
            raise ValueError(
                f"Missing {ANNOTATION_COL_CLUSTER!r} in {annotation_tsv}. "
                f"Available: {list(reader.fieldnames)}"
            )

        mapping: dict[str, str] = {}
        n_rows = 0
        for row in reader:
            n_rows += 1
            cluster = (row.get(ANNOTATION_COL_CLUSTER) or "").strip()
            for acc in _split_accessions(row.get(uniprot_col) or ""):
                if acc not in mapping or (not mapping[acc] and cluster):
                    mapping[acc] = cluster

    return mapping, n_rows


def fasta_to_csv(
    input_path: Path,
    output_path: Path,
    *,
    u_mode: str = "replace",
    u_replacement: str = "C",
    annotation_tsv: Path | None = None,
) -> tuple[int, UHandlingStats, AnnotationJoinStats | None]:
    records = [r for r in iter_fasta_records(input_path) if r.sequence]
    if not records:
        return (
            0,
            UHandlingStats(n_u_substituted=0, n_records_dropped_for_u=0),
            None,
        )

    cluster_by_acc: dict[str, str] | None = None
    join_stats: AnnotationJoinStats | None = None
    if annotation_tsv is not None:
        cluster_by_acc, n_ann_rows = load_cluster_by_uniprot(annotation_tsv)

    width = max(2, len(str(len(records))))
    n_u_substituted = 0
    n_dropped = 0
    n_matched = 0
    n_unmatched = 0
    fieldnames = ["id", "gene", "name", "sequence"]
    if cluster_by_acc is not None:
        fieldnames.append(OUTPUT_COL_CLUSTER)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        n_written = 0
        for rec in records:
            seq, substituted = _handle_u(rec.sequence, u_mode, u_replacement)
            if seq is None:
                n_dropped += 1
                continue
            if substituted:
                n_u_substituted += 1
            n_written += 1
            row = {
                "id": f"seq_{n_written:0{width}d}",
                "gene": rec.gene,
                "name": rec.name,
                "sequence": seq,
            }
            if cluster_by_acc is not None:
                acc = uniprot_accession(rec.gene)
                cluster = cluster_by_acc.get(acc, "")
                row[OUTPUT_COL_CLUSTER] = cluster
                if acc and acc in cluster_by_acc:
                    n_matched += 1
                else:
                    n_unmatched += 1
            writer.writerow(row)

    if cluster_by_acc is not None:
        join_stats = AnnotationJoinStats(
            n_annotation_rows=n_ann_rows,
            n_accessions_mapped=len(cluster_by_acc),
            n_records_matched=n_matched,
            n_records_unmatched=n_unmatched,
        )

    return (
        n_written,
        UHandlingStats(
            n_u_substituted=n_u_substituted, n_records_dropped_for_u=n_dropped
        ),
        join_stats,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FASTA to CSV (id, gene, name, sequence"
            f"[, {OUTPUT_COL_CLUSTER}])."
        )
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input FASTA file")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--annotation-tsv",
        type=Path,
        default=None,
        help=(
            "Optional HPA MS TSV with Uniprot + "
            f"'{ANNOTATION_COL_CLUSTER}' to join into the CSV."
        ),
    )
    parser.add_argument(
        "--u-mode",
        choices=["replace", "keep", "drop"],
        default="replace",
        help="How to handle selenocysteine U (default: replace).",
    )
    parser.add_argument(
        "--u-replacement",
        type=str,
        default="C",
        help="Single-letter residue used when --u-mode replace (default: C).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input FASTA not found: {args.input}")
    if args.annotation_tsv is not None and not args.annotation_tsv.is_file():
        raise SystemExit(f"Annotation TSV not found: {args.annotation_tsv}")
    repl = args.u_replacement.strip().upper()
    if args.u_mode == "replace":
        if len(repl) != 1:
            raise SystemExit("--u-replacement must be a single residue letter.")
    try:
        n, stats, join_stats = fasta_to_csv(
            args.input,
            args.output,
            u_mode=args.u_mode,
            u_replacement=repl,
            annotation_tsv=args.annotation_tsv,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if n == 0:
        raise SystemExit(f"No FASTA records found in: {args.input}")
    print(f"Wrote {n} sequences to {args.output}")
    if args.u_mode == "replace":
        print(f"U handling: substituted U->{repl} in {stats.n_u_substituted} records")
    elif args.u_mode == "drop":
        print(f"U handling: dropped {stats.n_records_dropped_for_u} records containing U")
    else:
        print("U handling: kept U unchanged")
    if join_stats is not None:
        print(
            f"Annotation join ({ANNOTATION_COL_CLUSTER}): "
            f"matched={join_stats.n_records_matched} "
            f"unmatched={join_stats.n_records_unmatched} "
            f"(tsv_rows={join_stats.n_annotation_rows}, "
            f"accessions={join_stats.n_accessions_mapped})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

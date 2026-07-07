"""Analyze why reference peptides fail paper_pddp or heuristic filters."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from proteome_candidate_generator.candidates import (  # noqa: E402
    hydrophobicity,
    net_charge,
)
from proteome_candidate_generator.cleavage import (  # noqa: E402
    ProteinCleavageSites,
    iter_pepsickle_tsv_sites,
)
from proteome_candidate_generator.fasta import iter_fasta_records  # noqa: E402
from proteome_candidate_generator.pddp_scoring import load_mapp_reference_scorer  # noqa: E402

POSITIVE = frozenset("RK")
NEGATIVE = frozenset("DE")
HYDROPHOBIC = frozenset("AILMFVPG")

FASTA = ROOT / "data/proteomes/uniprotkb_UP000005640_2026_05_13.fasta"
TABLE6 = ROOT / "data/proteomes/original_paper/Candidates Table 6 - Database of PDDPs.csv"
PAPER_CSV = ROOT / "data/proteomes/paper_pddp/final_candidates.csv"
CLEAVAGE_JSONL = ROOT / "data/proteomes/paper_pddp/generated/cleavage_sites.jsonl"
MAPP = ROOT / "data/proteomes/MAPP_database.csv"
PEPSICKLE_DIR = ROOT / "data/proteomes/paper_pddp/generated/pepsickle"


@dataclass
class FilterResult:
    sequence: str
    gene: str
    length: int
    charge: int
    hydro: float
    in_mapp: bool
    in_paper_pddp: bool
    fail_len_8_30: bool
    fail_len_10_50: bool
    fail_charge_2: bool
    fail_hydro_0_3: bool
    fail_cleavage_0_5: bool
    fail_cleavage_0_35: bool
    cleavage_detail: str
    protein_id: str | None


def load_proteins() -> dict[str, tuple[str, str]]:
    """Map UniProt accession -> (full protein_id, sequence)."""
    by_accession: dict[str, tuple[str, str]] = {}
    for record in iter_fasta_records(FASTA):
        parts = record.protein_id.split("|")
        if len(parts) >= 2:
            by_accession[parts[1]] = (record.protein_id, record.sequence.upper())
    return by_accession


def load_cleavage_sites() -> dict[str, ProteinCleavageSites]:
    sites: dict[str, ProteinCleavageSites] = {}
    with CLEAVAGE_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            probs = {int(k): float(v) for k, v in payload["site_probabilities"].items()}
            sites[payload["protein_id"]] = ProteinCleavageSites(
                payload["protein_id"], payload["length"], probs
            )
    return sites


def boundary_probs(
    sites: ProteinCleavageSites,
    start: int,
    end: int,
) -> tuple[float, float]:
    left = 1.0 if start in (0, sites.length) else sites.site_probabilities.get(start, 0.0)
    right = 1.0 if end in (0, sites.length) else sites.site_probabilities.get(end, 0.0)
    return left, right


def passes_cleavage(sites: ProteinCleavageSites, start: int, end: int, threshold: float) -> bool:
    left, right = boundary_probs(sites, start, end)
    return left > threshold and right > threshold


def load_boundary_probs_from_tsv(
    protein_id: str,
    start: int,
    end: int,
) -> tuple[float, float, str]:
    """Read raw pepsickle TSVs for one protein to get max boundary probabilities."""
    stem = protein_id.replace("|", "_").replace(" ", "_")
    left_max = 1.0 if start in (0,) else 0.0
    right_max = 0.0
    found_files = 0
    for path in PEPSICKLE_DIR.glob("*.tsv"):
        found_files += 1
        for site in iter_pepsickle_tsv_sites(path, model_name=path.stem.split(".")[-1], threshold=0.0):
            if site.protein_id != protein_id:
                continue
            if site.position == start:
                left_max = max(left_max, site.probability)
            if site.position == end:
                right_max = max(right_max, site.probability)
    length = None
    if protein_id in sites_cache:
        length = sites_cache[protein_id].length
    if length is not None:
        if start == 0:
            left_max = 1.0
        if end == length:
            right_max = max(right_max, 1.0)
    return left_max, right_max, f"scanned_{found_files}_tsv"


sites_cache: dict[str, ProteinCleavageSites] = {}


def analyze_row(
    row: pd.Series,
    paper_seq: set[str],
    mapp_scorer,
    proteins: dict[str, tuple[str, str]],
    cleavage: dict[str, ProteinCleavageSites],
) -> FilterResult:
    seq = str(row["Sequence"]).strip().upper()
    accession = str(row["Leading razor protein"]).strip()
    # Table 6 uses 1-based start and exclusive end matching Python slice after start-1.
    table_start = int(row["Start position"])
    table_end = int(row["End position"])
    start = table_start - 1
    end = table_end
    gene = str(row.get("Gene names", ""))
    length = len(seq)
    charge = net_charge(seq, positive_aa=POSITIVE, negative_aa=NEGATIVE)
    hydro = hydrophobicity(seq, hydrophobic_aa=HYDROPHOBIC)
    in_mapp = mapp_scorer.score_sequence(seq) > 0

    protein_id = None
    fail_cleavage_0_5 = True
    fail_cleavage_0_35 = True
    detail = "protein_not_found"

    match = proteins.get(accession)
    if match is not None:
        protein_id, full_seq = match
        fragment = full_seq[start:end]
        if fragment != seq:
            detail = f"coordinate_mismatch:fasta_has_{fragment[:20]}..."
        else:
            site_info = cleavage.get(protein_id)
            if site_info is None:
                detail = "no_cleavage_sites_for_protein"
            else:
                left, right = boundary_probs(site_info, start, end)
                fail_cleavage_0_5 = not passes_cleavage(site_info, start, end, 0.5)
                fail_cleavage_0_35 = not passes_cleavage(site_info, start, end, 0.35)
                detail = f"left_boundary={start} p={left:.4f}; right_boundary={end} p={right:.4f}"

    return FilterResult(
        sequence=seq,
        gene=gene,
        length=length,
        charge=charge,
        hydro=hydro,
        in_mapp=in_mapp,
        in_paper_pddp=seq in paper_seq,
        fail_len_8_30=length < 8 or length > 30,
        fail_len_10_50=length < 10 or length > 50,
        fail_charge_2=charge < 2,
        fail_hydro_0_3=hydro < 0.30,
        fail_cleavage_0_5=fail_cleavage_0_5,
        fail_cleavage_0_35=fail_cleavage_0_35,
        cleavage_detail=detail,
        protein_id=protein_id,
    )


def primary_blocker(r: FilterResult, *, paper_mode: bool) -> str:
    if paper_mode:
        if r.fail_len_10_50:
            return "length_10_50"
        if not r.in_mapp:
            return "not_in_mapp"
        if r.fail_cleavage_0_5:
            return "cleavage_0.5"
        return "unknown"
    reasons = []
    if r.fail_len_8_30:
        reasons.append("length_8_30")
    if r.fail_charge_2:
        reasons.append("min_charge_2")
    if r.fail_hydro_0_3:
        reasons.append("min_hydro_0.3")
    if reasons:
        return "+".join(reasons)
    if r.fail_cleavage_0_5:
        return "cleavage_0.5"
    return "rank_or_dedupe"


def main() -> None:
    global sites_cache
    table6 = pd.read_csv(TABLE6)
    paper = pd.read_csv(PAPER_CSV)
    paper_seq = set(paper["sequence"].astype(str).str.upper())
    mapp_scorer = load_mapp_reference_scorer(MAPP)
    proteins = load_proteins()
    sites_cache = load_cleavage_sites()
    cleavage = sites_cache

    results = [analyze_row(row, paper_seq, mapp_scorer, proteins, cleavage) for _, row in table6.iterrows()]
    df = pd.DataFrame([r.__dict__ for r in results])

    missing_paper = df[~df["in_paper_pddp"]].copy()
    print("=" * 72)
    print("TABLE 6 vs paper_pddp (threshold 0.5, MAPP match, len 10-50)")
    print("=" * 72)
    print(f"Total Table 6 peptides: {len(df)}")
    print(f"In paper_pddp: {df['in_paper_pddp'].sum()}")
    print(f"Missing from paper_pddp: {len(missing_paper)}")
    print()

    paper_blockers = Counter(primary_blocker(r, paper_mode=True) for r in results if not r.in_paper_pddp)
    print("Primary blocker for missing (paper_pddp @ 0.5, len 10-50):")
    for reason, count in paper_blockers.most_common():
        print(f"  {reason:20s} {count:4d}")

    would_pass_035 = missing_paper[
        ~missing_paper["fail_len_10_50"]
        & missing_paper["in_mapp"]
        & ~missing_paper["fail_cleavage_0_35"]
    ]
    print(f"\nMissing but would pass at cleavage 0.35 (len 10-50, MAPP): {len(would_pass_035)}")

    print()
    print("=" * 72)
    print("TABLE 6 vs heuristic filter (len 8-30, charge>=2, hydro>=0.3)")
    print("=" * 72)
    heuristic_fail = df["fail_len_8_30"] | df["fail_charge_2"] | df["fail_hydro_0_3"]
    print(f"Fail length 8-30: {df['fail_len_8_30'].sum()}")
    print(f"Fail min_charge 2: {df['fail_charge_2'].sum()}")
    print(f"Fail min_hydro 0.3: {df['fail_hydro_0_3'].sum()}")
    print(f"Fail any heuristic: {heuristic_fail.sum()}")
    print(f"Pass heuristics but fail cleavage 0.5: {(~heuristic_fail & df['fail_cleavage_0_5']).sum()}")
    print(f"In paper_pddp but fail heuristics: {(df['in_paper_pddp'] & heuristic_fail).sum()}")

    heur_blockers = Counter(
        primary_blocker(r, paper_mode=False)
        for r in results
        if r.fail_len_8_30 or r.fail_charge_2 or r.fail_hydro_0_3
    )
    print("\nHeuristic failure breakdown (can overlap):")
    for reason, count in heur_blockers.most_common():
        print(f"  {reason:20s} {count:4d}")

    print()
    print("=" * 72)
    print("SAMPLE missing peptides (notebook examples + diverse blockers)")
    print("=" * 72)
    sample_seqs = [
        "MRAKWRKKRMRRLK",
        "GFVKVVKNKAYFKRYQVKF",
        "GHQQLYWSHPRKFGQGSRSCRVCSNRHGLIRKYGLNMC",
        "LVRIPLHKFTSIRR",
        "ACVVLCVWWTRKRRKER",
        "RKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYR",
        "RMFRGSLYKRYPSLWRRL",
        "LSLVTKKKRFWCWQRPKYQFL",
    ]
    for seq in sample_seqs:
        row = df[df["sequence"] == seq]
        if row.empty:
            print(f"\n{seq}: not in Table 6")
            continue
        r = row.iloc[0]
        print(f"\n{seq} ({r['gene']}, len={r['length']}, charge={r['charge']}, hydro={r['hydro']:.3f})")
        print(f"  in paper_pddp: {bool(r['in_paper_pddp'])}")
        print(f"  fail len 8-30: {bool(r['fail_len_8_30'])} | fail len 10-50: {bool(r['fail_len_10_50'])}")
        print(f"  fail charge>=2: {bool(r['fail_charge_2'])} | fail hydro>=0.3: {bool(r['fail_hydro_0_3'])}")
        print(f"  fail cleavage 0.5: {bool(r['fail_cleavage_0_5'])} | fail cleavage 0.35: {bool(r['fail_cleavage_0_35'])}")
        print(f"  cleavage: {r['cleavage_detail']}")
        print(f"  paper blocker: {primary_blocker(FilterResult(**r.to_dict()), paper_mode=True)}")
        print(f"  heuristic blocker: {primary_blocker(FilterResult(**r.to_dict()), paper_mode=False)}")


if __name__ == "__main__":
    main()

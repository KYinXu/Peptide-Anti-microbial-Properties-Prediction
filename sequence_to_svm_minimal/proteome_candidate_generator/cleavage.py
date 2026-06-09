"""Parse pepsickle TSV output and union cleavage coordinates."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from proteome_candidate_generator.progress import progress_iter

PROTEIN_COLUMNS = ("protein_id", "protein", "id", "proteinid")
POSITION_COLUMNS = ("position", "pos", "site", "index")
PROBABILITY_COLUMNS = ("cleav_prob", "cleavage_probability", "probability", "prob", "p")


@dataclass(frozen=True)
class CleavageSite:
    protein_id: str
    position: int
    probability: float
    model_name: str


@dataclass(frozen=True)
class ProteinCleavageSites:
    protein_id: str
    length: int
    site_probabilities: dict[int, float]

    @property
    def sites(self) -> list[int]:
        return sorted(self.site_probabilities)


def _normalized_header(row: dict[str, str]) -> dict[str, str]:
    return {key.strip().lower(): key for key in row}


def _choose_column(row: dict[str, str], options: tuple[str, ...], label: str) -> str:
    normalized = _normalized_header(row)
    for option in options:
        if option in normalized:
            return normalized[option]
    raise ValueError(f"Missing {label} column. Found columns: {list(row)}")


<<<<<<< HEAD
def parse_pepsickle_tsv(path: Path, *, model_name: str, threshold: float) -> list[CleavageSite]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        return []
    protein_col = _choose_column(rows[0], PROTEIN_COLUMNS, "protein id")
    position_col = _choose_column(rows[0], POSITION_COLUMNS, "position")
    probability_col = _choose_column(rows[0], PROBABILITY_COLUMNS, "probability")

    sites: list[CleavageSite] = []
    for row in rows:
        probability = float(row[probability_col])
        if probability <= threshold:
            continue
        position = int(float(row[position_col]))
        if position <= 0:
            raise ValueError(f"Pepsickle position must be 1-based and positive in {path}: {position}")
        sites.append(
            CleavageSite(
=======
def iter_pepsickle_tsv_sites(
    path: Path,
    *,
    model_name: str,
    threshold: float,
):
    """Stream cleavage sites above threshold without loading the full TSV into memory."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        first_row = next(reader, None)
        if first_row is None:
            return
        protein_col = _choose_column(first_row, PROTEIN_COLUMNS, "protein id")
        position_col = _choose_column(first_row, POSITION_COLUMNS, "position")
        probability_col = _choose_column(first_row, PROBABILITY_COLUMNS, "probability")

        def emit(row: dict[str, str]):
            probability = float(row[probability_col])
            if probability <= threshold:
                return
            position = int(float(row[position_col]))
            if position <= 0:
                raise ValueError(f"Pepsickle position must be 1-based and positive in {path}: {position}")
            yield CleavageSite(
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
                protein_id=row[protein_col],
                position=position,
                probability=probability,
                model_name=model_name,
            )
<<<<<<< HEAD
        )
    return sites
=======

        yield from emit(first_row)
        for row in reader:
            yield from emit(row)


def parse_pepsickle_tsv(path: Path, *, model_name: str, threshold: float) -> list[CleavageSite]:
    return list(iter_pepsickle_tsv_sites(path, model_name=model_name, threshold=threshold))
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)


def inspect_pepsickle_schema(path: Path) -> dict[str, str]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        row = next(reader, None)
    if row is None:
        raise ValueError(f"Pepsickle output is empty: {path}")
    return {
        "protein_id": _choose_column(row, PROTEIN_COLUMNS, "protein id"),
        "position": _choose_column(row, POSITION_COLUMNS, "position"),
        "probability": _choose_column(row, PROBABILITY_COLUMNS, "probability"),
        "coordinate_convention": "1-based residue positions, converted to Python slice boundaries",
    }


<<<<<<< HEAD
=======
def _init_merged_sites(lengths: dict[str, int]) -> dict[str, ProteinCleavageSites]:
    return {
        protein_id: ProteinCleavageSites(protein_id, length, {})
        for protein_id, length in lengths.items()
    }


def _merge_site(merged: dict[str, ProteinCleavageSites], site: CleavageSite) -> None:
    if site.protein_id not in merged:
        return
    if site.position > merged[site.protein_id].length:
        raise ValueError(f"Cleavage position {site.position} exceeds length for {site.protein_id}")
    current = merged[site.protein_id].site_probabilities.get(site.position, 0.0)
    merged[site.protein_id].site_probabilities[site.position] = max(current, site.probability)


>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
def union_sites(
    site_groups: list[list[CleavageSite]],
    lengths: dict[str, int],
) -> dict[str, ProteinCleavageSites]:
<<<<<<< HEAD
    merged = {
        protein_id: ProteinCleavageSites(protein_id, length, {})
        for protein_id, length in lengths.items()
    }
    for group in site_groups:
        for site in group:
            if site.protein_id not in merged:
                continue
            if site.position > merged[site.protein_id].length:
                raise ValueError(
                    f"Cleavage position {site.position} exceeds length for {site.protein_id}"
                )
            current = merged[site.protein_id].site_probabilities.get(site.position, 0.0)
            merged[site.protein_id].site_probabilities[site.position] = max(current, site.probability)
=======
    merged = _init_merged_sites(lengths)
    for group in site_groups:
        for site in group:
            _merge_site(merged, site)
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    return merged


def load_union_from_outputs(
    output_paths: list[tuple[Path, str]],
    *,
    lengths: dict[str, int],
    threshold: float,
    show_progress: bool = False,
) -> dict[str, ProteinCleavageSites]:
<<<<<<< HEAD
    iterator = output_paths
    if show_progress:
        iterator = progress_iter(output_paths, desc="Parsing pepsickle TSVs", total=len(output_paths))
    groups = [
        parse_pepsickle_tsv(path, model_name=model_name, threshold=threshold)
        for path, model_name in iterator
    ]
    return union_sites(groups, lengths)
=======
    merged = _init_merged_sites(lengths)
    iterator = output_paths
    if show_progress:
        iterator = progress_iter(output_paths, desc="Parsing pepsickle TSVs", total=len(output_paths))
    for path, model_name in iterator:
        for site in iter_pepsickle_tsv_sites(path, model_name=model_name, threshold=threshold):
            _merge_site(merged, site)
    return merged


def read_sites_jsonl(path: Path, *, lengths: dict[str, int] | None = None) -> dict[str, ProteinCleavageSites]:
    sites: dict[str, ProteinCleavageSites] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            protein_id = payload["protein_id"]
            length = payload["length"]
            if lengths is not None and protein_id in lengths and lengths[protein_id] != length:
                raise ValueError(
                    f"Cleavage cache length mismatch for {protein_id}: "
                    f"cache={length}, fasta={lengths[protein_id]}"
                )
            site_probabilities = {int(site): float(prob) for site, prob in payload["site_probabilities"].items()}
            sites[protein_id] = ProteinCleavageSites(protein_id, length, site_probabilities)
    if lengths is not None:
        missing = set(lengths) - set(sites)
        if missing:
            raise ValueError(
                f"Cleavage cache at {path} is missing {len(missing)} protein(s); "
                "delete the cache and rerun without --reuse-cleavage-sites."
            )
    return sites
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)


def write_sites_jsonl(sites: dict[str, ProteinCleavageSites], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_id in sorted(sites):
            item = sites[protein_id]
            handle.write(
                json.dumps(
                    {
                        "protein_id": item.protein_id,
                        "length": item.length,
                        "sites": item.sites,
                        "site_probabilities": {
                            str(site): item.site_probabilities[site] for site in item.sites
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

"""Collapse scored mutation variants into per-peptide robustness scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, pstdev

from ..mutants.common import MutantRecord


@dataclass(frozen=True)
class Cell:
    """One (scheme, p) cell of replicates for a single parent."""

    scheme: str
    p: float
    n: int
    k_requested: int
    k_applied: int
    retention: float
    mean_prob_AMP: float
    sd_prob_AMP: float
    mean_delta_prob_AMP: float
    mean_logit_margin: float
    sd_logit_margin: float


def summarize_parent(
    records: list[MutantRecord],
    predictions: list,
    reference,
    backbone: str,
) -> tuple[list[dict], list[dict]]:
    """Collapse one parent's scored variants into per-scheme and per-cell rows."""
    grouped: dict[tuple[str, float], list[tuple[MutantRecord, object]]] = defaultdict(list)
    for record, prediction in zip(records, predictions):
        if record.scheme != "parent":
            grouped[(record.scheme, record.p)].append((record, prediction))
    if not grouped:
        return [], []

    cells_by_scheme: dict[str, list[Cell]] = defaultdict(list)
    margins_by_scheme: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (scheme, p), members in sorted(grouped.items()):
        cells_by_scheme[scheme].append(_cell(scheme, p, members, reference))
        margins_by_scheme[scheme].extend(
            (record.p, prediction.logit_margin) for record, prediction in members
        )

    parent = records[0]
    anchor = float(reference.pred) if reference is not None else 1.0
    summary_rows = [
        _summary_row(parent, scheme, cells, margins_by_scheme[scheme], reference, anchor, backbone)
        for scheme, cells in cells_by_scheme.items()
    ]
    curve_rows = [
        _curve_row(parent, cell, backbone)
        for cells in cells_by_scheme.values()
        for cell in cells
    ]
    return summary_rows, curve_rows


def _cell(scheme: str, p: float, members: list[tuple[MutantRecord, object]], reference) -> Cell:
    probabilities = [prediction.prob_AMP for _, prediction in members]
    margins = [prediction.logit_margin for _, prediction in members]
    predictions = [prediction.pred for _, prediction in members]
    return Cell(
        scheme=scheme,
        p=p,
        n=len(members),
        k_requested=members[0][0].k_requested,
        k_applied=members[0][0].k_applied,
        retention=fmean(predictions),
        mean_prob_AMP=fmean(probabilities),
        sd_prob_AMP=pstdev(probabilities),
        mean_delta_prob_AMP=(
            fmean(probabilities) - reference.prob_AMP if reference is not None else ""
        ),
        mean_logit_margin=fmean(margins),
        sd_logit_margin=pstdev(margins),
    )


def _summary_row(
    parent: MutantRecord,
    scheme: str,
    cells: list[Cell],
    margins: list[tuple[float, float]],
    reference,
    anchor: float,
    backbone: str,
) -> dict:
    row = {
        "parent_id": parent.parent_id,
        "scheme": scheme,
        "backbone": backbone,
        "sequence": parent.parent_sequence,
        "length": len(parent.parent_sequence),
        "parent_pred": reference.pred if reference is not None else "",
        "parent_prob_AMP": reference.prob_AMP if reference is not None else "",
        "parent_logit_margin": reference.logit_margin if reference is not None else "",
        "n_variants": sum(cell.n for cell in cells),
        "n_p_levels": len(cells),
        "mean_k_requested": fmean([cell.k_requested for cell in cells]),
        "mean_k_applied": fmean([cell.k_applied for cell in cells]),
        "retention_auc": _auc([(cell.p, cell.retention) for cell in cells]),
        "retention_first_p": cells[0].retention,
        "retention_last_p": cells[-1].retention,
        "p50_retention": _p50(cells, anchor),
        "mean_prob_AMP": _weighted_mean([(cell.n, cell.mean_prob_AMP) for cell in cells]),
        "mean_delta_prob_AMP": (
            _weighted_mean([(cell.n, cell.mean_delta_prob_AMP) for cell in cells])
            if reference is not None
            else ""
        ),
        "margin_auc": _auc([(cell.p, cell.mean_logit_margin) for cell in cells]),
        "margin_slope": _slope(margins),
    }
    row.update(parent.extras)
    return row


def _curve_row(parent: MutantRecord, cell: Cell, backbone: str) -> dict:
    return {
        "parent_id": parent.parent_id,
        "scheme": cell.scheme,
        "backbone": backbone,
        "p": cell.p,
        "n_variants": cell.n,
        "k_requested": cell.k_requested,
        "k_applied": cell.k_applied,
        "retention": cell.retention,
        "mean_prob_AMP": cell.mean_prob_AMP,
        "sd_prob_AMP": cell.sd_prob_AMP,
        "mean_delta_prob_AMP": cell.mean_delta_prob_AMP,
        "mean_logit_margin": cell.mean_logit_margin,
        "sd_logit_margin": cell.sd_logit_margin,
    }


def _auc(points: list[tuple[float, float]]) -> float:
    """Trapezoidal integral over p, normalized by the tested p span."""
    span = points[-1][0] - points[0][0]
    if span == 0:
        return fmean([value for _, value in points])
    area = sum(
        (points[index + 1][0] - points[index][0])
        * (points[index + 1][1] + points[index][1])
        / 2
        for index in range(len(points) - 1)
    )
    return area / span


def _p50(cells: list[Cell], anchor: float) -> float | str:
    """Lowest p where retention crosses 0.5, interpolating from (p=0, parent call)."""
    previous_p, previous_retention = 0.0, anchor
    for cell in cells:
        if cell.retention < 0.5 <= previous_retention:
            drop = cell.retention - previous_retention
            if drop == 0:
                return cell.p
            return previous_p + (0.5 - previous_retention) * (cell.p - previous_p) / drop
        previous_p, previous_retention = cell.p, cell.retention
    return ""


def _slope(pairs: list[tuple[float, float]]) -> float | str:
    """OLS slope of logit margin against p."""
    if len(pairs) < 2:
        return ""
    mean_p = fmean([p for p, _ in pairs])
    mean_value = fmean([value for _, value in pairs])
    denominator = sum((p - mean_p) ** 2 for p, _ in pairs)
    if denominator == 0:
        return ""
    return sum((p - mean_p) * (value - mean_value) for p, value in pairs) / denominator


def _weighted_mean(pairs: list[tuple[int, float]]) -> float:
    total = sum(count for count, _ in pairs)
    return sum(count * value for count, value in pairs) / total


def rank_by_worst_scheme(summary_rows: list[dict]) -> list[dict]:
    """Rank AMP-called parents by their weakest scheme's fixed-backbone retention AUC.

    Each record carries every scheme's score and applied-k, so a peptide that scores
    well only because its k was capped stays visible.
    """
    parents: dict[str, dict] = {}
    for row in summary_rows:
        if row["backbone"] != "fixed" or row["parent_pred"] != 1:
            continue
        record = parents.setdefault(
            row["parent_id"],
            {
                "parent_id": row["parent_id"],
                "length": row["length"],
                "mean_k_requested": row["mean_k_requested"],
                "by_scheme": {},
            },
        )
        record["by_scheme"][row["scheme"]] = {
            "retention_auc": row["retention_auc"],
            "mean_k_applied": row["mean_k_applied"],
        }
    ranked = []
    for record in parents.values():
        worst = min(record["by_scheme"].items(), key=lambda item: item[1]["retention_auc"])
        ranked.append({**record, "worst_scheme": worst[0], "worst_retention_auc": worst[1]["retention_auc"]})
    ranked.sort(key=lambda record: record["worst_retention_auc"], reverse=True)
    return ranked

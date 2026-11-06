"""Collapse scored deletion variants into per-peptide robustness scores.

Deletion has no replicate axis, so a (terminus, p) cell holds exactly one variant and
the curve rows carry its raw score rather than a mean over replicates.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from ..variants.common import DeletionRecord


def summarize_parent(
    records: list[DeletionRecord],
    predictions: list,
    reference,
) -> tuple[list[dict], list[dict]]:
    """Collapse one parent's scored truncations into per-terminus and per-cell rows."""
    grouped: dict[str, list[tuple[DeletionRecord, object]]] = defaultdict(list)
    for record, prediction in zip(records, predictions):
        if record.terminus != "parent":
            grouped[record.terminus].append((record, prediction))
    if not grouped:
        return [], []

    summary_rows = []
    curve_rows = []
    for terminus, members in sorted(grouped.items()):
        members = sorted(members, key=lambda item: item[0].p)
        curve_rows.extend(_curve_row(record, prediction, reference) for record, prediction in members)
        summary_rows.append(_summary_row(terminus, members, reference))
    return summary_rows, curve_rows


def _curve_row(record: DeletionRecord, prediction, reference) -> dict:
    row = {
        "parent_id": record.parent_id,
        "terminus": record.terminus,
        "p": record.p,
        "k": record.k,
        "parent_length": record.parent_length,
        "variant_length": record.variant_length,
        "sequence": record.sequence,
        "pred": prediction.pred,
        "prob_AMP": prediction.prob_AMP,
        "logit_margin": prediction.logit_margin,
        "delta_prob_AMP": (
            prediction.prob_AMP - reference.prob_AMP if reference is not None else ""
        ),
        "delta_logit_margin": (
            prediction.logit_margin - reference.logit_margin if reference is not None else ""
        ),
    }
    row.update(record.extras)
    return row


def _summary_row(terminus: str, members: list[tuple[DeletionRecord, object]], reference) -> dict:
    records = [record for record, _ in members]
    predictions = [prediction for _, prediction in members]
    points = [(record.p, prediction.logit_margin) for record, prediction in members]
    parent = records[0]
    row = {
        "parent_id": parent.parent_id,
        "terminus": terminus,
        "sequence": parent.parent_sequence,
        "length": parent.parent_length,
        "parent_pred": reference.pred if reference is not None else "",
        "parent_prob_AMP": reference.prob_AMP if reference is not None else "",
        "parent_logit_margin": reference.logit_margin if reference is not None else "",
        "n_variants": len(members),
        "n_p_levels": len({record.p for record in records}),
        "mean_k": fmean([record.k for record in records]),
        "retention": fmean([prediction.pred for prediction in predictions]),
        "retention_first_p": predictions[0].pred,
        "retention_last_p": predictions[-1].pred,
        "mean_prob_AMP": fmean([prediction.prob_AMP for prediction in predictions]),
        "mean_delta_prob_AMP": (
            fmean([prediction.prob_AMP - reference.prob_AMP for prediction in predictions])
            if reference is not None
            else ""
        ),
        "margin_auc": _auc(points),
        "margin_slope": _slope(points),
    }
    row.update(parent.extras)
    return row


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


def _slope(points: list[tuple[float, float]]) -> float | str:
    """OLS slope of logit margin against p."""
    if len(points) < 2:
        return ""
    mean_p = fmean([p for p, _ in points])
    mean_value = fmean([value for _, value in points])
    denominator = sum((p - mean_p) ** 2 for p, _ in points)
    if denominator == 0:
        return ""
    return sum((p - mean_p) * (value - mean_value) for p, value in points) / denominator

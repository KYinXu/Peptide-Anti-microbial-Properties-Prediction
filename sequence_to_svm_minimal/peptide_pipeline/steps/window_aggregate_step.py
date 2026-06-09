"""Join window metadata with SVM / model-comparison outputs; summarize per parent sequence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from peptide_pipeline.config import RunConfig
from peptide_pipeline.context import RunContext


def _safe_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def step_window_aggregate(
    ctx: RunContext,
    cfg: RunConfig,
<<<<<<< HEAD
    svm_preds: Path | None,
=======
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
) -> None:
    if not cfg.uses_windowing():
        return

    wmap = ctx.inputs_dir / "window_map.csv"
    if not wmap.is_file():
        return

    df = pd.read_csv(wmap)
    if df.empty:
        return

    joined = df.copy()

<<<<<<< HEAD
    if svm_preds is not None and svm_preds.is_file():
        svm = pd.read_csv(svm_preds)
        if "seqIndex" in svm.columns:
            joined = joined.merge(svm, on="seqIndex", how="left")

=======
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
    comp = ctx.work_dir / "model_comparison_latest.csv"
    if comp.is_file():
        mc = pd.read_csv(comp)
        if "peptide_id" in mc.columns:
            overlap = (set(joined.columns) & set(mc.columns)) - {"peptide_id"}
            mc2 = mc.drop(columns=list(overlap), errors="ignore")
            joined = joined.merge(mc2, on="peptide_id", how="left")

    out_join = ctx.work_dir / "window_predictions_joined.csv"
    joined.to_csv(out_join, index=False)

    parent_rows: list[dict] = []
    for parent_id, g in joined.groupby("parent_id", sort=True):
        g2 = g.reset_index(drop=True)
        row: dict = {"parent_id": parent_id, "n_windows": int(len(g2))}
<<<<<<< HEAD
        if "P(+1)" in g2.columns:
            s = pd.to_numeric(g2["P(+1)"], errors="coerce")
            arr = s.to_numpy(dtype=float)
            if np.isfinite(arr).any():
                imax = int(np.nanargmax(arr))
                row["svm_max_P_plus1"] = _safe_float(s.iloc[imax])
                row["svm_mean_P_plus1"] = float(np.nanmean(arr))
                row["svm_top_seqIndex"] = int(g2.iloc[imax]["seqIndex"])
                row["svm_top_window_id"] = str(g2.iloc[imax]["window_id"])
                row["svm_top_start"] = int(g2.iloc[imax]["start"])
                row["svm_top_length"] = int(g2.iloc[imax]["length"])
                row["svm_top_sequence"] = str(g2.iloc[imax]["sequence"])
=======
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
        prob_cols = [c for c in g2.columns if c.endswith("_prob_AMP")]
        for col in prob_cols:
            s = pd.to_numeric(g2[col], errors="coerce")
            arr = s.to_numpy(dtype=float)
            if not np.isfinite(arr).any():
                continue
            imax = int(np.nanargmax(arr))
            prefix = col.replace("_prob_AMP", "")
            row[f"{prefix}_max_prob_AMP"] = _safe_float(s.iloc[imax])
            row[f"{prefix}_mean_prob_AMP"] = float(np.nanmean(arr))
            row[f"{prefix}_top_seqIndex"] = int(g2.iloc[imax]["seqIndex"])
            row[f"{prefix}_top_window_id"] = str(g2.iloc[imax]["window_id"])
        parent_rows.append(row)

    parent_df = pd.DataFrame(parent_rows)
    out_parent = ctx.work_dir / "parent_summary.csv"
    parent_df.to_csv(out_parent, index=False)

    ctx.manifest["window_aggregate"] = {
        "window_predictions_joined": str(out_join.resolve()),
        "parent_summary": str(out_parent.resolve()),
    }

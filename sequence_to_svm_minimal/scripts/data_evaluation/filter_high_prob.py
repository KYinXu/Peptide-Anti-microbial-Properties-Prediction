import argparse
import pandas as pd
import sys

SUFFIX_PROB = "_prob_AMP"


def _pick_id_columns(columns):
    return [c for c in ("peptide_id", "seqIndex") if c in columns]


def _pick_prob_columns(columns, *, contains=None):
    cols = [c for c in columns if c.endswith(SUFFIX_PROB)]
    if contains:
        cols = [c for c in cols if contains in c]
    return sorted(cols)


def _summary_columns(columns, *, prob_cols, include_sequence):
    out = []
    out.extend(_pick_id_columns(columns))
    if include_sequence and "sequence" in columns:
        out.append("sequence")
    out.extend(prob_cols)
    return out


def _truncate_text(val, max_len):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "."
    return s[: max_len - 1] + "..."


def _fmt_num(val, decimals):
    if pd.isna(val):
        return "n/a"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)[:16]


def _col_widths(headers, rows):
    w = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            w[j] = max(w[j], len(cell))
    return w


def _row_line(cells, widths, rjust=None):
    rjust = rjust or [False] * len(widths)
    segs = []
    for i in range(len(widths)):
        c = cells[i]
        w = widths[i]
        pad = c.rjust(w) if rjust[i] else c.ljust(w)
        segs.append(f" {pad} ")
    return "|" + "|".join(segs) + "|"


def _sep_line(widths):
    segs = ["-" * (w + 2) for w in widths]
    return "+" + "+".join(segs) + "+"


def _format_table(filtered_df, summary_cols, *, prob_cols, seq_max_len, decimals):
    id_cols = [c for c in ("peptide_id", "seqIndex", "sequence") if c in summary_cols]

    header_labels = []
    for c in id_cols:
        if c == "seqIndex":
            header_labels.append("seq index")
        elif c == "peptide_id":
            header_labels.append("peptide id")
        else:
            header_labels.append("sequence")

    for pc in prob_cols:
        model = pc[: -len(SUFFIX_PROB)]
        header_labels.append(f"{model}  P(AMP)")

    rows = []
    for _, row in filtered_df.iterrows():
        cells = []
        for c in id_cols:
            val = row[c]
            if c == "sequence":
                cells.append(_truncate_text(val, seq_max_len))
            else:
                cells.append("" if pd.isna(val) else str(val).strip())
        for pc in prob_cols:
            cells.append(_fmt_num(row[pc], decimals))
        rows.append(cells)

    widths = _col_widths(header_labels, rows)
    n_id = len(id_cols)
    rjust_data = [False] * n_id + [True] * (len(widths) - n_id)

    print(_sep_line(widths))
    print(_row_line(header_labels, widths))
    print(_sep_line(widths))
    for r in rows:
        print(_row_line(r, widths, rjust=rjust_data))
    print(_sep_line(widths))


def _filter_any_p_amp_over_threshold(df, prob_cols, threshold):
    if not prob_cols:
        return df.iloc[0:0]
    # Row kept if any selected P(AMP) column exceeds threshold; NaNs do not pass.
    mx = df[prob_cols].max(axis=1, skipna=True)
    return df[mx > threshold]


def main():
    parser = argparse.ArgumentParser(
        description="Print a compact summary of candidates above a probability threshold."
    )
    parser.add_argument("input_csv", help="Path to the comparison CSV file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold (default: 0.9)")
    parser.add_argument(
        "--model-contains",
        default="",
        help="Only include *_prob_AMP columns whose name contains this substring (default: empty = all models). Example: Geo for GNN/geo models only.",
    )
    parser.add_argument(
        "--include-sequence",
        action="store_true",
        help="Include the sequence column (default: off, to reduce clutter).",
    )
    parser.add_argument("--seq-max-len", type=int, default=36, help="Max characters for sequence column (default: 36)")
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for probabilities (default: 3)")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_csv}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    contains = args.model_contains if args.model_contains else None
    prob_cols = _pick_prob_columns(df.columns, contains=contains)
    if not prob_cols:
        hint = f" containing '{args.model_contains}'" if args.model_contains else ""
        print(
            f"Error: No model probability columns (*{SUFFIX_PROB}) found{hint}.",
            file=sys.stderr,
        )
        sys.exit(1)

    filtered_df = _filter_any_p_amp_over_threshold(df, prob_cols, args.threshold)
    summary_cols = _summary_columns(
        filtered_df.columns, prob_cols=prob_cols, include_sequence=args.include_sequence
    )
    if not summary_cols:
        print(f"Error: No identifier/sequence columns or model columns (*{SUFFIX_PROB}) found.", file=sys.stderr)
        sys.exit(1)

    print()
    if args.model_contains:
        print(f"  Models:  *{args.model_contains}*  (columns ending with {SUFFIX_PROB})")
    else:
        print(f"  Models:  all  (columns ending with {SUFFIX_PROB})")
    print(f"  Filter:  any model P(AMP) (*{SUFFIX_PROB})  >  {args.threshold}")
    print(f"  Rows:    {len(filtered_df)}")
    print()

    if filtered_df.empty:
        print("  (no matching rows)")
        print()
        return

    _format_table(
        filtered_df,
        summary_cols,
        prob_cols=prob_cols,
        seq_max_len=args.seq_max_len,
        decimals=args.decimals,
    )
    print()


if __name__ == "__main__":
    main()

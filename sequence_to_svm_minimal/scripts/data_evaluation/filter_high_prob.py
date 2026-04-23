import argparse
import pandas as pd
import sys

SUFFIX_PROB = "_prob_AMP"


def _summary_columns(columns):
    out = [c for c in ("peptide_id", "seqIndex", "sequence") if c in columns]
    prob_cols = sorted(c for c in columns if c.endswith(SUFFIX_PROB))
    for pc in prob_cols:
        out.append(pc)
        prefix = pc[: -len(SUFFIX_PROB)]
        conf = f"{prefix}_confidence"
        if conf in columns:
            out.append(conf)
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


def _format_table(filtered_df, summary_cols, *, seq_max_len, decimals):
    id_cols = [c for c in ("peptide_id", "seqIndex", "sequence") if c in summary_cols]
    prob_cols = sorted(c for c in summary_cols if c.endswith(SUFFIX_PROB))

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
        conf_col = f"{model}_confidence"
        if conf_col in summary_cols:
            header_labels.append(f"{model}  confidence")

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
            conf_col = f"{pc[: -len(SUFFIX_PROB)]}_confidence"
            if conf_col in summary_cols:
                cells.append(_fmt_num(row[conf_col], decimals))
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


def main():
    parser = argparse.ArgumentParser(description="Filter comparison CSV data for entries with probability > 0.9")
    parser.add_argument("input_csv", help="Path to the comparison CSV file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold (default: 0.9)")
    parser.add_argument("--prob-column", default="SVM_prob_AMP", help="Name of the probability column (default: SVM_prob_AMP)")
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

    if args.prob_column not in df.columns:
        print(f"Error: Column '{args.prob_column}' not found in the CSV. Available columns: {', '.join(df.columns)}", file=sys.stderr)
        sys.exit(1)

    filtered_df = df[df[args.prob_column] > args.threshold]
    summary_cols = _summary_columns(filtered_df.columns)
    if not summary_cols:
        print(f"Error: No identifier/sequence columns or model columns (*{SUFFIX_PROB}) found.", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"  Filter:  {args.prob_column}  >  {args.threshold}")
    print(f"  Rows:    {len(filtered_df)}")
    print()

    if filtered_df.empty:
        print("  (no matching rows)")
        print()
        return

    _format_table(filtered_df, summary_cols, seq_max_len=args.seq_max_len, decimals=args.decimals)
    print()


if __name__ == "__main__":
    main()

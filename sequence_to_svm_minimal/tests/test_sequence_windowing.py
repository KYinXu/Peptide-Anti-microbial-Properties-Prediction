"""Tests for sequence_io, windowing, and normalize parity."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptide_pipeline.esmfold_sequences import (
    ParseStats,
    iter_esmfold_sequence_file,
    summarize_esmfold_work,
)
from peptide_pipeline.sequence_io import (
    concatenate_canonical_files,
    iter_sequence_records,
    read_sequence_records,
    write_canonical,
)
from peptide_pipeline.steps.normalize import normalize_to_canonical
from peptide_pipeline.windowing import expand_records_to_windows, iter_window_slices


class TestSequenceWindowing(unittest.TestCase):
    def test_read_txt_skips_comments(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text(
                "# header\n1 ACDF\n# skip\n2 EFGH\n",
                encoding="utf-8",
            )
            rec = read_sequence_records(p)
            self.assertEqual(len(rec), 2)
            self.assertEqual(rec[0], ("1", "ACDF"))
            self.assertEqual(rec[1], ("2", "EFGH"))

    def test_read_txt_skips_nonstandard_and_x(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text(
                "1 ACDF\n2 ACX\n3 EFGH\n4 JJJ\n",
                encoding="utf-8",
            )
            inv = {}
            rec = read_sequence_records(p, invalid_stats=inv)
            self.assertEqual(rec, [("1", "ACDF"), ("3", "EFGH")])
            self.assertEqual(inv.get("n_skipped_invalid"), 2)

    def test_expand_windows_count_and_ids(self) -> None:
        rec = [("p1", "ABCDEF")]
        out, meta = expand_records_to_windows(rec, min_len=3, max_len=4, stride=1)
        # len 3: starts 0,1,2,3 = 4; len 4: starts 0,1,2 = 7
        self.assertEqual(len(out), 7)
        self.assertEqual(len(meta), 7)
        self.assertEqual(meta[0].seq_index, 1)
        self.assertEqual(meta[0].peptide_id, "SEQ_1")
        self.assertEqual(meta[0].window_id, "p1__s0__l3")
        self.assertEqual(out[0], ("1", "ABC"))

    def test_iter_window_slices_stride(self) -> None:
        s = list(iter_window_slices("ABCDE", min_len=3, max_len=3, stride=2))
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0], (0, 3, "ABC"))
        self.assertEqual(s[1], (2, 3, "CDE"))

    def test_normalize_matches_read_then_filter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            inp = t / "in.txt"
            out = t / "out.txt"
            inp.write_text("1 WWW\n", encoding="utf-8")
            st = normalize_to_canonical(inp, out, min_len=3, max_len=None)
            self.assertEqual(st["n_written"], 1)
            self.assertEqual(st["n_skipped_len"], 0)
            self.assertEqual(st.get("n_skipped_invalid", 0), 0)

    def test_normalize_skips_invalid_aa(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            inp = t / "in.txt"
            out = t / "out.txt"
            inp.write_text("1 WWW\n2 ACX\n", encoding="utf-8")
            st = normalize_to_canonical(inp, out, min_len=3, max_len=None)
            self.assertEqual(st["n_written"], 1)
            self.assertEqual(st["n_skipped_invalid"], 1)
            self.assertEqual(out.read_text(encoding="utf-8").strip(), "1 WWW")

    def test_write_canonical_from_generator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "c.txt"

            def gen():
                yield "1", "ACDE"
                yield "2", "FGHI"

            write_canonical(out, gen())
            self.assertEqual(out.read_text(encoding="utf-8"), "1 ACDE\n2 FGHI\n")

    def test_concatenate_canonical_files_streams(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            a = t / "a.txt"
            b = t / "b.txt"
            dest = t / "out.txt"
            a.write_text("1 AAAA\n", encoding="utf-8")
            b.write_text("2 CCCC", encoding="utf-8")
            concatenate_canonical_files(dest, (a, b))
            self.assertEqual(dest.read_text(encoding="utf-8"), "1 AAAA\n2 CCCC\n")

    def test_iter_sequence_records_is_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text("1 ACDE\n2 FGHI\n", encoding="utf-8")
            it = iter_sequence_records(p)
            self.assertEqual(next(it), ("1", "ACDE"))
            self.assertEqual(next(it), ("2", "FGHI"))

    def test_esmfold_stream_ids_and_invalid_skip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text(
                "1 ACDE\nACX\nFGHI\nQ6FI13 KLMN\n",
                encoding="utf-8",
            )
            stats = ParseStats()
            recs = list(iter_esmfold_sequence_file(p, label=0, prefix="SEQ", stats=stats))
            self.assertEqual(stats.n_skipped_invalid, 1)
            self.assertEqual(recs[0], ("SEQ_1", "1", "ACDE", 0))
            self.assertEqual(recs[1][0], "SEQ_2")
            self.assertEqual(recs[1][2], "FGHI")
            self.assertEqual(recs[2][0], "Q6FI13")
            self.assertEqual(recs[2][2], "KLMN")

    def test_summarize_esmfold_work_does_not_require_materialized_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.txt"
            p.write_text("1 ACDE\n2 AAAAAAAAAA\n3 FGHI\n", encoding="utf-8")
            summary = summarize_esmfold_work(
                iter_esmfold_sequence_file(p, 0, "SEQ"),
                completed_ids={"SEQ_1"},
                max_length=5,
            )
            self.assertEqual(summary.n_valid, 3)
            self.assertEqual(summary.n_remaining, 2)
            self.assertEqual(summary.n_foldable, 1)


if __name__ == "__main__":
    unittest.main()

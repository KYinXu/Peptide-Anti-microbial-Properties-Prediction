"""Tests for sequence_io, windowing, and normalize parity."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptide_pipeline.sequence_io import read_sequence_records
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
            self.assertEqual(rec[0][0], "1")
            self.assertEqual(rec[1][0], "2")

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


if __name__ == "__main__":
    unittest.main()

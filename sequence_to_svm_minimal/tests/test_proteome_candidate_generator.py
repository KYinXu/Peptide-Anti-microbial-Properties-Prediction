"""Tests for the pepsickle proteome candidate generator."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proteome_candidate_generator.candidates import (
    generate_candidates,
    hydrophobicity,
    net_charge,
    write_candidates_csv,
    write_pipeline_txt,
)
from proteome_candidate_generator.cleavage import parse_pepsickle_tsv, union_sites
from proteome_candidate_generator.fasta import ProteinRecord, read_valid_proteins
from proteome_candidate_generator.pepsickle_runner import BatchFile, build_task


class TestProteomeCandidateGenerator(unittest.TestCase):
    def test_pepsickle_command_omits_threshold_flag(self) -> None:
        task = build_task(
            BatchFile(index=1, path=Path("batch_00001.fasta"), n_records=1),
            Path("out"),
            model_name="constitutive",
            pepsickle_bin="pepsickle",
        )
        self.assertNotIn("-t", task.command)
        self.assertEqual(task.command[:5], ["pepsickle", "-f", "batch_00001.fasta", "-m", "in-vitro"])

    def test_fasta_filter_skips_nonstandard_residues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "input.fasta"
            path.write_text(">p1\nACDE\n>p2\nACXU\n>p3\nRRKK\n", encoding="utf-8")
            records, stats = read_valid_proteins(path)
        self.assertEqual(records, [ProteinRecord("p1", "ACDE"), ProteinRecord("p3", "RRKK")])
        self.assertEqual(stats["skipped_invalid"], 1)

    def test_pepsickle_tsv_union_uses_max_probability(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first = Path(td) / "const.tsv"
            second = Path(td) / "immuno.tsv"
            first.write_text(
                "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                "8\tL\t0.60\tTrue\tp1\n"
                "9\tL\t0.20\tFalse\tp1\n",
                encoding="utf-8",
            )
            second.write_text(
                "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                "8\tL\t0.80\tTrue\tp1\n"
                "12\tG\t0.70\tTrue\tp1\n",
                encoding="utf-8",
            )
            groups = [
                parse_pepsickle_tsv(first, model_name="constitutive", threshold=0.5),
                parse_pepsickle_tsv(second, model_name="immunoproteasome", threshold=0.5),
            ]
        merged = union_sites(groups, {"p1": 20})
        self.assertEqual(merged["p1"].sites, [8, 12])
        self.assertEqual(merged["p1"].site_probabilities[8], 0.8)

    def test_candidate_generation_filters_dedupes_and_scores(self) -> None:
        records = [
            ProteinRecord("p1", "RRKAAAILLLGGDD"),
            ProteinRecord("p2", "RRKAAAILLLGGEE"),
        ]
        groups = [
            parse_pepsickle_tsv(
                _write_tsv(
                    "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                    "8\tL\t0.90\tTrue\tp1\n"
                    "8\tL\t0.95\tTrue\tp2\n"
                ),
                model_name="constitutive",
                threshold=0.5,
            )
        ]
        sites = union_sites(groups, {"p1": len(records[0].sequence), "p2": len(records[1].sequence)})
        candidates, stats = generate_candidates(
            records,
            sites,
            min_len=8,
            max_len=8,
            min_charge=2,
            min_hydrophobicity=0.30,
            top_n=10,
            include_terminal_boundaries=True,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].sequence, "RRKAAAIL")
        self.assertEqual(candidates[0].peptide_id, "PEP_1")
        self.assertEqual(candidates[0].start, 0)
        self.assertEqual(candidates[0].end, 8)
        self.assertEqual(candidates[0].net_charge, 3)
        self.assertGreater(candidates[0].rank_score, 0)
        self.assertEqual(stats.duplicate_sequences, 1)

    def test_metrics_and_output_formatting(self) -> None:
        self.assertEqual(net_charge("RRKDDEE"), -1)
        self.assertAlmostEqual(hydrophobicity("AILR"), 0.75)
        records = [ProteinRecord("p1", "RRKAAAIL")]
        sites = union_sites(
            [parse_pepsickle_tsv(_write_tsv("position\tresidue\tcleav_prob\tcleaved\tprotein_id\n8\tL\t0.9\tTrue\tp1\n"), model_name="constitutive", threshold=0.5)],
            {"p1": 8},
        )
        candidates, _ = generate_candidates(
            records,
            sites,
            min_len=8,
            max_len=8,
            min_charge=2,
            min_hydrophobicity=0.30,
            top_n=None,
            include_terminal_boundaries=True,
        )
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "candidates.csv"
            txt_path = Path(td) / "candidates.txt"
            write_candidates_csv(candidates, csv_path)
            write_pipeline_txt(candidates, txt_path)
            csv_text = csv_path.read_text(encoding="utf-8")
            txt_text = txt_path.read_text(encoding="utf-8")
        self.assertIn("peptide_id,sequence,source_protein_id", csv_text)
        self.assertEqual(txt_text.strip(), "PEP_1 RRKAAAIL")


def _write_tsv(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()

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
    generate_paper_candidates,
    has_cationic_cterm,
    hydrophobicity,
    net_charge,
    write_candidates_csv,
    write_pipeline_txt,
)
from proteome_candidate_generator.cli import _layout
from proteome_candidate_generator.cli import _build_paper_scorer
from proteome_candidate_generator.cli import _run_build_candidates
from proteome_candidate_generator.cleavage import parse_pepsickle_tsv, union_sites
from proteome_candidate_generator.fasta import ProteinRecord, read_valid_proteins
from proteome_candidate_generator.pddp_scoring import (
    compute_nonzero_mean_threshold,
    is_mapp_database,
    load_known_amp_sequences,
    load_mapp_reference_scorer,
    load_score_matrix,
)
from proteome_candidate_generator.pepsickle_runner import BatchFile, build_task


class TestProteomeCandidateGenerator(unittest.TestCase):
    def test_layout_places_artifacts_under_generated_subdirectory(self) -> None:
        paths = _layout(Path("data/proteomes/paper_pddp"))
        self.assertEqual(paths["root"].name, "generated")
        self.assertEqual(paths["root"].parent.name, "paper_pddp")
        self.assertEqual(paths["final_txt"].name, "final_candidates.txt")

    def test_build_candidates_requires_generated_batches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fasta = root / "input.fasta"
            fasta.write_text(">p1\nRRKAAAILLLGGDD\n", encoding="utf-8")

            class Args:
                input = fasta
                output_dir = root / "run"
                limit_proteins = None
                no_progress = True
                threshold = 0.5

            with self.assertRaisesRegex(FileNotFoundError, "Run the `preprocess` step first"):
                _run_build_candidates(Args())

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

    def test_paper_scoring_threshold_excludes_zero_scores(self) -> None:
        matrix = _write_score_matrix({aa: 1.0 for aa in "ACDEFGHIKLMNPQRSTVWY"})
        score_matrix = load_score_matrix(matrix)
        with tempfile.TemporaryDirectory() as td:
            known = Path(td) / "known.txt"
            known.write_text("amp1 AAAAA\namp2 CCCCC\n", encoding="utf-8")
            sequences = load_known_amp_sequences([known])
        self.assertEqual(sequences, ["AAAAA", "CCCCC"])
        self.assertEqual(compute_nonzero_mean_threshold(sequences, score_matrix), 5.0)

    def test_paper_candidate_generation_score_overlap_and_cterm(self) -> None:
        records = [ProteinRecord("p1", "MMMMMMMMMMAAAAAAAAAAKCCCCCCCCCCR")]
        sites = union_sites(
            [
                parse_pepsickle_tsv(
                    _write_tsv(
                        "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                        "10\tM\t0.9\tTrue\tp1\n"
                        "20\tA\t0.9\tTrue\tp1\n"
                        "32\tR\t0.9\tTrue\tp1\n"
                    ),
                    model_name="constitutive",
                    threshold=0.5,
                )
            ],
            {"p1": len(records[0].sequence)},
        )
        matrix = load_score_matrix(_write_score_matrix({"A": 1.0, "C": 2.0, "K": 3.0, "R": 4.0}))
        candidates, stats = generate_paper_candidates(
            records,
            sites,
            min_len=10,
            max_len=50,
            scorer=matrix,
            score_threshold=10.0,
            require_cationic_cterm=True,
            cationic_cterm_residues="KR",
            overlap_policy="top_score",
            include_terminal_boundaries=False,
        )
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0].sequence.endswith("R"))
        self.assertEqual(candidates[0].peptide_id, "PEP_1")
        self.assertGreater(candidates[0].pddp_score, 10.0)
        self.assertTrue(candidates[0].passes_cationic_cterm)
        self.assertEqual(stats.score_filtered, 1)
        self.assertEqual(stats.overlap_removed, 1)
        self.assertEqual(stats.cterm_filtered, 0)
        self.assertTrue(has_cationic_cterm("AAAAK", "KR"))

    def test_paper_candidate_generation_can_keep_all_overlaps(self) -> None:
        records = [ProteinRecord("p1", "MMMMMMMMMMAAAAAAAAAAKCCCCCCCCCCR")]
        sites = union_sites(
            [
                parse_pepsickle_tsv(
                    _write_tsv(
                        "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                        "10\tM\t0.9\tTrue\tp1\n"
                        "20\tA\t0.9\tTrue\tp1\n"
                        "32\tR\t0.9\tTrue\tp1\n"
                    ),
                    model_name="constitutive",
                    threshold=0.5,
                )
            ],
            {"p1": len(records[0].sequence)},
        )
        matrix = load_score_matrix(_write_score_matrix({"A": 1.0, "C": 2.0, "K": 3.0, "R": 4.0}))
        candidates, stats = generate_paper_candidates(
            records,
            sites,
            min_len=10,
            max_len=50,
            scorer=matrix,
            score_threshold=10.0,
            require_cationic_cterm=False,
            cationic_cterm_residues="KR",
            overlap_policy="keep_all",
            include_terminal_boundaries=False,
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(stats.overlap_removed, 0)

    def test_paper_candidate_streaming_matches_in_memory(self) -> None:
        records = [ProteinRecord("p1", "MMMMMMMMMMAAAAAAAAAAKCCCCCCCCCCR")]
        sites = union_sites(
            [
                parse_pepsickle_tsv(
                    _write_tsv(
                        "position\tresidue\tcleav_prob\tcleaved\tprotein_id\n"
                        "10\tM\t0.9\tTrue\tp1\n"
                        "20\tA\t0.9\tTrue\tp1\n"
                        "32\tR\t0.9\tTrue\tp1\n"
                    ),
                    model_name="constitutive",
                    threshold=0.5,
                )
            ],
            {"p1": len(records[0].sequence)},
        )
        matrix = load_score_matrix(_write_score_matrix({"A": 1.0, "C": 2.0, "K": 3.0, "R": 4.0}))
        kwargs = dict(
            records=records,
            sites_by_protein=sites,
            min_len=10,
            max_len=50,
            scorer=matrix,
            score_threshold=10.0,
            require_cationic_cterm=False,
            cationic_cterm_residues="KR",
            overlap_policy="keep_all",
            include_terminal_boundaries=False,
        )
        in_memory, in_memory_stats = generate_paper_candidates(**kwargs)
        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            streamed, streamed_stats = generate_paper_candidates(
                **kwargs,
                finalize_outputs=(output_dir / "final_candidates", output_dir / "final_candidates.txt", "csv"),
            )
            self.assertEqual(streamed, [])
            self.assertEqual(streamed_stats, in_memory_stats)
            txt_lines = (output_dir / "final_candidates.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                txt_lines,
                [f"{candidate.peptide_id} {candidate.sequence}" for candidate in in_memory],
            )

    def test_mapp_database_can_score_exact_sequence_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mapp = Path(td) / "MAPP_database.csv"
            mapp.write_text(
                "Sequence,Leading razor protein,Start position,End position,Gene names,Treatment\n"
                "AAAAK,P1,1,5,GENE,Wt-alpha6:0;KO-alpha6:25;KO-iso:5\n"
                "CCCCR,P2,1,5,GENE,Wt-alpha6:0;KO-alpha6:0;KO-iso:0\n",
                encoding="utf-8",
            )
            scorer = load_mapp_reference_scorer(mapp)
            self.assertTrue(is_mapp_database(mapp))
        self.assertEqual(scorer.score_sequence("AAAAK"), 30.0)
        self.assertEqual(scorer.score_sequence("CCCCR"), 0.0)
        self.assertEqual(scorer.score_sequence("RRRRR"), 0.0)



def _write_tsv(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8")
    with handle:
        handle.write(text)
    return Path(handle.name)


def _write_score_matrix(overrides: dict[str, float]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    aas = "ACDEFGHIKLMNPQRSTVWY"
    with handle:
        handle.write("position," + ",".join(aas) + "\n")
        for pos in range(1, 11):
            values = [str(overrides.get(aa, 0.0)) for aa in aas]
            handle.write(f"{pos}," + ",".join(values) + "\n")
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()

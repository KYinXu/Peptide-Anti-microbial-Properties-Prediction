"""Unit tests for length_analysis loader, provenance, and variant generation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from length_analysis.data_loader import SubsetSequenceDataset
from length_analysis.provenance import load_candidate_index, load_fasta_sequences, recover_parents
from length_analysis.variants.common import ParentRecord, SubsetRecord
from length_analysis.variants.generation import generate_variants, generate_variants_for_parent


class TestSubsetLoader(unittest.TestCase):
    def test_loads_normalized_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subset.csv"
            path.write_text("id,sequence,note\nPEP_1,AAA,x\n", encoding="utf-8")
            dataset = SubsetSequenceDataset.from_csv(path)
            records = list(dataset.records())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].id, "PEP_1")
            self.assertEqual(records[0].sequence, "AAA")
            self.assertEqual(records[0].extras["note"], "x")


class TestVariantGeneration(unittest.TestCase):
    def _parent(self) -> ParentRecord:
        return ParentRecord(
            id="PEP_1",
            sequence="AAAA",
            source_protein_id="sp|P1|TEST",
            start=2,
            end=6,
            left_flank="XY",
            right_flank="ZW",
        )

    def test_includes_original_and_cartesian_product(self) -> None:
        variants = list(
            generate_variants_for_parent(self._parent(), max_flank_fraction=0.5, stride=1)
        )
        # max_ext = floor(0.5*4)=2, left_limit=2, right_limit=2 -> 3x3 = 9
        self.assertEqual(len(variants), 9)
        by_id = {variant.variant_id: variant for variant in variants}
        self.assertEqual(by_id["PEP_1_L0_R0"].sequence, "AAAA")
        self.assertEqual(by_id["PEP_1_L2_R1"].sequence, "XYAAAAZ")
        self.assertEqual(by_id["PEP_1_L1_R2"].sequence, "YAAAAZW")

    def test_clamps_to_available_flank(self) -> None:
        parent = ParentRecord(
            id="PEP_1",
            sequence="AAAAAAAA",
            source_protein_id="p",
            start=1,
            end=9,
            left_flank="A",
            right_flank="",
        )
        variants = list(generate_variants([parent], max_flank_fraction=0.5, stride=1))
        # max_ext=4, left_limit=1, right_limit=0 -> 2x1 = 2
        self.assertEqual(len(variants), 2)
        self.assertEqual({v.left_added for v in variants}, {0, 1})
        self.assertEqual({v.right_added for v in variants}, {0})

    def test_stride_skips_values(self) -> None:
        variants = list(
            generate_variants_for_parent(self._parent(), max_flank_fraction=0.5, stride=2)
        )
        pairs = {(v.left_added, v.right_added) for v in variants}
        self.assertEqual(pairs, {(0, 0), (0, 2), (2, 0), (2, 2)})


class TestProvenance(unittest.TestCase):
    def test_recover_parents_from_table_and_fasta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "final_candidates.csv"
            fasta = root / "proteome.fasta"
            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "peptide_id",
                        "sequence",
                        "source_protein_id",
                        "start",
                        "end",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "peptide_id": "PEP_1",
                        "sequence": "AAAA",
                        "source_protein_id": "sp|P1|TEST",
                        "start": "2",
                        "end": "6",
                    }
                )
            fasta.write_text(">sp|P1|TEST desc\nXXAAAAZZ\n", encoding="utf-8")

            index = load_candidate_index(candidates)
            parents = recover_parents(
                [SubsetRecord(id="PEP_1", sequence="AAAA")],
                index,
                fasta,
            )
            self.assertEqual(len(parents), 1)
            self.assertEqual(parents[0].left_flank, "XX")
            self.assertEqual(parents[0].right_flank, "ZZ")
            self.assertEqual(parents[0].source_protein_id, "sp|P1|TEST")

    def test_sequence_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "final_candidates.csv"
            fasta = root / "proteome.fasta"
            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "peptide_id",
                        "sequence",
                        "source_protein_id",
                        "start",
                        "end",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "peptide_id": "PEP_1",
                        "sequence": "AAAA",
                        "source_protein_id": "sp|P1|TEST",
                        "start": "2",
                        "end": "6",
                    }
                )
            fasta.write_text(">sp|P1|TEST\nXXAAAAZZ\n", encoding="utf-8")
            index = load_candidate_index(candidates)
            with self.assertRaises(ValueError):
                recover_parents(
                    [SubsetRecord(id="PEP_1", sequence="BBBB")],
                    index,
                    fasta,
                )

    def test_missing_protein_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidates = root / "final_candidates.csv"
            fasta = root / "proteome.fasta"
            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "peptide_id",
                        "sequence",
                        "source_protein_id",
                        "start",
                        "end",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "peptide_id": "PEP_1",
                        "sequence": "AAAA",
                        "source_protein_id": "sp|MISSING|X",
                        "start": "0",
                        "end": "4",
                    }
                )
            fasta.write_text(">sp|P1|TEST\nAAAA\n", encoding="utf-8")
            index = load_candidate_index(candidates)
            with self.assertRaises(KeyError):
                recover_parents(
                    [SubsetRecord(id="PEP_1", sequence="AAAA")],
                    index,
                    fasta,
                )

    def test_load_fasta_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fasta = Path(temporary) / "p.fasta"
            fasta.write_text(
                ">a\nAAA\n>b\nBBB\n>c\nCCC\n",
                encoding="utf-8",
            )
            sequences = load_fasta_sequences(fasta, ["b"])
            self.assertEqual(sequences, {"b": "BBB"})


class TestResultsIo(unittest.TestCase):
    def test_writer_includes_extras_and_run_id(self) -> None:
        from length_analysis.scoring.results_io import LengthPredictionCsvWriter

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.csv"
            with LengthPredictionCsvWriter(path, extra_columns=["note"], run_id="run1") as writer:
                writer.write_row(
                    {
                        "variant_id": "PEP_1_L0_R0",
                        "parent_id": "PEP_1",
                        "source_protein_id": "p",
                        "start": 0,
                        "end": 4,
                        "left_added": 0,
                        "right_added": 0,
                        "core_length": 4,
                        "variant_length": 4,
                        "sequence": "AAAA",
                        "pred": 1,
                        "prob_AMP": 0.9,
                        "confidence": 0.9,
                        "logit_AMP": 1.0,
                        "logit_nonAMP": 0.0,
                        "logit_margin": 1.0,
                        "score_z": 0.0,
                        "note": "x",
                    }
                )
            text = path.read_text(encoding="utf-8")
            self.assertIn("run_id", text)
            self.assertIn("note", text)
            self.assertIn("PEP_1_L0_R0", text)


class TestRealProteomeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.candidates = (
            root
            / "sequence_to_svm_minimal/data/proteomes/pane_filtered_candidates/generated/7-11-465k/final_candidates.csv"
        )
        cls.fasta = (
            root
            / "sequence_to_svm_minimal/data/proteomes/uniprotkb_UP000005640_AND_model_organis_2026_06_27.fasta"
        )
        cls.peptide_id = "PEP_40497"

    def test_recover_and_expand_real_candidate(self) -> None:
        if not self.candidates.is_file() or not self.fasta.is_file():
            self.skipTest("Proteome candidate artifacts not present on disk")

        sequence = None
        reader = pd.read_csv(
            self.candidates,
            usecols=["peptide_id", "sequence"],
            chunksize=100_000,
        )
        try:
            for chunk in reader:
                hit = chunk[chunk["peptide_id"].astype(str) == self.peptide_id]
                if len(hit):
                    sequence = str(hit.iloc[0]["sequence"])
                    break
        finally:
            reader.close()
        if sequence is None:
            self.skipTest(f"{self.peptide_id} not found in candidates table")

        index = load_candidate_index(self.candidates)
        parents = recover_parents(
            [SubsetRecord(id=self.peptide_id, sequence=sequence)],
            index,
            self.fasta,
        )
        self.assertEqual(len(parents), 1)
        self.assertTrue(parents[0].left_flank)
        self.assertTrue(parents[0].right_flank)
        variants = list(generate_variants(parents, max_flank_fraction=0.5, stride=1))
        self.assertGreater(len(variants), 1)
        self.assertEqual(variants[0].variant_id, f"{self.peptide_id}_L0_R0")
        self.assertEqual(variants[0].sequence, sequence)


if __name__ == "__main__":
    unittest.main()

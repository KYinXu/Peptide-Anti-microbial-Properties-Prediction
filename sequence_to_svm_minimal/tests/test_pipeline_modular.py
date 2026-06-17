"""Tests for peptide_pipeline config and dry-run runner (stdlib only)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peptide_pipeline.config import RunConfig, default_work_dir
from peptide_pipeline.manifest_paths import (
    gnn_final_training_paths_from_work_dir,
    resolve_generated_workspace,
)
from peptide_pipeline.runner import run_pipeline
<<<<<<< HEAD
<<<<<<< HEAD
=======
from peptide_pipeline.sequence_io import read_sequence_records
from peptide_pipeline.steps.normalize import normalize_to_canonical
>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16


class TestPipelineModular(unittest.TestCase):
    def test_default_work_dir(self) -> None:
        if os.name == "nt":
            inp = Path(r"C:\fake\user\data\seqs.txt")
            self.assertEqual(default_work_dir(inp), Path(r"C:\fake\user\data\generated"))
        else:
            inp = Path("/tmp/data/seqs.txt")
            self.assertEqual(default_work_dir(inp), Path("/tmp/data/generated"))

    def test_run_pipeline_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            inp = t / "seqs.txt"
            inp.write_text("ACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
            out = t / "workspace"
            cfg = RunConfig(mode="blind", input_path=inp, work_dir=out, dry_run=True)
            self.assertEqual(run_pipeline(cfg), 0)

    def test_run_pipeline_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(mode="blind", input_path=Path(td) / "missing.txt", dry_run=True)
            self.assertEqual(run_pipeline(cfg), 1)

<<<<<<< HEAD
<<<<<<< HEAD
=======
    def test_read_sequence_records_supports_name_seq_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seqs.csv"
            path.write_text(
                "name,seq,,,,\n"
                "pep1,acdef,,,,\n"
                "pep2,RRKX,,,,\n"
                "pep3, GIGKFLHSAK ,,,,\n",
                encoding="utf-8",
            )
            invalid: dict = {}
            records = read_sequence_records(path, invalid_stats=invalid)
        self.assertEqual(records, [("pep1", "ACDEF"), ("pep3", "GIGKFLHSAK")])
        self.assertEqual(invalid["n_skipped_invalid"], 1)

    def test_normalize_to_canonical_reports_csv_format(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inp = root / "seqs.csv"
            out = root / "canonical.txt"
            inp.write_text("name,seq\npep1,ACDEFG\n", encoding="utf-8")
            stats = normalize_to_canonical(inp, out)
            self.assertEqual(stats["format"], "csv")
            self.assertEqual(stats["n_written"], 1)
            self.assertEqual(out.read_text(encoding="utf-8"), "pep1 ACDEFG\n")

>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    def test_resolve_generated_workspace_parent_or_nested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gen = root / "generated"
            gen.mkdir()
            (gen / "pipeline_manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_generated_workspace(gen), gen.resolve())
            self.assertEqual(resolve_generated_workspace(root), gen.resolve())

<<<<<<< HEAD
<<<<<<< HEAD
=======
    def test_run_pipeline_features_only_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            inp = t / "seqs.txt"
            inp.write_text("ACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
            out = t / "workspace"
            cfg = RunConfig(
                mode="blind",
                input_path=inp,
                work_dir=out,
                dry_run=True,
                features_only=True,
            )
            self.assertEqual(run_pipeline(cfg), 0)

    def test_run_pipeline_features_only_windowed_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            inp = t / "seqs.txt"
            inp.write_text("ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY\n", encoding="utf-8")
            out = t / "workspace"
            cfg = RunConfig(
                mode="blind",
                input_path=inp,
                work_dir=out,
                dry_run=True,
                features_only=True,
                window_min_len=10,
                window_max_len=15,
                window_stride=5,
            )
            self.assertEqual(run_pipeline(cfg), 0)

    def test_compare_models_windowed_inherits_base(self) -> None:
        from configs.load_config import load_compare_models_config, load_compare_models_windowed_config

        base = load_compare_models_config()
        windowed = load_compare_models_windowed_config()
        for key in ("svm_pkl", "svm_z_file", "architecture", "esm_only_pt", "node_feature_groups"):
            self.assertEqual(windowed[key], base[key])
        self.assertEqual(windowed.get("qsar_mode"), "parent")

>>>>>>> 020bd7d (SVM window config fix, pddp lower filter run and misc additions to data)
=======
>>>>>>> f255595470cd527a24de3b686587977fb372fb16
    def test_gnn_final_paths_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            g = wd / "geo.csv"
            g.write_text("a\n", encoding="utf-8")
            (wd / "pipeline_manifest.json").write_text(
                json.dumps(
                    {
                        "geometric_features": str(g),
                        "structures_dir": str(wd / "pdb"),
                        "qsar12_descriptors": str(wd / "q.csv"),
                        "esm2_embeddings": str(wd / "e.csv"),
                    }
                ),
                encoding="utf-8",
            )
            p = gnn_final_training_paths_from_work_dir(wd)
            self.assertEqual(p["csv_path"], str(g.resolve()))
            self.assertTrue(p["pdb_dir"].endswith("pdb"))
            self.assertIn("esm2_residue_dir", p)
            self.assertTrue(str(p["esm2_residue_dir"]).replace("\\", "/").endswith("esm2_per_residue"))


if __name__ == "__main__":
    unittest.main()

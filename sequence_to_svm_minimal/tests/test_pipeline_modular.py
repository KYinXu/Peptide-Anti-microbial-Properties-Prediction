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
from peptide_pipeline.manifest_paths import gnn_final_training_paths_from_work_dir
from peptide_pipeline.runner import run_pipeline


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
            cfg = RunConfig(input_path=inp, work_dir=out, dry_run=True)
            self.assertEqual(run_pipeline(cfg), 0)

    def test_run_pipeline_missing_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(input_path=Path(td) / "missing.txt", dry_run=True)
            self.assertEqual(run_pipeline(cfg), 1)

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


if __name__ == "__main__":
    unittest.main()

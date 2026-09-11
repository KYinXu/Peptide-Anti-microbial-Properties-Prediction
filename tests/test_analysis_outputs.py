from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from analysis_outputs import RunCsvWriter, RunOutput, RunSpec, execute_csv_run
from sequence_analysis.models import SvmSequencePrediction
from sequence_analysis.record_predictions import write_predictions
from sequence_analysis.utils.data_loader import SequenceRecord
from window_sequence_analysis.sliding_windows.raw_results_io import RawWindowCsvWriter
from window_sequence_analysis.sliding_windows.results_io import ProfileCsvWriter


NOW = datetime(2026, 9, 9, 18, 11, tzinfo=timezone.utc)


def make_spec(tmp_path: Path, input_path: Path, model_path: Path) -> RunSpec:
    return RunSpec(
        output_root=tmp_path / "results",
        output_filename="predictions.csv",
        runner="tests.example",
        model="svm",
        config={"window": {"min_len": 10, "max_len": 35}},
        inputs={"sequences": input_path},
        model_files={"checkpoint": model_path},
    )


def write_one_row(run: RunOutput) -> int:
    with RunCsvWriter(run.csv_path, ["id", "score"], run.run_id) as writer:
        writer.write_row({"id": "seq-1", "score": 0.75})
    return 1


class TestAnalysisOutputs(unittest.TestCase):
    def test_execute_csv_run_writes_bound_csv_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, model_path = self._write_inputs(root)
            run = execute_csv_run(make_spec(root, input_path, model_path), write_one_row, now=NOW)

            self.assertEqual(run.directory.name, "2026-09-09_18-11")
            with run.csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"run_id": run.run_id, "id": "seq-1", "score": "0.75"}])

            manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["run_id"], run.run_id)
            self.assertEqual(manifest["config"]["window"]["min_len"], 10)
            self.assertTrue(manifest["inputs"]["sequences"]["sha256"])
            self.assertTrue(manifest["model"]["files"]["checkpoint"]["sha256"])
            self.assertEqual(manifest["output"]["path"], "predictions.csv")
            self.assertEqual(manifest["output"]["row_count"], 1)
            self.assertTrue(manifest["output"]["sha256"])

    def test_runs_in_same_minute_do_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, model_path = self._write_inputs(root)
            spec = make_spec(root, input_path, model_path)

            first = execute_csv_run(spec, write_one_row, now=NOW)
            second = execute_csv_run(spec, write_one_row, now=NOW)

            self.assertEqual(first.run_id, "2026-09-09_18-11")
            self.assertEqual(second.run_id, "2026-09-09_18-11_02")
            second_csv = second.csv_path.read_text(encoding="utf-8").replace(second.run_id, first.run_id)
            self.assertEqual(first.csv_path.read_text(encoding="utf-8"), second_csv)

    def test_failed_run_retains_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path, model_path = self._write_inputs(root)

            def fail(_run: RunOutput) -> int:
                raise RuntimeError("inference failed")

            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                execute_csv_run(make_spec(root, input_path, model_path), fail, now=NOW)

            manifest_path = root / "results" / "runs" / "2026-09-09_18-11" / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"], {"type": "RuntimeError", "message": "inference failed"})

    def test_analysis_writers_include_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_csv = root / "sequence.csv"
            profile_csv = root / "profile.csv"
            raw_csv = root / "raw.csv"
            record = SequenceRecord(id="seq-1", sequence="ACDE", extras={"name": "example"})
            prediction = SvmSequencePrediction(id="seq-1", sequence="ACDE", prediction=1, sigma=0.5, p_amp=0.7)

            write_predictions(sequence_csv, ["name"], [record], [prediction], run_id="run-1")
            with ProfileCsvWriter(profile_csv, ["name"], "run-1") as writer:
                writer.write_row({"id": "seq-1", "name": "example"})
            with RawWindowCsvWriter(raw_csv, "run-1") as writer:
                writer.write_row({"id": "seq-1", "prediction": 1, "sigma": 0.5, "p_amp": 0.7})

            for path in (sequence_csv, profile_csv, raw_csv):
                with path.open(newline="", encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["run_id"], "run-1")

    @staticmethod
    def _write_inputs(root: Path) -> tuple[Path, Path]:
        input_path = root / "input.csv"
        model_path = root / "model.pkl"
        input_path.write_text("id,sequence\nseq-1,ACDE\n", encoding="utf-8")
        model_path.write_bytes(b"model")
        return input_path, model_path

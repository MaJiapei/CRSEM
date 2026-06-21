from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from CRSEM.calibration_reporting import CalibrationReporter, CalibrationTracker
from CRSEM.calibration_api import save_calibration_results
from CRSEM.contracts import ParameterBatch


class CalibrationReportingTests(unittest.TestCase):
    def test_tracker_records_iteration_history(self):
        tracker = CalibrationTracker()
        metrics = {"sse": 1.5, "nse": 0.8}
        iteration = tracker.record(np.array([1.0, 2.0]), 0.2, metrics)
        self.assertEqual(iteration, 1)
        self.assertEqual(tracker.iteration_history, [1])
        self.assertEqual(len(tracker.candidate_records), 1)

    def test_reporter_console_mode_does_not_require_visualizer(self):
        tracker = CalibrationTracker()
        tracker.record(np.array([1.0]), 0.3, {"sse": 1.0, "nse": 0.7})
        reporter = CalibrationReporter(
            model_type="rusle",
            plot_progress=False,
            plot_every_iters=10,
            plot_min_interval_s=0.1,
            diagnostics_provider=lambda run_output: {"R": 1.0},
        )
        run_output = types.SimpleNamespace(SSF_pred=np.array([1.0, 2.0]), A_channel=np.array([0.1, 0.2]))
        reporter.report(
            params=np.array([1.0]),
            current_loss=0.3,
            convergence_val=None,
            run_output=run_output,
            tracker=tracker,
            ssf_obs=np.array([1.0, 2.0]),
            param_names=["a"],
        )
        self.assertIsNone(reporter.visualizer)

    def test_save_calibration_results_writes_parameter_batch_schema(self):
        batch = ParameterBatch(values=np.array([[1.0, 2.0], [3.0, 4.0]]), param_names=("a", "b"), weights=np.array([1.0, 3.0]))
        metrics = {"model_type": "crsem", "meteorological_dataset": "met", "station_name": "X"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_calibration_results(batch, metrics, save_dir=tmpdir)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("parameter_batch", payload)
        self.assertEqual(payload["parameter_batch"]["param_names"], ["a", "b"])
        self.assertEqual(len(payload["parameter_batch"]["values"]), 2)

    def test_save_calibration_results_accepts_explicit_json_file_path(self):
        batch = ParameterBatch(values=np.array([[1.0, 2.0]]), param_names=("a", "b"))
        metrics = {"model_type": "crsem"}
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "custom_params.json"
            path = save_calibration_results(batch, metrics, save_path=save_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(path, save_path)
        self.assertEqual(payload["parameter_batch"]["param_names"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()


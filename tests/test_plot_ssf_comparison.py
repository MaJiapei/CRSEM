from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scripts.plot_ssf_comparison import (
    get_ensemble_plot_data,
    load_ssf_data,
    plot_comparison,
    select_simulated_member,
    split_periods,
)


class PlotSsfComparisonTests(unittest.TestCase):
    def test_select_simulated_member_returns_mean_for_auto(self):
        time = pd.date_range("2000-01-31", periods=3, freq="ME")
        data = xr.DataArray(
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
            coords={"member": ["member_0", "member_1"], "time": time},
            dims=("member", "time"),
            name="SSF_pred",
        )

        selected = select_simulated_member(data, "auto")

        self.assertEqual(selected.dims, ("time",))
        np.testing.assert_allclose(selected.values, np.array([2.0, 3.0, 4.0], dtype=np.float32))

    def test_select_simulated_member_supports_index_and_label(self):
        time = pd.date_range("2000-01-31", periods=2, freq="ME")
        data = xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            coords={"member": ["member_0", "member_1"], "time": time},
            dims=("member", "time"),
            name="SSF_pred",
        )

        by_index = select_simulated_member(data, "1")
        by_label = select_simulated_member(data, "member_0")

        np.testing.assert_allclose(by_index.values, np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(by_label.values, np.array([1.0, 2.0], dtype=np.float32))

    def test_load_ssf_data_keeps_single_member_files_working(self):
        time = pd.date_range("2000-01-31", periods=3, freq="ME")
        sim_ds = xr.Dataset(
            {"SSF_pred": (("time",), np.array([2.0, 3.0, 4.0], dtype=np.float32))},
            coords={"time": time},
        )
        obs_ds = xr.Dataset(
            {"SSF": (("time",), np.array([1.0, 2.0, 3.0], dtype=np.float32))},
            coords={"time": time},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            simulated_path = Path(tmpdir) / "model_output.nc"
            observed_path = Path(tmpdir) / "observations.nc"
            sim_ds.to_netcdf(simulated_path)
            obs_ds.to_netcdf(observed_path)

            simulated, observed, _ = load_ssf_data(simulated_path, observed_path, member="auto")

        np.testing.assert_allclose(simulated.values, np.array([2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(observed.values, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_load_ssf_data_uses_mean_for_ensemble_auto(self):
        time = pd.date_range("2000-01-31", periods=3, freq="ME")
        sim_ds = xr.Dataset(
            {
                "SSF_pred": (
                    ("member", "time"),
                    np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=np.float32),
                )
            },
            coords={"member": ["member_0", "member_1"], "time": time},
        )
        obs_ds = xr.Dataset(
            {"SSF": (("time",), np.array([1.0, 2.0, 3.0], dtype=np.float32))},
            coords={"time": time},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            simulated_path = Path(tmpdir) / "model_outputs.nc"
            observed_path = Path(tmpdir) / "observations.nc"
            sim_ds.to_netcdf(simulated_path)
            obs_ds.to_netcdf(observed_path)

            simulated, _, _ = load_ssf_data(simulated_path, observed_path, member="auto")

        np.testing.assert_allclose(simulated.values, np.array([2.0, 3.0, 4.0], dtype=np.float32))

    def test_plot_comparison_accepts_ensemble_payload(self):
        time = pd.date_range("2000-01-31", periods=12, freq="ME")
        observed = pd.Series(np.linspace(1.0, 12.0, 12, dtype=np.float32), index=time)
        simulated = pd.Series(np.linspace(1.5, 12.5, 12, dtype=np.float32), index=time)
        sim_ds = xr.Dataset(
            {
                "SSF_pred": (
                    ("member", "time"),
                    np.vstack(
                        [
                            np.linspace(1.0, 12.0, 12, dtype=np.float32),
                            np.linspace(2.0, 13.0, 12, dtype=np.float32),
                        ]
                    ),
                )
            },
            coords={"member": ["member_0", "member_1"], "time": time},
        )
        ensemble_data = get_ensemble_plot_data(sim_ds)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ssf_comparison.png"
            with patch("matplotlib.pyplot.close") as mock_close:
                plot_comparison(
                    simulated=simulated,
                    observed=observed,
                    periods={"full": {"sim": simulated.values, "obs": observed.values, "time": time}},
                    output_path=output_path,
                    ensemble_data=ensemble_data,
                    title="Test",
                    should_split=False,
                )

            self.assertTrue(output_path.exists())
            fig = mock_close.call_args.args[0]
            self.assertEqual(len(fig.axes), 4)
        plt.close("all")

    def test_split_periods_excludes_calibration_years_from_validation(self):
        time = pd.date_range("1990-01-31", periods=14 * 12, freq="ME")
        simulated = pd.Series(np.arange(len(time), dtype=np.float32), index=time)
        observed = pd.Series(np.arange(len(time), dtype=np.float32), index=time)

        periods, should_split = split_periods(simulated, observed, cal_start=1990, cal_end=2000)

        self.assertTrue(should_split)
        self.assertIn("calibration", periods)
        self.assertIn("validation_after", periods)
        self.assertNotIn("validation_before", periods)
        self.assertEqual(periods["calibration"]["time"][0].year, 1990)
        self.assertEqual(periods["calibration"]["time"][-1].year, 2000)
        self.assertEqual(periods["validation_after"]["time"][0].year, 2001)
        self.assertTrue((periods["validation_after"]["time"].year > 2000).all())


if __name__ == "__main__":
    unittest.main()

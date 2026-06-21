from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.prepare_basin_drivers import (
    basin_mean_time_series,
    infer_basin_time_coverage,
    load_meteo_data,
    normalize_dataset_names,
)


def test_normalize_dataset_names_supports_strings_and_lists():
    assert normalize_dataset_names("ERA5-Land, TerraClimate ,") == ["ERA5-Land", "TerraClimate"]
    assert normalize_dataset_names([" ERA5-Land ", "TerraClimate"]) == ["ERA5-Land", "TerraClimate"]


def test_load_meteo_data_warns_for_multiple_requested_datasets_and_records_all_ranges(tmp_path):
    time = pd.date_range("2000-01-01", periods=4, freq="MS")
    latitude = np.array([35.0, 35.5], dtype=np.float32)
    longitude = np.array([100.0, 100.5, 101.0], dtype=np.float32)
    dataset_names = np.array(["ERA5-Land", "TerraClimate"], dtype=object)

    temp_values = np.empty((2, 4, 2, 3), dtype=np.float32)
    precip_values = np.empty((2, 4, 2, 3), dtype=np.float32)
    temp_values[0] = 273.15 + np.arange(4, dtype=np.float32)[:, None, None]
    temp_values[1] = 280.15 + np.arange(4, dtype=np.float32)[:, None, None]
    precip_values[0] = 10.0
    precip_values[1] = 20.0

    ds = xr.Dataset(
        {
            "2t": (("name", "time", "latitude", "longitude"), temp_values),
            "tp": (("name", "time", "latitude", "longitude"), precip_values),
        },
        coords={
            "name": dataset_names,
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        },
    )

    meteo_path = tmp_path / "meteo.nc"
    ds.to_netcdf(meteo_path, engine="h5netcdf")

    with pytest.warns(UserWarning, match="Only the first dataset|Only the first dataset|first dataset"):
        temperature, precipitation, time_info = load_meteo_data(
            path=meteo_path,
            dataset="ERA5-Land, TerraClimate",
            varnames=["2t", "tp"],
            year_range=None,
        )

    assert temperature.dims == ("time", "latitude", "longitude")
    assert precipitation.dims == ("time", "latitude", "longitude")
    assert set(time_info) == {"ERA5-Land", "TerraClimate"}
    assert time_info["ERA5-Land"]["n_months"] == 4
    assert time_info["TerraClimate"]["start"] == "2000"
    np.testing.assert_allclose(temperature.isel(time=0).values, np.zeros((2, 3), dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(precipitation.values, np.full((4, 2, 3), 10.0, dtype=np.float32))


def test_infer_basin_time_coverage_uses_basin_valid_cells_and_members():
    time = pd.date_range("2000-01-01", periods=4, freq="MS")
    y = np.array([35.0, 34.5], dtype=np.float32)
    x = np.array([100.0, 100.5, 101.0], dtype=np.float32)
    basin_mask = xr.DataArray(
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
    )

    values = np.full((2, 4, 2, 3), np.nan, dtype=np.float32)
    values[:, 1, 0, 0] = 0.2
    values[1, 3, 1, 1] = 0.4

    da = xr.DataArray(
        values,
        coords={"member": ["a", "b"], "time": time, "y": y, "x": x},
        dims=("member", "time", "y", "x"),
    )

    coverage = infer_basin_time_coverage(da, basin_mask)

    assert coverage is not None
    assert coverage[0] == pd.Timestamp("2000-02-01")
    assert coverage[1] == pd.Timestamp("2000-04-01")


def test_basin_mean_time_series_averages_only_valid_basin_cells():
    time = pd.date_range("2000-01-01", periods=2, freq="MS")
    y = np.array([0.0, 1.0], dtype=np.float32)
    x = np.array([10.0, 11.0], dtype=np.float32)
    basin_mask = xr.DataArray(
        np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        coords={"y": y, "x": x},
        dims=("y", "x"),
    )
    values = xr.DataArray(
        np.array(
            [
                [[1.0, 100.0], [3.0, 100.0]],
                [[5.0, 100.0], [7.0, 100.0]],
            ],
            dtype=np.float32,
        ),
        coords={"time": time, "y": y, "x": x},
        dims=("time", "y", "x"),
        name="Pre",
    )

    result = basin_mean_time_series(values, basin_mask)

    assert list(result.index) == list(time)
    np.testing.assert_allclose(result.values, np.array([2.0, 6.0], dtype=np.float32))

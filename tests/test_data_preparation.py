"""Tests for data preparation module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from netCDF4 import Dataset as NetCDFDataset

from CRSEM.data_preparation.builders import (
    build_dynamic_nc,
    build_observations_nc,
    build_static_nc,
)
from CRSEM.data_preparation.obs_preprocessing import (
    fill_monthly_observations_by_precip,
)
from CRSEM.data_preparation.spatial import (
    align_to_basin,
    calculate_basin_area,
    load_basin_template,
)


@pytest.fixture
def basin_template() -> xr.Dataset:
    """Create a simple basin template for testing."""
    y = np.arange(10)
    x = np.arange(15)
    lat = np.linspace(35, 36, 10)
    lon = np.linspace(100, 101, 15)

    # Create 2D lat/lon coordinates
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")

    # Create mask (1 in center, 0 outside)
    mask = np.zeros((10, 15), dtype="float32")
    mask[2:8, 3:12] = 1.0

    return xr.Dataset(
        {
            "BasinMask": (["y", "x"], mask),
            "latitude": (["y", "x"], lat_2d.astype("float32")),
            "longitude": (["y", "x"], lon_2d.astype("float32")),
        },
        coords={
            "y": y.astype("float32"),
            "x": x.astype("float32"),
        },
        attrs={
            "crs": "EPSG:4326",
            "target_resolution_m": 1000.0,
        },
    )


@pytest.fixture
def sample_static_data(basin_template) -> dict:
    """Create sample static data arrays."""
    y = basin_template["y"]
    x = basin_template["x"]

    return {
        "K": xr.DataArray(
            np.random.uniform(0.02, 0.05, (len(y), len(x))).astype("float32"),
            coords=[y, x],
            dims=["y", "x"],
        ),
        "LS": xr.DataArray(
            np.random.uniform(0.5, 10, (len(y), len(x))).astype("float32"),
            coords=[y, x],
            dims=["y", "x"],
        ),
        "IC": xr.DataArray(
            np.random.uniform(0.1, 0.9, (len(y), len(x))).astype("float32"),
            coords=[y, x],
            dims=["y", "x"],
        ),
    }


@pytest.fixture
def sample_dynamic_data(basin_template) -> dict:
    """Create sample dynamic data arrays."""
    y = basin_template["y"]
    x = basin_template["x"]
    time = pd.date_range("1990-01-01", periods=12, freq="MS")

    return {
        "T": xr.DataArray(
            np.random.uniform(-10, 30, (12, len(y), len(x))).astype("float32"),
            coords=[time, y, x],
            dims=["time", "y", "x"],
        ),
        "Pre": xr.DataArray(
            np.random.uniform(0, 0.1, (12, len(y), len(x))).astype("float32"),
            coords=[time, y, x],
            dims=["time", "y", "x"],
        ),
        "NDVI": xr.DataArray(
            np.random.uniform(0, 1, (12, len(y), len(x))).astype("float32"),
            coords=[time, y, x],
            dims=["time", "y", "x"],
        ),
    }


@pytest.fixture
def sample_observations() -> tuple:
    """Create sample observation time series."""
    time = pd.date_range("1990-01-01", periods=12, freq="MS")
    Q = pd.Series(np.random.uniform(100, 500, 12), index=time, name="Q")
    SSF = pd.Series(np.random.uniform(1000, 5000, 12), index=time, name="SSF")
    return Q, SSF


class TestSpatialProcessing:
    """Tests for spatial processing functions."""

    def test_calculate_basin_area(self, basin_template):
        """Test basin area calculation from mask."""
        mask = basin_template["BasinMask"]
        area = calculate_basin_area(mask, cell_size_km=1.0)

        # Mask has 6x9 = 54 cells set to 1
        expected_area = 54 * 1e6  # 54 km² in m²
        assert area == expected_area

    def test_calculate_basin_area_uses_latlon_when_available(self, basin_template):
        """Test basin area calculation on latitude/longitude grids."""
        mask = basin_template["BasinMask"]
        area = calculate_basin_area(
            mask,
            latitude=basin_template["latitude"],
            longitude=basin_template["longitude"],
        )

        lat = basin_template["latitude"].values[:, 0].astype("float64")
        lon = basin_template["longitude"].values[0, :].astype("float64")

        lat_bounds = np.concatenate(
            (
                [lat[0] - (lat[1] - lat[0]) / 2],
                (lat[:-1] + lat[1:]) / 2,
                [lat[-1] + (lat[-1] - lat[-2]) / 2],
            )
        )
        lon_bounds = np.concatenate(
            (
                [lon[0] - (lon[1] - lon[0]) / 2],
                (lon[:-1] + lon[1:]) / 2,
                [lon[-1] + (lon[-1] - lon[-2]) / 2],
            )
        )
        earth_radius_m = 6_371_008.8
        lat_south = np.deg2rad(lat_bounds[:-1])
        lat_north = np.deg2rad(lat_bounds[1:])
        dlon = np.abs(np.deg2rad(lon_bounds[1:] - lon_bounds[:-1]))
        cell_area = (
            earth_radius_m ** 2
            * np.abs(np.sin(lat_north) - np.sin(lat_south))[:, None]
            * dlon[None, :]
        )
        expected_area = cell_area[mask.values > 0].sum()
        np.testing.assert_allclose(area, expected_area)

    def test_load_basin_template(self, tmp_path, basin_template):
        """Test loading basin template from file."""
        # Save template
        template_path = tmp_path / "template.nc"
        basin_template.to_netcdf(template_path)

        # Load and verify
        loaded = load_basin_template(template_path)
        assert "BasinMask" in loaded
        assert loaded.attrs.get("crs") == "EPSG:4326"

    def test_load_basin_template_normalizes_bd_mask_with_latlon_dims(self, tmp_path):
        """Test loading a template that uses bd_mask and latitude/longitude dims."""
        lat = np.array([35.5, 35.0, 34.5], dtype="float32")
        lon = np.array([100.0, 100.5, 101.0, 101.5], dtype="float32")
        mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype="float32",
        )

        template = xr.Dataset(
            {
                "bd_mask": (["latitude", "longitude"], mask),
                "bd_ewi_mask": (["latitude", "longitude"], mask * 0.5),
            },
            coords={
                "latitude": lat,
                "longitude": lon,
            },
        )

        template_path = tmp_path / "bd_mask_template.nc"
        template.to_netcdf(template_path)

        loaded = load_basin_template(template_path)

        assert loaded["BasinMask"].dims == ("y", "x")
        assert loaded["latitude"].shape == mask.shape
        assert loaded["longitude"].shape == mask.shape
        np.testing.assert_allclose(loaded["BasinMask"].values, mask)
        np.testing.assert_allclose(loaded["y"].values, lat)
        np.testing.assert_allclose(loaded["x"].values, lon)
        assert loaded.attrs.get("crs") == "EPSG:4326"


class TestObservationPreprocessing:
    """Tests for observation gap filling helpers."""

    def test_fill_monthly_observations_by_precip_fills_missing_months(self):
        time = pd.date_range("2000-01-01", periods=36, freq="MS")
        precip = pd.Series(
            np.tile(np.arange(1, 13, dtype=float), 3),
            index=time,
        )
        q = pd.Series(
            precip.values * 2.0,
            index=time,
        )
        ssf = pd.Series(
            precip.values * 10.0,
            index=time,
        )

        q.loc["2001-03-01"] = np.nan
        ssf.loc["2001-01-01":"2001-12-01"] = np.nan

        filled_q, filled_ssf, summary = fill_monthly_observations_by_precip(
            q_series=q,
            ssf_series=ssf,
            precip_series=precip,
            year_range=(2000, 2002),
        )

        assert filled_q.isna().sum() == 0
        assert filled_ssf.isna().sum() == 0
        np.testing.assert_allclose(
            filled_q.loc["2001-03-01"],
            6.0,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            filled_ssf.loc["2001-07-01"],
            70.0,
            atol=1e-4,
        )
        assert summary["Q"]["filled_by_covariate_or_fallback"] == 1
        assert summary["SSF"]["filled_by_covariate_or_fallback"] == 12
        assert summary["SSF"]["remaining_missing"] == 0


class TestBuilders:
    """Tests for NC file builders."""

    def test_build_static_nc(self, tmp_path, basin_template, sample_static_data):
        """Test building static.nc file."""
        output_path = tmp_path / "static.nc"

        result = build_static_nc(
            output_path=output_path,
            basin_template=basin_template,
            k=sample_static_data["K"],
            ls=sample_static_data["LS"],
            ic=sample_static_data["IC"],
            basin_name="test_basin",
        )

        assert result == output_path
        assert output_path.exists()

        # Verify contents
        ds = xr.open_dataset(output_path)
        assert "K" in ds
        assert "LS" in ds
        assert "IC" in ds
        assert "P_f" in ds
        assert "mask" in ds
        assert ds.attrs["basin_name"] == "test_basin"
        assert "s_area_m2" in ds.attrs

    def test_build_dynamic_nc(self, tmp_path, basin_template, sample_dynamic_data):
        """Test building dynamic.nc file."""
        output_path = tmp_path / "dynamic.nc"

        result = build_dynamic_nc(
            output_path=output_path,
            basin_template=basin_template,
            temperature=sample_dynamic_data["T"],
            precipitation=sample_dynamic_data["Pre"],
            ndvi=sample_dynamic_data["NDVI"],
            basin_name="test_basin",
        )

        assert result == output_path
        assert output_path.exists()

        # Verify contents
        ds = xr.open_dataset(output_path)
        assert "T" in ds
        assert "Pre" in ds
        assert "NDVI" in ds
        assert ds.attrs["basin_name"] == "test_basin"
        assert ds.attrs["frequency"] == "monthly"

        with NetCDFDataset(output_path, mode="r") as nc:
            assert nc.variables["T"].filters()["zlib"] is True
            assert nc.variables["Pre"].filters()["zlib"] is True
            assert nc.variables["NDVI"].filters()["zlib"] is True

    def test_build_observations_nc(self, tmp_path, sample_observations):
        """Test building observations.nc file."""
        output_path = tmp_path / "observations.nc"
        Q, SSF = sample_observations

        result = build_observations_nc(
            output_path=output_path,
            q_series=Q,
            ssf_series=SSF,
            s_area=1e7,  # 10 km²
            basin_name="test_basin",
            station_name="test_station",
        )

        assert result == output_path
        assert output_path.exists()

        # Verify contents
        ds = xr.open_dataset(output_path)
        assert "Q" in ds
        assert "SSF" in ds
        assert ds.attrs["basin_name"] == "test_basin"
        assert ds.attrs["station_name"] == "test_station"
        assert ds.attrs["s_area"] == 1e7


class TestDriverNCMode:
    """Tests for BasinDriver NC file mode."""

    def test_from_nc_files(self, tmp_path, basin_template, sample_static_data, sample_dynamic_data, sample_observations):
        """Test loading BasinDriver from NC files."""
        from CRSEM.driver import BasinDriver

        # Build test files
        static_path = tmp_path / "static.nc"
        dynamic_path = tmp_path / "dynamic.nc"
        obs_path = tmp_path / "observations.nc"

        build_static_nc(
            output_path=static_path,
            basin_template=basin_template,
            k=sample_static_data["K"],
            ls=sample_static_data["LS"],
            ic=sample_static_data["IC"],
            basin_name="test_basin",
        )

        build_dynamic_nc(
            output_path=dynamic_path,
            basin_template=basin_template,
            temperature=sample_dynamic_data["T"],
            precipitation=sample_dynamic_data["Pre"],
            ndvi=sample_dynamic_data["NDVI"],
            basin_name="test_basin",
        )

        Q, SSF = sample_observations
        build_observations_nc(
            output_path=obs_path,
            q_series=Q,
            ssf_series=SSF,
            s_area=1e7,
            basin_name="test_basin",
            station_name="test_station",
        )

        # Load with BasinDriver
        driver = BasinDriver.from_nc_files(
            static_nc=static_path,
            dynamic_nc=dynamic_path,
            observations_nc=obs_path,
            station_name="test_station",
        )

        # Verify
        assert driver.model_inputs is not None
        assert driver.model_inputs.K is not None
        assert driver.model_inputs.T is not None
        assert driver.Q is not None
        assert driver.SSF is not None
        expected_area_m2 = calculate_basin_area(
            basin_template["BasinMask"],
            latitude=basin_template["latitude"],
            longitude=basin_template["longitude"],
        )
        assert driver.s_area == pytest.approx(expected_area_m2 / 10000.0)

        # Verify RunContext
        ctx = driver.to_run_context()
        assert ctx.inputs is not None
        assert ctx.q is not None
        assert ctx.ssf_obs is not None

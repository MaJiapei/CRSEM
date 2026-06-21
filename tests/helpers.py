from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from CRSEM.contracts import ParameterBatch, RunContext
from CRSEM.driver import BasinDriver
from CRSEM.model import ModelInputs
from CRSEM.parameters import CRSEMParameters, RUSLEParameters


def make_time_index(n_months: int = 24):
    return pd.date_range("2000-01-31", periods=n_months, freq="ME")


def make_point_context(n_months: int = 24) -> RunContext:
    time = make_time_index(n_months)
    inputs = ModelInputs(
        T=xr.DataArray(np.linspace(-10.0, 12.0, n_months, dtype=np.float32), coords={"time": time}, dims=("time",), name="T"),
        Pre=xr.DataArray(np.linspace(20.0, 120.0, n_months, dtype=np.float32), coords={"time": time}, dims=("time",), name="Pre"),
        NDVI=xr.DataArray(np.linspace(0.15, 0.75, n_months, dtype=np.float32), coords={"time": time}, dims=("time",), name="NDVI"),
        LS=xr.DataArray(np.float32(2.5), name="LS"),
        P_f=xr.DataArray(np.float32(0.9), name="P_f"),
        IC=xr.DataArray(np.float32(0.35), name="IC"),
        K=xr.DataArray(np.float32(0.03), name="K"),
    )
    q = pd.Series(np.linspace(5.0, 40.0, n_months, dtype=np.float32), index=time, name="Q")
    ssf = pd.Series(np.linspace(1.0, 12.0, n_months, dtype=np.float32), index=time, name="SSF")
    return RunContext(inputs=inputs, q=q, ssf_obs=ssf, s_area=1250.0, metadata={"station_name": "synthetic"})


def make_grid_context(n_months: int = 24, n_lat: int = 2, n_lon: int = 3) -> RunContext:
    time = make_time_index(n_months)
    latitude = np.linspace(45.0, 46.0, n_lat, dtype=np.float32)
    longitude = np.linspace(120.0, 122.0, n_lon, dtype=np.float32)
    shape = (n_months, n_lat, n_lon)
    base_time = np.linspace(0.0, 1.0, n_months, dtype=np.float32).reshape(n_months, 1, 1)
    lat_grid = latitude.reshape(1, n_lat, 1)
    lon_grid = longitude.reshape(1, 1, n_lon)

    T = -8.0 + 20.0 * base_time + 0.5 * lat_grid - 0.2 * lon_grid
    Pre = 30.0 + 90.0 * base_time + 2.0 * lat_grid + 1.5 * lon_grid
    NDVI = 0.2 + 0.4 * base_time + 0.02 * (lat_grid - latitude.mean()) - 0.01 * (lon_grid - longitude.mean())

    inputs = ModelInputs(
        T=xr.DataArray(T.astype(np.float32), coords={"time": time, "latitude": latitude, "longitude": longitude}, dims=("time", "latitude", "longitude"), name="T"),
        Pre=xr.DataArray(Pre.astype(np.float32), coords={"time": time, "latitude": latitude, "longitude": longitude}, dims=("time", "latitude", "longitude"), name="Pre"),
        NDVI=xr.DataArray(NDVI.astype(np.float32), coords={"time": time, "latitude": latitude, "longitude": longitude}, dims=("time", "latitude", "longitude"), name="NDVI"),
        LS=xr.DataArray(np.full((n_lat, n_lon), 2.5, dtype=np.float32), coords={"latitude": latitude, "longitude": longitude}, dims=("latitude", "longitude"), name="LS"),
        P_f=xr.DataArray(np.full((n_lat, n_lon), 0.9, dtype=np.float32), coords={"latitude": latitude, "longitude": longitude}, dims=("latitude", "longitude"), name="P_f"),
        IC=xr.DataArray(np.full((n_lat, n_lon), 0.35, dtype=np.float32), coords={"latitude": latitude, "longitude": longitude}, dims=("latitude", "longitude"), name="IC"),
        K=xr.DataArray(np.full((n_lat, n_lon), 0.03, dtype=np.float32), coords={"latitude": latitude, "longitude": longitude}, dims=("latitude", "longitude"), name="K"),
    )
    q = pd.Series(np.linspace(5.0, 40.0, n_months, dtype=np.float32), index=time, name="Q")
    ssf = pd.Series(np.linspace(1.0, 12.0, n_months, dtype=np.float32), index=time, name="SSF")
    return RunContext(inputs=inputs, q=q, ssf_obs=ssf, s_area=1250.0, metadata={"station_name": "synthetic_grid", "shape": shape})


def make_point_driver(n_months: int = 24) -> BasinDriver:
    context = make_point_context(n_months)
    driver = BasinDriver(
        station_name="synthetic",
        start_year=int(context.inputs.T.time.dt.year.values[0]),
        end_year=int(context.inputs.T.time.dt.year.values[-1]),
        model_inputs=context.inputs,
    )
    driver._Q = context.q
    driver._SSF = context.ssf_obs
    driver._s_area = context.s_area
    driver._data_loaded = True
    driver._river_data_loaded = True
    return driver


def make_parameter_batch(model_type: str, n_members: int = 2) -> ParameterBatch:
    if model_type == "crsem":
        base = CRSEMParameters.from_default().to_array()
        values = np.vstack([base + i * 0.01 for i in range(n_members)])
        return ParameterBatch(values=values, param_names=tuple(CRSEMParameters.DEFAULT_PARAMS.keys()))
    if model_type == "rusle":
        base = RUSLEParameters.from_default().to_array()
        values = np.vstack([base + i * 0.001 for i in range(n_members)])
        return ParameterBatch(values=values, param_names=tuple(RUSLEParameters.DEFAULT_PARAMS.keys()))
    raise ValueError(model_type)

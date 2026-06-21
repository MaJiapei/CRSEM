"""Driver class for encapsulating station data and model inputs.

This module provides a data driver that loads data from pre-prepared NetCDF files.
Use BasinDriver.from_nc_files() to create a driver instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr

from CRSEM.contracts import RunContext
from CRSEM.model import ModelInputs


class BasinDriver(object):
    """Data driver for calibration and model runs.

    The driver owns station metadata and loaded inputs from NetCDF files.
    Use BasinDriver.from_nc_files() to create an instance.
    """

    def __init__(
        self,
        station_name: str,
        start_year: int,
        end_year: int,
        model_inputs: Optional[ModelInputs] = None,
    ) -> None:
        """Initialize BasinDriver with metadata.

        This constructor is primarily for internal use. Use from_nc_files()
        to create a driver from prepared NetCDF files.
        """
        self.station_name = station_name
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.model_inputs = model_inputs

        self._river_data_loaded = False
        self._data_loaded = model_inputs is not None

        self._Q: Optional[pd.Series] = None
        self._SSF: Optional[pd.Series] = None
        self._s_area: Optional[float] = None
        self._basin_name: str = "unknown"

        # NC file paths
        self._static_nc: Optional[Path] = None
        self._dynamic_nc: Optional[Path] = None
        self._observations_nc: Optional[Path] = None

    @classmethod
    def from_nc_files(
        cls,
        static_nc: Path | str,
        dynamic_nc: Path | str,
        observations_nc: Path | str | None = None,
        station_name: str | None = None,
    ) -> "BasinDriver":
        """Create BasinDriver from pre-prepared NetCDF files.

        This is the primary way to load data. Data should be prepared using
        the scripts/prepare_basin_drivers.py script.

        Args:
            static_nc: Path to static.nc file (K, LS, IC, P_f, mask)
            dynamic_nc: Path to dynamic.nc file (T, Pre, NDVI with time dimension)
            observations_nc: Path to observations.nc file (Q, SSF). Optional.
            station_name: Station name for metadata. If not provided, extracted from file.

        Returns:
            BasinDriver with all data loaded from NetCDF files

        Example:
            >>> driver = BasinDriver.from_nc_files(
            ...     static_nc="outputs/tuotuohe/static.nc",
            ...     dynamic_nc="outputs/tuotuohe/dynamic.nc",
            ...     observations_nc="outputs/tuotuohe/observations.nc",
            ...     station_name="沱沱河"
            ... )
            >>> ctx = driver.to_run_context()
        """
        static_nc = Path(static_nc)
        dynamic_nc = Path(dynamic_nc)
        observations_nc = Path(observations_nc) if observations_nc else None

        # Load static data (force into memory for fast .values access)
        ds_static = xr.open_dataset(static_nc)
        ds_static.load()

        # Load dynamic data (force into memory for fast .values access)
        ds_dynamic = xr.open_dataset(dynamic_nc)
        ds_dynamic.load()

        # Create ModelInputs
        model_inputs = ModelInputs(
            K=ds_static["K"],
            LS=ds_static["LS"],
            IC=ds_static["IC"],
            P_f=ds_static["P_f"],
            T=ds_dynamic["T"],
            Pre=ds_dynamic["Pre"],
            NDVI=ds_dynamic["NDVI"],
        )

        # Extract time range from dynamic file
        time_coord = ds_dynamic["time"]
        start_year = int(pd.Timestamp(time_coord.values[0]).year)
        end_year = int(pd.Timestamp(time_coord.values[-1]).year)

        # Create driver instance
        driver = cls(
            station_name=station_name or "unknown",
            start_year=start_year,
            end_year=end_year,
            model_inputs=model_inputs,
        )

        # Store NC file paths
        driver._static_nc = static_nc
        driver._dynamic_nc = dynamic_nc
        driver._observations_nc = observations_nc

        # Extract metadata from static file
        driver._basin_name = ds_static.attrs.get("basin_name", "unknown")
        # Convert s_area from m² to hectares (ha) for model calculations
        # Model expects s_area in hectares: E_hillslope (t/ha) * s_area (ha) = SSF (tonnes)
        s_area_m2 = ds_static.attrs.get("s_area_m2", 0)
        driver._s_area = s_area_m2 / 10000.0 if s_area_m2 else None  # ha

        # Load observations if provided
        if observations_nc:
            ds_obs = xr.open_dataset(observations_nc)
            driver._Q = pd.Series(
                ds_obs["Q"].values,
                index=pd.to_datetime(ds_obs["time"].values),
                name="Q"
            )
            driver._SSF = pd.Series(
                ds_obs["SSF"].values,
                index=pd.to_datetime(ds_obs["time"].values),
                name="SSF"
            )
            if station_name is None:
                driver.station_name = ds_obs.attrs.get("station_name", "unknown")
            driver._river_data_loaded = True

            # Verify time alignment
            obs_start = driver._Q.index[0]
            obs_end = driver._Q.index[-1]
            dyn_start = pd.Timestamp(time_coord.values[0])
            dyn_end = pd.Timestamp(time_coord.values[-1])

            if obs_start != dyn_start or obs_end != dyn_end:
                import warnings
                warnings.warn(
                    f"Observation time range ({obs_start} to {obs_end}) "
                    f"does not match dynamic data ({dyn_start} to {dyn_end})"
                )

        driver._data_loaded = True
        return driver

    @property
    def Q(self) -> pd.Series:
        if self._Q is None:
            raise RuntimeError("River discharge 'Q' is not loaded.")
        return self._Q

    @property
    def SSF(self) -> pd.Series:
        if self._SSF is None:
            raise RuntimeError("Suspended sediment flux 'SSF' is not loaded.")
        return self._SSF

    @property
    def s_area(self) -> float:
        if self._s_area is None:
            raise RuntimeError("Basin area 's_area' is not set.")
        return self._s_area

    @property
    def basin_name(self) -> str:
        return self._basin_name

    @property
    def static_nc(self) -> Optional[Path]:
        """Path to static.nc file."""
        return self._static_nc

    @property
    def dynamic_nc(self) -> Optional[Path]:
        """Path to dynamic.nc file."""
        return self._dynamic_nc

    @property
    def observations_nc(self) -> Optional[Path]:
        """Path to observations.nc file."""
        return self._observations_nc

    def to_run_context(self) -> RunContext:
        """Convert driver to RunContext for model execution."""
        if self.model_inputs is None:
            raise RuntimeError("Model inputs are not loaded.")
        return RunContext(
            inputs=self.model_inputs,
            q=self._Q,
            ssf_obs=self._SSF,
            s_area=self._s_area,
            metadata={
                "station_name": self.station_name,
                "basin_name": self._basin_name,
                "start_year": self.start_year,
                "end_year": self.end_year,
            },
        )

    def _clone_with_inputs(self, model_inputs: ModelInputs) -> "BasinDriver":
        """Create a shallow driver clone with replaced model inputs."""
        cloned = BasinDriver(
            station_name=self.station_name,
            start_year=self.start_year,
            end_year=self.end_year,
            model_inputs=model_inputs,
        )
        cloned._Q = self._Q
        cloned._SSF = self._SSF
        cloned._s_area = self._s_area
        cloned._river_data_loaded = self._river_data_loaded
        cloned._static_nc = self._static_nc
        cloned._dynamic_nc = self._dynamic_nc
        cloned._observations_nc = self._observations_nc
        cloned._basin_name = self._basin_name
        cloned._data_loaded = self._data_loaded
        return cloned

    def collapse_ndvi_members(self, method: str = "mean") -> "BasinDriver":
        """Collapse NDVI ensemble members to a single NDVI field.

        Standard calibration and simulation paths expect `member` to represent
        parameter ensembles in the outputs. When dynamic NDVI inputs also carry a
        `member` dimension, we collapse them before the run to keep one NDVI
        forcing trajectory per execution. Attribution keeps its own explicit
        per-member workflow and should not use this helper.
        """
        ndvi = self.model_inputs.NDVI if self.model_inputs is not None else None
        if ndvi is None or "member" not in ndvi.dims:
            return self

        method_normalized = method.lower()
        if method_normalized != "mean":
            raise ValueError(f"Unsupported NDVI member collapse method: {method}")

        collapsed_inputs = ModelInputs(
            T=self.model_inputs.T,
            Pre=self.model_inputs.Pre,
            NDVI=ndvi.mean(dim="member"),
            LS=self.model_inputs.LS,
            P_f=self.model_inputs.P_f,
            IC=self.model_inputs.IC,
            K=self.model_inputs.K,
        )
        return self._clone_with_inputs(collapsed_inputs)

    def to_point_driver(self, lon: float = None, lat: float = None, keep_rivers: bool = False) -> "BasinDriver":
        """Create a point driver by averaging over spatial dimensions.

        Args:
            lon: Longitude for point selection (not implemented yet)
            lat: Latitude for point selection (not implemented yet)
            keep_rivers: Whether to copy river data to point driver

        Returns:
            New BasinDriver with spatially-averaged inputs
        """
        point_driver = BasinDriver(
            station_name=self.station_name,
            start_year=self.start_year,
            end_year=self.end_year,
        )

        if keep_rivers:
            point_driver._Q = self.Q
            point_driver._SSF = self.SSF
            point_driver._river_data_loaded = True

        if self._s_area is not None:
            point_driver._s_area = self._s_area
        point_driver._static_nc = self._static_nc
        point_driver._dynamic_nc = self._dynamic_nc
        point_driver._observations_nc = self._observations_nc
        point_driver._basin_name = self._basin_name

        # Detect spatial dimension names from data
        sample_da = self.model_inputs["T"]
        spatial_dims = [d for d in sample_da.dims if d not in ("time",)]
        y_dim = "y" if "y" in spatial_dims else "latitude"
        x_dim = "x" if "x" in spatial_dims else "longitude"

        # Average over spatial dimensions
        T = self.model_inputs["T"].mean(dim=[y_dim, x_dim])
        Pre = self.model_inputs["Pre"].mean(dim=[y_dim, x_dim])

        # NDVI: handle member dimension if present
        NDVI = self.model_inputs["NDVI"]
        if "member" in NDVI.dims:
            # Average over space, keep member dimension: (member, time)
            NDVI = NDVI.mean(dim=[y_dim, x_dim])
        else:
            NDVI = NDVI.mean(dim=[y_dim, x_dim])

        LS = self.model_inputs["LS"].mean(dim=[y_dim, x_dim])
        K = self.model_inputs["K"].mean(dim=[y_dim, x_dim])
        IC = self.model_inputs["IC"].mean(dim=[y_dim, x_dim])
        P_f = self.model_inputs["P_f"].mean(dim=[y_dim, x_dim])

        point_driver.model_inputs = ModelInputs(T=T, Pre=Pre, NDVI=NDVI, LS=LS, P_f=P_f, IC=IC, K=K)
        point_driver._data_loaded = True
        return point_driver

    def to_cf_driver(
        self,
        variable: str | list[str],
        baseline_start: int,
        baseline_end: int,
    ) -> "BasinDriver":
        """Create counterfactual driver with climatology for specified variable.

        The climatology is computed from all data within [baseline_start, baseline_end]
        (both inclusive). For each month (Jan-Dec), the mean value over the baseline
        period is computed and then repeated for each year in the simulation period.

        Args:
            variable: Variable name(s) to replace ('NDVI', 'T', 'Pre')
            baseline_start: Start year of baseline period (inclusive)
            baseline_end: End year of baseline period (inclusive)

        Returns:
            New BasinDriver with variable replaced by monthly climatology cycle

        Example:
            >>> driver_cf = driver.to_cf_driver('NDVI', 1982, 1992)
            # NDVI replaced with 1982-1992 monthly climatology cycle

            >>> driver_cf = driver.to_cf_driver(['T', 'Pre'], 1960, 1980)
            # Both T and Pre replaced with climatology
        """
        var_names = [variable] if isinstance(variable, str) else list(variable)

        cf_map: dict[str, xr.DataArray] = {}
        for v in var_names:
            if v not in ("T", "Pre", "NDVI"):
                raise ValueError(f"Variable '{v}' not supported. Must be 'T', 'Pre', or 'NDVI'")

            data_array = self.model_inputs[v]
            if data_array is None:
                raise ValueError(f"Variable '{v}' is not available in model inputs")

            years = data_array["time"].dt.year

            # Select baseline period (inclusive)
            baseline_mask = (years >= baseline_start) & (years <= baseline_end)
            hist = data_array.sel(time=baseline_mask)

            if hist.time.size == 0:
                raise ValueError(
                    f"No data in baseline period {baseline_start}-{baseline_end} for variable '{v}'"
                )

            # Compute monthly climatology
            climatology = hist.groupby("time.month").mean("time")

            # Expand to full time axis by selecting climatology for each month
            month_indexer = data_array["time"].dt.month
            cf_series = climatology.sel(month=month_indexer).drop_vars("month")

            # Preserve original time coordinate
            cf_series = cf_series.assign_coords(time=data_array.time)

            cf_map[v] = cf_series

        # Create new ModelInputs with modified variables
        new_inputs = ModelInputs(
            T=cf_map.get("T", self.model_inputs.T),
            Pre=cf_map.get("Pre", self.model_inputs.Pre),
            NDVI=cf_map.get("NDVI", self.model_inputs.NDVI),
            LS=self.model_inputs.LS,
            P_f=self.model_inputs.P_f,
            IC=self.model_inputs.IC,
            K=self.model_inputs.K,
        )

        # Create new driver instance
        cf_driver = BasinDriver(
            station_name=self.station_name,
            start_year=self.start_year,
            end_year=self.end_year,
            model_inputs=new_inputs,
        )

        # Copy river data if available
        if self._river_data_loaded:
            cf_driver._Q = self._Q
            cf_driver._SSF = self._SSF
            cf_driver._s_area = self._s_area
            cf_driver._river_data_loaded = True

        # Copy metadata
        cf_driver._static_nc = self._static_nc
        cf_driver._dynamic_nc = self._dynamic_nc
        cf_driver._observations_nc = self._observations_nc
        cf_driver._basin_name = self._basin_name
        cf_driver._data_loaded = True

        return cf_driver

    def crop_time_range(self, start_year: int, end_year: int, align_to_obs: bool = True) -> "BasinDriver":
        """Crop driver to a specific time range.

        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            align_to_obs: If True and river data available, align to observation time indices

        Returns:
            New BasinDriver with cropped time range
        """
        from CRSEM.model import ModelInputs

        # Determine the actual time range
        if align_to_obs and self._river_data_loaded and self._Q is not None:
            # Use observation time indices for alignment
            obs_start = self._Q.index[0]
            obs_end = self._Q.index[-1]
            # Adjust to requested range
            actual_start = max(pd.Timestamp(f"{start_year}-01-01"), obs_start)
            actual_end = min(pd.Timestamp(f"{end_year}-12-31"), obs_end)
            time_slice = slice(str(actual_start), str(actual_end))
        else:
            time_slice = slice(str(start_year), str(end_year))

        # Crop dynamic inputs
        T = self.model_inputs.T.sel(time=time_slice) if self.model_inputs.T is not None else None
        Pre = self.model_inputs.Pre.sel(time=time_slice) if self.model_inputs.Pre is not None else None
        NDVI = self.model_inputs.NDVI

        # Handle NDVI with member dimension
        if NDVI is not None:
            if "member" in NDVI.dims:
                NDVI = NDVI.sel(time=time_slice)
            else:
                NDVI = NDVI.sel(time=time_slice)

        cropped_inputs = ModelInputs(
            T=T,
            Pre=Pre,
            NDVI=NDVI,
            LS=self.model_inputs.LS,
            P_f=self.model_inputs.P_f,
            IC=self.model_inputs.IC,
            K=self.model_inputs.K,
        )

        # Get actual years from cropped data
        actual_start_year = T.time.values[0].astype('datetime64[Y]').astype(int) + 1970
        actual_end_year = T.time.values[-1].astype('datetime64[Y]').astype(int) + 1970

        cropped_driver = BasinDriver(
            station_name=self.station_name,
            start_year=actual_start_year,
            end_year=actual_end_year,
            model_inputs=cropped_inputs,
        )

        # Crop river data if available and alignment requested
        if align_to_obs and self._river_data_loaded and self._Q is not None:
            q_mask = (self._Q.index >= actual_start) & (self._Q.index <= actual_end)
            cropped_driver._Q = self._Q[q_mask]
            cropped_driver._SSF = self._SSF[q_mask] if self._SSF is not None else None
            cropped_driver._s_area = self._s_area
            cropped_driver._river_data_loaded = True
        elif self._s_area is not None:
            # Always copy s_area if available (needed for E_hillslope -> SSF conversion)
            cropped_driver._s_area = self._s_area
            cropped_driver._river_data_loaded = False

        # Copy metadata
        cropped_driver._static_nc = self._static_nc
        cropped_driver._dynamic_nc = self._dynamic_nc
        cropped_driver._observations_nc = self._observations_nc
        cropped_driver._basin_name = self._basin_name
        cropped_driver._data_loaded = True

        return cropped_driver

    def __repr__(self) -> str:
        return (
            f"Driver(station_name={self.station_name}, "
            f"start_year={self.start_year}, end_year={self.end_year}, "
            f"data loaded: {self._data_loaded}, "
            f"river data loaded: {self._river_data_loaded})"
        )

    def calculate_erosion_modulus(self) -> pd.DataFrame:
        """Calculate erosion modulus (SSF per unit area)."""
        df = pd.DataFrame({"Q": self.Q.values, "SSF": self.SSF.values}, index=self.Q.index)
        df.index.name = "Date"
        df["Erosion_modulus"] = df["SSF"] / self.s_area
        return df

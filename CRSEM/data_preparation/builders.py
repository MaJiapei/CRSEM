"""NetCDF file builders for CRSEM data preparation.

This module provides functions to build CF-compliant NetCDF files
for CRSEM model inputs.

Units:
    - Temperature: °C (degrees Celsius)
    - Precipitation: mm/month (monthly accumulated precipitation)
    - NDVI: dimensionless [0, 1]
    - Q: m³/s (cubic meters per second)
    - SSF: t/month (tons per month, suspended sediment flux)
"""

from pathlib import Path
from typing import Any, Optional
import json

import numpy as np
import pandas as pd
import xarray as xr

from CRSEM.netcdf_utils import build_netcdf_compression_encoding

from CRSEM.data_preparation.spatial import apply_basin_mask, calculate_basin_area
from CRSEM.data_preparation.quality import (
    assess_time_series_quality,
    assess_spatial_quality,
    generate_quality_report,
    print_quality_report,
    QualityReport,
)


# CF-compliant variable attributes for static.nc
STATIC_VAR_ATTRS = {
    "K": {"long_name": "Soil erodibility factor", "units": "t ha h ha-1 MJ-1 mm-1"},
    "LS": {"long_name": "Slope length and steepness factor", "units": "1"},
    "IC": {"long_name": "Index of connectivity for sediment delivery ratio", "units": "1"},
    "P_f": {"long_name": "Support practice factor", "units": "1"},
    "mask": {"long_name": "Basin mask", "units": "1"},
    "latitude": {"long_name": "latitude", "units": "degrees_north"},
    "longitude": {"long_name": "longitude", "units": "degrees_east"},
}

STATIC_COORD_ATTRS = {
    "y": {"long_name": "y coordinate of projection", "units": "m"},
    "x": {"long_name": "x coordinate of projection", "units": "m"},
}

# CF-compliant variable attributes for dynamic.nc
# Note: Precipitation is stored as mm/month (monthly accumulated)
DYNAMIC_VAR_ATTRS = {
    "T": {"long_name": "Surface temperature", "units": "degC"},
    "Pre": {"long_name": "Precipitation", "units": "mm month-1"},
    "NDVI": {"long_name": "Normalized Difference Vegetation Index", "units": "1"},
    "T_mean": {"long_name": "Basin-averaged surface temperature", "units": "degC"},
    "Pre_mean": {"long_name": "Basin-averaged precipitation", "units": "mm month-1"},
    "NDVI_mean": {"long_name": "Basin-averaged NDVI", "units": "1"},
}

DYNAMIC_COORD_ATTRS = {
    "time": {"long_name": "time"},
    "y": {"long_name": "y coordinate of projection", "units": "m"},
    "x": {"long_name": "x coordinate of projection", "units": "m"},
    "latitude": {"long_name": "latitude", "units": "degrees_north"},
    "longitude": {"long_name": "longitude", "units": "degrees_east"},
}

# CF-compliant variable attributes for observations.nc
OBS_VAR_ATTRS = {
    "Q": {"long_name": "Discharge", "units": "m3 s-1"},
    "SSF": {"long_name": "Suspended sediment flux", "units": "t month-1"},
}

OBS_COORD_ATTRS = {
    "time": {"long_name": "time"},
}


def _nanmean_last_axis(values: np.ndarray) -> np.ndarray:
    """Compute a NaN-aware mean over the last axis without runtime warnings."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=-1)
    totals = np.where(finite, values, 0.0).sum(axis=-1, dtype=np.float64)
    means = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means


def build_static_nc(
    output_path: Path | str,
    basin_template: xr.Dataset,
    k: xr.DataArray,
    ls: xr.DataArray,
    ic: xr.DataArray,
    p_f: xr.DataArray | None = None,
    basin_name: str | None = None,
    source_files: dict[str, str] | None = None,
) -> Path:
    """Build static.nc file containing basin spatial data.

    Args:
        output_path: Path to output file (e.g., "static.nc")
        basin_template: Basin template dataset with mask and coordinates
        k: Soil erodibility factor (K)
        ls: Slope length factor (LS)
        ic: SDR factor (IC)
        p_f: Support practice factor (P), defaults to all 1s
        basin_name: Name of the basin (stored in attributes)
        source_files: Dictionary of source file paths for provenance

    Returns:
        Path to created output file

    Output Format:
        Dimensions: (y, x)
        Variables:
            - K: (y, x) float32 - Soil erodibility factor
            - LS: (y, x) float32 - Slope length factor
            - IC: (y, x) float32 - SDR factor
            - P_f: (y, x) float32 - Support practice factor
            - mask: (y, x) float32 - Basin mask (1=inside, 0=outside)
            - latitude: (y, x) float32 - Latitude coordinates
            - longitude: (y, x) float32 - Longitude coordinates
        Attributes:
            - basin_name: Name of basin
            - crs: Coordinate reference system
            - resolution_m: Grid resolution in meters
            - source_files: JSON string of source paths
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get mask from template
    mask = basin_template["BasinMask"]

    # Get coordinates
    coords = {
        "y": basin_template["y"] if "y" in basin_template.coords else basin_template.coords["y"],
        "x": basin_template["x"] if "x" in basin_template.coords else basin_template.coords["x"],
    }

    # Add 2D lat/lon if available
    if "latitude" in basin_template:
        coords["latitude"] = (["y", "x"], basin_template["latitude"].values)
    if "longitude" in basin_template:
        coords["longitude"] = (["y", "x"], basin_template["longitude"].values)

    # Default P_f if not provided
    if p_f is None:
        p_f = xr.full_like(mask, 1.0, dtype="float32")

    # Apply basin mask to all variables (set outside basin to NaN)
    k_masked = apply_basin_mask(k, mask, mask_value=np.nan)
    ls_masked = apply_basin_mask(ls, mask, mask_value=np.nan)
    ic_masked = apply_basin_mask(ic, mask, mask_value=np.nan)
    p_f_masked = apply_basin_mask(p_f, mask, mask_value=np.nan)

    # Create dataset
    ds = xr.Dataset(
        {
            "K": (["y", "x"], k_masked.values.astype("float32")),
            "LS": (["y", "x"], ls_masked.values.astype("float32")),
            "IC": (["y", "x"], ic_masked.values.astype("float32")),
            "P_f": (["y", "x"], p_f_masked.values.astype("float32")),
            "mask": (["y", "x"], mask.values.astype("float32")),
        },
        coords=coords,
    )

    # Add CF-compliant variable attributes
    for var_name, attrs in STATIC_VAR_ATTRS.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs.update(attrs)
        elif var_name in ds.coords:
            ds.coords[var_name].attrs.update(attrs)
    for coord_name, attrs in STATIC_COORD_ATTRS.items():
        if coord_name in ds.coords:
            ds.coords[coord_name].attrs.update(attrs)

    crs = str(basin_template.attrs.get("crs", "")).upper()
    use_geographic_area = "4326" in crs
    if use_geographic_area:
        latitude = basin_template["latitude"] if "latitude" in basin_template else None
        longitude = basin_template["longitude"] if "longitude" in basin_template else None
    else:
        latitude = None
        longitude = None

    s_area = calculate_basin_area(
        mask,
        cell_size_km=1.0,
        latitude=latitude,
        longitude=longitude,
    )

    # Add attributes
    ds.attrs["basin_name"] = basin_name or "unknown"
    ds.attrs["s_area_m2"] = float(s_area)
    ds.attrs["s_area_km2"] = float(s_area / 1e6)
    ds.attrs["n_grid_cells"] = int(np.sum(mask.values > 0))

    # Copy CRS from template if available
    if "crs" in basin_template.attrs:
        ds.attrs["crs"] = basin_template.attrs["crs"]
    if "target_resolution_m" in basin_template.attrs:
        ds.attrs["resolution_m"] = basin_template.attrs["target_resolution_m"]

    # Store source files
    if source_files:
        import json
        ds.attrs["source_files"] = json.dumps(source_files)

    # Save to NetCDF
    ds.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=build_netcdf_compression_encoding(ds),
    )

    return output_path


def build_dynamic_nc(
    output_path: Path | str,
    basin_template: xr.Dataset,
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    ndvi: xr.DataArray,
    basin_name: str | None = None,
    source_files: dict[str, str] | None = None,
    quality_report_path: Path | str | None = None,
) -> Path:
    """Build dynamic.nc file containing time-varying data.

    Args:
        output_path: Path to output file (e.g., "dynamic.nc")
        basin_template: Basin template with spatial coordinates
        temperature: Temperature DataArray (time, y, x) in Celsius
        precipitation: Precipitation DataArray (time, y, x) in mm/month
        ndvi: NDVI DataArray (time, y, x) or (member, time, y, x) [0, 1]
        basin_name: Name of the basin
        source_files: Dictionary of source file paths
        quality_report_path: Optional path to save quality report

    Returns:
        Path to created output file

    Output Format:
        Dimensions: (time, y, x) or (member, time, y, x) for NDVI with ensemble
        Variables:
            - T: (time, y, x) float32 - Temperature (°C)
            - Pre: (time, y, x) float32 - Precipitation (mm/month)
            - NDVI: (time, y, x) or (member, time, y, x) float32 - NDVI [0, 1]
            - T_mean: (time,) float32 - Basin-averaged temperature (°C)
            - Pre_mean: (time,) float32 - Basin-averaged precipitation (mm/month)
            - NDVI_mean: (time,) float32 - Basin-averaged NDVI [0, 1]
        Coordinates:
            - time: datetime64[ns]
            - y, x: spatial coordinates
            - latitude, longitude: 2D coordinates
            - member: (optional) ensemble member names for NDVI
        Attributes:
            - basin_name: Name of basin
            - time_range: [start_time, end_time]
            - frequency: "monthly"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get mask from template
    mask = basin_template["BasinMask"]

    # Get spatial coordinates from template
    y = basin_template["y"]
    x = basin_template["x"]

    # Ensure all arrays have same time coordinate
    time_coord = temperature["time"]

    # Align precipitation to same time coordinate
    if not np.array_equal(precipitation["time"].values, time_coord.values):
        precipitation = precipitation.reindex(time=time_coord)  # NaN for missing times

    # Check if NDVI has member dimension
    has_member_dim = "member" in ndvi.dims

    # Determine spatial dimension names (could be y/x or latitude/longitude)
    spatial_dims = [d for d in ndvi.dims if d not in ("member", "time")]

    if has_member_dim:
        # NDVI has (member, time, spatial...) dims
        # Ensure correct dimension order: (member, time, y, x)
        target_dims = ("member", "time") + tuple(spatial_dims)
        if ndvi.dims != target_dims:
            ndvi = ndvi.transpose(*target_dims)
        # Align time coordinate (NaN for missing times)
        if not np.array_equal(ndvi["time"].values, time_coord.values):
            ndvi = ndvi.interp(time=time_coord)
    else:
        # NDVI has (time, spatial...) dims
        target_dims = ("time",) + tuple(spatial_dims)
        if ndvi.dims != target_dims:
            ndvi = ndvi.transpose(*target_dims)
        if not np.array_equal(ndvi["time"].values, time_coord.values):
            ndvi = ndvi.interp(time=time_coord)

    # Rename spatial dimensions to y/x if needed
    if "latitude" in ndvi.dims and "y" not in ndvi.dims:
        ndvi = ndvi.rename({"latitude": "y", "longitude": "x"})

    # Apply basin mask to dynamic variables (set outside basin to NaN)
    # Ensure proper dimension alignment before masking
    if temperature.dims != ("time", "y", "x"):
        temperature = temperature.transpose("time", "y", "x")
    if precipitation.dims != ("time", "y", "x"):
        precipitation = precipitation.transpose("time", "y", "x")

    # Validate spatial dimensions match mask
    if temperature.shape[1:] != mask.shape:
        raise ValueError(
            f"Temperature spatial dimensions {temperature.shape[1:]} "
            f"do not match basin mask {mask.shape}"
        )
    if precipitation.shape[1:] != mask.shape:
        raise ValueError(
            f"Precipitation spatial dimensions {precipitation.shape[1:]} "
            f"do not match basin mask {mask.shape}"
        )

    # Handle NDVI spatial validation
    if has_member_dim:
        if ndvi.shape[2:] != mask.shape:
            raise ValueError(
                f"NDVI spatial dimensions {ndvi.shape[2:]} "
                f"do not match basin mask {mask.shape}"
            )
    else:
        if ndvi.shape[1:] != mask.shape:
            raise ValueError(
                f"NDVI spatial dimensions {ndvi.shape[1:]} "
                f"do not match basin mask {mask.shape}"
            )

    # Apply mask using xarray where (memory-efficient broadcasting)
    temperature_masked = xr.where(mask > 0, temperature, np.nan).transpose("time", "y", "x").values
    precipitation_masked = xr.where(mask > 0, precipitation, np.nan).transpose("time", "y", "x").values

    # Apply mask to NDVI
    if has_member_dim:
        # NDVI: (member, time, y, x)
        ndvi = ndvi.transpose("member", "time", "y", "x")
        ndvi_masked = xr.where(mask > 0, ndvi, np.nan).transpose("member", "time", "y", "x").values
    else:
        ndvi = ndvi.transpose("time", "y", "x")
        ndvi_masked = xr.where(mask > 0, ndvi, np.nan).transpose("time", "y", "x").values

    # Calculate basin-averaged time series (spatial mean over valid cells)
    # Use 2D mask for spatial averaging
    mask_2d = mask.values > 0
    temp_mean = _nanmean_last_axis(temperature.values[:, mask_2d])
    precip_mean = _nanmean_last_axis(precipitation.values[:, mask_2d])

    # NDVI mean: average over space only (keep member dimension if present)
    if has_member_dim:
        # Average over space for each member: (member, time)
        ndvi_mean = _nanmean_last_axis(ndvi.values[:, :, mask_2d])  # (member, time)
    else:
        ndvi_mean = _nanmean_last_axis(ndvi.values[:, mask_2d])  # (time,)

    # Build coordinates dict
    coords = {
        "time": time_coord.values,
        "y": y.values,
        "x": x.values,
    }

    # Add member coordinate if NDVI has ensemble
    if has_member_dim:
        coords["member"] = ndvi["member"].values

    # Add 2D lat/lon if available in template
    if "latitude" in basin_template:
        coords["latitude"] = (["y", "x"], basin_template["latitude"].values)
    if "longitude" in basin_template:
        coords["longitude"] = (["y", "x"], basin_template["longitude"].values)

    # Create dataset with masked values
    data_vars = {
        "T": (["time", "y", "x"], temperature_masked.astype("float32")),
        "Pre": (["time", "y", "x"], precipitation_masked.astype("float32")),
        # Basin-averaged time series
        "T_mean": (["time"], temp_mean.astype("float32")),
        "Pre_mean": (["time"], precip_mean.astype("float32")),
    }

    # Add NDVI with appropriate dimensions
    if has_member_dim:
        data_vars["NDVI"] = (["member", "time", "y", "x"], ndvi_masked.astype("float32"))
        data_vars["NDVI_mean"] = (["member", "time"], ndvi_mean.astype("float32"))
    else:
        data_vars["NDVI"] = (["time", "y", "x"], ndvi_masked.astype("float32"))
        data_vars["NDVI_mean"] = (["time"], ndvi_mean.astype("float32"))

    ds = xr.Dataset(data_vars, coords=coords)

    # Add CF-compliant variable and coordinate attributes
    for var_name, attrs in DYNAMIC_VAR_ATTRS.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs.update(attrs)
    for coord_name, attrs in DYNAMIC_COORD_ATTRS.items():
        if coord_name in ds.coords:
            ds.coords[coord_name].attrs.update(attrs)

    # Add member coordinate attributes
    if "member" in ds.coords:
        ds.coords["member"].attrs["long_name"] = "Ensemble member name"

    # Add attributes
    ds.attrs["basin_name"] = basin_name or "unknown"
    ds.attrs["time_range"] = [
        str(time_coord.values[0]),
        str(time_coord.values[-1]),
    ]
    ds.attrs["frequency"] = "monthly"
    ds.attrs["n_time_steps"] = len(time_coord)

    if has_member_dim:
        ds.attrs["n_ndvi_members"] = len(ndvi["member"])

    # Store source files
    if source_files:
        ds.attrs["source_files"] = json.dumps(source_files)

    # Save to NetCDF
    ds.to_netcdf(
        output_path,
        engine="netcdf4",
        encoding=build_netcdf_compression_encoding(ds),
    )

    return output_path


def build_observations_nc(
    output_path: Path | str,
    q_series: pd.Series,
    ssf_series: pd.Series,
    s_area: float,
    basin_name: str | None = None,
    station_name: str | None = None,
    source_file: str | None = None,
) -> Path:
    """Build observations.nc file containing gauge observations.

    Args:
        output_path: Path to output file (e.g., "observations.nc")
        q_series: Discharge time series (m³/s) with DatetimeIndex
        ssf_series: Sediment flux time series (tons/day) with DatetimeIndex
        s_area: Basin area in square meters
        basin_name: Name of the basin
        station_name: Name of the gauge station
        source_file: Path to original observation file

    Returns:
        Path to created output file

    Output Format:
        Dimensions: (time,)
        Variables:
            - Q: (time,) float32 - Discharge (m³/s)
            - SSF: (time,) float32 - Suspended sediment flux (tons/day)
        Coordinates:
            - time: datetime64[ns]
        Attributes:
            - basin_name: Name of basin
            - station_name: Name of gauge station
            - s_area: Basin area in m²
            - s_area_km2: Basin area in km²
            - source_file: Path to original data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure both series have same index
    if not q_series.index.equals(ssf_series.index):
        # Reindex to common time range
        common_index = q_series.index.intersection(ssf_series.index)
        q_series = q_series.reindex(common_index)
        ssf_series = ssf_series.reindex(common_index)

    # Convert to xarray
    ds = xr.Dataset(
        {
            "Q": (["time"], q_series.values.astype("float32")),
            "SSF": (["time"], ssf_series.values.astype("float32")),
        },
        coords={
            "time": q_series.index,
        },
    )

    # Add CF-compliant variable and coordinate attributes
    for var_name, attrs in OBS_VAR_ATTRS.items():
        if var_name in ds.data_vars:
            ds[var_name].attrs.update(attrs)
    for coord_name, attrs in OBS_COORD_ATTRS.items():
        if coord_name in ds.coords:
            ds.coords[coord_name].attrs.update(attrs)

    # Add attributes
    ds.attrs["basin_name"] = basin_name or "unknown"
    ds.attrs["station_name"] = station_name or "unknown"
    ds.attrs["s_area"] = float(s_area)
    ds.attrs["s_area_km2"] = float(s_area / 1e6)
    ds.attrs["time_range"] = [
        str(q_series.index[0]),
        str(q_series.index[-1]),
    ]

    if source_file:
        ds.attrs["source_file"] = str(source_file)

    # Save to NetCDF
    ds.to_netcdf(output_path, engine="netcdf4")

    return output_path

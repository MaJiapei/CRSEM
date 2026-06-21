"""Legacy data loading functions compatible with existing data formats."""

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


def load_static_legacy(
    k_path: Path | str,
    ls_path: Path | str,
    ic_path: Path | str,
    basin_mask: xr.DataArray,
    **kwargs: Any,
) -> dict[str, xr.DataArray]:
    """Load static spatial data from legacy file formats.

    Args:
        k_path: Path to K factor NetCDF (e.g., K_combine_1km.nc)
        ls_path: Path to LS factor NetCDF (e.g., LS_1km.nc)
        ic_path: Path to IC factor GeoTIFF (e.g., IC_geos.tif)
        basin_mask: Basin mask DataArray for spatial subsetting
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        Dictionary with keys 'K', 'LS', 'IC', 'P_f'

    Example:
        >>> from CRSEM.data_preparation import load_basin_template
        >>> template = load_basin_template("basin_static.nc")
        >>> mask = template["BasinMask"]
        >>> static = load_static_legacy(
        ...     k_path="K_combine_1km.nc",
        ...     ls_path="LS_1km.nc",
        ...     ic_path="IC_geos.tif",
        ...     basin_mask=mask
        ... )
    """
    # Load K factor
    k_ds = xr.open_dataset(k_path).astype("float32")
    # Select "Dai" dataset if available
    if "name" in k_ds["K"].coords:
        k_da = k_ds["K"].sel(name="Dai")
    else:
        k_da = k_ds["K"]

    # Load LS factor
    ls_ds = xr.open_dataset(ls_path).astype("float32")
    ls_da = ls_ds["LS"]

    # Load IC factor (from GeoTIFF)
    ic_ds = xr.open_dataset(ic_path)
    # GeoTIFF loaded via xarray typically has band_data variable
    if "band_data" in ic_ds:
        ic_da = ic_ds["band_data"]
        # Select first band if multi-band
        if "band" in ic_da.dims:
            ic_da = ic_da.sel(band=1)
        # Rename coordinates to match standard
        coord_map = {}
        if "x" in ic_da.coords and "x" not in ic_da.dims:
            coord_map["x"] = "x"
        if "y" in ic_da.coords and "y" not in ic_da.dims:
            coord_map["y"] = "y"
        # Standardize coordinate names
        ic_da = ic_da.rename({"x": "longitude", "y": "latitude"} if "longitude" not in ic_da.coords else {})
    else:
        # Fallback: try to find the data variable
        data_vars = list(ic_ds.data_vars)
        if data_vars:
            ic_da = ic_ds[data_vars[0]]
        else:
            raise ValueError(f"Could not find IC data in {ic_path}")

    # Create P_f (default all 1s, matching K shape)
    p_f_da = xr.full_like(k_da, 1.0, dtype="float32")

    return {
        "K": k_da,
        "LS": ls_da,
        "IC": ic_da,
        "P_f": p_f_da,
    }


def load_meteo_legacy(
    path: Path | str,
    dataset: str,
    year_range: tuple[int, int],
    **kwargs: Any,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load meteorological data from legacy file formats.

    Args:
        path: Path to merged meteorology NetCDF
        dataset: Dataset name (e.g., "ERA5-Land", "TerraClimate")
        year_range: (start_year, end_year) tuple
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        Tuple of (temperature, precipitation) DataArrays

    Note:
        Temperature values are converted from Kelvin to Celsius if needed.
        Precipitation units are preserved from source.
    """
    start_year, end_year = year_range

    # Open dataset
    ds = xr.open_dataset(path, engine="h5netcdf")

    # Select time range and dataset
    time_slice = slice(str(start_year), str(end_year))

    if "name" in ds.coords:
        ds_selected = ds.sel(name=dataset, time=time_slice)
    else:
        ds_selected = ds.sel(time=time_slice)

    # Get temperature and precipitation
    temp = ds_selected.get("2t") or ds_selected.get("tmean") or ds_selected.get("tmp")
    precip = ds_selected.get("tp") or ds_selected.get("ppt") or ds_selected.get("pre")

    if temp is None or precip is None:
        raise ValueError(
            f"Could not find temperature/precipitation variables in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    # Convert temperature to Celsius if it appears to be in Kelvin
    if temp.mean() > 200:  # Rough check for Kelvin
        temp = temp - 273.15

    return temp.astype("float32"), precip.astype("float32")


def load_ndvi_legacy(
    path: Path | str,
    dataset: str,
    year_range: tuple[int, int],
    **kwargs: Any,
) -> xr.DataArray:
    """Load NDVI data from legacy file formats.

    Args:
        path: Path to merged NDVI NetCDF
        dataset: Dataset name (e.g., "AVHRR_GIMMS", "AVHRR_MODIS")
        year_range: (start_year, end_year) tuple
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        NDVI DataArray with values clipped to [0, 1] range
    """
    start_year, end_year = year_range

    # Open dataset
    ds = xr.open_dataset(path, engine="h5netcdf")

    # Select time range and dataset
    time_slice = slice(str(start_year), str(end_year))

    if "name" in ds.coords:
        da = ds["NDVI"].sel(name=dataset, time=time_slice)
    else:
        da = ds["NDVI"].sel(time=time_slice)

    # Clip to valid range [0, 1] and interpolate missing values
    da = xr.where((da >= 0) & (da <= 1), da, float("nan"))
    da = da.interpolate_na(dim="time", method="nearest", fill_value="extrapolate", limit=3)

    return da.astype("float32")


def load_observations_legacy(
    excel_path: Path | str,
    sheet_name: str,
    year_range: tuple[int, int],
    **kwargs: Any,
) -> tuple[pd.Series, pd.Series, float]:
    """Load river observations from legacy Excel format.

    Args:
        excel_path: Path to observations Excel file
        sheet_name: Sheet name (e.g., "沱沱河", "直门达")
        year_range: (start_year, end_year) tuple
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        Tuple of (Q_series, SSF_series, s_area)
        - Q: discharge in m³/s
        - SSF: suspended sediment flux in tons/day
        - s_area: basin area (placeholder, should be calculated from mask)

    Note:
        s_area should be calculated from basin mask, not from this function.
        This function returns a placeholder value.
    """
    # Read Excel directly
    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        parse_dates=True,
        index_col="Date",
    )

    # Calculate SSF if not present
    if "SSF.t_day" not in df.columns and "SSC.kgm-3" in df.columns and "Q.m3s-1" in df.columns:
        df["SSF.t_day"] = df["Q.m3s-1"] * df["SSC.kgm-3"] * 86400 / 1000

    # Filter to year range
    start_year, end_year = year_range
    df_filtered = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    # Rename columns to standardized names
    q_col = "Q.m3s-1" if "Q.m3s-1" in df_filtered.columns else "Q.m3s-1_month"
    ssf_col = "SSF.t_day" if "SSF.t_day" in df_filtered.columns else "SSC_load_t_month"

    Q = df_filtered[q_col]
    SSF = df_filtered[ssf_col]

    # Return placeholder s_area (will be calculated from mask)
    s_area = kwargs.get("s_area", 0.0)

    return Q, SSF, s_area


def load_observations_csv(
    csv_path: Path | str,
    date_col: str = "Date",
    q_col: str = "Q",
    ssf_col: str = "SSF",
    year_range: tuple[int, int] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Load river observations from CSV format.

    Args:
        csv_path: Path to CSV file
        date_col: Date column name
        q_col: Discharge column name
        ssf_col: Sediment flux column name
        year_range: Optional (start_year, end_year) for filtering

    Returns:
        Tuple of (Q_series, SSF_series)
    """
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df.set_index(date_col, inplace=True)

    Q = df[q_col]
    SSF = df[ssf_col]

    if year_range:
        start_year, end_year = year_range
        mask = (Q.index.year >= start_year) & (Q.index.year <= end_year)
        Q = Q[mask]
        SSF = SSF[mask]

    return Q, SSF

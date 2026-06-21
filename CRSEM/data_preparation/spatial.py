"""Spatial processing tools for data preparation."""

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


MASK_VARIABLE_CANDIDATES = ("BasinMask", "bd_mask", "mask", "basin_mask")


def _select_basin_mask_var(ds: xr.Dataset, mask_var: str | None = None) -> str:
    """Pick the basin mask variable from a template dataset."""
    if mask_var:
        if mask_var not in ds.data_vars:
            raise ValueError(
                f"Basin template missing requested mask variable: {mask_var}"
            )
        return mask_var

    for candidate in MASK_VARIABLE_CANDIDATES:
        if candidate in ds.data_vars:
            return candidate

    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))

    available = ", ".join(ds.data_vars)
    raise ValueError(
        "Basin template missing a recognized mask variable. "
        f"Tried {MASK_VARIABLE_CANDIDATES}; available variables: {available}"
    )


def _infer_template_crs(ds: xr.Dataset, mask_dims: tuple[str, ...]) -> str | None:
    """Infer CRS from attributes or spatial dimension names."""
    crs = ds.attrs.get("crs")
    if crs:
        return str(crs)

    if mask_dims == ("latitude", "longitude"):
        return "EPSG:4326"

    return None


def _build_latlon_coords(
    ds: xr.Dataset,
    mask: xr.DataArray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return 2D latitude/longitude arrays if they can be derived."""
    if "latitude" in ds and "longitude" in ds:
        lat = ds["latitude"]
        lon = ds["longitude"]

        if lat.ndim == 2 and lon.ndim == 2:
            return lat.values, lon.values

        if lat.ndim == 1 and lon.ndim == 1:
            lat_2d, lon_2d = np.meshgrid(lat.values, lon.values, indexing="ij")
            return lat_2d, lon_2d

    if {"y", "x"}.issubset(mask.coords):
        y_2d, x_2d = np.meshgrid(mask["y"].values, mask["x"].values, indexing="ij")
        return y_2d, x_2d

    return None, None


def load_basin_template(path: Path | str, mask_var: str | None = None) -> xr.Dataset:
    """Load basin template NetCDF file.

    The template defines the spatial extent and grid for all basin data.
    Supported mask variables:
        - BasinMask
        - bd_mask
        - mask / basin_mask

    The returned dataset is normalized to the internal convention:
        - BasinMask: 2D mask array (1 = inside basin, 0 = outside)
        - y/x: 1D spatial axes
        - latitude/longitude: optional 2D coordinate arrays

    Args:
        path: Path to basin template NetCDF file (e.g., basin_static.nc)
        mask_var: Optional explicit mask variable name

    Returns:
        Dataset containing basin mask and normalized spatial coordinates

    Example:
        >>> ds = load_basin_template("basin_static.nc")
        >>> print(ds.dims)  # Frozen({'y': 175, 'x': 256})
        >>> mask = ds["BasinMask"]
    """
    with xr.open_dataset(path) as raw_ds:
        ds = raw_ds.load()

    mask_name = _select_basin_mask_var(ds, mask_var=mask_var)
    mask = ds[mask_name]

    if mask.ndim != 2:
        raise ValueError(
            f"Basin mask '{mask_name}' must be 2D, got dims {mask.dims}"
        )

    if mask.dims == ("y", "x"):
        y = mask["y"].values
        x = mask["x"].values
    elif mask.dims == ("latitude", "longitude"):
        y = ds["latitude"].values
        x = ds["longitude"].values
        mask = mask.rename({"latitude": "y", "longitude": "x"})
    else:
        raise ValueError(
            "Basin template has unsupported spatial dimensions: "
            f"{mask.dims}. Expected ('y', 'x') or ('latitude', 'longitude')."
        )

    lat_2d, lon_2d = _build_latlon_coords(ds, mask)

    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {
        "BasinMask": (("y", "x"), mask.values.astype("float32")),
    }
    if lat_2d is not None and lon_2d is not None:
        data_vars["latitude"] = (("y", "x"), lat_2d.astype("float32"))
        data_vars["longitude"] = (("y", "x"), lon_2d.astype("float32"))

    normalized = xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": y.astype("float32"),
            "x": x.astype("float32"),
        },
        attrs=dict(ds.attrs),
    )

    crs = _infer_template_crs(ds, tuple(ds[mask_name].dims))
    if crs and "crs" not in normalized.attrs:
        normalized.attrs["crs"] = crs

    return normalized


def get_basin_crs(ds: xr.Dataset) -> str | None:
    """Extract CRS from basin dataset attributes.

    Args:
        ds: Basin dataset with attributes

    Returns:
        CRS string (e.g., "EPSG:32646") or None if not found
    """
    return ds.attrs.get("crs")


def _coordinate_bounds_from_centers(values: np.ndarray) -> np.ndarray:
    """Estimate cell bounds from cell-center coordinates."""
    coords = np.asarray(values, dtype="float64")
    if coords.ndim != 1 or coords.size < 2:
        raise ValueError("Geographic area calculation requires 1D coordinates with at least two points.")

    mids = 0.5 * (coords[:-1] + coords[1:])
    first = coords[0] - (mids[0] - coords[0])
    last = coords[-1] + (coords[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def _as_rectilinear_axis(values: xr.DataArray | np.ndarray, *, axis: str) -> np.ndarray:
    """Extract a 1D rectilinear axis from 1D or 2D lat/lon coordinates."""
    arr = np.asarray(values, dtype="float64")
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Unsupported {axis} coordinate rank: {arr.ndim}")

    if axis == "latitude":
        candidate = arr[:, 0]
        if not np.allclose(arr, candidate[:, None], equal_nan=True):
            raise ValueError("Latitude coordinates are not rectilinear; cannot infer cell areas.")
        return candidate

    candidate = arr[0, :]
    if not np.allclose(arr, candidate[None, :], equal_nan=True):
        raise ValueError("Longitude coordinates are not rectilinear; cannot infer cell areas.")
    return candidate


def _calculate_geographic_basin_area(
    basin_mask: xr.DataArray,
    latitude: xr.DataArray | np.ndarray,
    longitude: xr.DataArray | np.ndarray,
) -> float:
    """Calculate basin area on a latitude/longitude grid using spherical cells."""
    lat_axis = _as_rectilinear_axis(latitude, axis="latitude")
    lon_axis = _as_rectilinear_axis(longitude, axis="longitude")

    lat_bounds = _coordinate_bounds_from_centers(lat_axis)
    lon_bounds = _coordinate_bounds_from_centers(lon_axis)

    earth_radius_m = 6_371_008.8
    lat_south = np.deg2rad(lat_bounds[:-1])
    lat_north = np.deg2rad(lat_bounds[1:])
    dlon = np.abs(np.deg2rad(lon_bounds[1:] - lon_bounds[:-1]))
    cell_area = (
        earth_radius_m ** 2
        * np.abs(np.sin(lat_north) - np.sin(lat_south))[:, None]
        * dlon[None, :]
    )

    valid = np.asarray(basin_mask.values > 0)
    if cell_area.shape != valid.shape:
        raise ValueError(
            "Latitude/longitude coordinates do not match basin mask shape: "
            f"{cell_area.shape} vs {valid.shape}"
        )
    return float(cell_area[valid].sum())


def calculate_basin_area(
    basin_mask: xr.DataArray,
    cell_size_km: float = 1.0,
    latitude: xr.DataArray | np.ndarray | None = None,
    longitude: xr.DataArray | np.ndarray | None = None,
) -> float:
    """Calculate basin area from mask.

    Args:
        basin_mask: 2D mask array (1 = inside basin, 0 = outside)
        cell_size_km: Grid cell size in km for projected grids
        latitude: Optional latitude coordinates for geographic grids
        longitude: Optional longitude coordinates for geographic grids

    Returns:
        Basin area in square meters
    """
    if latitude is not None and longitude is not None:
        return _calculate_geographic_basin_area(
            basin_mask,
            latitude=latitude,
            longitude=longitude,
        )

    n_cells = int(np.sum(basin_mask.values > 0))
    cell_area_m2 = (cell_size_km * 1000) ** 2
    return float(n_cells * cell_area_m2)


def align_to_basin(
    src_da: xr.DataArray,
    template: xr.Dataset,
    method: str = "nearest",
    fill_value: Any = np.nan,
) -> xr.DataArray:
    """Align source data array to basin template grid.

    Uses nearest neighbor resampling to match the template's spatial grid.
    Handles coordinate system differences if rioxarray is available.

    Args:
        src_da: Source data array with spatial coordinates
                  Can have dims (time, y, x), (member, time, y, x), or (y, x)
        template: Basin template dataset with target grid
        method: Resampling method (default "nearest")
        fill_value: Value for areas outside source data extent

    Returns:
        Aligned data array matching template grid

    Raises:
        ValueError: If source data cannot be aligned
    """
    template_mask = template["BasinMask"]

    # Get target coordinates from template
    target_y = template_mask.y.values
    target_x = template_mask.x.values

    # Check if source already has compatible coordinates
    src_coords = set(src_da.coords.keys())
    has_yx = "y" in src_coords and "x" in src_coords
    has_latlon = "latitude" in src_coords and "longitude" in src_coords

    result = src_da

    # Check if data has time and/or member dimensions
    has_time = "time" in src_da.dims
    has_member = "member" in src_da.dims

    if has_yx:
        # Source has y/x coordinates, try direct reindex
        result = src_da.reindex(
            y=target_y,
            x=target_x,
            method=method,
        )
    elif has_latlon:
        # Source has lat/lon, need to handle reprojection
        template_lat = template.get("latitude")
        template_lon = template.get("longitude")

        if template_lat is not None and template_lon is not None:
            # Template has 2D lat/lon, use them for selection
            try:
                import rioxarray  # noqa: F401

                # Use xarray's interp - it will preserve member dimension if present
                result = src_da.interp(
                    latitude=template_lat,
                    longitude=template_lon,
                    method=method,
                )
            except ImportError:
                # Fallback: use interp
                result = src_da.interp(
                    latitude=template_lat,
                    longitude=template_lon,
                    method=method,
                )
        else:
            # Template only has y/x in projected CRS
            # Need to use rioxarray for reprojection
            try:
                import rioxarray  # noqa: F401

                src_crs = src_da.attrs.get("crs", "EPSG:4326")
                template_crs = get_basin_crs(template) or "EPSG:32646"

                # Determine the order of non-spatial dimensions
                non_spatial_dims = [d for d in src_da.dims if d not in ("latitude", "longitude", "y", "x")]
                spatial_dims = ["latitude", "longitude"] if "latitude" in src_da.dims else ["y", "x"]

                if non_spatial_dims:
                    # Transpose to put non-spatial dims first, then spatial
                    src_da_rio = src_da.transpose(*non_spatial_dims, *spatial_dims)
                else:
                    src_da_rio = src_da

                template_mask_rio = template_mask.rio.write_crs(template_crs)
                result = src_da_rio.rio.write_crs(src_crs).rio.reproject_match(
                    template_mask_rio
                )
            except ImportError:
                raise ImportError(
                    "rioxarray is required for coordinate reprojection. "
                    "Install with: pip install rioxarray"
                )
    else:
        raise ValueError(
            f"Source data has unsupported coordinates: {src_coords}. "
            "Expected ('y', 'x') or ('latitude', 'longitude')."
        )

    # Ensure coordinates match template exactly
    if "y" in result.dims:
        result = result.assign_coords(y=template_mask.y)
    if "x" in result.dims:
        result = result.assign_coords(x=template_mask.x)

    # Rename latitude/longitude to y/x for consistency
    if "latitude" in result.dims and "y" not in result.dims:
        result = result.rename({"latitude": "y", "longitude": "x"})
    if "latitude" in result.coords and "y" not in result.coords:
        result = result.rename({"latitude": "y", "longitude": "x"})

    return result


def apply_basin_mask(
    da: xr.DataArray,
    basin_mask: xr.DataArray,
    mask_value: Any = np.nan,
) -> xr.DataArray:
    """Apply basin mask to data array.

    Args:
        da: Input data array
        basin_mask: Mask array (1 = inside basin, 0 = outside)
        mask_value: Value to use for masked areas (default: NaN)

    Returns:
        Masked data array
    """
    return xr.where(basin_mask > 0, da, mask_value)


def get_spatial_coords(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Extract spatial coordinates from dataset.

    Args:
        ds: Dataset with spatial coordinates

    Returns:
        Dictionary of coordinate arrays (y, x, latitude, longitude)
    """
    coords = {}
    for coord_name in ["y", "x", "latitude", "longitude"]:
        if coord_name in ds.coords:
            coords[coord_name] = ds[coord_name]
    return coords

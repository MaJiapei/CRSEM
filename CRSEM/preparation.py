from __future__ import annotations

import numpy as np
import xarray as xr

from CRSEM.contracts import PreparedInputs, RunContext


_DYNAMIC_FIELDS = ("T", "Pre", "NDVI")
_STATIC_FIELDS = ("LS", "P_f", "IC", "K")


def _coord_values_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    return np.array_equal(np.asarray(left.values), np.asarray(right.values))



def _first_available_dynamic(inputs):
    for name in _DYNAMIC_FIELDS:
        value = getattr(inputs, name)
        if value is not None:
            return value
    raise ValueError("At least one dynamic input (T/Pre/NDVI) is required.")


def _validate_dynamic_inputs(inputs, ref_dynamic: xr.DataArray, spatial_dims: tuple[str, ...]) -> None:
    ref_time = ref_dynamic.coords["time"]
    ref_spatial_dims = tuple(dim for dim in ref_dynamic.dims if dim not in ("time", "member"))

    for name in _DYNAMIC_FIELDS:
        da = getattr(inputs, name)
        if da is None:
            continue
        if "time" not in da.dims:
            raise ValueError(f"Dynamic input '{name}' must include a 'time' dimension.")

        # Get spatial dims (exclude time and member)
        da_spatial_dims = tuple(dim for dim in da.dims if dim not in ("time", "member"))

        # Allow NDVI to have member dimension
        if name == "NDVI" and "member" in da.dims:
            # NDVI can have (member, time) or (member, time, ...) structure
            if da_spatial_dims != ref_spatial_dims:
                raise ValueError(
                    f"Dynamic input '{name}' has spatial dims {da_spatial_dims}, expected {ref_spatial_dims}."
                )
        elif da_spatial_dims != ref_spatial_dims:
            raise ValueError(
                f"Dynamic input '{name}' has spatial dims {da_spatial_dims}, expected {ref_spatial_dims}."
            )

        if not _coord_values_equal(da.coords["time"], ref_time):
            raise ValueError(f"Dynamic input '{name}' does not share the same time coordinate as the reference input.")
        for dim in spatial_dims:
            if dim not in da.coords:
                raise ValueError(f"Dynamic input '{name}' is missing coordinate '{dim}'.")
            if not _coord_values_equal(da.coords[dim], ref_dynamic.coords[dim]):
                raise ValueError(f"Dynamic input '{name}' does not share the same '{dim}' coordinate as the reference input.")


def _validate_static_inputs(inputs, ref_dynamic: xr.DataArray, spatial_dims: tuple[str, ...]) -> None:
    if not spatial_dims:
        for name in _STATIC_FIELDS:
            da = getattr(inputs, name)
            if da is None:
                continue
            if da.ndim > 1:
                raise ValueError(f"Point-mode static input '{name}' must be scalar or 1D, got shape {da.shape}.")
        return

    for name in _STATIC_FIELDS:
        da = getattr(inputs, name)
        if da is None:
            continue
        if tuple(da.dims) != spatial_dims:
            raise ValueError(f"Static input '{name}' has dims {tuple(da.dims)}, expected {spatial_dims}.")
        for dim in spatial_dims:
            if dim not in da.coords:
                raise ValueError(f"Static input '{name}' is missing coordinate '{dim}'.")
            if not _coord_values_equal(da.coords[dim], ref_dynamic.coords[dim]):
                raise ValueError(f"Static input '{name}' does not share the same '{dim}' coordinate as the reference input.")


def _validate_series_lengths(context: RunContext, n_time: int) -> None:
    if context.q is not None and len(context.q) != n_time:
        raise ValueError(f"RunContext.q length {len(context.q)} does not match time length {n_time}.")
    if context.ssf_obs is not None and len(context.ssf_obs) != n_time:
        raise ValueError(f"RunContext.ssf_obs length {len(context.ssf_obs)} does not match time length {n_time}.")



def _dynamic_to_numpy(da: xr.DataArray, spatial_dims: tuple[str, ...], has_member: bool = False) -> np.ndarray:
    """Convert dynamic DataArray to numpy.

    Args:
        da: Input DataArray
        spatial_dims: Spatial dimension names (excluding time and member)
        has_member: If True, preserve member dimension as first axis

    Returns:
        numpy array with shape:
            - Point mode, no member: (time,)
            - Point mode, with member: (member, time)
            - Grid mode, no member: (time, cell)
            - Grid mode, with member: (member, time, cell)
    """
    if has_member:
        if not spatial_dims:
            # Point mode with member: (member, time)
            return da.transpose("member", "time").values.astype(np.float32)
        else:
            # Grid mode with member: (member, time, cell)
            arr = da.transpose("member", "time", *spatial_dims).values.astype(np.float32)
            return arr.reshape(arr.shape[0], arr.shape[1], -1)
    else:
        if not spatial_dims:
            return da.transpose("time").values.astype(np.float32)
        if len(spatial_dims) != 2:
            raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
        arr = da.transpose("time", *spatial_dims).values.astype(np.float32)
        return arr.reshape(arr.shape[0], -1)



def _static_to_numpy(da: xr.DataArray, spatial_dims: tuple[str, ...]) -> np.ndarray:
    if not spatial_dims:
        return np.asarray(da.values, dtype=np.float32)
    if len(spatial_dims) != 2:
        raise ValueError(f"Unsupported spatial dimensions: {spatial_dims}")
    return da.transpose(*spatial_dims).values.astype(np.float32).reshape(-1)



def prepare_inputs(context: RunContext) -> PreparedInputs:
    """Convert xarray-backed RunContext into numpy-backed PreparedInputs."""

    ref_dynamic = _first_available_dynamic(context.inputs)
    # Exclude both 'time' and 'member' from spatial_dims
    spatial_dims = tuple(dim for dim in ref_dynamic.dims if dim not in ("time", "member"))
    is_point_mode = len(spatial_dims) == 0
    _validate_dynamic_inputs(context.inputs, ref_dynamic, spatial_dims)
    _validate_static_inputs(context.inputs, ref_dynamic, spatial_dims)
    _validate_series_lengths(context, ref_dynamic.sizes["time"])

    latitude = None
    longitude = None
    if not is_point_mode and len(spatial_dims) == 2:
        lat_dim, lon_dim = spatial_dims
        latitude = ref_dynamic.coords[lat_dim].values
        longitude = ref_dynamic.coords[lon_dim].values

    # Check if NDVI has member dimension
    ndvi_has_member = context.inputs.NDVI is not None and "member" in context.inputs.NDVI.dims

    prepared = PreparedInputs(
        T=None if context.inputs.T is None else _dynamic_to_numpy(context.inputs.T, spatial_dims),
        Pre=None if context.inputs.Pre is None else _dynamic_to_numpy(context.inputs.Pre, spatial_dims),
        NDVI=None if context.inputs.NDVI is None else _dynamic_to_numpy(context.inputs.NDVI, spatial_dims, has_member=ndvi_has_member),
        LS=None if context.inputs.LS is None else _static_to_numpy(context.inputs.LS, spatial_dims),
        P_f=None if context.inputs.P_f is None else _static_to_numpy(context.inputs.P_f, spatial_dims),
        IC=None if context.inputs.IC is None else _static_to_numpy(context.inputs.IC, spatial_dims),
        K=None if context.inputs.K is None else _static_to_numpy(context.inputs.K, spatial_dims),
        q=None if context.q is None else np.asarray(context.q.values, dtype=np.float32),
        time=ref_dynamic.coords["time"].values,
        latitude=latitude,
        longitude=longitude,
        spatial_dims=spatial_dims,
        dynamic_dims=("time",) if is_point_mode else ("time", "cell"),
        is_point_mode=is_point_mode,
    )
    return prepared

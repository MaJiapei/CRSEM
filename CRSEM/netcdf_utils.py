from __future__ import annotations

from typing import Any

import xarray as xr


def build_netcdf_compression_encoding(
    dataset: xr.Dataset,
    *,
    complevel: int = 4,
    shuffle: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build netCDF4 compression settings for numeric data variables."""
    encoding: dict[str, dict[str, Any]] = {}
    for name, data_array in dataset.data_vars.items():
        if data_array.dtype.kind in {"O", "U", "S"}:
            continue
        encoding[name] = {
            "zlib": True,
            "complevel": complevel,
            "shuffle": shuffle,
        }
    return encoding

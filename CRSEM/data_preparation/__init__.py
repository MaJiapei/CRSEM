"""Data preparation module for CRSEM.

This module provides tools to prepare and standardize input data
for the CRSEM erosion models. It handles:

- Spatial alignment to basin mask
- Time aggregation (daily to monthly)
- NetCDF file generation (static.nc, dynamic.nc, observations.nc)
- Observation preprocessing (gap filling, aggregation)

Example:
    >>> from CRSEM.data_preparation import build_basin_dataset
    >>> build_basin_dataset(
    ...     config_path="config/basin_data_sources.tuotuohe.yml",
    ...     basin_name="tuotuohe",
    ...     start_year=1990,
    ...     end_year=2000,
    ...     output_dir="outputs/tuotuohe_1990_2000"
    ... )
"""

from .builders import build_dynamic_nc, build_observations_nc, build_static_nc
from .io_legacy import load_meteo_legacy, load_ndvi_legacy, load_observations_legacy, load_static_legacy
from .spatial import align_to_basin, apply_basin_mask, load_basin_template
from .obs_preprocessing import (
    monthly_stack,
    year_complete,
    monthly_agg,
    stack_q,
    stack_ssf,
    interpolate_by_covariate,
    process_raw_observations,
    fill_monthly_observations_by_precip,
)

__all__ = [
    # Spatial tools
    "load_basin_template",
    "align_to_basin",
    "apply_basin_mask",
    # Legacy loaders
    "load_static_legacy",
    "load_meteo_legacy",
    "load_ndvi_legacy",
    "load_observations_legacy",
    # NC builders
    "build_static_nc",
    "build_dynamic_nc",
    "build_observations_nc",
    # Observation preprocessing
    "monthly_stack",
    "year_complete",
    "monthly_agg",
    "stack_q",
    "stack_ssf",
    "interpolate_by_covariate",
    "process_raw_observations",
    "fill_monthly_observations_by_precip",
]

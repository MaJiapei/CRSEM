"""Prepare basin driver data for CRSEM model.

This script prepares input data for any basin using a configuration file.
It is the standard data-preparation entry point for the project.

Usage:
    # With year range (crop data to specified years)
    python scripts/prepare_basin_drivers.py \
        --config config/basin_data_sources.tuotuohe.yml \
        --basin tuotuohe \
        --years 1990 2000 \
        --output example/tuotuohe_1990_2000

    # Without year range (use full data range)
    python scripts/prepare_basin_drivers.py \
        --config config/basin_data_sources.tuotuohe.yml \
        --basin tuotuohe \
        --output example/tuotuohe_full

    # With quality report
    python scripts/prepare_basin_drivers.py \
        --config config/basin_data_sources.tuotuohe.yml \
        --basin tuotuohe \
        --years 1990 2000 \
        --output example/tuotuohe_1990_2000 \
        --quality-report

Multi-dataset support:
    In the config YAML, you can specify multiple datasets for NDVI or meteorological
    data by using comma-separated names. The first dataset will be used as the primary
    data source, and all datasets' time ranges will be recorded in metadata.json.

    Example config:
        datasets:
          meteorological: "ERA5-Land,TerraClimate"
          ndvi: "AVHRR_GIMMS,MODIS"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CRSEM.config import load_runtime_config
from CRSEM.data_preparation.builders import (
    build_dynamic_nc,
    build_observations_nc,
    build_static_nc,
)
from CRSEM.data_preparation.spatial import (
    align_to_basin,
    calculate_basin_area,
    load_basin_template,
)
from CRSEM.data_preparation.quality import (
    generate_quality_report,
    print_quality_report,
)


def windows_to_wsl_path(win_path: str) -> Path:
    """Convert Windows path to WSL path.

    Examples:
        G:\\RTS_route\\file.nc -> /mnt/g/RTS_route/file.nc
        H:\\datasets\\file.nc -> /mnt/h/datasets/file.nc
    """
    match = re.match(r'^([A-Za-z]):([\\/])(.+)$', win_path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(3).replace('\\', '/')
        return Path(f'/mnt/{drive}/{rest}')
    return Path(win_path)


def detect_ensemble_dim(ds: xr.Dataset | xr.DataArray) -> str | None:
    """Detect ensemble/member dimension name in dataset.

    Supports: name, member, ensemble (case-insensitive)

    Args:
        ds: xarray Dataset or DataArray

    Returns:
        Dimension name if found, None otherwise
    """
    # Get all dimension names
    dims = list(ds.dims)

    # Check for ensemble dimension (case-insensitive)
    ensemble_names = ["name", "member", "ensemble"]
    for dim in dims:
        dim_lower = dim.lower()
        if dim_lower in ensemble_names:
            return dim  # Return original case

    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare basin input data for CRSEM"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to data sources config YAML file",
    )
    parser.add_argument(
        "--basin",
        type=str,
        required=True,
        help="Basin name (must match config entry)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Year range (optional, if not specified, use full data range)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Generate and print data quality report",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def load_basin_config(config_path: Path, basin_name: str) -> Dict[str, Any]:
    """Load basin-specific configuration.

    Args:
        config_path: Path to basin data sources YAML
        basin_name: Basin name from config

    Returns:
        Basin configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If basin not found in config
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for data sources config")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if "basins" not in config:
        raise ValueError("Config must contain 'basins' section")

    if basin_name not in config["basins"]:
        available = list(config["basins"].keys())
        raise ValueError(
            f"Basin '{basin_name}' not found in config. Available: {available}"
        )

    basin_config = config["basins"][basin_name]
    basin_config["settings"] = config.get("settings", {})

    return basin_config


def load_tuotuohe_observations(
    csv_path: Path,
    year_range: tuple[int, int] | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """Load observations from CSV.

    Expected CSV columns:
        - Date: datetime
        - Q.m3s-1_month: discharge (m³/s)
        - SSC_load_t_month: suspended sediment load (tons/month)

    Args:
        csv_path: Path to observation CSV
        year_range: (start_year, end_year) tuple. If None, use full data range.

    Returns:
        Tuple of (Q_series, SSF_series, time_info)
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    if year_range is not None:
        start_year, end_year = year_range
        mask = (df.index.year >= start_year) & (df.index.year <= end_year)
        df_filtered = df[mask].copy()
    else:
        df_filtered = df.copy()

    Q = df_filtered["Q.m3s-1_month"]
    SSF = df_filtered["SSC_load_t_month"]

    # Normalize time to first day of month
    Q.index = Q.index.to_period('M').to_timestamp()
    SSF.index = SSF.index.to_period('M').to_timestamp()

    time_info = {
        "start": str(Q.index[0].year),
        "end": str(Q.index[-1].year),
        "n_months": len(Q),
    }

    return Q, SSF, time_info


def load_meteo_data(
    path: Path,
    dataset: str | list[str],
    varnames: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Load meteorological data from merged NetCDF.

    Args:
        path: Path to meteorological data file
        dataset: Dataset name(s) to select. Can be a single name or comma-separated string.
        varnames: [temp_varname, precip_varname]. Default: ["2t", "tp"]
        year_range: (start_year, end_year) tuple. If None, use full data range.

    Returns:
        Tuple of (temperature, precipitation, time_info)
        time_info contains actual time range of each dataset
    """
    if varnames is None:
        varnames = ["2t", "tp"]
    temp_var, precip_var = varnames[0], varnames[1]

    ds = xr.open_dataset(path, engine="h5netcdf")

    # Detect ensemble dimension
    ens_dim = detect_ensemble_dim(ds)
    if ens_dim is None:
        raise ValueError(f"No ensemble dimension (name/member/ensemble) found in {path}")

    # Handle multi-dataset (comma-separated)
    if isinstance(dataset, str):
        dataset_names = [d.strip() for d in dataset.split(",")]
    else:
        dataset_names = dataset

    time_info = {}

    if year_range is not None:
        time_slice = slice(str(year_range[0]), str(year_range[1]))
    else:
        time_slice = None

    # Load first dataset
    first_dataset = dataset_names[0]
    sel_kwargs = {ens_dim: first_dataset}
    if time_slice is not None:
        sel_kwargs["time"] = time_slice

    temp = ds[temp_var].sel(**sel_kwargs)
    precip = ds[precip_var].sel(**sel_kwargs)

    # Record time range
    time_vals = temp.time.values
    time_info[first_dataset] = {
        "start": str(pd.Timestamp(time_vals[0]).year),
        "end": str(pd.Timestamp(time_vals[-1]).year),
        "n_months": len(time_vals),
    }

    # Transpose to (time, latitude, longitude)
    if temp.dims[0] != "time":
        temp = temp.transpose("time", "latitude", "longitude")
        precip = precip.transpose("time", "latitude", "longitude")

    # Convert temperature from Kelvin to Celsius if needed
    if float(temp.mean()) > 200:
        temp = temp - 273.15

    return temp.astype("float32"), precip.astype("float32"), time_info


def load_ndvi_data(
    path: Path,
    dataset: str | list[str],
    varname: str | None = None,
    year_range: tuple[int, int] | None = None,
) -> tuple[xr.DataArray, dict]:
    """Load NDVI data from merged NetCDF.

    Args:
        path: Path to NDVI data file
        dataset: Dataset name(s) to select. Can be a single name or comma-separated string.
                 For multiple datasets, returns all datasets stacked along 'member' dimension.
        varname: Variable name for NDVI. Default: "NDVI"
        year_range: (start_year, end_year) tuple. If None, use full data range.

    Returns:
        Tuple of (NDVI DataArray, time_info)
        If multiple datasets, DataArray has dims (member, time, latitude, longitude)
        time_info contains actual time range of each dataset
    """
    if varname is None:
        varname = "NDVI"

    ds = xr.open_dataset(path, engine="h5netcdf")

    # Detect ensemble dimension
    ens_dim = detect_ensemble_dim(ds)
    if ens_dim is None:
        raise ValueError(f"No ensemble dimension (name/member/ensemble) found in {path}")

    # Handle multi-dataset (comma-separated)
    if isinstance(dataset, str):
        dataset_names = [d.strip() for d in dataset.split(",")]
    else:
        dataset_names = dataset

    time_info = {}

    if year_range is not None:
        time_slice = slice(str(year_range[0]), str(year_range[1]))
    else:
        time_slice = None

    # Load all datasets
    data_arrays = []
    for ds_name in dataset_names:
        sel_kwargs = {ens_dim: ds_name}
        if time_slice is not None:
            sel_kwargs["time"] = time_slice

        da = ds[varname].sel(**sel_kwargs)

        # Record time range
        time_vals = da.time.values
        time_info[ds_name] = {
            "start": str(pd.Timestamp(time_vals[0]).year),
            "end": str(pd.Timestamp(time_vals[-1]).year),
            "n_months": len(time_vals),
        }

        data_arrays.append(da)

    # Stack along 'member' dimension if multiple datasets
    if len(data_arrays) == 1:
        da = data_arrays[0]
    else:
        da = xr.concat(data_arrays, dim="member")
        da = da.assign_coords(member=dataset_names)

    # Transpose to (member, time, latitude, longitude) or (time, latitude, longitude)
    if "member" in da.dims:
        # Desired order: member, time, lat, lon
        spatial_dims = [d for d in da.dims if d not in ("member", "time")]
        target_dims = ["member", "time"] + spatial_dims
        if list(da.dims) != target_dims:
            da = da.transpose(*target_dims)
    else:
        spatial_dims = [d for d in da.dims if d != "time"]
        target_dims = ["time"] + spatial_dims
        if list(da.dims) != target_dims:
            da = da.transpose(*target_dims)

    # Clip to valid range and interpolate along time
    da = xr.where((da >= 0) & (da <= 1), da, float("nan"))
    da = da.interpolate_na(
        dim="time",
        method="nearest",
        fill_value="extrapolate",
        limit=3
    )

    return da.astype("float32"), time_info


def load_static_data(
    k_path: Path,
    ls_path: Path,
    ic_path: Path,
) -> Dict[str, xr.DataArray]:
    """Load static spatial data.

    Args:
        k_path: Path to K factor file
        ls_path: Path to LS factor file
        ic_path: Path to IC factor file

    Returns:
        Dictionary with K, LS, IC DataArrays
    """
    # Load K
    k_ds = xr.open_dataset(k_path).astype("float32")
    if "name" in k_ds["K"].coords:
        k_da = k_ds["K"].sel(name="Dai")
    else:
        k_da = k_ds["K"]

    # Load LS
    ls_ds = xr.open_dataset(ls_path).astype("float32")
    ls_da = ls_ds["LS"]

    # Load IC (from GeoTIFF or NetCDF)
    ic_ds = xr.open_dataset(ic_path)
    if "band_data" in ic_ds:
        ic_da = ic_ds["band_data"].sel(band=1).drop_vars(
            ["band", "spatial_ref"], errors="ignore"
        )
        if "x" in ic_da.coords:
            ic_da = ic_da.rename({"x": "longitude", "y": "latitude"})
    else:
        ic_da = ic_ds[list(ic_ds.data_vars)[0]]

    return {"K": k_da, "LS": ls_da, "IC": ic_da}


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("CRSEM Basin Driver Data Preparation")
    print("=" * 60)

    # Load configuration
    print(f"\n[1/7] Loading configuration from {args.config}")
    basin_config = load_basin_config(args.config, args.basin)

    settings = basin_config.get("settings", {})
    cell_size_km = settings.get("cell_size_km", 1.0)
    quality_settings = settings.get("quality", {})

    # Get quality thresholds
    min_coverage = quality_settings.get("min_time_coverage", 0.95)
    max_missing = quality_settings.get("max_missing_rate", 0.05)
    max_consecutive = quality_settings.get("max_consecutive_missing", 3)

    # Handle optional year range
    year_range = tuple(args.years) if args.years is not None else None
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create organized output structure
    drivers_dir = output_dir / "drivers"
    drivers_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Basin: {basin_config.get('name', args.basin)}")
    if year_range:
        print(f"  Years: {year_range[0]}-{year_range[1]} (cropped)")
    else:
        print(f"  Years: using full data range (no crop)")
    print(f"  Output: {output_dir}")

    # Load basin template
    print(f"\n[2/7] Loading basin template")
    basin_template_path = windows_to_wsl_path(
        basin_config.get("basin_template", "")
    )
    print(f"  Template: {basin_template_path}")

    basin_template = load_basin_template(basin_template_path)
    mask = basin_template["BasinMask"]
    s_area = calculate_basin_area(mask, cell_size_km=cell_size_km)

    print(f"  Grid: {len(basin_template.y)} x {len(basin_template.x)}")
    print(f"  Valid cells: {int(mask.sum())}")
    print(f"  Basin area: {s_area/1e6:.2f} km²")

    # Load and build static data
    print(f"\n[3/7] Building static.nc")
    static_paths = basin_config.get("static", {})
    k_path = windows_to_wsl_path(static_paths.get("k_file", ""))
    ls_path = windows_to_wsl_path(static_paths.get("ls_file", ""))
    ic_path = windows_to_wsl_path(static_paths.get("ic_file", ""))

    print(f"  Loading K from {k_path}")
    print(f"  Loading LS from {ls_path}")
    print(f"  Loading IC from {ic_path}")

    static_data = load_static_data(k_path, ls_path, ic_path)

    print("  Aligning to basin grid...")
    k_aligned = align_to_basin(static_data["K"], basin_template)
    ls_aligned = align_to_basin(static_data["LS"], basin_template)
    ic_aligned = align_to_basin(static_data["IC"], basin_template)

    static_nc_path = drivers_dir / "static.nc"
    build_static_nc(
        output_path=static_nc_path,
        basin_template=basin_template,
        k=k_aligned,
        ls=ls_aligned,
        ic=ic_aligned,
        basin_name=args.basin,
        source_files={
            "k_file": str(k_path),
            "ls_file": str(ls_path),
            "ic_file": str(ic_path),
            "basin_template": str(basin_template_path),
        },
    )
    print(f"  Created: {static_nc_path}")

    # Load and build dynamic data
    print(f"\n[4/7] Building dynamic.nc")
    dynamic_paths = basin_config.get("dynamic", {})
    datasets = basin_config.get("datasets", {})
    varnames = basin_config.get("varnames", {})

    meteo_path = windows_to_wsl_path(dynamic_paths.get("meteo_file", ""))
    ndvi_path = windows_to_wsl_path(dynamic_paths.get("ndvi_file", ""))
    meteo_dataset = datasets.get("meteorological", "ERA5-Land")
    ndvi_dataset = datasets.get("ndvi", "AVHRR_GIMMS")

    # Get variable names from config
    meteo_varnames = varnames.get("meteo", ["2t", "tp"])
    ndvi_varname = varnames.get("ndvi", ["NDVI"])[0] if varnames.get("ndvi") else "NDVI"

    print(f"  Meteorological source: {meteo_dataset}")
    print(f"  NDVI source: {ndvi_dataset}")

    print(f"  Loading temperature and precipitation...")
    temperature, precipitation, meteo_time_info = load_meteo_data(
        path=meteo_path,
        dataset=meteo_dataset,
        varnames=meteo_varnames,
        year_range=year_range,
    )

    print(f"  Loading NDVI...")
    ndvi, ndvi_time_info = load_ndvi_data(
        path=ndvi_path,
        dataset=ndvi_dataset,
        varname=ndvi_varname,
        year_range=year_range,
    )

    print("  Aligning to basin grid...")
    temp_aligned = align_to_basin(temperature, basin_template)
    precip_aligned = align_to_basin(precipitation, basin_template)
    ndvi_aligned = align_to_basin(ndvi, basin_template)

    dynamic_nc_path = drivers_dir / "dynamic.nc"
    build_dynamic_nc(
        output_path=dynamic_nc_path,
        basin_template=basin_template,
        temperature=temp_aligned,
        precipitation=precip_aligned,
        ndvi=ndvi_aligned,
        basin_name=args.basin,
        source_files={
            "meteo_file": str(meteo_path),
            "ndvi_file": str(ndvi_path),
        },
    )
    print(f"  Created: {dynamic_nc_path}")

    # Load and build observations
    print(f"\n[5/7] Building observations.nc")
    obs_csv_path = windows_to_wsl_path(
        basin_config.get("observation_csv", "")
    )
    print(f"  Loading from {obs_csv_path}")

    Q, SSF, obs_time_info = load_tuotuohe_observations(obs_csv_path, year_range)
    print(f"  Loaded {len(Q)} months of observations")

    obs_nc_path = drivers_dir / "observations.nc"
    build_observations_nc(
        output_path=obs_nc_path,
        q_series=Q,
        ssf_series=SSF,
        s_area=s_area,
        basin_name=args.basin,
        station_name=basin_config.get("name", args.basin),
        source_file=str(obs_csv_path),
    )
    print(f"  Created: {obs_nc_path}")

    # Generate quality report if requested
    quality_reports = {}
    if args.quality_report:
        print(f"\n[6/7] Generating quality report")

        quality_reports = generate_quality_report(
            Q=Q,
            SSF=SSF,
            temperature=temp_aligned,
            precipitation=precip_aligned,
            ndvi=ndvi_aligned,
            basin_mask=mask,
            min_time_coverage=min_coverage,
            max_missing_rate=max_missing,
            max_consecutive_missing=max_consecutive,
        )

        # Save quality report to JSON
        quality_report_path = output_dir / "quality_report.json"
        report_data = {
            "basin": args.basin,
            "years": list(year_range) if year_range else None,
            "reports": {
                name: report.to_dict()
                for name, report in quality_reports.items()
            },
            "overall_passed": all(
                r.passed for r in quality_reports.values()
            ),
        }

        with open(quality_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {quality_report_path}")

        # Print to console
        print_quality_report(*quality_reports.values())

    # Determine actual time range: intersection of observations and dynamic variables
    dynamic_coverages = [
        infer_basin_time_coverage(temp_aligned, mask),
        infer_basin_time_coverage(precip_aligned, mask),
        infer_basin_time_coverage(ndvi_aligned, mask),
    ]

    dynamic_coverages = [coverage for coverage in dynamic_coverages if coverage is not None]
    obs_start = Q.index[0]
    obs_end = Q.index[-1]

    if dynamic_coverages:
        dynamic_start = max(coverage[0] for coverage in dynamic_coverages)
        dynamic_end = min(coverage[1] for coverage in dynamic_coverages)
    else:
        dynamic_start = obs_start
        dynamic_end = obs_end

    # Compute intersection
    actual_start = max(dynamic_start, obs_start)
    actual_end = min(dynamic_end, obs_end)

    # Check if there's valid overlap
    if actual_start > actual_end:
        import warnings
        warnings.warn(
            f"No time overlap between dynamic data ({dynamic_start.year}-{dynamic_end.year}) "
            f"and observations ({obs_start.year}-{obs_end.year})"
        )
        actual_years = None
    else:
        actual_years = [actual_start.year, actual_end.year]

    # Write metadata
    print(f"\n[{7 if args.quality_report else 6}/7] Writing metadata")
    metadata = {
        "basin_name": args.basin,
        "station_name": basin_config.get("name", args.basin),
        "requested_years": list(year_range) if year_range else None,
        "actual_years": actual_years,
        "s_area_m2": float(s_area),
        "s_area_km2": float(s_area / 1e6),
        "n_grid_cells": int(mask.sum()),
        "crs": basin_template.attrs.get("crs", "unknown"),
        "files": {
            "static_nc": str(static_nc_path),
            "dynamic_nc": str(dynamic_nc_path),
            "observations_nc": str(obs_nc_path),
        },
        "data_sources": {
            "meteorological": {
                "datasets": normalize_dataset_names(meteo_dataset),
                "varnames": meteo_varnames,
                "time_ranges": meteo_time_info,
            },
            "ndvi": {
                "datasets": normalize_dataset_names(ndvi_dataset),
                "varname": ndvi_varname,
                "time_ranges": ndvi_time_info,
            },
            "observations": {
                "time_range": obs_time_info,
            },
        },
    }

    if quality_reports:
        metadata["quality"] = {
            name: report.passed for name, report in quality_reports.items()
        }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Created: {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)

    print(f"\nOutput directory: {output_dir}")
    if actual_years:
        print(f"Effective time range (intersection): {actual_years[0]}-{actual_years[1]}")
    else:
        print("Warning: No time overlap between dynamic data and observations!")
    print(f"\nDriver files:")
    print(f"  Static:     {static_nc_path}")
    print(f"  Dynamic:    {dynamic_nc_path}")
    print(f"  Obs:        {obs_nc_path}")
    print(f"  Metadata:   {metadata_path}")

    if args.quality_report:
        all_passed = all(r.passed for r in quality_reports.values())
        status = "PASS" if all_passed else "FAIL"
        print(f"\nQuality check: {status}")
        if not all_passed:
            print("  See quality_report.json for details")

    print(f"\nTo use with CRSEM:")
    print(f"  python scripts/calibrate_parameters.py \\")
    print(f"    --static-nc {static_nc_path} \\")
    print(f"    --dynamic-nc {dynamic_nc_path} \\")
    print(f"    --observations-nc {obs_nc_path} \\")
    print(f"    --station-name \"{basin_config.get('name', args.basin)}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())

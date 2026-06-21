from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CRSEM.batch_runner import run_parameter_batch
from CRSEM.contracts import ParameterBatch
from CRSEM.driver import BasinDriver
from CRSEM.model import ModelFactory
from CRSEM.netcdf_utils import build_netcdf_compression_encoding
from CRSEM.result_aggregator import ResultAggregator

DEFAULT_OUTPUT_FILE = "__DEFAULT__"


VARIABLE_ATTRS = {
    "SSF_pred": {"units": "t month-1", "long_name": "Predicted suspended sediment flux"},
    "A_channel": {"units": "t month-1", "long_name": "Channel erosion or deposition contribution"},
    "E_hillslope": {"units": "t ha-1 month-1", "long_name": "Hillslope sediment delivery"},
    "E_hillslope_rain": {"units": "t ha-1 month-1", "long_name": "Rainfall-driven hillslope sediment delivery"},
    "E_hillslope_melt": {"units": "t ha-1 month-1", "long_name": "Snowmelt-driven hillslope sediment delivery"},
    "R_rain": {"units": "MJ mm ha-1 h-1 month-1", "long_name": "Rainfall erosivity factor"},
    "R_melt": {"units": "MJ mm ha-1 h-1 month-1", "long_name": "Snowmelt erosivity factor"},
    "K_factor": {"units": "t ha h ha-1 MJ-1 mm-1", "long_name": "Soil erodibility factor"},
    "C_factor": {"units": "1", "long_name": "Cover management factor"},
    "SDR": {"units": "1", "long_name": "Sediment delivery ratio"},
}

COORD_ATTRS = {
    "time": {"long_name": "time"},
    "latitude": {"units": "degrees_north", "long_name": "latitude"},
    "longitude": {"units": "degrees_east", "long_name": "longitude"},
    "y": {"long_name": "y coordinate"},
    "x": {"long_name": "x coordinate"},
    "member": {"long_name": "ensemble member"},
}


def annotate_output_dataset(
    dataset,
    *,
    model_type: str,
    run_method: str,
    run_mode: str,
    calibration_info: dict | None = None,
):
    for name, attrs in VARIABLE_ATTRS.items():
        if name in dataset.data_vars:
            dataset[name].attrs.update(attrs)

    for coord_name, attrs in COORD_ATTRS.items():
        if coord_name in dataset.coords:
            dataset.coords[coord_name].attrs.update(attrs)

    attrs = {
        "title": "CRSEM model outputs",
        "model_type": model_type,
        "run_method": run_method,
        "run_mode": run_mode,
        "Conventions": "CF-1.10",
    }

    # Add calibration info if available
    if calibration_info:
        if "start_year" in calibration_info:
            attrs["calibration_start_year"] = calibration_info["start_year"]
        if "end_year" in calibration_info:
            attrs["calibration_end_year"] = calibration_info["end_year"]
        if "station_name" in calibration_info:
            attrs["station_name"] = calibration_info["station_name"]
        if "NSE" in calibration_info:
            attrs["calibration_NSE"] = calibration_info["NSE"]
        if "KGE" in calibration_info:
            attrs["calibration_KGE"] = calibration_info["KGE"]
        if "R2" in calibration_info:
            attrs["calibration_R2"] = calibration_info["R2"]
        if "PBIAS" in calibration_info:
            attrs["calibration_PBIAS"] = calibration_info["PBIAS"]

    dataset.attrs.update(attrs)
    return dataset


def infer_model_type(parameter_batch: ParameterBatch, saved_metrics: dict | None) -> str:
    if saved_metrics is not None:
        model_type = saved_metrics.get("model_type")
        if isinstance(model_type, str) and model_type.lower() in {"crsem", "rusle"}:
            return model_type.lower()

    param_names = tuple(parameter_batch.param_names)
    candidates: list[str] = []
    for model_type in ("crsem", "rusle"):
        names, _ = ModelFactory.get_parameter_info(model_type)
        if tuple(names) == param_names:
            candidates.append(model_type)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        "Could not infer model type from params file. "
        "Ensure the params.json metrics include 'model_type' or the parameter names match a known model."
    )


def resolve_output_file_arg(output_arg: str | None, *, static_nc: Path) -> Path | None:
    if output_arg is None:
        return None
    if output_arg == DEFAULT_OUTPUT_FILE:
        return static_nc.parent / "model_output.nc"

    output_path = Path(output_arg)
    if output_path.suffix.lower() == ".nc":
        return output_path
    return output_path / "model_output.nc"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CRSEM with a saved ParameterBatch using pre-prepared NetCDF files."
    )

    # Required NC files
    parser.add_argument("--static-nc", type=Path, required=True, help="Path to static.nc file.")
    parser.add_argument("--dynamic-nc", type=Path, required=True, help="Path to dynamic.nc file.")
    parser.add_argument("--observations-nc", type=Path, required=True, help="Path to observations.nc file.")
    parser.add_argument("--station-name", default="unknown", help="Station name for metadata.")
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Simulation start year (inclusive). Defaults to the driver start year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Simulation end year (inclusive). Defaults to the driver end year.",
    )

    # Model parameters
    parser.add_argument("--params-file", type=Path, required=True, help="Path to a saved ParameterBatch JSON file.")
    parser.add_argument(
        "--run-method",
        default="run_hillslope_river",
        choices=("run_hillslope", "run_hillslope_river"),
        help="Model execution entry point.",
    )
    parser.add_argument(
        "--run-mode",
        default="gridded",
        choices=("point", "gridded"),
        help="Use basin-mean point inputs or the original gridded inputs.",
    )
    parser.add_argument(
        "--aggregate",
        default="none",
        help="Aggregation method over member dimension. Use 'none' to keep all members.",
    )
    parser.add_argument(
        "--output-file",
        nargs="?",
        const=DEFAULT_OUTPUT_FILE,
        default=None,
        help="Write model output to NetCDF. Use '--output-file' to write model_output.nc beside static.nc, '--output-file DIR' to write DIR/model_output.nc, or '--output-file FILE.nc' to choose a custom file path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Determine output file from static_nc path
    output_file = resolve_output_file_arg(args.output_file, static_nc=Path(args.static_nc))

    parameter_batch, saved_metrics = ParameterBatch.from_file(args.params_file)
    model_type = infer_model_type(parameter_batch, saved_metrics)

    # Load driver from NC files
    driver = BasinDriver.from_nc_files(
        static_nc=args.static_nc,
        dynamic_nc=args.dynamic_nc,
        observations_nc=args.observations_nc,
        station_name=args.station_name,
    )

    sim_start = args.start_year if args.start_year is not None else driver.start_year
    sim_end = args.end_year if args.end_year is not None else driver.end_year
    if sim_start > sim_end:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    if sim_start != driver.start_year or sim_end != driver.end_year:
        driver = driver.crop_time_range(
            sim_start,
            sim_end,
            align_to_obs=args.run_method == "run_hillslope_river",
        )

    source = driver.to_point_driver(keep_rivers=args.run_method == "run_hillslope_river") if args.run_mode == "point" else driver
    if source.model_inputs.NDVI is not None and "member" in source.model_inputs.NDVI.dims:
        print("Run detected NDVI ensemble input; averaging across NDVI members before execution.")
        source = source.collapse_ndvi_members()
    result = run_parameter_batch(model_type, source, parameter_batch, run_method=args.run_method)

    dataset = result.to_dataset()
    if args.aggregate.lower() != "none":
        dataset = ResultAggregator.aggregate(dataset, method=args.aggregate, weights=result.weights)

    # Extract calibration info from saved_metrics
    calibration_info = None
    if saved_metrics is not None:
        calibration_info = saved_metrics

    dataset = annotate_output_dataset(
        dataset,
        model_type=model_type,
        run_method=args.run_method,
        run_mode=args.run_mode,
        calibration_info=calibration_info,
    )

    print("\nParameter batch payload:")
    pprint(parameter_batch.to_payload())

    if saved_metrics is not None:
        print("\nSaved calibration metrics:")
        pprint(saved_metrics)

    print("\nRun result:")
    print(dataset)

    # Save results to the output directory when requested
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_netcdf(
            output_file,
            engine="netcdf4",
            encoding=build_netcdf_compression_encoding(dataset),
        )
        print(f"\nSaved run result: {output_file}")


if __name__ == "__main__":
    main()

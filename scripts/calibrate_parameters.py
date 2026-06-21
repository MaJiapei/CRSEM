from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CRSEM.calibration_api import refine_parameters, save_calibration_results
from CRSEM.driver import BasinDriver

DEFAULT_SAVE_TARGET = "__DEFAULT__"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CRSEM parameter calibration using pre-prepared NetCDF files."
    )

    # Required NC files
    parser.add_argument("--static-nc", type=Path, required=True, help="Path to static.nc file.")
    parser.add_argument("--dynamic-nc", type=Path, required=True, help="Path to dynamic.nc file.")
    parser.add_argument("--observations-nc", type=Path, required=True, help="Path to observations.nc file.")
    parser.add_argument("--station-name", default="unknown", help="Station name for metadata.")
    parser.add_argument(
        "--calibration-start",
        type=int,
        default=None,
        help="Calibration start year (inclusive). Defaults to the observation start year when observations are available.",
    )
    parser.add_argument(
        "--calibration-end",
        type=int,
        default=None,
        help="Calibration end year (inclusive). Defaults to the observation end year when observations are available.",
    )

    # Calibration options
    parser.add_argument("--model-type", default="crsem", choices=("crsem", "rusle"), help="Model type to calibrate.")
    parser.add_argument(
        "--run-mode",
        default="point",
        choices=("point", "gridded"),
        help="Calibration input mode. 'point' is faster; 'gridded' retains spatial heterogeneity.",
    )
    parser.add_argument(
        "--optimizer",
        default="differential_evolution",
        choices=("differential_evolution", "glue"),
        help="Calibration optimizer.",
    )
    parser.add_argument("--objective-method", default="nse", help="Objective method.")
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML parameter configuration file. If not provided, uses built-in defaults.")
    parser.add_argument(
        "--selector",
        default=None,
        choices=("best_only", "aic", "glue"),
        help="Post-calibration selector. Defaults to 'aic' for differential evolution and 'glue' for the sampling optimizer.",
    )
    parser.add_argument(
        "--aic-numbers",
        type=int,
        default=None,
        help="Force the AIC selector to return exactly this many ensemble members.",
    )
    parser.add_argument(
        "--aic-max-numbers",
        type=int,
        default=None,
        help="Automatic upper bound for AIC-selected ensemble size.",
    )
    parser.add_argument(
        "--aic-delta-threshold",
        type=float,
        default=None,
        help="AIC delta threshold for automatic ensemble selection.",
    )
    parser.add_argument(
        "--aic-cum-weight",
        type=float,
        default=None,
        help="Cumulative AIC weight threshold for automatic ensemble selection.",
    )
    parser.add_argument("--maxiter", type=int, default=100, help="Maximum optimizer iterations.")
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help="Differential evolution population size multiplier. If omitted, uses optimizer default.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for gridded differential evolution (-1 uses all CPUs). Supported only in gridded mode.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of Monte Carlo samples for the GLUE optimizer. If omitted, falls back to --maxiter.",
    )
    parser.add_argument(
        "--sampling-method",
        default="sobol",
        choices=("sobol", "lhs", "random"),
        help="Parameter-space sampling method used by the GLUE optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stochastic calibration components such as differential evolution and scrambled GLUE sampling.",
    )
    parser.add_argument(
        "--polish",
        action="store_true",
        help="Enable final local search (L-BFGS-B polish). Disabled by default for speed.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_SAVE_TARGET,
        default=None,
        metavar="PATH",
        help="Save calibration results. Use '--save' to write params.json beside static.nc, or '--save PATH' to choose a directory or .json file path.",
    )
    parser.add_argument(
        "--plot-progress",
        action="store_true",
        help="Enable calibration progress plotting. Default: disabled.",
    )
    parser.add_argument(
        "--glue-threshold",
        type=float,
        default=None,
        help="Behavioral threshold for the GLUE selector. For NSE/KGE/R2 this is a lower bound; for RMSE/MAE it is an upper bound.",
    )
    parser.add_argument(
        "--glue-top-fraction",
        type=float,
        default=None,
        help="Fallback top fraction for the GLUE selector when no explicit or default threshold is available.",
    )
    parser.add_argument(
        "--glue-max-members",
        type=int,
        default=None,
        help="Maximum number of behavioral members returned by the GLUE selector.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.run_mode == "gridded" and args.plot_progress:
        raise ValueError("Grid-mode calibration does not support --plot-progress. Use point mode for plotting.")
    if args.run_mode == "point" and args.workers is not None:
        raise ValueError("Point-mode calibration does not accept --workers. Use gridded mode for parallel workers.")
    if args.optimizer != "differential_evolution" and args.workers is not None:
        raise ValueError("Only differential evolution supports --workers.")
    if args.optimizer != "differential_evolution" and args.polish:
        raise ValueError("Only differential evolution supports --polish.")

    # Determine output directory from static_nc path
    output_dir = Path(args.static_nc).parent

    # Load driver from NC files
    driver = BasinDriver.from_nc_files(
        static_nc=args.static_nc,
        dynamic_nc=args.dynamic_nc,
        observations_nc=args.observations_nc,
        station_name=args.station_name,
    )

    if driver._river_data_loaded and driver._Q is not None:
        obs_start_year = int(driver._Q.index[0].year)
        obs_end_year = int(driver._Q.index[-1].year)
    else:
        obs_start_year = driver.start_year
        obs_end_year = driver.end_year

    calibration_start = args.calibration_start if args.calibration_start is not None else obs_start_year
    calibration_end = args.calibration_end if args.calibration_end is not None else obs_end_year

    if calibration_start > calibration_end:
        raise ValueError("--calibration-start must be less than or equal to --calibration-end.")

    if (
        calibration_start != driver.start_year
        or calibration_end != driver.end_year
        or (driver._river_data_loaded and driver._Q is not None)
    ):
        driver = driver.crop_time_range(
            calibration_start,
            calibration_end,
            align_to_obs=driver._river_data_loaded and driver._Q is not None,
        )

    source_driver = driver.to_point_driver(keep_rivers=True) if args.run_mode == "point" else driver
    if source_driver.model_inputs.NDVI is not None and "member" in source_driver.model_inputs.NDVI.dims:
        print("Calibration detected NDVI ensemble input; averaging across NDVI members before optimization.")
        source_driver = source_driver.collapse_ndvi_members()
    print(source_driver)

    plot_progress = args.plot_progress
    calibration_output_mode = "full" if plot_progress else "compact"
    selector_name = args.selector or ("glue" if args.optimizer == "glue" else "aic")

    optimizer_kwargs = {}
    if args.seed is not None:
        optimizer_kwargs["seed"] = args.seed
    if args.optimizer == "differential_evolution" and args.popsize is not None:
        optimizer_kwargs["popsize"] = args.popsize
    if args.run_mode == "gridded" and args.workers is not None:
        optimizer_kwargs["workers"] = args.workers
    if args.optimizer == "glue":
        optimizer_kwargs["sampler"] = args.sampling_method
        if args.n_samples is not None:
            optimizer_kwargs["n_samples"] = args.n_samples
    if args.optimizer == "differential_evolution" and args.polish:
        optimizer_kwargs["polish"] = True

    selector_kwargs = {}
    if selector_name == "aic":
        if args.aic_numbers is not None:
            selector_kwargs["exact_members"] = args.aic_numbers
        if args.aic_max_numbers is not None:
            selector_kwargs["max_members"] = args.aic_max_numbers
        if args.aic_delta_threshold is not None:
            selector_kwargs["delta_aic_threshold"] = args.aic_delta_threshold
        if args.aic_cum_weight is not None:
            selector_kwargs["cumulative_weight_threshold"] = args.aic_cum_weight
    elif selector_name == "glue":
        if args.glue_threshold is not None:
            selector_kwargs["threshold"] = args.glue_threshold
        if args.glue_top_fraction is not None:
            selector_kwargs["top_fraction"] = args.glue_top_fraction
        if args.glue_max_members is not None:
            selector_kwargs["max_members"] = args.glue_max_members

    selected_batch, metrics = refine_parameters(
        source_driver,
        model_type=args.model_type,
        optimizer=args.optimizer,
        ensemble_para=(selector_name != "best_only"),
        selector_name=selector_name,
        selector_kwargs=selector_kwargs,
        plot_progress=plot_progress,
        maxiter=args.maxiter,
        objective_method=args.objective_method,
        calibration_output_mode=calibration_output_mode,
        optimizer_kwargs=optimizer_kwargs,
        config_path=args.config,
    )

    # Print calibration metrics summary
    print("\n" + "=" * 60)
    print("Calibration Summary")
    print("=" * 60)

    # Key metrics
    print(f"\nPerformance Metrics:")
    print(f"  NSE:  {metrics.get('NSE', 'N/A'):.4f}" if isinstance(metrics.get('NSE'), (int, float)) else f"  NSE:  {metrics.get('NSE', 'N/A')}")
    print(f"  KGE:  {metrics.get('KGE', 'N/A'):.4f}" if isinstance(metrics.get('KGE'), (int, float)) else f"  KGE:  {metrics.get('KGE', 'N/A')}")
    print(f"  R²:   {metrics.get('R2', 'N/A'):.4f}" if isinstance(metrics.get('R2'), (int, float)) else f"  R²:   {metrics.get('R2', 'N/A')}")
    print(f"  RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if isinstance(metrics.get('RMSE'), (int, float)) else f"  RMSE: {metrics.get('RMSE', 'N/A')}")
    print(f"  MAE:  {metrics.get('MAE', 'N/A'):.4f}" if isinstance(metrics.get('MAE'), (int, float)) else f"  MAE:  {metrics.get('MAE', 'N/A')}")

    # Ensemble info (if applicable)
    if metrics.get('selected_n_members', 1) > 1:
        print(f"\nEnsemble Information:")
        print(f"  Selected members: {metrics.get('selected_n_members', 'N/A')}")
        if 'ensemble_info' in metrics:
            info = metrics['ensemble_info']
            print(f"  Selection method: {info.get('selection', 'N/A')}")
            print(f"  Candidates evaluated: {info.get('n_candidates', 'N/A')}")
            print(f"  Valid candidates: {info.get('n_valid_candidates', 'N/A')}")
    archived_candidates = metrics.get("n_archived_candidates")
    if archived_candidates is not None:
        print(f"\nCandidate Archive:")
        print(f"  Archived unique candidates: {archived_candidates}")

    # File paths
    print(f"\nData Files:")
    print(f"  Static:       {metrics.get('static', 'N/A')}")
    print(f"  Dynamic:      {metrics.get('dynamic', 'N/A')}")
    print(f"  Observations: {metrics.get('observations', 'N/A')}")

    # Save results to the output directory when requested
    if args.save is not None:
        save_target = output_dir if args.save == DEFAULT_SAVE_TARGET else Path(args.save)
        save_path = save_calibration_results(selected_batch, metrics, save_path=save_target)
        print(f"\nSaved calibration file: {save_path}")


if __name__ == "__main__":
    main()

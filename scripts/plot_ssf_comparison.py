"""Plot simulated vs observed SSF with calibration/validation period separation.

Usage:
    python scripts/plot_ssf_comparison.py \
        --simulated example/tuotuohe_1985_2015/model_output.nc \
        --observed example/tuotuohe_1985_2015/observations.nc \
        --calibration-start 1990 \
        --calibration-end 2000 \
        --output figures/ssf_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from CRSEM.calibration_evaluation import calculate_metrics


def setup_chinese_font():
    """Configure matplotlib for Chinese font support (optional)."""
    # Don't modify global font settings - keep default for Latin characters
    # The script works fine with English labels
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot simulated vs observed SSF comparison"
    )
    parser.add_argument(
        "--simulated",
        type=Path,
        required=True,
        help="Path to simulated SSF NetCDF file (model output)",
    )
    parser.add_argument(
        "--observed",
        type=Path,
        required=True,
        help="Path to observed SSF NetCDF file",
    )
    parser.add_argument(
        "--calibration-start",
        type=int,
        default=None,
        help="Start year of calibration period (default: read from model_output.nc)",
    )
    parser.add_argument(
        "--calibration-end",
        type=int,
        default=None,
        help="End year of calibration period (default: read from model_output.nc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: same dir as simulated, ssf_comparison.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (default: auto-generated)",
    )
    parser.add_argument(
        "--member",
        default="auto",
        help="Member selection for ensemble outputs: 'auto'/'mean', a zero-based index, or a member label. Non-ensemble files ignore this.",
    )
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Force period split even if simulation equals calibration period",
    )
    return parser.parse_args()


def get_calibration_info(simulated_nc: Path) -> dict[str, Any]:
    """Read calibration info from model output NetCDF attributes."""
    ds = xr.open_dataset(simulated_nc)
    info = {}

    # Read calibration period from attributes
    if "calibration_start_year" in ds.attrs:
        info["start_year"] = ds.attrs["calibration_start_year"]
    if "calibration_end_year" in ds.attrs:
        info["end_year"] = ds.attrs["calibration_end_year"]
    if "station_name" in ds.attrs:
        info["station_name"] = ds.attrs["station_name"]
    if "calibration_NSE" in ds.attrs:
        info["NSE"] = ds.attrs["calibration_NSE"]
    if "calibration_KGE" in ds.attrs:
        info["KGE"] = ds.attrs["calibration_KGE"]
    if "calibration_R2" in ds.attrs:
        info["R2"] = ds.attrs["calibration_R2"]

    return info


def load_ssf_data(
    simulated_nc: Path,
    observed_nc: Path,
    member: str = "auto",
) -> tuple[pd.Series, pd.Series, xr.Dataset]:
    """Load simulated and observed SSF data.

    Returns:
        Tuple of (simulated_series, observed_series, sim_dataset) with DatetimeIndex
    """
    # Load observed
    obs_ds = xr.open_dataset(observed_nc)
    obs_time = pd.to_datetime(obs_ds.time.values)
    obs_ssf = obs_ds["SSF"].values
    observed = pd.Series(obs_ssf, index=obs_time, name="observed")

    # Load simulated
    sim_ds = xr.open_dataset(simulated_nc)

    # Handle ensemble members
    if "member" in sim_ds.dims:
        sim_ssf = select_simulated_member(sim_ds["SSF_pred"], member).values
    else:
        sim_ssf = sim_ds["SSF_pred"].values

    sim_time = pd.to_datetime(sim_ds.time.values)
    simulated = pd.Series(sim_ssf, index=sim_time, name="simulated")

    return simulated, observed, sim_ds


def get_ensemble_plot_data(sim_ds: xr.Dataset) -> dict[str, Any] | None:
    """Build ensemble plotting payload from a simulated dataset."""
    if "member" not in sim_ds.dims:
        return None

    ssf_da = sim_ds["SSF_pred"]
    members = np.asarray(ssf_da.values, dtype=float)
    return {
        "time": pd.to_datetime(sim_ds.time.values),
        "members": members,
        "member_labels": [str(value) for value in ssf_da.coords["member"].values],
        "mean": members.mean(axis=0),
        "lower": members.min(axis=0),
        "upper": members.max(axis=0),
    }


def select_simulated_member(simulated_da: xr.DataArray, member: str) -> xr.DataArray:
    """Select a single member series or reduce an ensemble series."""
    if "member" not in simulated_da.dims:
        return simulated_da

    selection = str(member).strip()
    if selection in {"", "auto", "mean"}:
        return simulated_da.mean(dim="member")

    try:
        index = int(selection)
    except ValueError:
        member_labels = {str(value) for value in simulated_da.coords["member"].values}
        if selection not in member_labels:
            raise ValueError(
                f"Unknown member selection '{selection}'. Use 'auto', 'mean', a zero-based index, "
                f"or one of {sorted(member_labels)}."
            ) from None
        return simulated_da.sel(member=selection)

    n_members = int(simulated_da.sizes["member"])
    if index < 0 or index >= n_members:
        raise ValueError(f"Member index {index} is out of range for {n_members} ensemble members.")
    return simulated_da.isel(member=index)


def align_time_series(
    simulated: pd.Series,
    observed: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align simulated and observed to common time index."""
    # Find common time range
    common_index = simulated.index.intersection(observed.index)

    if len(common_index) == 0:
        raise ValueError("No overlapping time period between simulated and observed data")

    simulated_aligned = simulated.reindex(common_index)
    observed_aligned = observed.reindex(common_index)

    return simulated_aligned, observed_aligned


def split_periods(
    simulated: pd.Series,
    observed: pd.Series,
    cal_start: int | None,
    cal_end: int | None,
    force_split: bool = False,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Split data into calibration and validation periods.

    Logic:
        - If simulation period is longer than calibration period: split into periods
        - If simulation period equals calibration period: no split (single period)
        - If force_split is True: always split

    Returns:
        Tuple of (periods dict, should_split bool)
        periods dict contains 'sim', 'obs', and 'time' arrays for each period
    """
    periods = {}
    should_split = False

    # Get simulation time range
    sim_start_year = simulated.index.year.min()
    sim_end_year = simulated.index.year.max()

    # If no calibration info, return full period only
    if cal_start is None or cal_end is None:
        periods["full"] = {
            "sim": simulated.values,
            "obs": observed.values,
            "time": simulated.index,
        }
        return periods, False

    # Determine if simulation period is longer than calibration period
    # Consider both start and end differences
    sim_is_longer = (sim_start_year < cal_start) or (sim_end_year > cal_end)

    if force_split:
        should_split = True
    elif sim_is_longer:
        should_split = True
    else:
        should_split = False

    if not should_split:
        # No split needed - simulation period equals calibration period
        periods["full"] = {
            "sim": simulated.values,
            "obs": observed.values,
            "time": simulated.index,
            "label": f"Calibration ({cal_start}-{cal_end})",
        }
        return periods, False

    # Split into periods
    # Calibration period
    cal_mask = (simulated.index.year >= cal_start) & (simulated.index.year <= cal_end)
    if cal_mask.any():
        periods["calibration"] = {
            "sim": simulated[cal_mask].values,
            "obs": observed[cal_mask].values,
            "time": simulated.index[cal_mask],
            "label": f"Calibration ({cal_start}-{cal_end})",
        }

    # Validation period (before calibration)
    val_before_mask = simulated.index.year < cal_start
    if val_before_mask.any():
        periods["validation_before"] = {
            "sim": simulated[val_before_mask].values,
            "obs": observed[val_before_mask].values,
            "time": simulated.index[val_before_mask],
            "label": f"Validation ({sim_start_year}-{cal_start-1})",
        }

    # Validation period (after calibration)
    val_after_mask = simulated.index.year > cal_end
    if val_after_mask.any():
        periods["validation_after"] = {
            "sim": simulated[val_after_mask].values,
            "obs": observed[val_after_mask].values,
            "time": simulated.index[val_after_mask],
            "label": f"Validation ({cal_end+1}-{sim_end_year})",
        }

    return periods, True


def format_metrics_text(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary as text string."""
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ["PBIAS"]:
                lines.append(f"{prefix}{key}: {value:+.2f}%")
            elif key in ["RMSE", "MAE"]:
                lines.append(f"{prefix}{key}: {value:.2e} t")
            elif key == "n months":
                lines.append(f"{prefix}n: {int(value)}")
            else:
                lines.append(f"{prefix}{key}: {value:.3f}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def plot_comparison(
    simulated: pd.Series,
    observed: pd.Series,
    periods: dict[str, dict[str, Any]],
    output_path: Path,
    ensemble_data: dict[str, Any] | None = None,
    title: str | None = None,
    should_split: bool = True,
    cal_start: int | None = None,
    cal_end: int | None = None,
) -> None:
    """Create the comparison plot."""
    # Ensemble outputs are easier to read as a single time-series panel.
    if ensemble_data is not None:
        fig, ax_ts = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
        ax_scatter = None
        ax_clim = None
        ax_annual = None
    else:
        fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
        ax_ts = fig.add_subplot(gs[0, :])
        ax_scatter = fig.add_subplot(gs[1, 0])
        ax_clim = fig.add_subplot(gs[1, 1])
        ax_annual = fig.add_subplot(gs[2, :])

    # Color scheme
    colors = {
        "observed": "#1f77b4",
        "simulated": "#d62728",
        "calibration": "#ffbb33",
        "validation": "#28a745",
    }

    # === Time Series Plot ===
    ax_ts.plot(
        observed.index, observed.values,
        color=colors["observed"], linewidth=1.5, label="Observed", alpha=0.8
    )
    if ensemble_data is not None:
        for idx, member_values in enumerate(ensemble_data["members"]):
            ax_ts.plot(
                ensemble_data["time"],
                member_values,
                color=colors["simulated"],
                linewidth=0.8,
                alpha=0.16,
                label="Ensemble members" if idx == 0 else None,
            )
        ax_ts.fill_between(
            ensemble_data["time"],
            ensemble_data["lower"],
            ensemble_data["upper"],
            color=colors["simulated"],
            alpha=0.12,
            label="Ensemble range",
        )
    ax_ts.plot(
        simulated.index, simulated.values,
        color=colors["simulated"],
        linewidth=1.8,
        linestyle="--",
        label="Simulated mean" if ensemble_data is not None else "Simulated",
        alpha=0.9,
    )

    # Shade calibration period only if should_split is True
    if should_split and "calibration" in periods:
        cal_time = periods["calibration"]["time"]

        # For monthly data, time points are at month start
        # Extend right boundary by 1 month to cover the full last month
        from pandas.tseries.offsets import DateOffset
        cal_end_boundary = cal_time[-1] + DateOffset(months=1)

        ax_ts.axvspan(
            cal_time[0], cal_end_boundary,
            alpha=0.2, color=colors["calibration"], label="Calibration period"
        )

        # Add vertical lines for period boundaries
        ax_ts.axvline(cal_time[0], color=colors["calibration"], linestyle=":", alpha=0.7)
        ax_ts.axvline(cal_end_boundary, color=colors["calibration"], linestyle=":", alpha=0.7)

    ax_ts.set_ylabel("SSF (t month$^{-1}$)")
    ax_ts.legend(loc="upper right")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))

    # Calculate metrics for each period
    metrics_texts = []

    # Full period metrics
    full_sim = simulated.values
    full_obs = observed.values
    full_metrics = calculate_metrics(full_sim, full_obs)

    if should_split:
        # Split mode: show Full, Calibration, and Validation separately
        metrics_texts.append(("Full Period", full_metrics))

        # Calibration metrics
        if "calibration" in periods:
            cal_metrics = calculate_metrics(
                periods["calibration"]["sim"],
                periods["calibration"]["obs"]
            )
            metrics_texts.append(("Calibration", cal_metrics))

        # Validation metrics
        if "validation_before" in periods or "validation_after" in periods:
            val_sim = np.concatenate([
                periods.get("validation_before", {}).get("sim", np.array([])),
                periods.get("validation_after", {}).get("sim", np.array([]))
            ])
            val_obs = np.concatenate([
                periods.get("validation_before", {}).get("obs", np.array([])),
                periods.get("validation_after", {}).get("obs", np.array([]))
            ])
            if len(val_sim) > 0:
                val_metrics = calculate_metrics(val_sim, val_obs)
                metrics_texts.append(("Validation", val_metrics))
    else:
        # No split: show single period with calibration label if available
        label = "Calibration" if cal_start and cal_end else "Full Period"
        metrics_texts.append((label, full_metrics))

    # Add metrics text box
    metrics_box = ""
    for label, metrics in metrics_texts:
        metrics_box += f"=== {label} ===\n"
        metrics_box += f"  NSE: {metrics['NSE']:.3f}\n"
        metrics_box += f"  KGE: {metrics['KGE']:.3f}\n"
        metrics_box += f"  PBIAS: {metrics['PBIAS']:+.1f}%\n"
        metrics_box += f"  R²: {metrics['R2']:.3f}\n"
        metrics_box += f"  n: {int(metrics['n months'])}\n\n"

    ax_ts.text(
        0.02, 0.98, metrics_box.strip(),
        transform=ax_ts.transAxes,
        fontsize=9, family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    )

    if title:
        ax_ts.set_title(title)
    else:
        ax_ts.set_title("Simulated vs Observed SSF")

    if ensemble_data is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {output_path}")
        plt.close(fig)
        return

    # === Scatter Plot ===
    # Plot all points
    ax_scatter.scatter(
        observed.values, simulated.values,
        alpha=0.5, s=20, color="steelblue", edgecolors="none"
    )

    # Highlight calibration points only in split mode
    if should_split and "calibration" in periods:
        ax_scatter.scatter(
            periods["calibration"]["obs"],
            periods["calibration"]["sim"],
            alpha=0.7, s=25, color=colors["calibration"],
            edgecolors="black", linewidths=0.5,
            label="Calibration"
        )

    # 1:1 line
    max_val = max(observed.max(), simulated.max())
    ax_scatter.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="1:1 line")

    ax_scatter.set_xlabel("Observed SSF (t month$^{-1}$)")
    ax_scatter.set_ylabel("Simulated SSF (t month$^{-1}$)")
    ax_scatter.set_title("Scatter Plot")
    ax_scatter.legend(loc="upper left", fontsize=8)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(True, alpha=0.3)

    # Add R² to scatter
    r2 = full_metrics["R2"]
    ax_scatter.text(
        0.98, 0.02, f"R² = {r2:.3f}",
        transform=ax_scatter.transAxes,
        ha="right", va="bottom",
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # === Monthly Climatology ===
    # Calculate monthly means
    obs_monthly = observed.groupby(observed.index.month).mean()
    sim_monthly = simulated.groupby(simulated.index.month).mean()

    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    x = np.arange(12)
    width = 0.35

    ax_clim.bar(x - width/2, obs_monthly.values, width, label="Observed", color=colors["observed"], alpha=0.7)
    ax_clim.bar(x + width/2, sim_monthly.values, width, label="Simulated", color=colors["simulated"], alpha=0.7)
    if ensemble_data is not None:
        ensemble_frame = pd.DataFrame(
            ensemble_data["members"].T,
            index=simulated.index,
            columns=ensemble_data["member_labels"],
        )
        ensemble_monthly = ensemble_frame.groupby(ensemble_frame.index.month).mean()
        ax_clim.errorbar(
            x + width / 2,
            sim_monthly.values,
            yerr=ensemble_monthly.std(axis=1).fillna(0.0).values,
            fmt="none",
            ecolor="darkred",
            elinewidth=1.0,
            capsize=2,
            alpha=0.7,
        )

    ax_clim.set_xticks(x)
    ax_clim.set_xticklabels(months)
    ax_clim.set_xlabel("Month")
    ax_clim.set_ylabel("Mean SSF (t month$^{-1}$)")
    ax_clim.set_title("Monthly Climatology")
    ax_clim.legend(loc="upper right", fontsize=8)
    ax_clim.grid(True, alpha=0.3, axis="y")

    # === Annual Totals ===
    obs_annual = observed.groupby(observed.index.year).sum()
    sim_annual = simulated.groupby(simulated.index.year).sum()

    years = obs_annual.index
    x_years = np.arange(len(years))
    width = 0.35

    # Color bars by period
    bar_colors_obs = []
    bar_colors_sim = []
    for year in years:
        if should_split and "calibration" in periods:
            cal_start_year = periods["calibration"]["time"][0].year
            cal_end_year = periods["calibration"]["time"][-1].year
            if cal_start_year <= year <= cal_end_year:
                bar_colors_obs.append(colors["calibration"])
                bar_colors_sim.append("darkorange")
            else:
                bar_colors_obs.append(colors["observed"])
                bar_colors_sim.append(colors["simulated"])
        else:
            bar_colors_obs.append(colors["observed"])
            bar_colors_sim.append(colors["simulated"])

    ax_annual.bar(x_years - width/2, obs_annual.values, width, label="Observed", color=bar_colors_obs, alpha=0.7)
    ax_annual.bar(x_years + width/2, sim_annual.values, width, label="Simulated", color=bar_colors_sim, alpha=0.7)
    if ensemble_data is not None:
        ensemble_frame = pd.DataFrame(
            ensemble_data["members"].T,
            index=simulated.index,
            columns=ensemble_data["member_labels"],
        )
        ensemble_annual = ensemble_frame.groupby(ensemble_frame.index.year).sum().reindex(years)
        ax_annual.errorbar(
            x_years + width / 2,
            sim_annual.values,
            yerr=ensemble_annual.std(axis=1).fillna(0.0).values,
            fmt="none",
            ecolor="darkred",
            elinewidth=1.0,
            capsize=2,
            alpha=0.7,
        )

    ax_annual.set_xticks(x_years[::2])
    ax_annual.set_xticklabels(years[::2], rotation=45)
    ax_annual.set_xlabel("Year")
    ax_annual.set_ylabel("Annual SSF (t yr$^{-1}$)")
    ax_annual.set_title("Annual Sediment Flux")
    ax_annual.legend(loc="upper right", fontsize=8)
    ax_annual.grid(True, alpha=0.3, axis="y")

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")

    plt.close(fig)


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("SSF Comparison Plot")
    print("=" * 60)

    # Load data
    print(f"\nLoading simulated data: {args.simulated}")
    print(f"Loading observed data: {args.observed}")

    simulated, observed, sim_ds = load_ssf_data(
        args.simulated,
        args.observed,
        member=args.member,
    )
    ensemble_data = get_ensemble_plot_data(sim_ds)

    print(f"  Simulated: {simulated.index[0]} to {simulated.index[-1]} ({len(simulated)} months)")
    print(f"  Observed:  {observed.index[0]} to {observed.index[-1]} ({len(observed)} months)")
    if ensemble_data is not None:
        print(f"  Ensemble members: {len(ensemble_data['member_labels'])}")

    # Align time series
    simulated, observed = align_time_series(simulated, observed)
    print(f"  Common period: {simulated.index[0]} to {simulated.index[-1]} ({len(simulated)} months)")

    # Get calibration info from command line or model output attributes
    cal_start = args.calibration_start
    cal_end = args.calibration_end

    if cal_start is None or cal_end is None:
        # Try to read from model output attributes
        cal_info = get_calibration_info(args.simulated)
        if cal_start is None and "start_year" in cal_info:
            cal_start = cal_info["start_year"]
        if cal_end is None and "end_year" in cal_info:
            cal_end = cal_info["end_year"]

        if cal_start is not None and cal_end is not None:
            print(f"\nCalibration period from model output: {cal_start} - {cal_end}")

    # Split periods with automatic detection
    periods, should_split = split_periods(
        simulated, observed,
        cal_start=cal_start,
        cal_end=cal_end,
        force_split=args.force_split,
    )

    if should_split and "calibration" in periods:
        cal_time = periods["calibration"]["time"]
        print(f"Mode: Split (simulation period differs from calibration)")
        print(f"Calibration period: {cal_time[0].year} - {cal_time[-1].year}")
    elif cal_start and cal_end:
        print(f"Mode: No split (simulation period equals calibration: {cal_start}-{cal_end})")
    else:
        print(f"Mode: No calibration info available")

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = args.simulated.parent / "ssf_comparison.png"

    # Generate title
    title = args.title
    if title is None:
        title = "Simulated vs Observed Suspended Sediment Flux"

    # Create plot
    print(f"\nCreating plot...")
    plot_comparison(
        simulated, observed, periods, output_path,
        ensemble_data=ensemble_data,
        title=title,
        should_split=should_split,
        cal_start=cal_start,
        cal_end=cal_end,
    )

    # Print metrics summary
    print("\n" + "=" * 60)
    print("Metrics Summary")
    print("=" * 60)

    full_metrics = calculate_metrics(simulated.values, observed.values)

    if should_split:
        print(f"\nFull Period ({simulated.index[0].year}-{simulated.index[-1].year}):")
        for k, v in full_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        if "calibration" in periods:
            cal_metrics = calculate_metrics(
                periods["calibration"]["sim"],
                periods["calibration"]["obs"]
            )
            cal_time = periods["calibration"]["time"]
            print(f"\nCalibration ({cal_time[0].year}-{cal_time[-1].year}):")
            for k, v in cal_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

        # Validation period
        val_sim_list = []
        val_obs_list = []
        for key in ["validation_before", "validation_after"]:
            if key in periods:
                val_sim_list.append(periods[key]["sim"])
                val_obs_list.append(periods[key]["obs"])

        if val_sim_list:
            val_sim = np.concatenate(val_sim_list)
            val_obs = np.concatenate(val_obs_list)
            val_metrics = calculate_metrics(val_sim, val_obs)
            print(f"\nValidation:")
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
    else:
        label = "Calibration" if cal_start and cal_end else "Full Period"
        print(f"\n{label} ({simulated.index[0].year}-{simulated.index[-1].year}):")
        for k, v in full_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

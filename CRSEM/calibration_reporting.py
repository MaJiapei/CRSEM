from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from CRSEM._calibration_plot import CalibrationVisualizer, PlotState


@dataclass(slots=True)
class CalibrationTracker:
    """Track calibration histories and candidate records."""

    iteration_history: list[int] = field(default_factory=list)
    sse_history: list[float] = field(default_factory=list)
    nse_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    candidate_records: list[dict] = field(default_factory=list)

    def record(self, params, current_loss: float, metrics: dict, evaluation: dict | None = None) -> int:
        iteration = len(self.iteration_history) + 1
        self.iteration_history.append(iteration)
        self.sse_history.append(float(metrics["sse"]))
        self.nse_history.append(float(metrics["nse"]))
        self.loss_history.append(float(current_loss))
        evaluation = {} if evaluation is None else dict(evaluation)
        penalty_payload = {
            key: float(value)
            for key, value in evaluation.items()
            if key.endswith("_penalty")
        }
        self.candidate_records.append(
            {
                "loss": float(current_loss),
                "objective_value": float(evaluation.get("objective_value", current_loss)),
                "penalties": penalty_payload,
                "params": np.asarray(params, dtype=float).copy(),
                "metrics": metrics,
            }
        )
        return iteration


class CalibrationReporter:
    """Handle calibration progress visualization and console reporting."""

    def __init__(
        self,
        *,
        model_type: str,
        plot_progress: bool,
        plot_every_iters: int,
        plot_min_interval_s: float,
        diagnostics_provider: Callable,
    ) -> None:
        self.model_type = model_type
        self.plot_progress = plot_progress
        self.plot_every_iters = plot_every_iters
        self.plot_min_interval_s = plot_min_interval_s
        self.diagnostics_provider = diagnostics_provider
        self.last_plot_time = 0.0
        self.visualizer = CalibrationVisualizer(model_type) if plot_progress else None

    def initialize(self) -> None:
        if self.visualizer:
            self.visualizer.initialize()

    def finalize(self, *, block: bool = False) -> None:
        if self.visualizer:
            self.visualizer.finalize(block=block)

    def print_run_configuration(
        self,
        *,
        optimizer: str,
        model_type: str,
        plot_progress: bool,
        param_names: list[str] | tuple[str, ...],
        param_bounds,
    ) -> None:
        print(f"\nStarting optimization with {optimizer} algorithm...")
        print(f"Model name {model_type}, Number of parameters: {len(param_bounds)}")
        print(f"Real-time plotting: {'Enabled' if plot_progress else 'Disabled'}")
        print("Parameter bounds:")
        for name, bound in zip(param_names, param_bounds):
            print(f"  {name:<20}: [{bound[0]:.4f}, {bound[1]:.4f}]")

    def print_final_summary(
        self,
        *,
        optimizer: str,
        result,
        param_names: list[str] | tuple[str, ...],
        param_bounds,
        param_defaults: dict[str, float] | None = None,
        final_metrics: dict[str, float],
    ) -> None:
        """Print final calibration summary in a formatted table.

        Args:
            optimizer: Optimizer name
            result: Optimization result object
            param_names: List of parameter names
            param_bounds: List of (min, max) tuples for each parameter
            param_defaults: Optional dictionary of default/initial parameter values
            final_metrics: Dictionary with calibration metrics
        """
        print("\n" + "=" * 60)
        print("\nOptimization Results\n")
        print("=" * 60)
        print(f"Optimizer: {optimizer}  |  Success: {result.success}")
        if hasattr(result, "message"):
            print(f"Message: {result.message}")
        print(f"Final objective value: {result.fun:.4e}")
        print(f"\nFinal NSE: {final_metrics['nse']:.4f}")
        print(f"Amplitude ratio (pred/obs): {final_metrics['std_ratio']:.4f}")
        print(f"Channel contribution ratio (channel/hillslope): {final_metrics['channel_ratio']:.4f}")

        # Print calibrated parameters table
        print("\nCalibrated Parameters:")
        print(self._format_parameters_table(param_names, result.x, param_bounds, param_defaults))

    def _format_parameters_table(
        self,
        param_names: list[str] | tuple[str, ...],
        param_values: list[float],
        param_bounds: list[tuple[float, float]],
        param_defaults: dict[str, float] | None = None,
        boundary_threshold: float = 0.05,
    ) -> str:
        """Format parameters as a table with boundary indicators.

        Args:
            param_names: List of parameter names
            param_values: List of calibrated parameter values
            param_bounds: List of (min, max) tuples for each parameter
            param_defaults: Optional dictionary of default/initial parameter values
            boundary_threshold: Threshold (as fraction of range) to flag boundary proximity.
                               Default 5% means only flag if within 5% of range.

        Returns:
            Formatted table string
        """
        lines = []

        # Table header
        lines.append(f"{'Parameter':<18} {'Initial':>14} {'Calibrated':>14} {'Bounds':>22} {'→':>6}")
        lines.append("-" * 80)

        for i, (name, value, (lower, upper)) in enumerate(zip(param_names, param_values, param_bounds)):
            # Calculate position within bounds (0-1 scale)
            range_size = upper - lower
            normalized_pos = (value - lower) / range_size if range_size > 0 else 0.5

            # Determine boundary indicator
            indicator = ""
            if normalized_pos < boundary_threshold:
                # Close to lower bound
                indicator = "↓"
            elif normalized_pos > (1 - boundary_threshold):
                # Close to upper bound
                indicator = "↑"

            # Get initial value from defaults or use midpoint
            if param_defaults and name in param_defaults:
                initial = param_defaults[name]
            else:
                initial = (lower + upper) / 2

            # Format bounds string
            bounds_str = f"[{lower:.4f}, {upper:.4f}]"

            # Format the row
            row = f"{name:<18} {initial:>14.6f} {value:>14.6f} {bounds_str:>22} {indicator:>6}"
            lines.append(row)

        # Add legend
        lines.append("")
        lines.append(f"Legend: ↓ = near lower bound (within {boundary_threshold*100:.0f}% of range)")
        lines.append(f"        ↑ = near upper bound (within {boundary_threshold*100:.0f}% of range)")

        return "\n".join(lines)

    def report(
        self,
        *,
        params,
        current_loss: float,
        convergence_val,
        run_output,
        tracker: CalibrationTracker,
        ssf_obs: np.ndarray,
        param_names: list[str] | tuple[str, ...],
    ) -> None:
        if self.plot_progress and self.visualizer:
            now = time.monotonic()
            time_ok = (now - self.last_plot_time) >= self.plot_min_interval_s
            iter_ok = len(tracker.iteration_history) % self.plot_every_iters == 0
            if time_ok or iter_ok:
                valid_mask = ~np.isnan(ssf_obs.flatten())
                obs_valid = ssf_obs.flatten()[valid_mask]
                pred_valid = np.asarray(run_output.SSF_pred).flatten()[valid_mask]
                plot_state = PlotState(
                    iteration=len(tracker.iteration_history),
                    params=params,
                    param_names=list(param_names),
                    nse=tracker.nse_history[-1] if tracker.nse_history else -np.inf,
                    loss=current_loss,
                    iteration_history=tracker.iteration_history,
                    nse_history=tracker.nse_history,
                    loss_history=tracker.loss_history,
                    obs_valid=obs_valid,
                    pred_valid=pred_valid,
                    E_rain=getattr(run_output, "E_hillslope_rain", getattr(run_output, "E_hillslope", None)),
                    E_melt=getattr(run_output, "E_hillslope_melt", None),
                    A_channel=run_output.A_channel,
                    R_rain=getattr(run_output, "R_rain", None),
                    R_melt=getattr(run_output, "R_melt", None),
                    diagnostics=self.diagnostics_provider(run_output),
                )
                self.visualizer.update(plot_state)
                self.last_plot_time = now
            return

        current_nse = tracker.nse_history[-1] if tracker.nse_history else -np.inf
        latest_record = tracker.candidate_records[-1] if tracker.candidate_records else {}
        penalty = float(sum(latest_record.get("penalties", {}).values()))
        step = len(tracker.iteration_history)
        conv_str = f"{convergence_val:.3f}" if convergence_val is not None else "N/A"
        print(f"Iter: {step:<5} | NSE: {current_nse: .4f} | Loss: {current_loss: .4f} | Penalty: {penalty: .4f} | Conv: {conv_str}")

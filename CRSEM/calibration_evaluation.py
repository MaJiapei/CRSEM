"""Calibration evaluation tools for CRSEM.

This module provides:
- Objective functions (NSE, KGE, RMSE, MAE, R²)
- Penalty terms (channel ratio, annual R-factor)
- Metrics extraction and diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Optional, Dict

import numpy as np


# =============================================================================
# Performance Metrics
# =============================================================================

def nse(sim, obs):
    """Nash-Sutcliffe Efficiency."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) == 0:
        return np.nan

    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0:
        if numerator == 0:
            return 1.0
        else:
            return -np.inf
    return 1 - (numerator / denominator)


def kge(sim, obs):
    """Kling-Gupta Efficiency."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) < 2:
        return np.nan

    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs)
    std_sim = np.std(sim)

    if std_obs == 0 or std_sim == 0 or mean_obs == 0:
        return np.nan

    r_matrix = np.corrcoef(obs, sim)
    if np.isnan(r_matrix).any():
        return np.nan
    r = r_matrix[0, 1]

    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(sim, obs):
    """Root Mean Squared Error."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) == 0:
        return np.nan

    return np.sqrt(np.mean((sim - obs) ** 2))


def mae(sim, obs):
    """Mean Absolute Error."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) == 0:
        return np.nan

    return np.mean(np.abs(sim - obs))


def r2(sim, obs):
    """Coefficient of Determination."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) < 2:
        return np.nan

    corr_matrix = np.corrcoef(obs, sim)
    if np.isnan(corr_matrix).any():
        return np.nan

    return corr_matrix[0, 1] ** 2


def pbias(sim, obs):
    """Percent Bias."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) == 0:
        return np.nan

    numerator = np.sum(sim - obs)
    denominator = np.sum(obs)
    if denominator == 0:
        return np.nan
    return 100 * (numerator / denominator)


def rsr(sim, obs):
    """RMSE-observations standard deviation ratio."""
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]

    if len(obs) == 0:
        return np.nan

    std_obs = np.std(obs)
    if std_obs == 0:
        return np.nan

    return rmse(sim, obs) / std_obs


def calculate_metrics(sim, obs):
    """Calculate a dictionary of performance metrics.

    Args:
        sim: Simulated values
        obs: Observed values

    Returns:
        Dictionary with NSE, PBIAS, RSR, KGE, RMSE, R2, MAE, and sample count
    """
    valid_indices = ~np.isnan(np.asarray(obs)) & ~np.isnan(np.asarray(sim))

    return {
        'NSE': nse(sim, obs),
        'PBIAS': pbias(sim, obs),
        'RSR': rsr(sim, obs),
        'KGE': kge(sim, obs),
        'RMSE': rmse(sim, obs),
        'R2': r2(sim, obs),
        'MAE': mae(sim, obs),
        'n months': int(np.sum(valid_indices))
    }


class Objective(Protocol):
    """Score the primary calibration target."""

    name: str

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        ...


class Penalty(Protocol):
    """Compute an auxiliary penalty term from a model run."""

    name: str

    def evaluate(self, run_output, ssf_pred: np.ndarray) -> dict[str, float]:
        ...


@dataclass(slots=True)
class NSEObjective:
    name: str = "nse"

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(nse(ssf_pred_valid, ssf_obs_valid))
        return metric_val, 1.0 - metric_val


@dataclass(slots=True)
class NSEPBIASObjective:
    name: str = "nse_pbias"
    pbias_scale: float = 100.0

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(nse(ssf_pred_valid, ssf_obs_valid))
        pbias_val = float(pbias(ssf_pred_valid, ssf_obs_valid))
        if not np.isfinite(pbias_val):
            pbias_penalty = float("inf")
        else:
            pbias_penalty = abs(pbias_val) / self.pbias_scale
        return metric_val, (1.0 - metric_val) + pbias_penalty


@dataclass(slots=True)
class KGEObjective:
    name: str = "kge"

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(kge(ssf_pred_valid, ssf_obs_valid))
        return metric_val, 1.0 - metric_val


@dataclass(slots=True)
class KGEPBIASObjective:
    name: str = "kge_pbias"
    pbias_scale: float = 100.0

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(kge(ssf_pred_valid, ssf_obs_valid))
        pbias_val = float(pbias(ssf_pred_valid, ssf_obs_valid))
        if not np.isfinite(pbias_val):
            pbias_penalty = float("inf")
        else:
            pbias_penalty = abs(pbias_val) / self.pbias_scale
        return metric_val, (1.0 - metric_val) + pbias_penalty


@dataclass(slots=True)
class RMSEObjective:
    name: str = "rmse"

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(rmse(ssf_pred_valid, ssf_obs_valid))
        return metric_val, metric_val / std_obs


@dataclass(slots=True)
class MAEObjective:
    name: str = "mae"

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(mae(ssf_pred_valid, ssf_obs_valid))
        return metric_val, metric_val / std_obs


@dataclass(slots=True)
class R2Objective:
    name: str = "r2"

    def evaluate(self, ssf_pred_valid: np.ndarray, ssf_obs_valid: np.ndarray, std_obs: float) -> tuple[float, float]:
        metric_val = float(r2(ssf_pred_valid, ssf_obs_valid))
        return metric_val, 1.0 - metric_val


@dataclass(slots=True)
class ChannelRatioPenalty:
    name: str = "channel_penalty"
    lower_bound: float = -0.6
    upper_bound: float = 0.3

    def evaluate(self, run_output, ssf_pred: np.ndarray) -> dict[str, float]:
        channel_ratio = _channel_contribution_ratio(run_output)
        penalty = float(
            max(0.0, self.lower_bound - channel_ratio) ** 2 +
            max(0.0, channel_ratio - self.upper_bound) ** 2
        )
        return {
            "channel_ratio": channel_ratio,
            self.name: penalty,
        }


@dataclass(slots=True)
class AnnualRFactorPenalty:
    name: str = "rain_penalty"
    lower_normal: float = 100.0
    upper_normal: float = 200.0

    def evaluate(self, run_output, ssf_pred: np.ndarray) -> dict[str, float]:
        if not hasattr(run_output, "R_rain") or not hasattr(run_output, "R_melt"):
            return {self.name: 0.0}

        r_rain = np.asarray(run_output.R_rain, dtype=float)
        r_melt = np.asarray(run_output.R_melt, dtype=float)
        if r_rain.ndim > 1:
            r_rain = np.nanmean(r_rain, axis=tuple(range(1, r_rain.ndim)))
        if r_melt.ndim > 1:
            r_melt = np.nanmean(r_melt, axis=tuple(range(1, r_melt.ndim)))
        n_months = len(r_rain)
        n_years = n_months // 12
        mean_r_annual = 0.0
        if n_years > 0:
            r_total = (r_rain + r_melt)[: n_years * 12].reshape(n_years, 12)
            mean_r_annual = float(np.mean(np.sum(r_total, axis=1)))

        if mean_r_annual < self.lower_normal:
            penalty = float(((self.lower_normal - mean_r_annual) / self.lower_normal) ** 2)
        elif mean_r_annual > self.upper_normal:
            penalty = float(((mean_r_annual - self.upper_normal) / self.upper_normal) ** 2)
        else:
            penalty = 0.0

        return {
            "mean_R_annual": mean_r_annual,
            self.name: penalty,
        }


def create_objective(method: str) -> Objective:
    method_key = (method or "nse").lower()
    objective_cls = OBJECTIVE_REGISTRY.get(method_key, NSEObjective)
    return objective_cls()


OBJECTIVE_REGISTRY: dict[str, type[Objective]] = {
    "nse": NSEObjective,
    "nse_pbias": NSEPBIASObjective,
    "kge": KGEObjective,
    "kge_pbias": KGEPBIASObjective,
    "rmse": RMSEObjective,
    "mae": MAEObjective,
    "r2": R2Objective,
}

PENALTY_REGISTRY: dict[str, type[Penalty]] = {
    "channel_ratio": ChannelRatioPenalty,
    "annual_r_factor": AnnualRFactorPenalty,
}

MODEL_PENALTY_REGISTRY: dict[str, tuple[str, ...]] = {
    "default": ("channel_ratio",),
    "crsem": ("channel_ratio", "annual_r_factor"),
    "rusle": ("channel_ratio",),
}


def register_objective(name: str, objective_cls: type[Objective]) -> None:
    OBJECTIVE_REGISTRY[name.lower()] = objective_cls


def register_penalty(name: str, penalty_cls: type[Penalty]) -> None:
    PENALTY_REGISTRY[name.lower()] = penalty_cls


def register_model_penalties(model_type: str, penalty_names: list[str] | tuple[str, ...]) -> None:
    MODEL_PENALTY_REGISTRY[model_type.lower()] = tuple(name.lower() for name in penalty_names)


def create_penalties(model_type: str, penalty_settings: Optional[Dict[str, Dict[str, Any]]] = None) -> list[Penalty]:
    """Create penalty instances for a model type.

    Args:
        model_type: Model type ('crsem', 'rusle', etc.)
        penalty_settings: Optional dictionary with penalty configurations.
                         If provided, overrides default penalty bounds.
                         Format: {"channel_ratio": {"lower_bound": -0.5, "upper_bound": 0.4}, ...}

    Returns:
        List of Penalty instances
    """
    penalty_names = MODEL_PENALTY_REGISTRY.get(model_type.lower(), MODEL_PENALTY_REGISTRY["default"])

    penalties = []
    for name in penalty_names:
        if name not in PENALTY_REGISTRY:
            continue

        penalty_cls = PENALTY_REGISTRY[name]
        config = None

        # Apply custom settings if provided
        if penalty_settings and name in penalty_settings:
            config = penalty_settings[name]

        # Create penalty instance with optional custom bounds
        if config:
            penalties.append(penalty_cls(**config))
        else:
            penalties.append(penalty_cls())

    return penalties


def _extract_hillslope_components(run_output) -> tuple[np.ndarray | None, np.ndarray | None]:
    if hasattr(run_output, "E_hillslope_rain"):
        e_rain = np.asarray(run_output.E_hillslope_rain)
        if hasattr(run_output, "E_hillslope_melt") and run_output.E_hillslope_melt is not None:
            e_melt = np.asarray(run_output.E_hillslope_melt)
        else:
            e_melt = np.zeros_like(e_rain)
        return e_rain, e_melt
    if hasattr(run_output, "E_hillslope"):
        e_rain = np.asarray(run_output.E_hillslope)
        return e_rain, np.zeros_like(e_rain)
    return None, None


def _total_hillslope_sediment(run_output) -> float:
    if hasattr(run_output, "E_hillslope") and run_output.E_hillslope is not None:
        total_hillslope = float(np.nansum(np.asarray(run_output.E_hillslope, dtype=float)))
        if np.isfinite(total_hillslope):
            return total_hillslope

    e_rain, e_melt = _extract_hillslope_components(run_output)
    if e_rain is None:
        return 0.0
    total_hillslope = float(np.nansum(np.asarray(e_rain, dtype=float) + np.asarray(e_melt, dtype=float)))
    if not np.isfinite(total_hillslope):
        return float("nan")
    return total_hillslope


def _channel_contribution_ratio(run_output) -> float:
    """Net channel contribution normalized by hillslope sediment supply."""
    total_channel = float(np.nansum(np.asarray(run_output.A_channel, dtype=float)))
    total_hillslope = _total_hillslope_sediment(run_output)
    tiny = 1e-9

    if not np.isfinite(total_channel) or not np.isfinite(total_hillslope):
        return float("nan")
    if abs(total_hillslope) <= tiny:
        if abs(total_channel) <= tiny:
            return 0.0
        return float(np.copysign(np.inf, total_channel))
    return float(total_channel / total_hillslope)


def _require_output_field(run_output, field_name: str) -> np.ndarray:
    if not hasattr(run_output, field_name):
        raise ValueError(f"Run output is missing required field '{field_name}'.")
    value = getattr(run_output, field_name)
    if value is None:
        raise ValueError(f"Run output field '{field_name}' cannot be None.")
    return np.asarray(value)


@dataclass(slots=True)
class ObjectiveEvaluator:
    """Compose a primary objective and optional penalty terms."""

    objective_method: str
    model_type: str
    penalty_settings: Optional[Dict[str, Any]] = None
    objective: Objective = field(init=False)
    penalties: list[Penalty] = field(init=False)

    def __post_init__(self) -> None:
        self.objective = create_objective(self.objective_method)
        self.penalties = create_penalties(self.model_type, self.penalty_settings)

    def evaluate(self, run_output, ssf_obs: np.ndarray) -> dict[str, float]:
        valid_mask = ~np.isnan(ssf_obs)
        if not np.any(valid_mask):
            empty_payload = {
                "loss": float(np.inf),
                "metric_value": float("-inf"),
                "nse": float("-inf"),
                "std_ratio": float("inf"),
                "objective_value": float("inf"),
            }
            for penalty in self.penalties:
                empty_payload[penalty.name] = 0.0
                if penalty.name == "channel_penalty":
                    empty_payload["channel_ratio"] = 0.0
                if penalty.name == "rain_penalty":
                    empty_payload["mean_R_annual"] = 0.0
            return empty_payload

        ssf_obs_valid = ssf_obs[valid_mask]
        ssf_pred = np.asarray(run_output.SSF_pred, dtype=float)

        # Check for NaN/Inf or unreasonably large values in the ENTIRE simulation output
        # This catches runs that are numerically unstable even if explosion occurs
        # only in months without observations
        MAX_REASONABLE_SSF = 1e12  # Maximum reasonable SSF value (1 trillion tonnes)
        has_invalid = not np.all(np.isfinite(ssf_pred))
        has_overflow = np.any(np.abs(ssf_pred) > MAX_REASONABLE_SSF)

        if has_invalid or has_overflow:
            return {
                "loss": float("inf"),
                "metric_value": float("-inf"),
                "nse": float("-inf"),
                "std_ratio": float("inf"),
                "objective_value": float("inf"),
                "channel_ratio": 0.0,
                "mean_R_annual": 0.0,
                "channel_penalty": 0.0,
                "rain_penalty": 0.0,
            }

        # Extract valid months for metric calculation (after passing the full-series check)
        ssf_pred_valid = ssf_pred[valid_mask]

        # Clip extreme values ONLY for metric calculation stability
        # The raw output check above already ensures fundamentally valid runs
        clip_limit = 1e10
        ssf_pred_clipped = np.clip(ssf_pred_valid, -clip_limit, clip_limit)
        ssf_obs_clipped = np.clip(ssf_obs_valid, -clip_limit, clip_limit)

        std_obs = float(np.std(ssf_obs_clipped))
        if std_obs == 0.0:
            std_obs = 1.0

        metric_value, objective_value = self.objective.evaluate(ssf_pred_clipped, ssf_obs_clipped, std_obs)

        # Handle NaN/Inf from objective evaluation
        if not np.isfinite(metric_value):
            metric_value = float("-inf") if self.objective.name in ("nse", "kge", "r2") else float("inf")
        if not np.isfinite(objective_value):
            objective_value = float("inf")

        current_nse = float(metric_value if self.objective.name == "nse" else nse(ssf_pred_clipped, ssf_obs_clipped))
        std_ratio = float(np.std(ssf_pred_clipped) / std_obs) if std_obs > 0 else 1.0

        # Ensure std_ratio is finite
        if not np.isfinite(std_ratio):
            std_ratio = float("inf")

        penalty_payload: dict[str, float] = {}
        total_penalty = 0.0
        for penalty in self.penalties:
            penalty_result = penalty.evaluate(run_output, ssf_pred)
            penalty_payload.update(penalty_result)
            penalty_val = float(penalty_result.get(penalty.name, 0.0))
            # Non-finite penalty indicates invalid run - must be rejected
            if not np.isfinite(penalty_val):
                return {
                    "loss": float("inf"),
                    "metric_value": float("-inf"),
                    "nse": float("-inf"),
                    "std_ratio": float("inf"),
                    "objective_value": float(objective_value),
                    **penalty_payload,
                }
            total_penalty += penalty_val

        total_loss = float(objective_value + total_penalty)

        return {
            "loss": total_loss,
            "metric_value": float(metric_value),
            "nse": current_nse,
            "std_ratio": std_ratio,
            "objective_value": float(objective_value),
            **penalty_payload,
        }


@dataclass(slots=True)
class MetricsExtractor:
    """Extract calibration metrics from a model run."""

    # Threshold for detecting unreasonably large SSF values that indicate overflow
    MAX_REASONABLE_SSF = 1e12  # 1 trillion tonnes

    def extract(self, run_output, ssf_obs: np.ndarray) -> dict[str, Any]:
        valid_mask = ~np.isnan(ssf_obs)
        if np.any(valid_mask):
            ssf_obs_valid = ssf_obs[valid_mask]
            ssf_pred = np.asarray(run_output.SSF_pred)

            # Check for NaN/Inf or unreasonably large values in the ENTIRE simulation output
            # This catches runs that are numerically unstable even if explosion occurs
            # only in months without observations
            has_invalid = not np.all(np.isfinite(ssf_pred))
            has_overflow = np.any(np.abs(ssf_pred) > self.MAX_REASONABLE_SSF)

            if has_invalid or has_overflow:
                # Return infinite/NaN metrics to mark this run as invalid
                # AICSelector and other consumers will exclude these
                return {
                    "nse": float("-inf"),
                    "sse": float("inf"),
                    "std_ratio": float("inf"),
                    "channel_ratio": 0.0,
                    "kge": float("-inf"),
                    "rmse": float("inf"),
                    "mae": float("inf"),
                    "r2": float("-inf"),
                    "SSF_pred": ssf_pred,
                    "A_channel": np.asarray(run_output.A_channel),
                    "E_rain": None,
                    "E_melt": None,
                }

            # Extract valid months for metric calculation (after passing the full-series check)
            ssf_pred_valid = ssf_pred[valid_mask]

            sse = float(np.sum((ssf_obs_valid - ssf_pred_valid) ** 2))
            nse_val = float(nse(ssf_pred_valid, ssf_obs_valid))
            kge_val = float(kge(ssf_pred_valid, ssf_obs_valid))
            rmse_val = float(rmse(ssf_pred_valid, ssf_obs_valid))
            mae_val = float(mae(ssf_pred_valid, ssf_obs_valid))
            r2_val = float(r2(ssf_pred_valid, ssf_obs_valid))
            std_obs = float(np.std(ssf_obs_valid))
            std_ratio = float(np.std(ssf_pred_valid) / std_obs) if std_obs > 0 else float("inf")
        else:
            sse, nse_val, std_ratio = float("inf"), float("-inf"), float("inf")
            kge_val, rmse_val, mae_val, r2_val = float("-inf"), float("inf"), float("inf"), float("-inf")

        ssf_pred = np.asarray(run_output.SSF_pred)
        a_channel = np.asarray(run_output.A_channel)
        e_rain, e_melt = _extract_hillslope_components(run_output)

        channel_ratio = _channel_contribution_ratio(run_output)
        return {
            "nse": nse_val,
            "sse": sse,
            "std_ratio": std_ratio,
            "channel_ratio": channel_ratio,
            "kge": kge_val,
            "rmse": rmse_val,
            "mae": mae_val,
            "r2": r2_val,
            "SSF_pred": ssf_pred,
            "A_channel": a_channel,
            "E_rain": e_rain,
            "E_melt": e_melt,
        }


@dataclass(slots=True)
class DiagnosticsExtractor:
    """Build summary diagnostics for visualization and reporting."""

    def compute(self, run_output, model_inputs, s_area: float | None, ssf_obs: np.ndarray) -> dict[str, float]:
        diag: dict[str, float] = {}
        if hasattr(run_output, "R_rain") and run_output.R_rain is not None:
            r_all = np.asarray(run_output.R_rain)
            if hasattr(run_output, "R_melt") and run_output.R_melt is not None:
                r_all = r_all + np.asarray(run_output.R_melt)
            diag["R"] = self._annual_mean(r_all, is_flux=False, s_area=s_area)
        elif hasattr(run_output, "R_factor"):
            diag["R"] = self._mean_value(run_output.R_factor)

        ls = model_inputs["LS"]
        if ls is None:
            raise ValueError("Model inputs are missing required field 'LS' for diagnostics.")
        diag["LS"] = self._mean_value(ls)
        if hasattr(run_output, "C_factor") and run_output.C_factor is not None:
            diag["C"] = self._mean_value(run_output.C_factor)
        if hasattr(run_output, "K_factor") and run_output.K_factor is not None:
            diag["K"] = self._mean_value(run_output.K_factor)
        if hasattr(run_output, "SDR") and run_output.SDR is not None:
            diag["SDR"] = self._mean_value(run_output.SDR)

        e_rain, e_melt = _extract_hillslope_components(run_output)
        diag["E_rain_modulus"] = self._annual_mean(e_rain, is_flux=True, s_area=s_area)
        diag["E_melt_modulus"] = self._annual_mean(e_melt, is_flux=True, s_area=s_area)

        diag["A_channel_modulus"] = self._annual_mean(_require_output_field(run_output, "A_channel"), is_flux=True, s_area=s_area)
        diag["SSF_pred_modulus"] = self._annual_mean(_require_output_field(run_output, "SSF_pred"), is_flux=True, s_area=s_area)
        diag["SSF_obs_modulus"] = self._annual_mean(ssf_obs, is_flux=True, s_area=s_area)
        return diag

    def _annual_mean(self, arr, *, is_flux: bool, s_area: float | None) -> float:
        if arr is None:
            return 0.0
        values = np.asarray(arr)
        if values.ndim != 1:
            return 0.0
        n_years = len(values) // 12
        if n_years == 0:
            return 0.0
        arr_2d = values[: n_years * 12].reshape(n_years, 12)
        annual_totals = np.nansum(arr_2d, axis=1)
        val = float(np.nanmean(annual_totals))
        if is_flux and s_area:
            val = val / s_area
        return val

    def _mean_value(self, val) -> float:
        if hasattr(val, "values"):
            val = val.values
        return float(np.nanmean(val))

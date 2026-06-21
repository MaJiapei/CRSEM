"""Sensitivity analysis tools for CRSEM.

This module provides functions to analyze model sensitivity
to parameters and climate drivers.

Features:
    - One-at-a-time (OAT) parameter sensitivity
    - Climate/NDVI sensitivity using regression and SHAP analysis
    - Point-mode validation for driver data
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any
from itertools import combinations
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression
import shap

from CRSEM.model import ModelFactory


def _fit_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Fit a linear model and return in-sample R²."""
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _compute_vif(X: np.ndarray, feature_cols: Sequence[str]) -> Dict[str, float]:
    """Compute variance inflation factors for each feature."""
    vif = {}
    for idx, col in enumerate(feature_cols):
        other_idx = [j for j in range(len(feature_cols)) if j != idx]
        r2 = _fit_r2(X[:, other_idx], X[:, idx])
        if np.isnan(r2):
            vif[col] = float("nan")
        elif r2 >= 1.0:
            vif[col] = float("inf")
        else:
            vif[col] = float(1.0 / (1.0 - r2))
    return vif


def _residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Return target residuals after regressing on covariates."""
    if covariates.size == 0:
        return target - np.mean(target)
    model = LinearRegression()
    model.fit(covariates, target)
    return target - model.predict(covariates)


def _compute_partial_and_unique_stats(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Compute partial correlations and unique R² contribution per feature."""
    full_r2 = _fit_r2(X, y)
    stats: Dict[str, Dict[str, float]] = {}
    for idx, col in enumerate(feature_cols):
        other_idx = [j for j in range(len(feature_cols)) if j != idx]
        x_resid = _residualize(X[:, idx], X[:, other_idx])
        y_resid = _residualize(y, X[:, other_idx])

        x_std = float(np.std(x_resid))
        y_std = float(np.std(y_resid))
        if x_std <= 0 or y_std <= 0:
            partial_corr = float("nan")
        else:
            partial_corr = float(np.corrcoef(x_resid, y_resid)[0, 1])

        reduced_r2 = _fit_r2(X[:, other_idx], y)
        unique_r2 = float(full_r2 - reduced_r2) if not np.isnan(full_r2) and not np.isnan(reduced_r2) else float("nan")
        stats[col] = {
            "partial_corr": partial_corr,
            "unique_r2": unique_r2,
            "unique_fraction_of_full_r2": float(unique_r2 / full_r2) if full_r2 not in (0.0, np.nan) and not np.isnan(unique_r2) else float("nan"),
        }
    return stats


def _compute_commonality(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    """Partition model R² into unique and shared commonality components."""
    subset_r2: Dict[tuple[str, ...], float] = {}
    for subset_size in range(1, len(feature_cols) + 1):
        for subset in combinations(feature_cols, subset_size):
            indices = [feature_cols.index(name) for name in subset]
            subset_r2[subset] = _fit_r2(X[:, indices], y)

    commonality_raw: Dict[str, float] = {}
    for subset_size in range(1, len(feature_cols) + 1):
        for subset in combinations(feature_cols, subset_size):
            value = subset_r2[subset]
            if subset_size > 1:
                for smaller_size in range(1, subset_size):
                    for smaller in combinations(subset, smaller_size):
                        value -= commonality_raw["+".join(smaller)]
            commonality_raw["+".join(subset)] = float(value)

    full_key = tuple(feature_cols)
    full_r2 = subset_r2[full_key]
    commonality_relative = {}
    for key, value in commonality_raw.items():
        commonality_relative[key] = float(value / full_r2) if full_r2 not in (0.0, np.nan) else float("nan")

    return {
        "subset_r2": {"+".join(key): float(value) for key, value in subset_r2.items()},
        "commonality_raw": commonality_raw,
        "commonality_relative": commonality_relative,
    }


def _compute_residualized_driver_contribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: Sequence[str],
    target_feature: str,
    control_features: Sequence[str],
) -> Dict[str, float]:
    """Measure the contribution of target_feature after removing control-feature signal."""
    target_idx = feature_cols.index(target_feature)
    control_idx = [feature_cols.index(name) for name in control_features]

    target_raw = X[:, target_idx]
    controls = X[:, control_idx]
    target_resid = _residualize(target_raw, controls)
    y_resid = _residualize(y, controls)

    target_std = float(np.std(target_resid))
    y_std = float(np.std(y_resid))
    if target_std <= 0 or y_std <= 0:
        resid_corr = float("nan")
        resid_beta = float("nan")
        incremental_r2 = float("nan")
    else:
        resid_corr = float(np.corrcoef(target_resid, y_resid)[0, 1])
        simple_model = LinearRegression()
        simple_model.fit(target_resid.reshape(-1, 1), y_resid)
        resid_beta = float(simple_model.coef_[0] * (target_std / y_std))
        incremental_r2 = _fit_r2(target_resid.reshape(-1, 1), y_resid)

    target_total_std = float(np.std(target_raw))
    if target_total_std <= 0:
        resid_std_fraction = float("nan")
    else:
        resid_std_fraction = float(target_std / target_total_std)

    return {
        "residualized_std": target_std,
        "residualized_std_fraction_of_original": resid_std_fraction,
        "partial_corr": resid_corr,
        "standardized_beta_on_residualized_target": resid_beta,
        "incremental_r2_on_residualized_target": float(incremental_r2),
    }


def _safe_log_transform(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-transform a vector, applying a positive shift only when required."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot log-transform an all-NaN vector.")
    min_val = float(np.min(finite))
    shift = 0.0
    if min_val <= 0:
        shift = 1.0 - min_val
    return np.log(arr + shift), shift


def _compute_elasticity_analysis(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Estimate a semi-log climate elasticity model with residualized NDVI elasticity."""
    y_log, y_shift = _safe_log_transform(df[target].to_numpy(dtype=float))
    pre_log, pre_shift = _safe_log_transform(df["Pre"].to_numpy(dtype=float))
    ndvi_log, ndvi_shift = _safe_log_transform(df["NDVI"].to_numpy(dtype=float))
    t_values = df["T"].to_numpy(dtype=float)

    X = np.column_stack([t_values, pre_log, ndvi_log])
    model = LinearRegression()
    model.fit(X, y_log)
    y_pred = model.predict(X)

    y_mean = float(np.mean(y_log))
    ss_res = float(np.sum((y_log - y_pred) ** 2))
    ss_tot = float(np.sum((y_log - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    coeffs = {
        "T": float(model.coef_[0]),
        "Pre_log": float(model.coef_[1]),
        "NDVI_log": float(model.coef_[2]),
    }

    ndvi_control = np.column_stack([t_values, pre_log])
    ndvi_log_resid = _residualize(ndvi_log, ndvi_control)
    y_log_resid = _residualize(y_log, ndvi_control)

    ndvi_resid_std = float(np.std(ndvi_log_resid))
    y_resid_std = float(np.std(y_log_resid))
    if ndvi_resid_std <= 0 or y_resid_std <= 0:
        ndvi_partial = float("nan")
        ndvi_beta = float("nan")
        ndvi_incremental_r2 = float("nan")
    else:
        ndvi_partial = float(np.corrcoef(ndvi_log_resid, y_log_resid)[0, 1])
        ndvi_model = LinearRegression()
        ndvi_model.fit(ndvi_log_resid.reshape(-1, 1), y_log_resid)
        ndvi_beta = float(ndvi_model.coef_[0])
        ndvi_incremental_r2 = _fit_r2(ndvi_log_resid.reshape(-1, 1), y_log_resid)

    ndvi_total_std = float(np.std(ndvi_log))
    ndvi_resid_std_frac = float(ndvi_resid_std / ndvi_total_std) if ndvi_total_std > 0 else float("nan")

    contributions = {}
    feature_means = {
        "T": float(np.mean(t_values)),
        "Pre_log": float(np.mean(pre_log)),
        "NDVI_log": float(np.mean(ndvi_log)),
        "NDVI_log_resid": float(np.mean(np.abs(ndvi_log_resid))),
    }
    for key, coef in coeffs.items():
        contributions[key] = float(coef * feature_means[key])
    contributions["NDVI_log_resid"] = float(ndvi_beta * feature_means["NDVI_log_resid"]) if not np.isnan(ndvi_beta) else float("nan")

    return {
        "model_form": "log(target) ~ T + log(Pre_shifted) + log(NDVI_shifted)",
        "target_log_shift": float(y_shift),
        "feature_log_shifts": {
            "Pre": float(pre_shift),
            "NDVI": float(ndvi_shift),
        },
        "r2": float(r2),
        "coefficients": coeffs,
        "contribution_proxies": contributions,
        "residualized_ndvi_elasticity": {
            "partial_corr": ndvi_partial,
            "beta": ndvi_beta,
            "incremental_r2": float(ndvi_incremental_r2),
            "residualized_std_fraction_of_original_log_ndvi": ndvi_resid_std_frac,
        },
    }


# =============================================================================
# Climate/NDVI Sensitivity Analysis
# =============================================================================

def to_monthly_series(data_array: xr.DataArray, name: str) -> pd.Series:
    """Convert a 1D time series DataArray to a monthly pandas Series.

    Args:
        data_array: xarray DataArray with time dimension
        name: Name for the output Series

    Returns:
        Monthly pandas Series with timestamp index
    """
    if not isinstance(data_array, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray for {name}, got {type(data_array).__name__}")

    series = data_array.to_series()
    idx = pd.DatetimeIndex(series.index).to_period('M').to_timestamp('M')
    ser = pd.Series(series.values, index=idx, name=name)
    return ser.sort_index()


def prepare_sensitivity_dataset(
    temp_da: xr.DataArray,
    pre_da: xr.DataArray,
    ndvi_da: xr.DataArray,
    q_series: pd.Series,
    ssf_series: pd.Series,
    months: Optional[Sequence[int]] = None,
    freq: str = 'M'
) -> tuple[pd.DataFrame, Optional[set]]:
    """Prepare dataset for sensitivity analysis by merging and resampling time series.

    Args:
        temp_da: Temperature DataArray (time,)
        pre_da: Precipitation DataArray (time,)
        ndvi_da: NDVI DataArray (time,)
        q_series: Discharge time series (m³/s)
        ssf_series: Suspended sediment flux time series (t/month)
        months: Optional month filter (1-12)
        freq: Resampling frequency (default: 'M' for monthly)

    Returns:
        Tuple of (DataFrame with merged data, months filter set)
    """
    freq = (freq or 'M').upper()

    # Convert inputs to monthly series
    t_ser = to_monthly_series(temp_da, 'T')
    p_ser = to_monthly_series(pre_da, 'Pre')
    n_ser = to_monthly_series(ndvi_da, 'NDVI')

    # Prepare river targets
    q_ser = q_series.copy()
    ssf_ser = ssf_series.copy()
    q_ser.index = pd.DatetimeIndex(q_ser.index)
    ssf_ser.index = pd.DatetimeIndex(ssf_ser.index)

    # Derive Monthly SSC (kg/m³)
    month_seconds = q_ser.index.days_in_month * 24 * 3600
    water_volume = q_ser.values * month_seconds

    with np.errstate(divide='ignore', invalid='ignore'):
        ssc_values = (ssf_ser.values * 1000.0) / water_volume

    ssc_ser = pd.Series(ssc_values, index=q_ser.index, name='SSC')
    ssc_ser.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Merge all variables
    df = pd.concat([
        t_ser, p_ser, n_ser,
        ssf_ser.rename('SSF'),
        ssc_ser,
        pd.Series(water_volume, index=q_ser.index, name='water_volume_m3')
    ], axis=1).dropna()

    # Filter by months
    months_set = None
    if months is not None:
        months_set = {int(m) for m in months if 1 <= int(m) <= 12}
        if not months_set:
            raise ValueError("Months filter is empty or invalid. Provide month numbers in 1-12.")
        df = df[df.index.month.isin(months_set)]

    if df.empty:
        raise ValueError("No data remaining after applying months filter.")

    # Resample if needed
    if freq != 'M':
        agg_ops = {
            'T': 'mean',
            'Pre': 'mean',
            'NDVI': 'mean',
            'SSF': 'sum',
            'SSC': 'mean',
            'water_volume_m3': 'sum',
        }
        df = df.resample(freq).agg(agg_ops).dropna()

        # Recompute SSC for aggregated data
        with np.errstate(divide='ignore', invalid='ignore'):
            df['SSC'] = (df['SSF'] * 1000.0) / df['water_volume_m3']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna()

    # Remove helper column
    if 'water_volume_m3' in df:
        df = df.drop(columns=['water_volume_m3'])

    return df, months_set


def analyze_sensitivity(
    df: pd.DataFrame,
    min_samples: int = 24,
    return_dataset: bool = False,
    months_set: Optional[set] = None
) -> Dict[str, Any]:
    """Perform regression and SHAP analysis on the prepared dataset.

    Args:
        df: DataFrame from prepare_sensitivity_dataset()
        min_samples: Minimum samples required for analysis
        return_dataset: If True, include dataset in results
        months_set: Months filter set for reference

    Returns:
        Dictionary with sensitivity analysis results including:
            - samples_used, time_span, months_filter
            - feature_stats (mean, std for each feature)
            - targets: SSF and SSC regression/SHAP results
    """
    if len(df) < min_samples:
        raise ValueError(
            f"Not enough overlapping samples for regression "
            f"(found {len(df)}, require at least {min_samples})."
        )

    feature_cols = ['T', 'Pre', 'NDVI']
    target_cols = ['SSF', 'SSC']

    # Standardize Features
    X = df[feature_cols].to_numpy(dtype=float)
    feature_stats = {}
    X_std = np.zeros_like(X)

    for idx, col in enumerate(feature_cols):
        mean = float(np.mean(X[:, idx]))
        std = float(np.std(X[:, idx]))
        if std <= 0:
            raise ValueError(f"Feature '{col}' has zero variance, cannot standardize.")
        X_std[:, idx] = (X[:, idx] - mean) / std
        feature_stats[col] = {'mean': mean, 'std': std}

    # Initialize results
    results = {
        'samples_used': int(len(df)),
        'time_span': {
            'start': df.index[0].isoformat(),
            'end': df.index[-1].isoformat(),
        },
        'months_filter': sorted(months_set) if months_set else None,
        'feature_stats': feature_stats,
        'feature_collinearity': {
            'vif': _compute_vif(X, feature_cols),
        },
        'targets': {},
    }

    if return_dataset:
        df_copy = df.copy()
        df_copy.index.name = 'time'
        results['dataset'] = df_copy

    # Process each target
    for target in target_cols:
        y = df[target].to_numpy(dtype=float)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))

        if y_std <= 0:
            raise ValueError(f"Target '{target}' has zero variance, cannot standardize.")

        # Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        coeffs_raw = model.coef_
        intercept = float(model.intercept_)

        # Metrics
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

        # SHAP Analysis
        explainer = shap.LinearExplainer(
            model, X, feature_perturbation="correlation_dependent"
        )
        shap_vals = explainer.shap_values(X)

        # Calculate stats
        shap_mean = np.mean(shap_vals, axis=0)
        shap_abs_mean = np.mean(np.abs(shap_vals), axis=0)
        total_abs_shap = float(np.sum(shap_abs_mean))

        shap_rel = (
            {col: float(shap_abs_mean[i] / total_abs_shap) for i, col in enumerate(feature_cols)}
            if total_abs_shap > 0 else {col: float('nan') for col in feature_cols}
        )

        # Pearson Correlation
        pearson_corr = {}
        for col in feature_cols:
            corr = np.corrcoef(df[col].to_numpy(dtype=float), y)[0, 1]
            pearson_corr[col] = float(corr)

        partial_stats = _compute_partial_and_unique_stats(X, y, feature_cols)
        commonality = _compute_commonality(X, y, feature_cols)
        residualized_ndvi = _compute_residualized_driver_contribution(
            X=X,
            y=y,
            feature_cols=feature_cols,
            target_feature="NDVI",
            control_features=["T", "Pre"],
        )

        # Standardized Coefficients
        standardized_coeffs = {
            col: float(coeffs_raw[i] * (feature_stats[col]['std'] / y_std))
            for i, col in enumerate(feature_cols)
        }

        results['targets'][target] = {
            'standardized_coefficients': standardized_coeffs,
            'raw_coefficients': {col: float(coeffs_raw[i]) for i, col in enumerate(feature_cols)},
            'intercept': intercept,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'partial_corr': {col: partial_stats[col]['partial_corr'] for col in feature_cols},
            'unique_r2': {col: partial_stats[col]['unique_r2'] for col in feature_cols},
            'unique_fraction_of_full_r2': {
                col: partial_stats[col]['unique_fraction_of_full_r2'] for col in feature_cols
            },
            'commonality': commonality,
            'residualized_contribution': {
                'NDVI_given_T_Pre': residualized_ndvi,
            },
            'elasticity_model': _compute_elasticity_analysis(df, target),
            'mean': y_mean,
            'std': y_std,
            'residual_sum_squares': ss_res,
            'rank': len(feature_cols),
            'shapley': {
                'mean': {col: float(shap_mean[i]) for i, col in enumerate(feature_cols)},
                'mean_abs': {col: float(shap_abs_mean[i]) for i, col in enumerate(feature_cols)},
                'relative_importance': shap_rel,
                'baseline': float(np.atleast_1d(explainer.expected_value)[0]),
            }
        }

    return results


# =============================================================================
# Parameter Sensitivity Analysis
# =============================================================================

def run_oat_sensitivity_analysis(
    calibrated_params,
    objective_function,
    *,
    model_type: str = "crsem",
    n_samples: int = 10,
):
    """Run one-at-a-time sensitivity analysis around a calibrated parameter vector.

    Args:
        calibrated_params: Calibrated parameter values (1D array)
        objective_function: Function to evaluate (lower = better)
        model_type: Model type for parameter info ('crsem' or 'rusle')
        n_samples: Number of samples along each parameter dimension

    Returns:
        Dictionary with:
            - 'baseline': baseline objective value
            - {param_name}: (param_range, objective_values) for each parameter
    """
    print("\n--- Starting One-at-a-Time (OAT) Sensitivity Analysis ---")
    sensitivity_results = {}
    baseline_objective = objective_function(calibrated_params)
    print(f"Baseline objective value: {baseline_objective:.4e}")
    sensitivity_results["baseline"] = baseline_objective

    param_names, bounds = ModelFactory.get_parameter_info(model_type)
    calibrated = np.asarray(calibrated_params, dtype=float)
    if calibrated.ndim != 1:
        raise ValueError(f"calibrated_params must be 1D, got shape {calibrated.shape}.")
    if calibrated.shape[0] != len(param_names):
        raise ValueError(
            f"calibrated_params length {calibrated.shape[0]} does not match "
            f"expected parameter count {len(param_names)} for model_type='{model_type}'."
        )

    for i, param_name in enumerate(param_names):
        print(f"Analyzing sensitivity of parameter: {param_name}")
        temp_params = np.copy(calibrated)
        lower_bound, upper_bound = bounds[i]
        param_range = np.linspace(lower_bound, upper_bound, n_samples)
        objective_values = []

        for val in param_range:
            temp_params[i] = val
            objective_values.append(objective_function(temp_params))

        sensitivity_results[param_name] = (param_range, objective_values)

    return sensitivity_results


def validate_point_mode(driver) -> None:
    """Validate that driver is in point mode (no spatial dimensions).

    Args:
        driver: Driver object with model_inputs attribute

    Raises:
        ValueError: If any variable has spatial dimensions
    """
    spatial_dims = {"latitude", "longitude", "lat", "lon"}
    for var_name in ["T", "Pre", "NDVI"]:
        data_array = driver.model_inputs[var_name]
        if data_array is None:
            continue
        found_spatial = [dim for dim in data_array.dims if dim in spatial_dims]
        if found_spatial:
            raise ValueError(
                f"Data is not in point mode: '{var_name}' has spatial dimensions {found_spatial}. "
                f"Please call `to_point_driver(keep_rivers=True)` first to convert basin-scale data to point mode."
            )


def analyze_climate_ndvi_sensitivity(
    driver,
    *,
    min_samples: int = 24,
    freq: str = "M",
    months: Optional[Sequence[int]] = None,
    return_dataset: bool = False,
):
    """Run climate/NDVI sensitivity analysis for a point-mode driver.

    Args:
        driver: Point-mode driver with T, Pre, NDVI, Q, SSF data
        min_samples: Minimum samples required for analysis
        freq: Resampling frequency
        months: Optional month filter (1-12)
        return_dataset: If True, include dataset in results

    Returns:
        Sensitivity analysis results from analyze_sensitivity()
    """
    validate_point_mode(driver)
    df, months_set = prepare_sensitivity_dataset(
        temp_da=driver.model_inputs["T"],
        pre_da=driver.model_inputs["Pre"],
        ndvi_da=driver.model_inputs["NDVI"],
        q_series=driver.Q,
        ssf_series=driver.SSF,
        months=months,
        freq=freq,
    )
    return analyze_sensitivity(df=df, min_samples=min_samples, return_dataset=return_dataset, months_set=months_set)

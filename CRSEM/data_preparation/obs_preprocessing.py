"""Observation data preprocessing for CRSEM.

This module provides functions to process raw gauge observations
into monthly time series suitable for model calibration.

Features:
    - Daily to monthly aggregation (Q and SSF)
    - Missing data interpolation using precipitation as covariate
    - Year-month matrix completion
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Optional


def power_func(P: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power function for sediment-precipitation relationship.

    SSF = a * P^b

    Args:
        P: Precipitation values
        a: Coefficient
        b: Exponent

    Returns:
        Computed SSF values
    """
    return a * P ** b


def interpolate_by_covariate(
    target: np.ndarray,
    covariate: np.ndarray,
    min_samples: int = 3,
) -> np.ndarray:
    """Interpolate missing values in target using covariate relationship.

    Uses month-by-month power function fitting: SSF = a * P^b

    Args:
        target: 2D array (n_years, 12 months) with missing values
        covariate: 2D array (n_years, 12 months) as predictor
        min_samples: Minimum samples required for fitting

    Returns:
        Filled array with same shape as target
    """
    target_filled = target.copy()
    n_years, n_months = target.shape

    for m in range(n_months):
        tgt_month = target[:, m]
        cov_month = covariate[:, m]

        # Only fit using non-missing values
        mask = (
            np.isfinite(tgt_month)
            & np.isfinite(cov_month)
            & (cov_month > 0)
            & (tgt_month >= 0)
        )
        if mask.sum() < min_samples:
            continue

        try:
            popt, _ = curve_fit(
                power_func, cov_month[mask], tgt_month[mask], maxfev=10000
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue

        # Interpolate missing values
        missing = np.isnan(tgt_month) & np.isfinite(cov_month) & (cov_month > 0)
        if missing.any():
            target_filled[missing, m] = power_func(cov_month[missing], *popt)

    return target_filled


def _normalize_monthly_series(series: pd.Series) -> pd.Series:
    """Normalize a monthly time series to month-start timestamps."""
    normalized = pd.Series(series.copy(), dtype=float)
    normalized.index = pd.to_datetime(normalized.index).to_period("M").to_timestamp()
    normalized.index.name = "time"
    return normalized.sort_index()


def _fill_monthly_matrix(
    target_df: pd.DataFrame,
    covariate_df: pd.DataFrame,
    min_samples: int = 3,
    clip_min: float | None = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """Fill a year-month matrix using covariate fitting and monthly fallback."""
    original_missing = int(target_df.isna().sum().sum())
    interpolated = interpolate_by_covariate(
        target_df.values.astype(float),
        covariate_df.values.astype(float),
        min_samples=min_samples,
    )
    filled_df = pd.DataFrame(interpolated, index=target_df.index, columns=target_df.columns)

    remaining_before_fallback = int(filled_df.isna().sum().sum())
    monthly_fallback = target_df.median(axis=0, skipna=True)
    for month in filled_df.columns:
        missing = filled_df[month].isna()
        fallback_value = monthly_fallback.get(month)
        if missing.any() and pd.notna(fallback_value):
            filled_df.loc[missing, month] = float(fallback_value)

    if clip_min is not None:
        filled_df = filled_df.clip(lower=clip_min)

    remaining_after = int(filled_df.isna().sum().sum())
    filled_count = original_missing - remaining_after
    summary = {
        "original_missing": original_missing,
        "filled_by_covariate_or_fallback": filled_count,
        "remaining_missing": remaining_after,
        "remaining_before_fallback": remaining_before_fallback,
        "fallback_used": max(remaining_before_fallback - remaining_after, 0),
    }
    return filled_df, summary


def _monthly_matrix_to_series(df: pd.DataFrame) -> pd.Series:
    """Convert a year-month matrix back to a flat monthly series."""
    values = []
    index = []
    for year in df.index:
        for month in df.columns:
            index.append(pd.Timestamp(year=int(year), month=int(month), day=1))
            values.append(df.loc[year, month])
    series = pd.Series(values, index=pd.DatetimeIndex(index, name="time"), dtype=float)
    return series.sort_index()


def monthly_stack(series: pd.Series, agg_fun: callable = np.nanmean) -> pd.DataFrame:
    """Stack time series into year x month matrix.

    Args:
        series: Time series with DatetimeIndex
        agg_fun: Aggregation function (default: nanmean)

    Returns:
        DataFrame with years as index, months (1-12) as columns
    """
    df = series.to_frame(name='value')
    df['year'] = df.index.year
    df['month'] = df.index.month
    monthly_agg = df.groupby(['year', 'month'])['value'].agg(agg_fun).unstack()
    return monthly_agg


def year_complete(
    year_index: pd.DatetimeIndex,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Complete year-month matrix with NaN for missing years.

    Args:
        year_index: DatetimeIndex with year range (YS frequency)
        df: Year-month matrix to complete

    Returns:
        Completed DataFrame with all years in index
    """
    template = pd.DataFrame(
        index=year_index.year,
        columns=np.arange(1, 13),
        dtype=float
    )
    return template.combine_first(df)


def monthly_agg(group: pd.DataFrame) -> pd.Series:
    """Aggregate daily gauge data to monthly values.

    This function processes raw daily observations into monthly
    discharge and sediment load.

    Args:
        group: DataFrame for one month with columns:
            - Q.m3s-1: Daily discharge (m³/s)
            - SSC.kgm-3: Suspended sediment concentration (kg/m³)

    Returns:
        Series with:
            - Q.m3s-1_month: Monthly discharge
            - SSC_load_t_month: Monthly sediment load (tons/month)

    Notes:
        Q aggregation:
            - If >20 valid daily values: sum (treat as daily averages)
            - If >0 valid values: mean × days_in_month
            - Otherwise: NaN

        Sediment load calculation:
            Daily load = Q × SSC × 86400 (kg/day)
            Monthly total = sum of daily loads / 1000 (tons)
    """
    # Get days in month
    if len(group) > 0:
        try:
            n_days = int(group.index.days_in_month[0])
        except Exception:
            n_days = int(pd.Timestamp(group.index[0]).days_in_month)
    else:
        if group.name is not None:
            n_days = int(pd.Timestamp(group.name).days_in_month)
        else:
            n_days = np.nan

    # Process Q.m3s-1
    q = group.get('Q.m3s-1')
    Q_month = np.nan
    if q is not None:
        valid_count = int(q.notna().sum())
        if valid_count > 20:
            Q_month = np.nansum(q)
        elif valid_count > 0:
            Q_month = np.nanmean(q) * n_days

    # Process SSC and compute sediment load
    ssc = group.get('SSC.kgm-3')
    sediment_load_t = np.nan
    if (q is not None) and (ssc is not None):
        valid = np.isfinite(q.values) & np.isfinite(ssc.values)
        if valid.any():
            # Daily load: Q (m³/s) × SSC (kg/m³) × 86400 (s/day) = kg/day
            sediment_load_kg = np.nansum(q.values[valid] * ssc.values[valid] * 86400.0)
            sediment_load_t = sediment_load_kg / 1000.0

    return pd.Series({
        'Q.m3s-1_month': Q_month,
        'SSC_load_t_month': sediment_load_t
    })


def stack_ssf(
    ssf: pd.Series,
    df_pre: pd.DataFrame,
    sub_year: pd.DatetimeIndex,
    convert_to_monthly_total: bool = False,
) -> pd.DataFrame:
    """Aggregate daily SSF to monthly values with gap filling.

    Args:
        ssf: Daily SSF time series (t/day)
        df_pre: Precipitation DataFrame for interpolation
        sub_year: Date range (YS frequency) for target years
        convert_to_monthly_total: If True, convert daily mean to monthly total
                                  by multiplying with days_in_month

    Returns:
        Monthly SSF DataFrame (years × 12 months) with filled values
    """
    if convert_to_monthly_total:
        ssf_m = monthly_stack(ssf, np.nanmean)
        days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        for month in ssf_m.columns:
            ssf_m[month] = ssf_m[month] * days_in_month[month - 1]
    else:
        ssf_m = monthly_stack(ssf, np.sum)

    # Replace zeros with NaN
    ssf_m = ssf_m.replace({0: np.nan})

    # Complete year matrix
    ssf_m = year_complete(sub_year, ssf_m)

    # Interpolate missing values using precipitation
    ssf_m = interpolate_by_covariate(ssf_m.values, df_pre.values)
    ssf_m = pd.DataFrame(ssf_m, index=df_pre.index, columns=range(1, 13))

    return ssf_m


def stack_q(
    q: pd.Series,
    df_pre: pd.DataFrame,
    sub_year: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate daily Q to monthly values with gap filling.

    Args:
        q: Daily discharge time series (m³/s)
        df_pre: Precipitation DataFrame for interpolation
        sub_year: Date range (YS frequency) for target years

    Returns:
        Monthly Q DataFrame (years × 12 months) with filled values
    """
    q_m = monthly_stack(q, np.mean)
    q_m = q_m.replace({0: np.nan})
    q_m = year_complete(sub_year, q_m)
    q_m = interpolate_by_covariate(q_m.values, df_pre.values)
    q_m = pd.DataFrame(q_m, index=df_pre.index, columns=range(1, 13))

    return q_m


def process_raw_observations(
    daily_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    year_range: tuple[int, int],
    q_col: str = "Q.m3s-1",
    ssc_col: str = "SSC.kgm-3",
) -> tuple[pd.Series, pd.Series]:
    """Process raw daily gauge data to monthly time series.

    This is the main entry point for observation preprocessing.

    Args:
        daily_df: DataFrame with daily observations
                  Must have DatetimeIndex and columns: Q.m3s-1, SSC.kgm-3
        precip_df: Monthly precipitation DataFrame for interpolation
                   Shape: (n_years, 12 months)
        year_range: (start_year, end_year) tuple
        q_col: Column name for discharge (default: "Q.m3s-1")
        ssc_col: Column name for SSC (default: "SSC.kgm-3")

    Returns:
        Tuple of (Q_series, SSF_series) as flat time series
    """
    # Create year index
    sub_year = pd.date_range(
        f"{year_range[0]}-01-01",
        f"{year_range[1]}-12-31",
        freq="YS"
    )

    # Ensure columns exist
    required_cols = [q_col, ssc_col]
    for col in required_cols:
        if col not in daily_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Aggregate daily to monthly
    monthly = daily_df.resample('M').apply(monthly_agg)

    # Remove rows with all NaN
    monthly = monthly.dropna(how='all')

    # Extract and process Q and SSF
    Q_monthly = stack_q(
        monthly['Q.m3s-1_month'],
        precip_df,
        sub_year
    )

    SSF_monthly = stack_ssf(
        monthly['SSC_load_t_month'],
        precip_df,
        sub_year,
        convert_to_monthly_total=False
    )

    # Convert back to flat Series with period index
    Q_series = pd.Series(
        Q_monthly.values.flatten(),
        index=pd.MultiIndex.from_product(
            [Q_monthly.index, Q_monthly.columns],
            names=['year', 'month']
        )
    )

    SSF_series = pd.Series(
        SSF_monthly.values.flatten(),
        index=pd.MultiIndex.from_product(
            [SSF_monthly.index, SSF_monthly.columns],
            names=['year', 'month']
        )
    )

    return Q_series, SSF_series


def fill_monthly_observations_by_precip(
    q_series: pd.Series,
    ssf_series: pd.Series,
    precip_series: pd.Series,
    year_range: tuple[int, int] | None = None,
    fill_q: bool = True,
    fill_ssf: bool = True,
    min_samples: int = 3,
) -> tuple[pd.Series, pd.Series, dict]:
    """Fill monthly Q/SSF gaps using monthly precipitation as covariate.

    Missing values are first estimated using month-specific power-law fits
    against precipitation. Any unresolved gaps fall back to the monthly median
    climatology derived from the original observations.
    """
    q_monthly = _normalize_monthly_series(q_series)
    ssf_monthly = _normalize_monthly_series(ssf_series)
    precip_monthly = _normalize_monthly_series(precip_series)

    if year_range is None:
        candidate_years = []
        if len(q_monthly.index) > 0:
            candidate_years.extend([q_monthly.index.min().year, q_monthly.index.max().year])
        if len(ssf_monthly.index) > 0:
            candidate_years.extend([ssf_monthly.index.min().year, ssf_monthly.index.max().year])
        if not candidate_years:
            raise ValueError("At least one observation series must be non-empty.")
        year_range = (min(candidate_years), max(candidate_years))

    sub_year = pd.date_range(
        f"{year_range[0]}-01-01",
        f"{year_range[1]}-12-31",
        freq="YS",
    )

    target_years = sub_year.year

    precip_df = monthly_stack(precip_monthly, np.nanmean)
    precip_df = year_complete(sub_year, precip_df).reindex(index=target_years, columns=range(1, 13))

    q_df = monthly_stack(q_monthly, np.nanmean)
    q_df = year_complete(sub_year, q_df).reindex(index=target_years, columns=range(1, 13))
    ssf_df = monthly_stack(ssf_monthly, np.nansum)
    ssf_df = ssf_df.replace({0: np.nan})
    ssf_df = year_complete(sub_year, ssf_df).reindex(index=target_years, columns=range(1, 13))

    q_summary = {
        "original_missing": int(q_df.isna().sum().sum()),
        "filled_by_covariate_or_fallback": 0,
        "remaining_missing": int(q_df.isna().sum().sum()),
        "remaining_before_fallback": int(q_df.isna().sum().sum()),
        "fallback_used": 0,
    }
    ssf_summary = {
        "original_missing": int(ssf_df.isna().sum().sum()),
        "filled_by_covariate_or_fallback": 0,
        "remaining_missing": int(ssf_df.isna().sum().sum()),
        "remaining_before_fallback": int(ssf_df.isna().sum().sum()),
        "fallback_used": 0,
    }

    q_filled_df = q_df.copy()
    ssf_filled_df = ssf_df.copy()

    if fill_q:
        q_filled_df, q_summary = _fill_monthly_matrix(
            target_df=q_df,
            covariate_df=precip_df,
            min_samples=min_samples,
            clip_min=0.0,
        )

    if fill_ssf:
        ssf_filled_df, ssf_summary = _fill_monthly_matrix(
            target_df=ssf_df,
            covariate_df=precip_df,
            min_samples=min_samples,
            clip_min=0.0,
        )

    filled_q = _monthly_matrix_to_series(q_filled_df)
    filled_ssf = _monthly_matrix_to_series(ssf_filled_df)

    summary = {
        "year_range": [int(year_range[0]), int(year_range[1])],
        "Q": q_summary,
        "SSF": ssf_summary,
    }
    return filled_q, filled_ssf, summary

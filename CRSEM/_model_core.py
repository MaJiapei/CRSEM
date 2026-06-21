"""
Core erosion model algorithms (pure numpy, no xarray/pandas dependencies).

This module provides the single source of truth for model physics calculations.
The canonical model implementations should call these functions.

Design Principle:
- All functions operate on numpy arrays (float32)
- No xarray/pandas dependencies (wrapping is caller's responsibility)
- Stateless pure functions where possible
- Well-documented physical meaning of parameters

Usage:
    # From xarray wrapper
    result_np = r_factor_rain_core(P.values, a_rain, threshold)
    result_da = xr.DataArray(result_np, coords=P.coords, dims=P.dims)
    
    # From fast numpy version
    result = r_factor_rain_core(P, a_rain, threshold)  # direct usage
"""
import numba
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# R-Factor (Rainfall/Snowmelt Erosivity)
# =============================================================================

def r_factor_rain_core(
    P_rain: np.ndarray,
    a_rain: float,
    threshold: float
) -> np.ndarray:
    """Calculate rainfall erosivity R-factor.
    
    R = max(0, a_rain * (P - threshold))
    
    Args:
        P_rain: Rainfall amount (mm), any shape
        a_rain: Rainfall erosivity coefficient
        threshold: Minimum rainfall to cause erosion (mm)
        
    Returns:
        R_rain: Rainfall erosivity factor, same shape as P_rain
    """
    return np.maximum(0.0, a_rain * (P_rain - threshold)).astype(np.float32)


def r_factor_melt_core(
    Melt: np.ndarray,
    a_melt: float,
    threshold: float
) -> np.ndarray:
    """Calculate snowmelt erosivity R-factor.
    
    R = max(0, a_melt * (Melt - threshold))
    
    Args:
        Melt: Snowmelt amount (mm), any shape
        a_melt: Snowmelt erosivity coefficient  
        threshold: Minimum snowmelt to cause erosion (mm)
        
    Returns:
        R_melt: Snowmelt erosivity factor, same shape as Melt
    """
    return np.maximum(0.0, a_melt * (Melt - threshold)).astype(np.float32)


def _snowmelt_accumulation_1d_core(
    P_snow: np.ndarray,
    T: np.ndarray,
    days_in_month: np.ndarray,
    k_melt: float,
    T_melt_start: float = 2.0
) -> np.ndarray:
    """Calculate snowmelt with snowpack accumulation for a single time series."""
    n_months = len(T)
    melt = np.zeros(n_months, dtype=np.float32)
    snowpack = np.float32(0.0)

    for i in range(n_months):
        snowpack += P_snow[i]
        potential_melt = k_melt * max(0.0, T[i] - T_melt_start) * days_in_month[i]
        actual_melt = min(snowpack, potential_melt)
        snowpack -= actual_melt
        melt[i] = actual_melt

    return melt


@numba.guvectorize(
    ["void(float32[:], float32[:], float32[:], float32, float32, float32[:])"],
    "(n),(n),(n),(),()->(n)",
    nopython=True,
    target="parallel",
)
def _snowmelt_accumulation_guvec(p_snow_1d, t_1d, days_in_month_1d, k_melt_factor, t_melt_start, melt_1d):
    snowpack = 0.0
    for i in range(p_snow_1d.shape[0]):
        snowpack += p_snow_1d[i]
        temp_diff = t_1d[i] - t_melt_start
        if temp_diff < 0.0:
            temp_diff = 0.0
        potential_melt = k_melt_factor * temp_diff * days_in_month_1d[i]
        actual_melt = snowpack if snowpack < potential_melt else potential_melt
        snowpack -= actual_melt
        melt_1d[i] = actual_melt


def snowmelt_accumulation_core(
    P_snow: np.ndarray,
    T: np.ndarray,
    days_in_month: np.ndarray,
    k_melt: float,
    T_melt_start: float = 2.0
) -> np.ndarray:
    """Calculate snowmelt with snowpack accumulation for 1D or batched time series.

    Args:
        P_snow: Monthly snowfall (mm), shape (n_months,) or (n_months, n_cells)
        T: Monthly temperature (°C), same shape as ``P_snow``
        days_in_month: Days in each month, shape (n_months,)
        k_melt: Degree-day melt factor (mm/°C/day)
        T_melt_start: Temperature threshold for melt to begin (°C)

    Returns:
        Melt: Monthly snowmelt (mm), same shape as ``P_snow``
    """
    p_snow = np.asarray(P_snow, dtype=np.float32)
    temperature = np.asarray(T, dtype=np.float32)
    days = np.asarray(days_in_month, dtype=np.float32)

    if p_snow.shape != temperature.shape:
        raise ValueError(f"P_snow and T must share the same shape, got {p_snow.shape} and {temperature.shape}.")
    if days.ndim != 1:
        raise ValueError(f"days_in_month must be 1D, got shape {days.shape}.")
    if p_snow.shape[0] != days.shape[0]:
        raise ValueError(
            f"days_in_month length {days.shape[0]} must match time dimension {p_snow.shape[0]}."
        )

    if p_snow.ndim == 1:
        return _snowmelt_accumulation_1d_core(p_snow, temperature, days, k_melt, T_melt_start)
    if p_snow.ndim == 2:
        # Handle NaN values in grid mode (masked cells)
        valid_mask = ~(np.isnan(p_snow) | np.isnan(temperature))
        p_snow_clean = np.where(valid_mask, p_snow, np.float32(0.0))
        temp_clean = np.where(valid_mask, temperature, np.float32(0.0))

        melt = _snowmelt_accumulation_guvec(
            p_snow_clean.T,
            temp_clean.T,
            days,
            np.float32(k_melt),
            np.float32(T_melt_start),
        )
        melt_out = np.asarray(melt.T, dtype=np.float32)
        # Restore NaN where inputs were NaN
        melt_out = np.where(valid_mask, melt_out, np.float32(np.nan))
        return melt_out
    raise ValueError(f"Unsupported snowfall array shape: {p_snow.shape}")


# =============================================================================
# K-Factor (Soil Erodibility with Freeze-Thaw Effects)
# =============================================================================

def k_factor_freeze_thaw_core(
    T: np.ndarray,
    K_base: float,
    alpha_K: float,
    K_min_ratio: float,
    K_max_ratio: float,
    T_0: float = 0.0,
    sigma_K: float = 2.5
) -> np.ndarray:
    """Calculate K-factor with freeze-thaw enhancement.
    
    K = K_base * (1 + alpha_K * F_i)
    where F_i = exp(-(T - T_0)^2 / (2 * sigma_K^2))
    
    The freeze-thaw effect peaks at T_0 (typically 0°C) and decreases
    as temperature moves away from freezing point.
    
    Args:
        T: Temperature (°C), any shape
        K_base: Base soil erodibility factor
        alpha_K: Freeze-thaw enhancement coefficient
        K_min_ratio: Minimum K as ratio of K_base
        K_max_ratio: Maximum K as ratio of K_base
        T_0: Temperature of maximum freeze-thaw effect (°C)
        sigma_K: Width of freeze-thaw temperature window (°C)
        
    Returns:
        K: Soil erodibility factor, same shape as T
    """
    F_i = np.exp(-((T - T_0) ** 2) / (2 * sigma_K ** 2))
    K_raw = K_base * (1 + alpha_K * F_i)
    return np.clip(K_raw, K_min_ratio * K_base, K_max_ratio * K_base).astype(np.float32)


def k_factor_static_core(K_base: float) -> float:
    """Return static K-factor (no freeze-thaw effects, for RUSLE)."""
    return np.float32(K_base)


# =============================================================================
# C-Factor (Vegetation Cover)
# =============================================================================

def c_factor_ndvi_core(
    NDVI: np.ndarray,
    alpha_C: float,
    NDVI_min: float = 0.05,
    NDVI_max: float = 0.95
) -> np.ndarray:
    """Calculate C-factor from NDVI using Van der Knijff (2000) equation.
    
    C = exp(-alpha_C * NDVI / (1 - NDVI))
    
    Args:
        NDVI: Vegetation index (0-1), any shape
        alpha_C: Vegetation cover effect coefficient
        NDVI_min: Minimum valid NDVI (avoid division by zero)
        NDVI_max: Maximum valid NDVI (cap extreme values)
        
    Returns:
        C: Cover management factor, same shape as NDVI
    """
    # Clip NDVI to valid range to avoid numerical issues
    NDVI_clipped = np.clip(NDVI, NDVI_min, NDVI_max)
    inv_gap = 1.0 - NDVI_clipped
    return np.exp(-alpha_C * NDVI_clipped / inv_gap).astype(np.float32)


def c_factor_simple_core(NDVI: np.ndarray, alpha_C: float) -> np.ndarray:
    """Simplified C-factor calculation (for RUSLE without gap term).
    
    C = exp(-alpha_C * NDVI)
    """
    return np.exp(-alpha_C * NDVI).astype(np.float32)


# =============================================================================
# SDR (Sediment Delivery Ratio)
# =============================================================================

def sdr_static_core(IC: float, ic0: float, k: float) -> float:
    """Calculate static SDR from connectivity index.
    
    SDR = 0.8 / (1 + exp((ic0 - IC) / k))
    
    Args:
        IC: Connectivity index (dimensionless)
        ic0: IC value where SDR = 0.4 (inflection point)
        k: Slope parameter controlling SDR sensitivity
        
    Returns:
        SDR: Sediment delivery ratio (0-1)
    """
    return np.float32(0.8 / (1 + np.exp((ic0 - IC) / k)))


def sdr_dynamic_core(
    IC: float,
    P_rain: np.ndarray,
    Melt: np.ndarray,
    ic0: float,
    k: float,
    beta_sdr: float,
    P_mean: float = 50.0
) -> np.ndarray:
    """Calculate dynamic SDR with precipitation adjustment.
    
    SDR = min(1, base_SDR * dynamic_factor)
    where:
        base_SDR = 0.8 / (1 + exp((ic0 - IC) / k))
        dynamic_factor = clip(1 + beta_sdr * (P_total / P_mean), 1, 3)
    
    Args:
        IC: Connectivity index (scalar)
        P_rain: Monthly rainfall (mm), shape (n_months,)
        Melt: Monthly snowmelt (mm), shape (n_months,)
        ic0: IC inflection point
        k: SDR slope parameter
        beta_sdr: Dynamic adjustment coefficient
        P_mean: Reference precipitation for normalization (mm)
        
    Returns:
        SDR: Dynamic sediment delivery ratio, shape (n_months,)
    """
    base_SDR = 0.8 / (1 + np.exp((ic0 - IC) / k))
    
    if beta_sdr == 0:
        return np.full_like(P_rain, base_SDR, dtype=np.float32)
    
    total_precip = P_rain + Melt
    dynamic_factor = np.clip(1.0 + beta_sdr * (total_precip / P_mean), 1.0, 3.0)
    return np.minimum(1.0, base_SDR * dynamic_factor).astype(np.float32)


# =============================================================================
# Channel Erosion
# =============================================================================

def channel_erosion_core(
    Q: np.ndarray,
    S_in: np.ndarray,
    c_base: float,
    n_chan: float,
    K_chan: float
) -> np.ndarray:
    """Calculate channel erosion/deposition using transport capacity model.
    
    Transport capacity: T_cap = c_base * Q^n_chan
    Erosion potential: E_potential = T_cap - S_in
    
    If E_potential > 0: erosion = E_potential * K_chan (detachment limited)
    If E_potential < 0: deposition = E_potential (transport limited)
    
    Args:
        Q: River discharge (m³/s), shape (n_months,)
        S_in: Incoming hillslope sediment (tonnes), shape (n_months,)
        c_base: Base transport capacity coefficient
        n_chan: Discharge exponent (typically 1-2)
        K_chan: Channel erodibility coefficient
        
    Returns:
        A_channel: Channel erosion (+) or deposition (-), shape (n_months,)
    """
    T_cap = np.maximum(0.0, c_base * (Q ** n_chan))
    E_potential = T_cap - S_in
    return np.where(E_potential > 0, E_potential * K_chan, E_potential).astype(np.float32)


# =============================================================================
# RUSLE Equation Assembly
# =============================================================================

def hillslope_erosion_core(
    R: np.ndarray,
    K: np.ndarray,
    LS: float,
    C: np.ndarray,
    P_factor: float
) -> np.ndarray:
    """Calculate hillslope erosion using RUSLE equation.
    
    A = R * K * LS * C * P
    
    Args:
        R: Rainfall/melt erosivity factor
        K: Soil erodibility factor
        LS: Topographic factor (scalar or array)
        C: Cover management factor
        P_factor: Practice factor (scalar or array)
        
    Returns:
        A: Hillslope erosion (t/ha), same shape as inputs
    """
    return (R * K * LS * C * P_factor).astype(np.float32)


def total_sediment_flux_core(
    E_hillslope: np.ndarray,
    SDR: np.ndarray,
    s_area: float,
    A_channel: np.ndarray
) -> np.ndarray:
    """Calculate total sediment flux at basin outlet.
    
    SSF = (E_hillslope * SDR * s_area) + A_channel
    
    Args:
        E_hillslope: Hillslope erosion (t/ha)
        SDR: Sediment delivery ratio
        s_area: Basin area (ha)
        A_channel: Channel erosion/deposition (tonnes)
        
    Returns:
        SSF: Total sediment flux (tonnes)
    """
    S_in = E_hillslope * SDR * s_area
    return (S_in + A_channel).astype(np.float32)


# =============================================================================
# Precipitation partitioning
# =============================================================================

def partition_precipitation_core(
    T: np.ndarray,
    P: np.ndarray,
    T_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition precipitation into rainfall and snowfall based on temperature.
    
    Args:
        T: Temperature (°C)
        P: Precipitation (mm)
        T_threshold: Temperature threshold for rain/snow partition (°C)
        
    Returns:
        Tuple of (P_rain, P_snow)
    """
    is_rain = T > T_threshold
    P_rain = np.where(is_rain, P, 0.0).astype(np.float32)
    P_snow = np.where(~is_rain, P, 0.0).astype(np.float32)
    return P_rain, P_snow

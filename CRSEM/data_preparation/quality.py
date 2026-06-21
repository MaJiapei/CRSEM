"""Data quality assessment tools for CRSEM data preparation.

This module provides functions to assess the quality of input data
before model simulation.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class QualityReport:
    """Data quality assessment report."""

    variable: str
    total_count: int = 0
    valid_count: int = 0
    missing_count: int = 0
    missing_rate: float = 0.0
    consecutive_missing_max: int = 0
    outlier_count: int = 0
    coverage: float = 0.0
    passed: bool = True
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "variable": self.variable,
            "total_count": self.total_count,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "missing_rate": round(self.missing_rate, 4),
            "consecutive_missing_max": self.consecutive_missing_max,
            "outlier_count": self.outlier_count,
            "coverage": round(self.coverage, 4),
            "passed": self.passed,
            "issues": self.issues,
        }


def assess_time_series_quality(
    series: pd.Series,
    min_coverage: float = 0.95,
    max_missing_rate: float = 0.05,
    max_consecutive_missing: int = 3,
) -> QualityReport:
    """Assess quality of a time series.

    Args:
        series: Time series to assess
        min_coverage: Minimum required coverage (0-1)
        max_missing_rate: Maximum allowed missing rate (0-1)
        max_consecutive_missing: Maximum allowed consecutive missing values

    Returns:
        QualityReport with assessment results
    """
    report = QualityReport(variable=series.name or "unnamed")

    total = len(series)
    valid = series.notna().sum()
    missing = total - valid

    report.total_count = int(total)
    report.valid_count = int(valid)
    report.missing_count = int(missing)
    report.missing_rate = float(missing / total) if total > 0 else 1.0
    report.coverage = float(valid / total) if total > 0 else 0.0

    # Calculate consecutive missing
    is_missing = series.isna()
    consecutive_counts = []
    current_count = 0
    for is_nan in is_missing:
        if is_nan:
            current_count += 1
        else:
            if current_count > 0:
                consecutive_counts.append(current_count)
            current_count = 0
    if current_count > 0:
        consecutive_counts.append(current_count)

    report.consecutive_missing_max = max(consecutive_counts) if consecutive_counts else 0

    # Detect outliers using IQR method
    if valid > 4:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        report.outlier_count = int(outliers)
        if outliers > 0:
            report.issues.append(f"Detected {outliers} potential outliers")

    # Check thresholds
    if report.coverage < min_coverage:
        report.passed = False
        report.issues.append(f"Coverage {report.coverage:.2%} < {min_coverage:.2%}")

    if report.missing_rate > max_missing_rate:
        report.passed = False
        report.issues.append(f"Missing rate {report.missing_rate:.2%} > {max_missing_rate:.2%}")

    if report.consecutive_missing_max > max_consecutive_missing:
        report.passed = False
        report.issues.append(
            f"Consecutive missing {report.consecutive_missing_max} > {max_consecutive_missing}"
        )

    return report


def assess_spatial_quality(
    data_array: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
    min_valid_rate: float = 0.9,
) -> QualityReport:
    """Assess quality of spatial data (2D or 3D).

    Args:
        data_array: Spatial data array to assess (2D: y,x or 3D: time,y,x)
        mask: Optional mask array (1=valid, 0=invalid)
        min_valid_rate: Minimum required valid data rate

    Returns:
        QualityReport with assessment results
    """
    report = QualityReport(variable=data_array.name or "unnamed")

    values = data_array.values

    if mask is not None:
        # Handle 3D data (time, y, x) by applying 2D spatial mask
        if values.ndim == 3:
            # Mask is 2D, need to apply it to each time step
            mask_2d = mask.values > 0
            # Extract valid spatial locations for all time steps
            values_masked = values[:, mask_2d]
        else:
            # 2D data
            values_masked = values[mask.values > 0]
    else:
        values_masked = values

    total = values_masked.size
    valid = np.sum(~np.isnan(values_masked))
    missing = total - valid

    report.total_count = int(total)
    report.valid_count = int(valid)
    report.missing_count = int(missing)
    report.missing_rate = float(missing / total) if total > 0 else 1.0
    report.coverage = float(valid / total) if total > 0 else 0.0

    # Detect outliers
    valid_values = values_masked.flatten()[~np.isnan(values_masked.flatten())]
    if len(valid_values) > 4:
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((valid_values < lower_bound) | (valid_values > upper_bound))
        report.outlier_count = int(outliers)
        if outliers > 0:
            report.issues.append(f"Detected {outliers} potential outliers")

    # Check thresholds
    if report.coverage < min_valid_rate:
        report.passed = False
        report.issues.append(f"Valid rate {report.coverage:.2%} < {min_valid_rate:.2%}")

    return report


def print_quality_report(*reports: QualityReport) -> None:
    """Print quality reports in a formatted table.

    Args:
        reports: One or more QualityReport objects to print
    """
    print("\n" + "=" * 80)
    print("Data Quality Report")
    print("=" * 80)

    header = f"{'Variable':<20} {'Coverage':>10} {'Missing':>10} {'Consecutive':>12} {'Outliers':>10} {'Status':>8}"
    print(header)
    print("-" * 80)

    for report in reports:
        status = "PASS" if report.passed else "FAIL"
        print(
            f"{report.variable:<20} "
            f"{report.coverage:>9.1%} "
            f"{report.missing_rate:>9.1%} "
            f"{report.consecutive_missing_max:>12} "
            f"{report.outlier_count:>10} "
            f"{status:>8}"
        )

    print("-" * 80)

    # Print issues
    has_issues = any(r.issues for r in reports)
    if has_issues:
        print("\nIssues detected:")
        for report in reports:
            for issue in report.issues:
                print(f"  - {report.variable}: {issue}")

    print("=" * 80)


def generate_quality_report(
    Q: pd.Series,
    SSF: pd.Series,
    temperature: xr.DataArray,
    precipitation: xr.DataArray,
    ndvi: xr.DataArray,
    basin_mask: xr.DataArray,
    **thresholds,
) -> dict:
    """Generate comprehensive quality report for all input data.

    Args:
        Q: Discharge time series
        SSF: Sediment flux time series
        temperature: Temperature data array
        precipitation: Precipitation data array
        ndvi: NDVI data array
        basin_mask: Basin mask array
        **thresholds: Quality threshold overrides

    Returns:
        Dictionary with all quality reports
    """
    # Default thresholds
    min_coverage = thresholds.get("min_time_coverage", 0.95)
    max_missing_rate = thresholds.get("max_missing_rate", 0.05)
    max_consecutive = thresholds.get("max_consecutive_missing", 3)
    min_spatial_valid = thresholds.get("min_spatial_valid_rate", 0.9)

    reports = {
        "Q": assess_time_series_quality(Q, min_coverage, max_missing_rate, max_consecutive),
        "SSF": assess_time_series_quality(SSF, min_coverage, max_missing_rate, max_consecutive),
        "T": assess_spatial_quality(temperature, basin_mask, min_spatial_valid),
        "Pre": assess_spatial_quality(precipitation, basin_mask, min_spatial_valid),
        "NDVI": assess_spatial_quality(ndvi, basin_mask, min_spatial_valid),
    }

    return reports

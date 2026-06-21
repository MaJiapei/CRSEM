from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


class ResultAggregator:
    """Aggregate member-wise xarray results along the ``member`` dimension."""

    @staticmethod
    def aggregate(
        data: xr.DataArray | xr.Dataset,
        method: str = "mean",
        weights: np.ndarray | xr.DataArray | None = None,
    ) -> xr.DataArray | xr.Dataset:
        if "member" not in data.dims:
            return data

        weights_da = ResultAggregator._normalize_weights(data, weights)
        if method == "mean":
            return ResultAggregator._weighted_mean(data, weights_da) if weights_da is not None else data.mean("member")
        if method == "weighted_mean":
            return ResultAggregator._weighted_mean(data, weights_da)
        if method == "median":
            return data.median("member")
        if method == "std":
            return ResultAggregator._weighted_std(data, weights_da) if weights_da is not None else data.std("member")
        if method == "var":
            return ResultAggregator._weighted_var(data, weights_da) if weights_da is not None else data.var("member")
        if method == "min":
            return data.min("member")
        if method == "max":
            return data.max("member")
        if method.startswith("quantile"):
            q = float(method.split("_")[1]) if "_" in method else 0.5
            if weights_da is not None:
                raise NotImplementedError("Weighted quantiles are not implemented.")
            return data.quantile(q, dim="member")
        raise ValueError(f"Unsupported aggregation method: {method}")

    @staticmethod
    def _normalize_weights(
        data: xr.DataArray | xr.Dataset,
        weights: np.ndarray | xr.DataArray | None,
    ) -> xr.DataArray | None:
        if weights is None:
            return None
        if isinstance(weights, xr.DataArray):
            return weights
        return xr.DataArray(np.asarray(weights, dtype=float), dims=["member"], coords={"member": data.coords["member"]})

    @staticmethod
    def _weighted_mean(data: xr.DataArray | xr.Dataset, weights: xr.DataArray | None) -> xr.DataArray | xr.Dataset:
        if weights is None:
            raise ValueError("weights are required for weighted aggregation")
        return (data * weights).sum("member") / weights.sum()

    @staticmethod
    def _weighted_var(data: xr.DataArray | xr.Dataset, weights: xr.DataArray | None) -> xr.DataArray | xr.Dataset:
        if weights is None:
            raise ValueError("weights are required for weighted aggregation")
        mean_val = ResultAggregator._weighted_mean(data, weights)
        return ((data - mean_val) ** 2 * weights).sum("member") / weights.sum()

    @staticmethod
    def _weighted_std(data: xr.DataArray | xr.Dataset, weights: xr.DataArray | None) -> xr.DataArray | xr.Dataset:
        variance = ResultAggregator._weighted_var(data, weights)
        return xr.apply_ufunc(np.sqrt, variance)

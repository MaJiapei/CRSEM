"""Base model definition and shared helpers."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from CRSEM._model_core import channel_erosion_core
from CRSEM.contracts import BatchRunResult, ParameterBatch, PreparedInputs, RunContext
from CRSEM.model import ModelOutputs
from CRSEM.parameters import BaseParameters
from CRSEM.preparation import prepare_inputs

NumericArray = Union[np.ndarray, List[float], Tuple[float, ...], Sequence[float]]


class BaseModel:
    """Unified model base class with shared helpers and interface."""

    PARAM_NAMES: Tuple[str, ...] = tuple()
    PARAM_BOUNDS: Tuple[Tuple[float, float], ...] = tuple()
    NDVI_MIN = 0.05
    NDVI_MAX = 0.95

    def __init__(self, params: Any) -> None:
        self._init_params(params)

    def _init_params(self, params: BaseParameters) -> None:
        raise NotImplementedError

    def _resolve_context(self, source) -> RunContext:
        if isinstance(source, RunContext):
            return source
        if hasattr(source, "to_run_context"):
            return source.to_run_context()
        raise TypeError(f"Unsupported model run source: {type(source)}")

    def _prepare_context(self, source) -> tuple[RunContext, PreparedInputs]:
        context = self._resolve_context(source)
        return context, prepare_inputs(context)

    def _coerce_parameter_batch(self, params_batch=None) -> ParameterBatch:
        if params_batch is None:
            return ParameterBatch(values=self.params.to_array(), param_names=tuple(self.PARAM_NAMES))
        return ParameterBatch.from_any(params_batch, param_names=tuple(self.PARAM_NAMES))

    def _time_coord(self, context: RunContext) -> np.ndarray:
        for name in ("T", "Pre", "NDVI"):
            data_array = context.inputs[name]
            if data_array is not None:
                return data_array.coords["time"].values
        raise ValueError("RunContext must contain at least one dynamic input with a time coordinate.")

    def _spatial_shape(self, prepared: PreparedInputs) -> tuple[int, int]:
        if prepared.latitude is None or prepared.longitude is None:
            raise ValueError("PreparedInputs must provide latitude/longitude for gridded outputs.")
        return len(prepared.latitude), len(prepared.longitude)

    def _cell_count(self, prepared: PreparedInputs) -> int:
        if prepared.is_point_mode:
            return 1
        n_lat, n_lon = self._spatial_shape(prepared)
        return n_lat * n_lon

    def _time_length(self, prepared: PreparedInputs, context: RunContext) -> int:
        if prepared.time is not None:
            return int(len(prepared.time))
        return int(len(self._time_coord(context)))

    def _prepared_field(
        self,
        prepared: PreparedInputs,
        field_name: str,
        *,
        required: bool = True,
        dtype=np.float32,
    ) -> np.ndarray | None:
        value = getattr(prepared, field_name)
        if value is None:
            if required:
                raise ValueError(f"PreparedInputs.{field_name} is required for {type(self).__name__} execution.")
            return None
        return np.asarray(value, dtype=dtype)

    def _prepared_fields(
        self,
        prepared: PreparedInputs,
        *field_names: str,
        dtype=np.float32,
    ) -> tuple[np.ndarray, ...]:
        return tuple(self._prepared_field(prepared, name, dtype=dtype) for name in field_names)

    def _to_output_data_array(self, context: RunContext, prepared: PreparedInputs, values, name: str) -> xr.DataArray:
        arr = np.asarray(values, dtype=np.float32)
        time_coord = self._time_coord(context)
        n_time = self._time_length(prepared, context)

        if arr.ndim == 0:
            return xr.DataArray(arr, name=name)

        if arr.ndim == 1:
            if arr.shape[0] == n_time:
                return xr.DataArray(arr, coords={"time": time_coord}, dims=("time",), name=name)
            if not prepared.is_point_mode and arr.shape[0] == self._cell_count(prepared):
                n_lat, n_lon = self._spatial_shape(prepared)
                return xr.DataArray(
                    arr.reshape(n_lat, n_lon),
                    coords={prepared.spatial_dims[0]: prepared.latitude, prepared.spatial_dims[1]: prepared.longitude},
                    dims=prepared.spatial_dims,
                    name=name,
                )

        if arr.ndim == 2 and not prepared.is_point_mode and arr.shape == (n_time, self._cell_count(prepared)):
            n_lat, n_lon = self._spatial_shape(prepared)
            return xr.DataArray(
                arr.reshape(n_time, n_lat, n_lon),
                coords={
                    "time": time_coord,
                    prepared.spatial_dims[0]: prepared.latitude,
                    prepared.spatial_dims[1]: prepared.longitude,
                },
                dims=("time", *prepared.spatial_dims),
                name=name,
            )

        raise ValueError(f"Cannot map output '{name}' with shape {arr.shape} back to xarray coordinates.")

    def _prepared_outputs_to_model_outputs(
        self,
        context: RunContext,
        prepared: PreparedInputs,
        data: dict[str, np.ndarray | float | None],
    ) -> ModelOutputs:
        payload = {}
        for name, values in data.items():
            if values is None:
                payload[name] = None
                continue
            payload[name] = self._to_output_data_array(context, prepared, values, name)
        return ModelOutputs(**payload)

    def _channel_erosion_numpy(self, q, s_in) -> np.ndarray:
        return channel_erosion_core(
            np.asarray(q, dtype=np.float32),
            np.asarray(s_in, dtype=np.float32),
            self.c_base,
            self.n_chan,
            self.K_chan,
        )

    def _basin_flux_from_hillslope(self, values, s_area: float) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        basin_area = np.float32(s_area)
        if arr.ndim == 1:
            return (arr * basin_area).astype(np.float32)
        if arr.ndim == 2:
            return (np.nanmean(arr, axis=1).astype(np.float32) * basin_area).astype(np.float32)
        raise ValueError(f"Unsupported hillslope output shape for basin aggregation: {arr.shape}")

    def _run_prepared_hillslope_numpy(self, prepared: PreparedInputs, *, context: RunContext | None = None) -> dict[str, np.ndarray | float]:
        raise NotImplementedError

    def _run_prepared_hillslope_river_numpy(
        self,
        prepared: PreparedInputs,
        *,
        context: RunContext | None = None,
        output_mode: str = "full",
    ) -> dict[str, np.ndarray | float]:
        raise NotImplementedError

    def run_hillslope(self, source):
        raise NotImplementedError

    def run_hillslope_river(self, source):
        raise NotImplementedError

    def run_batch(self, source, params_batch=None, run_method: str = "run_hillslope_river") -> BatchRunResult:
        context = self._resolve_context(source)
        batch = self._coerce_parameter_batch(params_batch)
        outputs = []
        current_params = np.asarray(self.params.to_array(), dtype=float)
        param_cls = type(self.params)
        prepared = prepare_inputs(context)

        prepared_runners = {
            "run_hillslope": lambda model: model._run_prepared_hillslope_numpy(prepared, context=context),
            "run_hillslope_river": lambda model: model._run_prepared_hillslope_river_numpy(prepared, context=context),
        }

        for member_values in batch.values:
            if np.allclose(member_values, current_params, equal_nan=True):
                member_model = self
            else:
                member_model = type(self)(param_cls.from_array(member_values))
            if run_method in prepared_runners:
                prepared_outputs = prepared_runners[run_method](member_model)
                outputs.append(member_model._prepared_outputs_to_model_outputs(context, prepared, prepared_outputs))
            else:
                outputs.append(getattr(member_model, run_method)(context))

        metadata = dict(context.metadata)
        metadata["run_method"] = run_method
        return BatchRunResult.from_outputs(
            outputs,
            weights=batch.weights,
            member_ids=batch.member_ids,
            metadata=metadata,
        )

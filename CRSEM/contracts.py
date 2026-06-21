from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from CRSEM.model import ModelInputs, ModelOutputs
from CRSEM.parameters import BaseParameters


@dataclass(slots=True)
class RunContext:
    """All inputs required for a model run."""

    inputs: ModelInputs
    q: pd.Series | None = None
    ssf_obs: pd.Series | None = None
    s_area: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedInputs:
    """Numpy-ready inputs for internal model execution."""

    T: np.ndarray | None = None
    Pre: np.ndarray | None = None
    NDVI: np.ndarray | None = None
    LS: np.ndarray | None = None
    P_f: np.ndarray | None = None
    IC: np.ndarray | None = None
    K: np.ndarray | None = None
    q: np.ndarray | None = None
    time: np.ndarray | None = None
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    spatial_dims: tuple[str, ...] = ()
    dynamic_dims: tuple[str, ...] = ()
    is_point_mode: bool = False


@dataclass(slots=True)
class ParameterBatch:
    """Single-parameter and multi-member parameter container."""

    values: np.ndarray
    param_names: tuple[str, ...]
    weights: np.ndarray | None = None
    member_ids: list[str] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("ParameterBatch.values must be 1D or 2D.")
        if self.param_names and values.shape[1] != len(self.param_names):
            raise ValueError(
                f"ParameterBatch received {values.shape[1]} columns but expected {len(self.param_names)} from param_names."
            )
        self.values = values

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 1 or weights.size != values.shape[0]:
                raise ValueError("ParameterBatch.weights must be 1D and match the number of members.")
            weight_sum = np.sum(weights)
            if weight_sum <= 0:
                raise ValueError("ParameterBatch.weights must sum to a positive value.")
            self.weights = weights / weight_sum

        if self.member_ids is None:
            self.member_ids = [f"member_{idx}" for idx in range(values.shape[0])]
        elif len(self.member_ids) != values.shape[0]:
            raise ValueError("ParameterBatch.member_ids must match the number of members.")

    @property
    def n_members(self) -> int:
        return int(self.values.shape[0])

    def to_payload(self) -> dict[str, Any]:
        return {
            "param_names": list(self.param_names),
            "values": np.asarray(self.values, dtype=float).tolist(),
            "weights": None if self.weights is None else np.asarray(self.weights, dtype=float).tolist(),
            "member_ids": list(self.member_ids) if self.member_ids is not None else None,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ParameterBatch":
        return cls(
            values=np.asarray(payload["values"], dtype=float),
            param_names=tuple(payload["param_names"]),
            weights=None if payload.get("weights") is None else np.asarray(payload["weights"], dtype=float),
            member_ids=payload.get("member_ids"),
        )

    def to_file(self, filepath: str | Path, *, metrics: dict[str, Any] | None = None) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"parameter_batch": self.to_payload()}
        if metrics is not None:
            payload["metrics"] = metrics
        path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    @classmethod
    def from_file(cls, filepath: str | Path) -> tuple["ParameterBatch", dict[str, Any] | None]:
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "parameter_batch" not in data:
            raise ValueError("Invalid parameter batch file format.")
        return cls.from_payload(data["parameter_batch"]), data.get("metrics")

    @classmethod
    def from_any(cls, params, param_names: tuple[str, ...]) -> "ParameterBatch":

        if isinstance(params, cls):
            return params
        if isinstance(params, BaseParameters):
            return cls(values=params.to_array(), param_names=tuple(param_names))

        arr = np.asarray(params, dtype=float)
        return cls(values=arr, param_names=tuple(param_names))


@dataclass(slots=True)
class BatchRunResult:
    """Unified batch result with explicit member dimension semantics."""

    variables: dict[str, np.ndarray]
    coords: dict[str, Any]
    dims: dict[str, tuple[str, ...]]
    weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 1:
                raise ValueError("BatchRunResult.weights must be 1D.")
            self.weights = weights

    @classmethod
    def from_outputs(
        cls,
        outputs: list[ModelOutputs],
        weights: np.ndarray | None = None,
        member_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BatchRunResult":
        if not outputs:
            member_coord = np.asarray(member_ids if member_ids is not None else [], dtype=object)
            return cls(variables={}, coords={"member": member_coord}, dims={}, weights=weights, metadata=metadata or {})

        if member_ids is None:
            member_ids = [f"member_{idx}" for idx in range(len(outputs))]

        datasets = [output.ds.expand_dims(member=[member_ids[idx]]) for idx, output in enumerate(outputs)]
        combined = xr.concat(datasets, dim="member")
        return cls(
            variables={name: np.asarray(combined[name].values) for name in combined.data_vars},
            coords={name: combined.coords[name].values for name in combined.coords},
            dims={name: tuple(combined[name].dims) for name in combined.data_vars},
            weights=weights,
            metadata=metadata or {},
        )

    @property
    def n_members(self) -> int:
        member_coord = self.coords.get("member")
        return 0 if member_coord is None else int(len(member_coord))

    def to_dataset(self) -> xr.Dataset:
        data_vars = {name: (self.dims[name], values) for name, values in self.variables.items()}
        coords = {}
        for name, values in self.coords.items():
            if name in ("member", "time", "y", "x"):
                coords[name] = (name, values)
            elif name in ("latitude", "longitude"):
                if values.ndim == 1:
                    coords[name] = (name, values)
                else:
                    coords[name] = (("y", "x"), values)
            else:
                coords[name] = (name, values) if values.ndim > 0 else values
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def to_model_outputs_list(self) -> list[ModelOutputs]:
        ds = self.to_dataset()
        if "member" not in ds.dims:
            return [ModelOutputs(**{name: ds[name] for name in ds.data_vars})]
        outputs = []
        for member in ds.coords["member"].values:
            subset = ds.sel(member=member, drop=True)
            outputs.append(ModelOutputs(**{name: subset[name] for name in subset.data_vars}))
        return outputs

    def select_member(self, index: int) -> ModelOutputs:
        return self.to_model_outputs_list()[index]


__all__ = [
    "RunContext",
    "PreparedInputs",
    "ParameterBatch",
    "BatchRunResult",
    "ModelInputs",
    "ModelOutputs",
    "BaseParameters",
]


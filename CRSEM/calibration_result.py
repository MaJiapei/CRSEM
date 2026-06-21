from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from CRSEM.contracts import ParameterBatch
from CRSEM.parameters import BaseParameters


@dataclass(slots=True)
class CalibrationResult:
    """Normalized record of calibration candidates and their scores."""

    candidates: np.ndarray
    losses: np.ndarray
    objective_values: np.ndarray
    penalties: list[dict[str, float]]
    metrics: list[dict[str, Any]]
    best_index: int
    param_names: tuple[str, ...]
    model_type: str
    param_cls: type[BaseParameters]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        *,
        records: list[dict[str, Any]],
        param_names: tuple[str, ...],
        model_type: str,
        param_cls: type[BaseParameters],
        metadata: dict[str, Any] | None = None,
    ) -> "CalibrationResult":
        candidates = np.vstack([np.asarray(record["params"], dtype=float) for record in records])
        losses = np.asarray([record["loss"] for record in records], dtype=float)
        objective_values = np.asarray([record["objective_value"] for record in records], dtype=float)
        penalties = [dict(record["penalties"]) for record in records]
        metrics = [record["metrics"] for record in records]
        best_index = int(np.argmin(losses)) if len(losses) else 0
        return cls(
            candidates=candidates,
            losses=losses,
            objective_values=objective_values,
            penalties=penalties,
            metrics=metrics,
            best_index=best_index,
            param_names=tuple(param_names),
            model_type=model_type,
            param_cls=param_cls,
            metadata={} if metadata is None else dict(metadata),
        )

    def best_params_array(self) -> np.ndarray:
        return np.asarray(self.candidates[self.best_index], dtype=float)

    def best_parameter_batch(self) -> ParameterBatch:
        return ParameterBatch(values=self.best_params_array(), param_names=self.param_names)

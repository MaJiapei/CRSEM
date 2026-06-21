from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from CRSEM.calibration_evaluation import MetricsExtractor, ObjectiveEvaluator
from CRSEM.contracts import RunContext
from CRSEM.model import ModelFactory
from CRSEM.preparation import prepare_inputs


@dataclass(slots=True)
class CandidateEvaluation:
    """Normalized payload for a single calibration candidate evaluation."""

    run_output: Any
    evaluation: dict[str, float]
    metrics: dict[str, Any] | None = None

    @property
    def loss(self) -> float:
        return float(self.evaluation["loss"])


class CalibrationModelRunner:
    """Run calibrated models against a cached prepared context."""

    def __init__(self, model_type: str, source, *, output_mode: str = "full") -> None:
        self.model_type = model_type.lower()
        self.context = self._resolve_context(source)
        self.prepared_inputs = prepare_inputs(self.context)
        self.output_mode = output_mode.lower()
        if self.output_mode not in {"full", "compact"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}. Expected 'full' or 'compact'.")
        self.param_names, self.param_bounds = ModelFactory.get_parameter_info(self.model_type)
        self.param_cls = ModelFactory.get_parameter_class(self.model_type)

    def _resolve_context(self, source) -> RunContext:
        if isinstance(source, RunContext):
            return source
        if hasattr(source, "to_run_context"):
            return source.to_run_context()
        raise TypeError(f"Unsupported calibration source: {type(source)}")

    def run(self, params):
        model = ModelFactory.create_model(self.model_type, params)
        outputs = model._run_prepared_hillslope_river_numpy(
            self.prepared_inputs,
            context=self.context,
            output_mode=self.output_mode,
        )
        return SimpleNamespace(**outputs)

    def evaluate(
        self,
        params,
        *,
        objective_evaluator: ObjectiveEvaluator,
        metrics_extractor: MetricsExtractor | None = None,
        ssf_obs,
        include_metrics: bool = False,
    ) -> CandidateEvaluation:
        run_output = self.run(params)
        evaluation = objective_evaluator.evaluate(run_output, ssf_obs)
        metrics = metrics_extractor.extract(run_output, ssf_obs) if include_metrics and metrics_extractor is not None else None
        return CandidateEvaluation(
            run_output=run_output,
            evaluation=evaluation,
            metrics=metrics,
        )

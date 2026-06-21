import numpy as np

from CRSEM.batch_runner import run_parameter_batch
from CRSEM.calibration_evaluation import DiagnosticsExtractor, MetricsExtractor, ObjectiveEvaluator
from CRSEM.calibration_optimizer import OPTIMIZER_REGISTRY, create_optimizer
from CRSEM.calibration_reporting import CalibrationReporter, CalibrationTracker
from CRSEM.calibration_result import CalibrationResult
from CRSEM.calibration_runner import CalibrationModelRunner, CandidateEvaluation
from CRSEM.contracts import BatchRunResult
from CRSEM.driver import BasinDriver
from CRSEM.ensemble_selector import create_selector
from CRSEM.model import ModelOutputs
from CRSEM.result_aggregator import ResultAggregator


class Calibrator:
    """Encapsulates calibration execution while delegating selection to selectors."""

    def __init__(
        self,
        driver: BasinDriver,
        model_type: str,
        plot_progress: bool = True,
        ensemble_para: bool = False,
        plot_every_iters: int = 10,
        plot_min_interval_s: float = 0.5,
        objective_method: str = "nse",
        selector_name: str | None = None,
        selector_kwargs: dict | None = None,
        penalty_settings: dict | None = None,
        calibration_output_mode: str = "full",
        optimizer_kwargs: dict | None = None,
    ):
        self.driver = driver
        self.model_type = model_type.lower()
        self.plot_progress = plot_progress
        self.ensemble_para = ensemble_para
        self.objective_method = objective_method.lower()
        self.plot_every_iters = plot_every_iters
        self.plot_min_interval_s = plot_min_interval_s
        self.selector_name = selector_name or ("aic" if ensemble_para else "best_only")
        self.selector_kwargs = dict(selector_kwargs or {})
        self.penalty_settings = penalty_settings
        self.calibration_output_mode = calibration_output_mode
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.calibration_result = None
        self.selected_batch = None

        self.runner = CalibrationModelRunner(self.model_type, driver, output_mode=self.calibration_output_mode)
        self.context = self.runner.context
        self.model_inputs = driver.model_inputs
        self.SSF_obs = driver.SSF.values
        self.Q = driver.Q.values
        self.s_area = driver.s_area
        self.param_names = self.runner.param_names
        self.param_bounds = self.runner.param_bounds
        self.param_cls = self.runner.param_cls

        self.objective_evaluator = ObjectiveEvaluator(self.objective_method, self.model_type, self.penalty_settings)
        self.metrics_extractor = MetricsExtractor()
        self.diagnostics_extractor = DiagnosticsExtractor()
        self.optimizer_adapter = self._create_optimizer_adapter("differential_evolution")
        self.tracker = CalibrationTracker()
        self.iteration_history = self.tracker.iteration_history
        self.sse_history = self.tracker.sse_history
        self.nse_history = self.tracker.nse_history
        self.loss_history = self.tracker.loss_history
        self.candidate_records = self.tracker.candidate_records
        self.evaluation_records: list[dict] = []
        self._evaluation_record_keys: set[tuple[float, ...]] = set()
        self.reporter = CalibrationReporter(
            model_type=self.model_type,
            plot_progress=self.plot_progress,
            plot_every_iters=self.plot_every_iters,
            plot_min_interval_s=self.plot_min_interval_s,
            diagnostics_provider=self._compute_diagnostics,
        )
        self.visualizer = self.reporter.visualizer
        self._last_candidate_key: tuple[float, ...] | None = None
        self._last_candidate: CandidateEvaluation | None = None

    @staticmethod
    def _candidate_key(params) -> tuple[float, ...]:
        arr = np.asarray(params, dtype=float).reshape(-1)
        return tuple(np.round(arr, 12))

    def _record_evaluated_candidate(self, params, candidate: CandidateEvaluation) -> None:
        key = self._candidate_key(params)
        if key in self._evaluation_record_keys:
            return
        metrics = candidate.metrics
        if metrics is None:
            metrics = self.metrics_extractor.extract(candidate.run_output, self.SSF_obs)
            candidate = CandidateEvaluation(
                run_output=candidate.run_output,
                evaluation=candidate.evaluation,
                metrics=metrics,
            )
            self._last_candidate = candidate
        self._evaluation_record_keys.add(key)
        self.evaluation_records.append(
            {
                "loss": candidate.loss,
                "objective_value": float(candidate.evaluation["objective_value"]),
                "penalties": {
                    name: float(value)
                    for name, value in candidate.evaluation.items()
                    if name.endswith("_penalty")
                },
                "params": np.asarray(params, dtype=float).copy(),
                "metrics": metrics,
            }
        )

    def _create_optimizer_adapter(self, optimizer_name: str):
        optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name.lower()]
        field_names = set(getattr(optimizer_cls, "__dataclass_fields__", {}).keys())
        filtered_kwargs = {
            key: value
            for key, value in self.optimizer_kwargs.items()
            if key in field_names
        }
        return create_optimizer(optimizer_name, **filtered_kwargs)

    def _evaluate_candidate(self, params, *, include_metrics: bool = False):
        key = self._candidate_key(params)
        if key == self._last_candidate_key and self._last_candidate is not None:
            candidate = self._last_candidate
        else:
            candidate = self.runner.evaluate(
                params,
                objective_evaluator=self.objective_evaluator,
                metrics_extractor=self.metrics_extractor,
                ssf_obs=self.SSF_obs,
                include_metrics=True,
            )
            self._last_candidate_key = key
            self._last_candidate = candidate
            self._record_evaluated_candidate(params, candidate)
        payload = {
            "run_output": candidate.run_output,
            "evaluation": candidate.evaluation,
            "loss": candidate.loss,
        }
        if include_metrics and candidate.metrics is not None:
            payload["metrics"] = candidate.metrics
        return payload

    def _objective_function(self, params, verbose=False):
        candidate = self._evaluate_candidate(params)
        evaluation = candidate["evaluation"]
        total_loss = candidate["loss"]

        if verbose and len(self.iteration_history) % 10 == 0:
            print(
                f"Iter {len(self.iteration_history)}: {self.objective_method.upper()}={evaluation['metric_value']:.4f}, "
                f"NSE={evaluation['nse']:.4f}, Chan={evaluation['channel_ratio']:.2%}, "
                f"Std={evaluation['std_ratio']:.2f}, Obj={total_loss:.4f}"
            )
        return total_loss

    def _compute_diagnostics(self, run_output):
        return self.diagnostics_extractor.compute(run_output, self.model_inputs, self.s_area, self.SSF_obs)

    def _update_progress(self, xk, current_loss, convergence_val, run_output):
        self.reporter.report(
            params=xk,
            current_loss=current_loss,
            convergence_val=convergence_val,
            run_output=run_output,
            tracker=self.tracker,
            ssf_obs=self.SSF_obs,
            param_names=self.param_names,
        )

    def _callback(self, xk, convergence=None, context=None):
        if len(self.iteration_history) == 0:
            print("\n" + "=" * 60)
            print("Initialization complete! Starting optimization iterations...")
            print("=" * 60 + "\n")

        candidate = self._evaluate_candidate(xk, include_metrics=True)
        current_loss = float(candidate["loss"])
        self.tracker.record(xk, current_loss, candidate["metrics"], evaluation=candidate["evaluation"])
        self._update_progress(xk, current_loss, convergence, candidate["run_output"])
        return False

    def _build_calibration_result(self, optimizer: str) -> CalibrationResult:
        return CalibrationResult.from_records(
            records=list(self.evaluation_records),
            param_names=tuple(self.param_names),
            model_type=self.model_type,
            param_cls=self.param_cls,
            metadata={
                "n_obs": int(np.sum(~np.isnan(self.SSF_obs.flatten()))),
                "station_name": self.driver.station_name,
                "objective_method": self.objective_method,
                "optimizer": optimizer,
                "n_archived_candidates": int(len(self.evaluation_records)),
            },
        )

    def run_selected_batch(self, source=None, run_method: str = "run_hillslope_river") -> BatchRunResult:
        if self.selected_batch is None:
            raise RuntimeError("No selected parameter batch is available. Run calibration first.")
        source = self.context if source is None else source
        return run_parameter_batch(self.model_type, source, self.selected_batch, run_method=run_method)

    def _evaluate_selected_batch(self) -> tuple[dict, dict]:
        if self.selected_batch is None:
            raise RuntimeError("No selected parameter batch is available. Run calibration first.")

        run_result = self.run_selected_batch()
        if run_result.n_members == 1:
            run_output = run_result.select_member(0)
        else:
            dataset = run_result.to_dataset()
            aggregate_method = "weighted_mean" if run_result.weights is not None else "mean"
            aggregated = ResultAggregator.aggregate(dataset, method=aggregate_method, weights=run_result.weights)
            run_output = ModelOutputs(**{name: aggregated[name] for name in aggregated.data_vars})

        final_metrics = self.metrics_extractor.extract(run_output, self.SSF_obs)
        final_evaluation = self.objective_evaluator.evaluate(run_output, self.SSF_obs)
        return final_metrics, final_evaluation

    def _build_run_metrics(self, optimizer: str, result, final_metrics: dict, final_evaluation: dict) -> dict:
        return {
            "NSE": final_metrics["nse"],
            "KGE": final_metrics["kge"],
            "RMSE": final_metrics["rmse"],
            "MAE": final_metrics["mae"],
            "R2": final_metrics["r2"],
            "SSE": final_metrics["sse"],
            "objective": final_evaluation["loss"],
            "optimizer_best_objective": result.fun,
            "objective_value": final_evaluation["objective_value"],
            "penalties": {
                key: float(value)
                for key, value in final_evaluation.items()
                if key.endswith("_penalty")
            },
            "std_ratio": final_metrics["std_ratio"],
            "channel_ratio": final_metrics["channel_ratio"],
            "success": result.success,
            "optimizer": optimizer,
            "n_archived_candidates": int(len(self.evaluation_records)),
            "station_name": self.driver.station_name,
            "start_year": self.driver.start_year,
            "end_year": self.driver.end_year,
            "model_type": self.model_type,
            "dynamic": str(self.driver.dynamic_nc) if self.driver.dynamic_nc else None,
            "static": str(self.driver.static_nc) if self.driver.static_nc else None,
            "observations": str(self.driver.observations_nc) if self.driver.observations_nc else None,
        }

    def _select_parameter_batch(self) -> dict[str, float | int | str | None] | None:
        selector = create_selector(self.selector_name, **self.selector_kwargs)
        self.selected_batch = selector.select(self.calibration_result)
        if self.selector_name.lower() == "best_only":
            return None
        return getattr(selector, "last_selection_info", None)

    def run(self, optimizer: str = "differential_evolution", maxiter: int = 40):
        self.optimizer_adapter = self._create_optimizer_adapter(optimizer)
        self.reporter.initialize()
        self.reporter.print_run_configuration(
            optimizer=optimizer,
            model_type=self.model_type,
            plot_progress=self.plot_progress,
            param_names=self.param_names,
            param_bounds=self.param_bounds,
        )

        result = self.optimizer_adapter.optimize(
            self._objective_function,
            self.param_bounds,
            callback=self._callback,
            maxiter=maxiter,
        )

        self.reporter.finalize(block=self.plot_progress)

        self.calibration_result = self._build_calibration_result(optimizer)
        selection_info = self._select_parameter_batch()
        final_metrics, final_evaluation = self._evaluate_selected_batch()

        # Get default parameter values for display
        param_defaults = None
        if hasattr(self.param_cls, 'DEFAULT_PARAMS'):
            param_defaults = self.param_cls.DEFAULT_PARAMS

        self.reporter.print_final_summary(
            optimizer=optimizer,
            result=result,
            param_names=self.param_names,
            param_bounds=self.param_bounds,
            param_defaults=param_defaults,
            final_metrics=final_metrics,
        )

        metrics = self._build_run_metrics(optimizer, result, final_metrics, final_evaluation)
        if selection_info is not None:
            metrics["ensemble_info"] = selection_info
        metrics["selected_n_members"] = self.selected_batch.n_members
        return self.selected_batch, metrics

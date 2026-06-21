from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CRSEM.calibration_result import CalibrationResult
from CRSEM.contracts import ParameterBatch


SELECTOR_REGISTRY: dict[str, type] = {}


def register_selector(name: str, selector_cls: type) -> None:
    SELECTOR_REGISTRY[name.lower()] = selector_cls


def create_selector(name: str, **kwargs):
    selector_cls = SELECTOR_REGISTRY.get(name.lower())
    if selector_cls is None:
        raise ValueError(f"Unknown selector: {name}")
    return selector_cls(**kwargs)


def _dedupe_candidates(
    calibration_result: CalibrationResult,
    *,
    round_decimals: int,
) -> tuple[list[np.ndarray], list[dict]]:
    seen: set[tuple[float, ...]] = set()
    unique_candidates: list[np.ndarray] = []
    unique_metrics: list[dict] = []
    for candidate, metrics in zip(calibration_result.candidates, calibration_result.metrics):
        arr = np.asarray(candidate, dtype=float)
        key = tuple(np.round(arr, round_decimals))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(arr)
        unique_metrics.append(metrics)
    return unique_candidates, unique_metrics


@dataclass(slots=True)
class BestOnlySelector:
    """Select the single best candidate from a calibration result."""

    last_selection_info: dict[str, float | int | str | None] | None = None

    def select(self, calibration_result: CalibrationResult) -> ParameterBatch:
        self.last_selection_info = {
            "selection": "best_only",
            "n_candidates": int(len(calibration_result.candidates)),
            "n_selected": 1,
        }
        return calibration_result.best_parameter_batch()


@dataclass(slots=True)
class AICSelector:
    """Select a small weighted ensemble from calibration candidates using AIC."""

    exact_members: int | None = None
    max_members: int = 5
    delta_aic_threshold: float = 4.0
    cumulative_weight_threshold: float = 0.95
    round_decimals: int = 6
    last_selection_info: dict[str, float | int | str | None] | None = None

    def select(self, calibration_result: CalibrationResult) -> ParameterBatch:
        unique_candidates, unique_metrics = _dedupe_candidates(
            calibration_result,
            round_decimals=self.round_decimals,
        )
        n_candidates = len(unique_candidates)
        n_obs = int(calibration_result.metadata.get("n_obs", 0))

        sse_list: list[float] = []
        valid_candidates: list[np.ndarray] = []
        for candidate, metrics in zip(unique_candidates, unique_metrics):
            sse = float(metrics.get("sse", np.nan))
            if np.isfinite(sse) and sse > 0.0:
                sse_list.append(sse)
                valid_candidates.append(candidate)

        if n_obs <= 0 or not valid_candidates:
            self.last_selection_info = {
                "selection": "fallback_best_only",
                "n_candidates": n_candidates,
                "n_valid_candidates": 0,
                "n_selected": 1,
                "min_dAIC": None,
                "max_dAIC_selected": None,
            }
            return calibration_result.best_parameter_batch()

        sse_arr = np.asarray(sse_list, dtype=float)
        n_params = len(calibration_result.param_names)
        aic = 2.0 * n_params + n_obs * np.log(sse_arr / n_obs)
        daic = aic - np.min(aic)
        weights = np.exp(-0.5 * daic)
        weights /= np.sum(weights)

        order = np.argsort(daic)
        weights_sorted = weights[order]
        cumw = np.cumsum(weights_sorted)
        if self.exact_members is not None:
            n_selected = max(1, min(int(self.exact_members), len(order)))
            selection_name = "AIC_exact"
        else:
            n_by_cum = int(np.searchsorted(cumw, self.cumulative_weight_threshold) + 1)
            n_by_cum = min(n_by_cum, len(order))
            n_by_delta = int(np.sum(daic <= self.delta_aic_threshold))
            n_selected = max(1, min(n_by_delta, n_by_cum, self.max_members))
            selection_name = "AIC_topk"
        sel_idx = order[:n_selected]

        selected_arrays = np.vstack([valid_candidates[i] for i in sel_idx])
        selected_weights = weights[sel_idx]
        selected_weights /= np.sum(selected_weights)

        self.last_selection_info = {
            "selection": selection_name,
            "n_candidates": n_candidates,
            "n_valid_candidates": len(valid_candidates),
            "n_selected": n_selected,
            "min_dAIC": float(np.min(daic)),
            "max_dAIC_selected": float(np.max(daic[sel_idx])) if n_selected > 0 else None,
        }
        return ParameterBatch(
            values=selected_arrays,
            param_names=calibration_result.param_names,
            weights=selected_weights,
        )


@dataclass(slots=True)
class GLUESelector:
    """Select behavioral parameter sets from sampled candidates."""

    metric_name: str | None = None
    threshold: float | None = None
    top_fraction: float | None = None
    max_members: int | None = 50
    channel_ratio_lower: float | None = None
    channel_ratio_upper: float | None = None
    round_decimals: int = 6
    last_selection_info: dict[str, float | int | str | None] | None = None

    def select(self, calibration_result: CalibrationResult) -> ParameterBatch:
        unique_candidates, unique_metrics = _dedupe_candidates(
            calibration_result,
            round_decimals=self.round_decimals,
        )
        n_candidates = len(unique_candidates)
        metric_name = (
            self.metric_name
            or calibration_result.metadata.get("objective_method")
            or "nse"
        ).lower()
        higher_is_better = metric_name in {"nse", "kge", "r2"}

        valid_candidates: list[np.ndarray] = []
        valid_scores: list[float] = []
        for candidate, metrics in zip(unique_candidates, unique_metrics):
            score = metrics.get(metric_name, np.nan)
            score = float(score) if score is not None else np.nan
            if not np.isfinite(score):
                continue
            channel_ratio = metrics.get("channel_ratio", np.nan)
            channel_ratio = float(channel_ratio) if channel_ratio is not None else np.nan
            if self.channel_ratio_lower is not None:
                if not np.isfinite(channel_ratio) or channel_ratio < float(self.channel_ratio_lower):
                    continue
            if self.channel_ratio_upper is not None:
                if not np.isfinite(channel_ratio) or channel_ratio > float(self.channel_ratio_upper):
                    continue
            valid_candidates.append(candidate)
            valid_scores.append(score)

        if not valid_candidates:
            self.last_selection_info = {
                "selection": "fallback_best_only",
                "metric_name": metric_name,
                "n_candidates": n_candidates,
                "n_valid_candidates": 0,
                "n_selected": 1,
                "threshold": None,
                "channel_ratio_lower": self.channel_ratio_lower,
                "channel_ratio_upper": self.channel_ratio_upper,
            }
            return calibration_result.best_parameter_batch()

        scores = np.asarray(valid_scores, dtype=float)
        threshold = self.threshold
        selection_name = "GLUE_threshold"

        if threshold is not None:
            selected_order = self._select_by_threshold(scores, higher_is_better, threshold)
        elif self.top_fraction is not None:
            selected_order = self._select_by_top_fraction(scores, higher_is_better, self.top_fraction)
            selection_name = "GLUE_top_fraction"
            threshold = None
        else:
            threshold = self._default_threshold(metric_name)
            if threshold is not None:
                selected_order = self._select_by_threshold(scores, higher_is_better, threshold)
            else:
                top_fraction = 0.1 if self.top_fraction is None else float(self.top_fraction)
                selected_order = self._select_by_top_fraction(scores, higher_is_better, top_fraction)
                selection_name = "GLUE_top_fraction"

        if selected_order.size == 0:
            best_idx = int(np.argmax(scores) if higher_is_better else np.argmin(scores))
            selected_order = np.asarray([best_idx], dtype=int)
            selection_name = "fallback_best_only"

        if self.max_members is not None:
            selected_order = selected_order[: max(1, int(self.max_members))]

        selected_arrays = np.vstack([valid_candidates[idx] for idx in selected_order])
        selected_scores = scores[selected_order]
        selected_weights = self._build_weights(
            selected_scores,
            threshold=threshold if selection_name == "GLUE_threshold" else None,
            higher_is_better=higher_is_better,
        )

        self.last_selection_info = {
            "selection": selection_name,
            "metric_name": metric_name,
            "n_candidates": n_candidates,
            "n_valid_candidates": len(valid_candidates),
            "n_selected": int(len(selected_order)),
            "threshold": None if threshold is None else float(threshold),
            "top_fraction": None if self.top_fraction is None else float(self.top_fraction),
            "channel_ratio_lower": None if self.channel_ratio_lower is None else float(self.channel_ratio_lower),
            "channel_ratio_upper": None if self.channel_ratio_upper is None else float(self.channel_ratio_upper),
            "best_score": float(np.max(scores) if higher_is_better else np.min(scores)),
        }
        return ParameterBatch(
            values=selected_arrays,
            param_names=calibration_result.param_names,
            weights=selected_weights,
        )

    @staticmethod
    def _default_threshold(metric_name: str) -> float | None:
        return {
            "nse": 0.5,
            "kge": 0.5,
            "r2": 0.5,
        }.get(metric_name)

    @staticmethod
    def _select_by_threshold(scores: np.ndarray, higher_is_better: bool, threshold: float) -> np.ndarray:
        if higher_is_better:
            order = np.argsort(-scores)
            mask = scores[order] >= threshold
        else:
            order = np.argsort(scores)
            mask = scores[order] <= threshold
        return order[mask]

    @staticmethod
    def _select_by_top_fraction(scores: np.ndarray, higher_is_better: bool, top_fraction: float) -> np.ndarray:
        frac = min(max(float(top_fraction), 0.0), 1.0)
        n_selected = max(1, int(np.ceil(len(scores) * frac)))
        order = np.argsort(-scores if higher_is_better else scores)
        return order[:n_selected]

    @staticmethod
    def _build_weights(
        scores: np.ndarray,
        *,
        threshold: float | None,
        higher_is_better: bool,
    ) -> np.ndarray:
        if threshold is not None:
            if higher_is_better:
                raw = np.maximum(scores - threshold, 0.0)
            else:
                raw = np.maximum(threshold - scores, 0.0)
        else:
            if higher_is_better:
                raw = scores - np.min(scores)
            else:
                raw = np.max(scores) - scores
        raw = np.asarray(raw, dtype=float)
        raw[~np.isfinite(raw)] = 0.0
        if np.all(raw <= 0.0):
            raw = np.ones_like(scores, dtype=float)
        return raw / np.sum(raw)


register_selector("best_only", BestOnlySelector)
register_selector("aic", AICSelector)
register_selector("glue", GLUESelector)

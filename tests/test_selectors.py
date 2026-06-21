from __future__ import annotations

import unittest

import numpy as np

from CRSEM.calibration_result import CalibrationResult
from CRSEM.ensemble_selector import AICSelector, BestOnlySelector, GLUESelector, create_selector, register_selector
from CRSEM.parameters import CRSEMParameters


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.param_names = tuple(CRSEMParameters.DEFAULT_PARAMS.keys())
        base = CRSEMParameters.from_default().to_array()
        self.candidates = np.vstack([
            base,
            base + 0.01,
            base + 0.02,
        ])
        self.metrics = [
            {"sse": 10.0, "nse": 0.82},
            {"sse": 12.0, "nse": 0.71},
            {"sse": 30.0, "nse": 0.21},
        ]
        self.result = CalibrationResult(
            candidates=self.candidates,
            losses=np.array([10.0, 12.0, 30.0]),
            objective_values=np.array([10.0, 12.0, 30.0]),
            penalties=[{}, {}, {}],
            metrics=self.metrics,
            best_index=0,
            param_names=self.param_names,
            model_type="crsem",
            param_cls=CRSEMParameters,
            metadata={"n_obs": 24, "objective_method": "nse"},
        )

    def test_best_only_selector_returns_single_member_batch(self):
        selector = BestOnlySelector()
        batch = selector.select(self.result)
        self.assertEqual(batch.n_members, 1)
        np.testing.assert_allclose(batch.values[0], self.candidates[0])
        self.assertEqual(selector.last_selection_info["selection"], "best_only")

    def test_aic_selector_returns_weighted_batch(self):
        selector = AICSelector(max_members=2)
        batch = selector.select(self.result)
        info = selector.last_selection_info
        self.assertGreaterEqual(batch.n_members, 1)
        self.assertLessEqual(batch.n_members, 2)
        self.assertEqual(info["selection"], "AIC_topk")
        self.assertIsNotNone(batch.weights)
        np.testing.assert_allclose(np.sum(batch.weights), 1.0)

    def test_aic_selector_supports_exact_member_count(self):
        selector = AICSelector(exact_members=2, max_members=1)
        batch = selector.select(self.result)
        info = selector.last_selection_info
        self.assertEqual(batch.n_members, 2)
        self.assertEqual(info["selection"], "AIC_exact")

    def test_create_selector_uses_registry(self):
        self.assertIsInstance(create_selector("best_only"), BestOnlySelector)
        selector = create_selector(
            "aic",
            max_members=2,
            exact_members=1,
            delta_aic_threshold=6.0,
            cumulative_weight_threshold=0.9,
        )
        self.assertIsInstance(selector, AICSelector)
        self.assertEqual(selector.max_members, 2)
        self.assertEqual(selector.exact_members, 1)
        self.assertEqual(selector.delta_aic_threshold, 6.0)
        self.assertEqual(selector.cumulative_weight_threshold, 0.9)
        self.assertIsInstance(create_selector("glue"), GLUESelector)

    def test_glue_selector_filters_behavioral_candidates(self):
        selector = GLUESelector(threshold=0.7, max_members=5)
        batch = selector.select(self.result)
        info = selector.last_selection_info
        self.assertEqual(batch.n_members, 2)
        self.assertEqual(info["selection"], "GLUE_threshold")
        self.assertEqual(info["metric_name"], "nse")
        np.testing.assert_allclose(np.sum(batch.weights), 1.0)

    def test_glue_selector_can_enforce_channel_ratio_bounds(self):
        self.result.metrics = [
            {"sse": 10.0, "nse": 0.82, "channel_ratio": -0.2},
            {"sse": 12.0, "nse": 0.71, "channel_ratio": -0.9},
            {"sse": 30.0, "nse": 0.68, "channel_ratio": 0.1},
        ]
        selector = GLUESelector(threshold=0.65, channel_ratio_lower=-0.6, channel_ratio_upper=0.3)
        batch = selector.select(self.result)
        info = selector.last_selection_info
        self.assertEqual(batch.n_members, 2)
        self.assertEqual(info["n_valid_candidates"], 2)
        self.assertEqual(info["channel_ratio_lower"], -0.6)
        self.assertEqual(info["channel_ratio_upper"], 0.3)

    def test_glue_selector_prefers_explicit_top_fraction_over_default_threshold(self):
        self.result.metrics = [
            {"sse": 10.0, "nse": 0.42, "channel_ratio": -0.2},
            {"sse": 12.0, "nse": 0.35, "channel_ratio": -0.1},
            {"sse": 30.0, "nse": 0.21, "channel_ratio": 0.1},
        ]
        selector = GLUESelector(top_fraction=0.34, channel_ratio_lower=-0.6, channel_ratio_upper=0.3)
        batch = selector.select(self.result)
        info = selector.last_selection_info
        self.assertEqual(batch.n_members, 2)
        self.assertEqual(info["selection"], "GLUE_top_fraction")

    def test_register_selector_supports_custom_selector(self):
        class DummySelector:
            def __init__(self, value):
                self.value = value

        register_selector("dummy", DummySelector)
        selector = create_selector("dummy", value=3)
        self.assertEqual(selector.value, 3)


if __name__ == "__main__":
    unittest.main()

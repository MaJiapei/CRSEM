from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from CRSEM.model import ModelFactory
from CRSEM.sensitivity import (
    run_oat_sensitivity_analysis,
    validate_point_mode,
    prepare_sensitivity_dataset,
    analyze_sensitivity,
)
from tests.helpers import make_grid_context, make_point_driver


class SensitivityTests(unittest.TestCase):
    def test_run_oat_sensitivity_analysis_returns_baseline_and_params(self):
        param_names, _ = ModelFactory.get_parameter_info("rusle")
        params = np.linspace(0.1, 0.1 * len(param_names), len(param_names), dtype=float)

        def objective(x):
            return float(np.sum(np.asarray(x, dtype=float) ** 2))

        results = run_oat_sensitivity_analysis(params, objective, model_type="rusle", n_samples=4)
        self.assertIn("baseline", results)
        self.assertGreater(len(results), 1)

    def test_run_oat_sensitivity_analysis_rejects_param_length_mismatch(self):
        params = np.array([0.1, 0.2, 0.3], dtype=float)

        def objective(x):
            return float(np.sum(np.asarray(x, dtype=float) ** 2))

        with self.assertRaises(ValueError):
            run_oat_sensitivity_analysis(params, objective, model_type="rusle", n_samples=4)

    def test_validate_point_mode_accepts_point_driver(self):
        driver = make_point_driver()
        validate_point_mode(driver)

    def test_validate_point_mode_rejects_grid_context_like_driver(self):
        class DummyDriver:
            def __init__(self, context):
                self.model_inputs = context.inputs

        with self.assertRaises(ValueError):
            validate_point_mode(DummyDriver(make_grid_context()))

    def test_analyze_sensitivity_reports_partial_and_commonality_stats(self):
        n = 120
        t = np.linspace(-1.0, 1.0, n)
        pre = 0.9 * t + np.linspace(-0.2, 0.2, n)
        ndvi = np.sin(np.linspace(0.0, 6.0 * np.pi, n))
        ssf = 3.0 * ndvi + 0.2 * pre
        ssc = 2.0 * ndvi - 0.1 * t

        df = pd.DataFrame(
            {
                "T": t,
                "Pre": pre,
                "NDVI": ndvi,
                "SSF": ssf,
                "SSC": ssc,
            },
            index=pd.date_range("2000-01-31", periods=n, freq="ME"),
        )

        results = analyze_sensitivity(df, min_samples=24)

        self.assertIn("feature_collinearity", results)
        self.assertIn("vif", results["feature_collinearity"])
        self.assertGreater(results["feature_collinearity"]["vif"]["T"], 1.0)

        ssf_target = results["targets"]["SSF"]
        self.assertIn("partial_corr", ssf_target)
        self.assertIn("unique_r2", ssf_target)
        self.assertIn("commonality", ssf_target)
        self.assertIn("residualized_contribution", ssf_target)
        self.assertIn("elasticity_model", ssf_target)
        self.assertGreater(ssf_target["partial_corr"]["NDVI"], 0.9)
        self.assertGreater(ssf_target["unique_r2"]["NDVI"], 0.7)
        self.assertIn("T+Pre+NDVI", ssf_target["commonality"]["commonality_raw"])
        ndvi_resid = ssf_target["residualized_contribution"]["NDVI_given_T_Pre"]
        self.assertGreater(ndvi_resid["partial_corr"], 0.9)
        self.assertGreater(ndvi_resid["incremental_r2_on_residualized_target"], 0.7)
        elasticity = ssf_target["elasticity_model"]
        self.assertIn("coefficients", elasticity)
        self.assertGreater(elasticity["coefficients"]["NDVI_log"], 0.0)
        self.assertGreater(elasticity["residualized_ndvi_elasticity"]["beta"], 0.0)


if __name__ == "__main__":
    unittest.main()

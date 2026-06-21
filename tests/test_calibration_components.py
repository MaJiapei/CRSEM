from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import xarray as xr

from CRSEM.calibration_evaluation import (
    AnnualRFactorPenalty,
    ChannelRatioPenalty,
    DiagnosticsExtractor,
    KGEPBIASObjective,
    KGEObjective,
    MetricsExtractor,
    NSEPBIASObjective,
    NSEObjective,
    ObjectiveEvaluator,
    RMSEObjective,
    create_objective,
    create_penalties,
    register_model_penalties,
    register_objective,
    register_penalty,
)
from CRSEM.calibration_optimizer import DifferentialEvolutionOptimizer, SamplingOptimizer
from CRSEM.calibration_result import CalibrationResult
from CRSEM.calibration_runner import CalibrationModelRunner
from CRSEM.calibrator import Calibrator
from scripts.calibrate_parameters import build_parser
from scripts import calibrate_parameters as calibrate_parameters_script
from scripts import run_model as run_model_script
from scripts.run_model import build_parser as build_run_parser, infer_model_type, resolve_output_file_arg
from CRSEM.parameters import CRSEMParameters
from CRSEM.contracts import BatchRunResult, ParameterBatch
from CRSEM.driver import BasinDriver
from CRSEM.model import ModelInputs
from tests.helpers import make_grid_context, make_point_context, make_point_driver


class CalibrationComponentTests(unittest.TestCase):
    def setUp(self):
        self.context = make_point_context()
        self.ssf_obs = self.context.ssf_obs.values
        self.run_output = types.SimpleNamespace(
            SSF_pred=self.ssf_obs * 1.05,
            A_channel=np.full_like(self.ssf_obs, 0.5),
            E_hillslope_rain=np.full_like(self.ssf_obs, 1.0),
            E_hillslope_melt=np.full_like(self.ssf_obs, 0.3),
            E_hillslope=np.full_like(self.ssf_obs, 1.3),
            R_rain=np.full_like(self.ssf_obs, 12.0),
            R_melt=np.full_like(self.ssf_obs, 4.0),
            K_factor=np.full_like(self.ssf_obs, 0.03),
            C_factor=np.full_like(self.ssf_obs, 0.5),
            SDR=np.full_like(self.ssf_obs, 0.4),
        )

    def test_objective_classes_return_finite_values(self):
        valid_obs = self.ssf_obs[~np.isnan(self.ssf_obs)]
        valid_pred = np.asarray(self.run_output.SSF_pred)[~np.isnan(self.ssf_obs)]
        std_obs = float(np.std(valid_obs)) or 1.0
        for objective in (NSEObjective(), NSEPBIASObjective(), KGEObjective(), KGEPBIASObjective(), RMSEObjective()):
            metric_value, objective_value = objective.evaluate(valid_pred, valid_obs, std_obs)
            self.assertTrue(np.isfinite(metric_value))
            self.assertTrue(np.isfinite(objective_value))

    def test_nse_pbias_objective_penalizes_bias_with_same_nse_shape_metric(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        low_bias = np.array([1.1, 2.1, 3.1, 4.1], dtype=float)
        high_bias = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)
        objective = NSEPBIASObjective()

        low_metric, low_loss = objective.evaluate(low_bias, obs, float(np.std(obs)))
        high_metric, high_loss = objective.evaluate(high_bias, obs, float(np.std(obs)))

        self.assertGreater(low_metric, high_metric)
        self.assertLess(low_loss, high_loss)

    def test_create_objective_supports_nse_pbias(self):
        objective = create_objective("nse_pbias")
        self.assertIsInstance(objective, NSEPBIASObjective)

    def test_kge_pbias_objective_penalizes_bias_with_same_kge_shape_metric(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        low_bias = np.array([1.1, 2.1, 3.1, 4.1], dtype=float)
        high_bias = np.array([1.5, 2.5, 3.5, 4.5], dtype=float)
        objective = KGEPBIASObjective()

        low_metric, low_loss = objective.evaluate(low_bias, obs, float(np.std(obs)))
        high_metric, high_loss = objective.evaluate(high_bias, obs, float(np.std(obs)))

        self.assertGreater(low_metric, high_metric)
        self.assertLess(low_loss, high_loss)

    def test_create_objective_supports_kge_pbias(self):
        objective = create_objective("kge_pbias")
        self.assertIsInstance(objective, KGEPBIASObjective)

    def test_penalties_return_named_payload(self):
        channel_penalty = ChannelRatioPenalty().evaluate(self.run_output, np.asarray(self.run_output.SSF_pred))
        annual_penalty = AnnualRFactorPenalty().evaluate(self.run_output, np.asarray(self.run_output.SSF_pred))
        self.assertIn("channel_penalty", channel_penalty)
        self.assertIn("rain_penalty", annual_penalty)
        self.assertAlmostEqual(channel_penalty["channel_ratio"], 0.5 / 1.3)

    def test_annual_r_factor_penalty_handles_gridded_time_series(self):
        run_output = types.SimpleNamespace(
            R_rain=np.ones((24, 2, 3), dtype=float) * 10.0,
            R_melt=np.ones((24, 2, 3), dtype=float) * 2.0,
        )
        penalty = AnnualRFactorPenalty().evaluate(run_output, np.ones(24, dtype=float))
        self.assertIn("rain_penalty", penalty)
        self.assertTrue(np.isfinite(penalty["rain_penalty"]))

    def test_objective_evaluator_returns_loss_payload(self):
        evaluator = ObjectiveEvaluator("nse", "crsem")
        result = evaluator.evaluate(self.run_output, self.ssf_obs)
        self.assertIn("loss", result)
        self.assertIn("objective_value", result)
        self.assertIn("channel_penalty", result)
        self.assertIn("rain_penalty", result)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_metrics_extractor_returns_summary_metrics(self):
        extractor = MetricsExtractor()
        metrics = extractor.extract(self.run_output, self.ssf_obs)
        self.assertIn("nse", metrics)
        self.assertIn("kge", metrics)
        self.assertEqual(metrics["SSF_pred"].shape, self.ssf_obs.shape)
        self.assertAlmostEqual(metrics["channel_ratio"], 0.5 / 1.3)

    def test_channel_ratio_falls_back_to_component_sum_when_total_hillslope_missing(self):
        run_output = types.SimpleNamespace(
            SSF_pred=self.ssf_obs * 1.05,
            A_channel=np.full_like(self.ssf_obs, -0.26),
            E_hillslope_rain=np.full_like(self.ssf_obs, 1.0),
            E_hillslope_melt=np.full_like(self.ssf_obs, 0.3),
            R_rain=np.full_like(self.ssf_obs, 12.0),
            R_melt=np.full_like(self.ssf_obs, 4.0),
        )

        channel_penalty = ChannelRatioPenalty().evaluate(run_output, np.asarray(run_output.SSF_pred))
        metrics = MetricsExtractor().extract(run_output, self.ssf_obs)

        self.assertAlmostEqual(channel_penalty["channel_ratio"], -0.2)
        self.assertAlmostEqual(metrics["channel_ratio"], -0.2)

    def test_diagnostics_extractor_returns_expected_keys(self):
        extractor = DiagnosticsExtractor()
        diagnostics = extractor.compute(self.run_output, self.context.inputs, self.context.s_area, self.ssf_obs)
        self.assertIn("R", diagnostics)
        self.assertIn("SSF_pred_modulus", diagnostics)
        self.assertIn("E_rain_modulus", diagnostics)

    def test_diagnostics_extractor_raises_for_missing_required_output(self):
        extractor = DiagnosticsExtractor()
        bad_output = types.SimpleNamespace(SSF_pred=self.run_output.SSF_pred)
        with self.assertRaises(ValueError):
            extractor.compute(bad_output, self.context.inputs, self.context.s_area, self.ssf_obs)

    def test_optimizer_adapter_accepts_explicit_runtime_configuration(self):
        optimizer = DifferentialEvolutionOptimizer(popsize=5, polish=True, workers=2)
        self.assertEqual(optimizer.popsize, 5)
        self.assertTrue(optimizer.polish)
        self.assertEqual(optimizer.workers, 2)

    def test_sampling_optimizer_accepts_explicit_runtime_configuration(self):
        optimizer = SamplingOptimizer(sampler="lhs", n_samples=12, callback_interval=3)
        self.assertEqual(optimizer.sampler, "lhs")
        self.assertEqual(optimizer.n_samples, 12)
        self.assertEqual(optimizer.callback_interval, 3)

    def test_calibration_model_runner_returns_candidate_evaluation(self):
        """Test that CalibrationModelRunner evaluates candidates and returns finite loss for valid parameters."""
        runner = CalibrationModelRunner("crsem", self.context)
        self.assertIs(runner.param_cls, CRSEMParameters)

        # Use parameters within reasonable bounds to get finite loss
        # These values are chosen to produce valid model output
        params = np.array([
            0.5,   # a_rain (bound: 0.5-1.0)
            10.0,  # r_th (bound: 1-20)
            0.1,   # a_melt (bound: 0.1-1)
            0.0,   # m_th (bound: 0-10)
            3.0,   # k_melt (bound: 1-5)
            0.35,  # alpha_K (bound: 0.1-0.8)
            0.7,   # K_min_r (bound: 0.4-1.0)
            1.5,   # K_max_r (bound: 1.0-2)
            2.0,   # alpha_C (bound: 1-5)
            0.5,   # ic0 (bound: 0.1-1.0)
            2.5,   # k (bound: 0.5-4)
            0.5,   # beta_sdr (bound: 0.3-1.0)
            5.0,   # c_base (bound: 0.1-20)
            1.5,   # n_chan (bound: 1-2)
            0.5,   # K_chan (bound: 0.1-1)
        ], dtype=float)

        candidate = runner.evaluate(
            params,
            objective_evaluator=ObjectiveEvaluator("nse", "crsem"),
            metrics_extractor=MetricsExtractor(),
            ssf_obs=self.ssf_obs,
            include_metrics=True,
        )
        self.assertIn("loss", candidate.evaluation)
        self.assertIsNotNone(candidate.metrics)
        self.assertTrue(np.isfinite(candidate.loss))

    def test_calibration_model_runner_rejects_unsupported_source(self):
        with self.assertRaises(TypeError):
            CalibrationModelRunner("crsem", object())

    def test_calibration_model_runner_compact_mode_keeps_core_and_drops_secondary_diagnostics(self):
        runner = CalibrationModelRunner("crsem", make_grid_context(), output_mode="compact")
        run_output = runner.run(CRSEMParameters.from_default().to_array())
        self.assertTrue(hasattr(run_output, "SSF_pred"))
        self.assertTrue(hasattr(run_output, "A_channel"))
        self.assertTrue(hasattr(run_output, "R_rain"))
        self.assertTrue(hasattr(run_output, "R_melt"))
        self.assertIsNone(getattr(run_output, "K_factor", None))
        self.assertIsNone(getattr(run_output, "C_factor", None))
        self.assertIsNone(getattr(run_output, "SDR", None))

    def test_calibration_model_runner_compact_mode_works_for_point_mode(self):
        runner = CalibrationModelRunner("crsem", make_point_context(), output_mode="compact")
        run_output = runner.run(CRSEMParameters.from_default().to_array())
        self.assertTrue(hasattr(run_output, "SSF_pred"))
        self.assertTrue(hasattr(run_output, "A_channel"))
        self.assertTrue(hasattr(run_output, "R_rain"))
        self.assertTrue(hasattr(run_output, "R_melt"))
        self.assertIsNone(getattr(run_output, "K_factor", None))
        self.assertIsNone(getattr(run_output, "C_factor", None))
        self.assertIsNone(getattr(run_output, "SDR", None))

    def test_extractors_adapt_to_rusle_like_outputs(self):
        run_output = types.SimpleNamespace(
            SSF_pred=self.ssf_obs * 0.95,
            A_channel=np.full_like(self.ssf_obs, 0.2),
            E_hillslope=np.full_like(self.ssf_obs, 1.1),
            R_factor=np.full_like(self.ssf_obs, 8.0),
            K_factor=np.full_like(self.ssf_obs, 0.02),
            C_factor=np.full_like(self.ssf_obs, 0.4),
            SDR=np.full_like(self.ssf_obs, 0.3),
        )
        metrics = MetricsExtractor().extract(run_output, self.ssf_obs)
        diagnostics = DiagnosticsExtractor().compute(run_output, self.context.inputs, self.context.s_area, self.ssf_obs)
        self.assertEqual(metrics["E_rain"].shape, self.ssf_obs.shape)
        self.assertTrue(np.allclose(metrics["E_melt"], 0.0))
        self.assertIn("E_rain_modulus", diagnostics)
        self.assertEqual(diagnostics["E_melt_modulus"], 0.0)

    def test_calibrator_evaluates_selected_batch_with_weighted_ensemble_metrics(self):
        calibrator = Calibrator(
            driver=make_point_driver(),
            model_type="crsem",
            plot_progress=False,
            ensemble_para=True,
            selector_name="glue",
            calibration_output_mode="compact",
        )
        n_time = self.ssf_obs.size
        weights = np.array([0.75, 0.25], dtype=float)
        calibrator.selected_batch = ParameterBatch(
            values=np.vstack([
                np.zeros(len(calibrator.param_names), dtype=float),
                np.ones(len(calibrator.param_names), dtype=float),
            ]),
            param_names=tuple(calibrator.param_names),
            weights=weights,
        )

        member0_ssf = self.ssf_obs.copy()
        member1_ssf = self.ssf_obs * 2.0
        member0_channel = np.full(n_time, -1.0, dtype=float)
        member1_channel = np.full(n_time, -2.0, dtype=float)
        member0_hillslope = np.full(n_time, 10.0, dtype=float)
        member1_hillslope = np.full(n_time, 20.0, dtype=float)
        zeros = np.zeros((2, n_time), dtype=float)

        run_result = BatchRunResult(
            variables={
                "SSF_pred": np.vstack([member0_ssf, member1_ssf]),
                "A_channel": np.vstack([member0_channel, member1_channel]),
                "E_hillslope": np.vstack([member0_hillslope, member1_hillslope]),
                "R_rain": zeros,
                "R_melt": zeros,
            },
            coords={"member": np.array(["member_0", "member_1"], dtype=object), "time": np.arange(n_time)},
            dims={
                "SSF_pred": ("member", "time"),
                "A_channel": ("member", "time"),
                "E_hillslope": ("member", "time"),
                "R_rain": ("member", "time"),
                "R_melt": ("member", "time"),
            },
            weights=weights,
        )
        calibrator.run_selected_batch = MagicMock(return_value=run_result)

        metrics, evaluation = calibrator._evaluate_selected_batch()
        expected_ssf = member0_ssf * weights[0] + member1_ssf * weights[1]
        expected_nse = MetricsExtractor().extract(
            types.SimpleNamespace(
                SSF_pred=expected_ssf,
                A_channel=member0_channel * weights[0] + member1_channel * weights[1],
                E_hillslope=member0_hillslope * weights[0] + member1_hillslope * weights[1],
                R_rain=np.zeros(n_time, dtype=float),
                R_melt=np.zeros(n_time, dtype=float),
            ),
            self.ssf_obs,
        )["nse"]

        self.assertAlmostEqual(metrics["nse"], expected_nse)
        self.assertAlmostEqual(metrics["channel_ratio"], -0.1)
        self.assertAlmostEqual(evaluation["channel_ratio"], -0.1)

    def test_penalty_registry_supports_model_registration(self):
        class ConstantPenalty:
            name = "constant_penalty"

            def evaluate(self, run_output, ssf_pred):
                return {self.name: 1.0}

        register_penalty("constant_penalty", ConstantPenalty)
        register_model_penalties("test_model", ["channel_ratio", "constant_penalty"])
        penalties = create_penalties("test_model")
        self.assertEqual([penalty.name for penalty in penalties], ["channel_penalty", "constant_penalty"])

    def test_objective_registry_supports_custom_registration(self):
        class ConstantObjective:
            name = "constant"

            def evaluate(self, ssf_pred_valid, ssf_obs_valid, std_obs):
                return 0.5, 0.25

        register_objective("constant", ConstantObjective)
        objective = create_objective("constant")
        self.assertIsInstance(objective, ConstantObjective)
        metric_value, objective_value = objective.evaluate(self.ssf_obs, self.ssf_obs, 1.0)
        self.assertEqual((metric_value, objective_value), (0.5, 0.25))

    def test_calibration_result_from_records_selects_best_candidate(self):
        result = CalibrationResult.from_records(
            records=[
                {
                    "params": np.array([2.0, 3.0]),
                    "loss": 1.5,
                    "objective_value": 1.2,
                    "penalties": {"channel_penalty": 0.3},
                    "metrics": {"nse": 0.4},
                },
                {
                    "params": np.array([4.0, 5.0]),
                    "loss": 0.8,
                    "objective_value": 0.7,
                    "penalties": {"channel_penalty": 0.1},
                    "metrics": {"nse": 0.7},
                },
            ],
            param_names=("a", "b"),
            model_type="crsem",
            param_cls=CRSEMParameters,
        )
        self.assertEqual(result.best_index, 1)
        np.testing.assert_allclose(result.best_params_array(), np.array([4.0, 5.0]))

    def test_calibrator_passes_optimizer_kwargs_to_adapter(self):
        calibrator = Calibrator(
            driver=make_point_driver(),
            model_type="crsem",
            plot_progress=False,
            optimizer_kwargs={"popsize": 4, "polish": True},
        )
        self.assertEqual(calibrator.optimizer_adapter.popsize, 4)
        self.assertTrue(calibrator.optimizer_adapter.polish)

    def test_calibrator_reuses_cached_candidate_for_repeated_same_params(self):
        calibrator = Calibrator(driver=make_point_driver(), model_type="crsem", plot_progress=False)
        params = CRSEMParameters.from_default().to_array()

        original_evaluate = calibrator.runner.evaluate
        calibrator.runner.evaluate = MagicMock(side_effect=original_evaluate)

        first = calibrator._evaluate_candidate(params, include_metrics=False)
        second = calibrator._evaluate_candidate(params, include_metrics=True)

        self.assertEqual(calibrator.runner.evaluate.call_count, 1)
        self.assertIn("metrics", second)
        self.assertEqual(first["loss"], second["loss"])

    def test_calibrator_archives_unique_candidates_outside_progress_callback(self):
        calibrator = Calibrator(driver=make_point_driver(), model_type="crsem", plot_progress=False)
        params = CRSEMParameters.from_default().to_array()

        calibrator._evaluate_candidate(params)
        calibrator._evaluate_candidate(params)
        calibrator._evaluate_candidate(params * 1.01)

        self.assertEqual(len(calibrator.evaluation_records), 2)

    def test_calibration_cli_parser_preserves_runtime_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--run-mode", "gridded",
            ]
        )
        self.assertEqual(args.run_mode, "gridded")
        self.assertFalse(args.plot_progress)
        self.assertFalse(args.polish)
        self.assertIsNone(args.save)
        self.assertIsNone(args.selector)

    def test_calibration_cli_parser_uses_default_save_target_when_requested_without_path(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--save",
            ]
        )
        self.assertEqual(args.save, "__DEFAULT__")

    def test_calibration_cli_parser_accepts_explicit_save_path(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--save", "custom/params.json",
            ]
        )
        self.assertEqual(args.save, "custom/params.json")

    def test_calibration_plot_is_not_allowed_for_gridded_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--run-mode", "gridded",
                "--plot-progress",
            ]
        )
        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files"):
                with self.assertRaisesRegex(ValueError, "Grid-mode calibration does not support --plot-progress"):
                    with patch("argparse.ArgumentParser.parse_args", return_value=args):
                        calibrate_parameters_script.main()

    def test_calibration_point_mode_rejects_workers(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--workers", "3",
            ]
        )
        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files"):
                with self.assertRaisesRegex(ValueError, "Point-mode calibration does not accept --workers"):
                    with patch("argparse.ArgumentParser.parse_args", return_value=args):
                        calibrate_parameters_script.main()

    def test_calibration_cli_passes_workers_to_optimizer_kwargs_for_gridded_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--run-mode", "gridded",
                "--workers", "3",
            ]
        )
        driver = MagicMock()
        dummy_batch = MagicMock()
        dummy_metrics = {
            "NSE": 0.5,
            "KGE": 0.4,
            "R2": 0.3,
            "RMSE": 1.2,
            "MAE": 0.8,
            "static": "static.nc",
            "dynamic": "dynamic.nc",
            "observations": "observations.nc",
            "selected_n_members": 1,
        }

        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files", return_value=driver):
                with patch("scripts.calibrate_parameters.refine_parameters", return_value=(dummy_batch, dummy_metrics)) as mock_refine:
                    with patch("argparse.ArgumentParser.parse_args", return_value=args):
                        calibrate_parameters_script.main()

        self.assertEqual(mock_refine.call_args.kwargs["optimizer_kwargs"]["workers"], 3)

    def test_calibration_cli_passes_sampling_kwargs_for_glue_optimizer(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--optimizer", "glue",
                "--sampling-method", "lhs",
                "--n-samples", "24",
                "--seed", "123",
                "--glue-channel-ratio-lower", "-0.6",
                "--glue-channel-ratio-upper", "0.3",
            ]
        )
        driver = MagicMock()
        point_driver = MagicMock()
        driver.to_point_driver.return_value = point_driver
        dummy_batch = MagicMock()
        dummy_metrics = {
            "NSE": 0.5,
            "KGE": 0.4,
            "R2": 0.3,
            "RMSE": 1.2,
            "MAE": 0.8,
            "static": "static.nc",
            "dynamic": "dynamic.nc",
            "observations": "observations.nc",
            "selected_n_members": 1,
        }

        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files", return_value=driver):
                with patch("scripts.calibrate_parameters.refine_parameters", return_value=(dummy_batch, dummy_metrics)) as mock_refine:
                    with patch("argparse.ArgumentParser.parse_args", return_value=args):
                        calibrate_parameters_script.main()

        self.assertEqual(mock_refine.call_args.kwargs["optimizer_kwargs"]["sampler"], "lhs")
        self.assertEqual(mock_refine.call_args.kwargs["optimizer_kwargs"]["n_samples"], 24)
        self.assertEqual(mock_refine.call_args.kwargs["optimizer_kwargs"]["seed"], 123)
        self.assertEqual(mock_refine.call_args.kwargs["selector_name"], "glue")
        self.assertEqual(mock_refine.call_args.kwargs["selector_kwargs"]["channel_ratio_lower"], -0.6)
        self.assertEqual(mock_refine.call_args.kwargs["selector_kwargs"]["channel_ratio_upper"], 0.3)

    def test_calibration_cli_passes_explicit_save_path_to_reporting_api(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--save", "custom/params.json",
            ]
        )
        driver = MagicMock()
        point_driver = MagicMock()
        driver.to_point_driver.return_value = point_driver
        dummy_batch = MagicMock()
        dummy_metrics = {
            "NSE": 0.5,
            "KGE": 0.4,
            "R2": 0.3,
            "RMSE": 1.2,
            "MAE": 0.8,
            "static": "static.nc",
            "dynamic": "dynamic.nc",
            "observations": "observations.nc",
            "selected_n_members": 1,
        }

        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files", return_value=driver):
                with patch("scripts.calibrate_parameters.refine_parameters", return_value=(dummy_batch, dummy_metrics)):
                    with patch("scripts.calibrate_parameters.save_calibration_results") as mock_save:
                        with patch("argparse.ArgumentParser.parse_args", return_value=args):
                            calibrate_parameters_script.main()

        self.assertEqual(mock_save.call_args.kwargs["save_path"], Path("custom/params.json"))

    def test_run_model_parser_defaults_to_gridded_mode(self):
        parser = build_run_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--params-file", "params.json",
            ]
        )
        self.assertEqual(args.run_mode, "gridded")
        self.assertIsNone(args.output_file)

    def test_run_model_parser_uses_default_output_file_when_requested_without_path(self):
        parser = build_run_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--params-file", "params.json",
                "--output-file",
            ]
        )
        self.assertEqual(args.output_file, "__DEFAULT__")

    def test_run_model_parser_accepts_explicit_output_file_path(self):
        parser = build_run_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--params-file", "params.json",
                "--output-file", "custom/model_output.nc",
            ]
        )
        self.assertEqual(args.output_file, "custom/model_output.nc")

    def test_run_model_resolves_default_output_file_to_static_directory(self):
        output_file = resolve_output_file_arg("__DEFAULT__", static_nc=Path("example/drivers/static.nc"))
        self.assertEqual(output_file, Path("example/drivers/model_output.nc"))

    def test_run_model_resolves_directory_output_file_to_default_filename(self):
        output_file = resolve_output_file_arg("example/tuotuohe_1990_2000", static_nc=Path("example/drivers/static.nc"))
        self.assertEqual(output_file, Path("example/tuotuohe_1990_2000/model_output.nc"))

    def test_run_model_resolves_explicit_nc_output_file_unchanged(self):
        output_file = resolve_output_file_arg("custom/model_output.nc", static_nc=Path("example/drivers/static.nc"))
        self.assertEqual(output_file, Path("custom/model_output.nc"))

    def test_run_model_output_write_uses_netcdf_compression(self):
        dataset = xr.Dataset({"SSF_pred": (("time",), np.array([1.0, 2.0], dtype=np.float32))}, coords={"time": [0, 1]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model_output.nc"
            dataset.to_netcdf(
                output_path,
                engine="netcdf4",
                encoding=run_model_script.build_netcdf_compression_encoding(dataset),
            )
            import netCDF4

            with netCDF4.Dataset(output_path, mode="r") as nc:
                self.assertTrue(nc.variables["SSF_pred"].filters()["zlib"])

    def test_run_model_infers_model_type_from_metrics(self):
        batch = ParameterBatch(values=np.array([1.0] * 15), param_names=tuple(CRSEMParameters.DEFAULT_PARAMS.keys()))
        model_type = infer_model_type(batch, {"model_type": "crsem"})
        self.assertEqual(model_type, "crsem")

    def test_run_model_infers_model_type_from_param_names(self):
        batch = ParameterBatch(values=np.array([1.0] * 15), param_names=tuple(CRSEMParameters.DEFAULT_PARAMS.keys()))
        model_type = infer_model_type(batch, None)
        self.assertEqual(model_type, "crsem")

    def test_driver_collapse_ndvi_members_reduces_member_dimension(self):
        time = xr.date_range("2000-01-01", periods=3, freq="MS")
        ndvi = xr.DataArray(
            np.array([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]], dtype=np.float32),
            dims=("member", "time"),
            coords={"member": ["a", "b"], "time": time},
        )
        driver = BasinDriver(
            station_name="test",
            start_year=2000,
            end_year=2000,
            model_inputs=ModelInputs(
                T=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
                Pre=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
                NDVI=ndvi,
                LS=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
                P_f=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
                IC=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
                K=xr.DataArray(np.ones(3, dtype=np.float32), dims=("time",), coords={"time": time}),
            ),
        )

        collapsed = driver.collapse_ndvi_members()

        self.assertEqual(collapsed.model_inputs.NDVI.dims, ("time",))
        np.testing.assert_allclose(collapsed.model_inputs.NDVI.values, np.array([0.3, 0.5, 0.7], dtype=np.float32))

    def test_calibration_cli_collapses_ndvi_members_before_refine(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--run-mode", "gridded",
            ]
        )
        driver = MagicMock()
        driver.start_year = 1990
        driver.end_year = 2000
        driver.model_inputs.NDVI = MagicMock(dims=("member", "time"))
        driver.crop_time_range.return_value = driver
        collapsed_driver = MagicMock()
        collapsed_driver.model_inputs.NDVI = MagicMock(dims=("time",))
        driver.collapse_ndvi_members.return_value = collapsed_driver
        dummy_batch = MagicMock()
        dummy_metrics = {"selected_n_members": 1}

        with patch("scripts.calibrate_parameters.build_parser", return_value=parser):
            with patch("scripts.calibrate_parameters.BasinDriver.from_nc_files", return_value=driver):
                with patch("scripts.calibrate_parameters.refine_parameters", return_value=(dummy_batch, dummy_metrics)) as mock_refine:
                    with patch("argparse.ArgumentParser.parse_args", return_value=args):
                        calibrate_parameters_script.main()

        driver.collapse_ndvi_members.assert_called_once_with()
        self.assertIs(mock_refine.call_args.args[0], collapsed_driver)

    def test_run_model_cli_collapses_ndvi_members_before_execution(self):
        parser = build_run_parser()
        args = parser.parse_args(
            [
                "--static-nc", "static.nc",
                "--dynamic-nc", "dynamic.nc",
                "--observations-nc", "observations.nc",
                "--params-file", "params.json",
            ]
        )
        driver = MagicMock()
        driver.start_year = 1990
        driver.end_year = 2000
        driver.model_inputs.NDVI = MagicMock(dims=("member", "time"))
        collapsed_driver = MagicMock()
        collapsed_driver.model_inputs.NDVI = MagicMock(dims=("time",))
        driver.collapse_ndvi_members.return_value = collapsed_driver
        parameter_batch = ParameterBatch(values=np.array([1.0] * 15), param_names=tuple(CRSEMParameters.DEFAULT_PARAMS.keys()))
        result = MagicMock()
        result.to_dataset.return_value = xr.Dataset({"SSF_pred": (("member", "time"), np.ones((1, 2), dtype=np.float32))}, coords={"member": ["m0"], "time": [0, 1]})
        result.weights = None

        with patch("scripts.run_model.build_parser", return_value=parser):
            with patch("scripts.run_model.ParameterBatch.from_file", return_value=(parameter_batch, {"model_type": "crsem"})):
                with patch("scripts.run_model.BasinDriver.from_nc_files", return_value=driver):
                    with patch("scripts.run_model.run_parameter_batch", return_value=result) as mock_run:
                        with patch("argparse.ArgumentParser.parse_args", return_value=args):
                            run_model_script.main()

        driver.collapse_ndvi_members.assert_called_once_with()
        self.assertIs(mock_run.call_args.args[1], collapsed_driver)


if __name__ == "__main__":
    unittest.main()

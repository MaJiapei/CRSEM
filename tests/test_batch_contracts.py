from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

import CRSEM._model_base as model_base_module
from CRSEM.batch_runner import run_parameter_batch
from CRSEM.contracts import BatchRunResult, ParameterBatch
from CRSEM.model import ModelFactory, ModelOutputs
from CRSEM.parameters import CRSEMParameters
from CRSEM.result_aggregator import ResultAggregator
from tests.helpers import make_grid_context, make_parameter_batch, make_point_context


class ParameterBatchTests(unittest.TestCase):
    def test_single_member_batch_is_normalized_to_2d(self):
        batch = ParameterBatch(values=np.array([1.0, 2.0, 3.0]), param_names=("a", "b", "c"))
        self.assertEqual(batch.values.shape, (1, 3))
        self.assertEqual(batch.n_members, 1)
        self.assertEqual(batch.member_ids, ["member_0"])

    def test_weights_are_normalized(self):
        batch = ParameterBatch(
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            param_names=("a", "b"),
            weights=np.array([2.0, 6.0]),
        )
        np.testing.assert_allclose(batch.weights, np.array([0.25, 0.75]))

    def test_parameter_batch_round_trips_through_file(self):
        batch = ParameterBatch(
            values=np.array([[1.0, 2.0], [3.0, 4.0]]),
            param_names=("a", "b"),
            weights=np.array([2.0, 6.0]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}\\batch.json"
            batch.to_file(path, metrics={"score": 1.23})
            restored, metrics = ParameterBatch.from_file(path)
        np.testing.assert_allclose(restored.values, batch.values)
        np.testing.assert_allclose(restored.weights, batch.weights)
        self.assertEqual(restored.param_names, batch.param_names)
        self.assertEqual(metrics["score"], 1.23)

    def test_from_any_accepts_parameter_object_and_array(self):
        params = CRSEMParameters.from_default()
        param_names = tuple(CRSEMParameters.DEFAULT_PARAMS.keys())
        batch_from_obj = ParameterBatch.from_any(params, param_names)
        batch_from_arr = ParameterBatch.from_any(params.to_array(), param_names)
        self.assertEqual(batch_from_obj.n_members, 1)
        self.assertEqual(batch_from_arr.n_members, 1)
        np.testing.assert_allclose(batch_from_obj.values, batch_from_arr.values)


class BatchRunResultTests(unittest.TestCase):
    def test_from_outputs_builds_member_dataset(self):
        time = np.array([0, 1, 2])
        outputs = [
            ModelOutputs(SSF_pred=xr.DataArray(np.array([1.0, 2.0, 3.0]), coords={"time": time}, dims=("time",), name="SSF_pred")),
            ModelOutputs(SSF_pred=xr.DataArray(np.array([4.0, 5.0, 6.0]), coords={"time": time}, dims=("time",), name="SSF_pred")),
        ]
        result = BatchRunResult.from_outputs(outputs, weights=np.array([0.4, 0.6]))
        ds = result.to_dataset()
        self.assertIn("member", ds.sizes)
        self.assertEqual(ds.sizes["member"], 2)
        self.assertEqual(ds["SSF_pred"].dims, ("member", "time"))
        np.testing.assert_allclose(result.weights, np.array([0.4, 0.6]))

    def test_select_member_round_trips_to_model_outputs(self):
        time = np.array([0, 1])
        outputs = [
            ModelOutputs(A_channel=xr.DataArray(np.array([2.0, 3.0]), coords={"time": time}, dims=("time",), name="A_channel")),
            ModelOutputs(A_channel=xr.DataArray(np.array([5.0, 7.0]), coords={"time": time}, dims=("time",), name="A_channel")),
        ]
        result = BatchRunResult.from_outputs(outputs)
        selected = result.select_member(1)
        np.testing.assert_allclose(selected.A_channel.values, np.array([5.0, 7.0]))

    def test_model_factory_create_execution_returns_base_model_and_batch(self):
        batch = make_parameter_batch("rusle", n_members=2)
        model, execution_batch = ModelFactory.create_execution("rusle", batch)
        self.assertEqual(type(model).__name__, "RUSLEModel")
        self.assertEqual(execution_batch.n_members, 2)

    def test_model_factory_create_model_rejects_multi_member_batch(self):
        batch = make_parameter_batch("rusle", n_members=2)
        with self.assertRaises(TypeError):
            ModelFactory.create_model("rusle", batch)

    def test_model_factory_coerce_parameter_batch_accepts_dict(self):
        template = ModelFactory.get_parameter_template("rusle")
        batch = ModelFactory.coerce_parameter_batch("rusle", template)
        self.assertEqual(batch.n_members, 1)
        self.assertEqual(batch.param_names, tuple(ModelFactory.get_parameter_info("rusle")[0]))

    def test_run_parameter_batch_handles_multi_member_without_ensemble_model_api(self):
        context = make_point_context()
        batch = make_parameter_batch("crsem", n_members=2)
        result = run_parameter_batch("crsem", context, batch)
        ds = result.to_dataset()
        self.assertEqual(result.n_members, 2)
        self.assertEqual(ds["SSF_pred"].dims, ("member", "time"))

    def test_run_parameter_batch_prepares_inputs_once_for_grid_multi_member(self):
        context = make_grid_context()
        batch = make_parameter_batch("crsem", n_members=3)
        with patch("CRSEM._model_base.prepare_inputs", wraps=model_base_module.prepare_inputs) as prepare_mock:
            result = run_parameter_batch("crsem", context, batch)
        self.assertEqual(result.n_members, 3)
        self.assertEqual(prepare_mock.call_count, 1)

    def test_run_parameter_batch_preserves_spatial_fields_for_gridded_river_mode(self):
        context = make_grid_context()
        batch = make_parameter_batch("crsem", n_members=1)
        result = run_parameter_batch("crsem", context, batch, run_method="run_hillslope_river")
        ds = result.to_dataset()

        self.assertEqual(ds["SSF_pred"].dims, ("member", "time"))
        self.assertEqual(ds["A_channel"].dims, ("member", "time"))
        self.assertEqual(ds["E_hillslope"].dims, ("member", "time", "latitude", "longitude"))
        self.assertEqual(ds["R_rain"].dims, ("member", "time", "latitude", "longitude"))
        self.assertEqual(ds["K_factor"].dims, ("member", "time", "latitude", "longitude"))


class ResultAggregatorTests(unittest.TestCase):
    def test_weighted_mean_uses_member_weights(self):
        data = xr.DataArray(
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=float),
            dims=("member", "time"),
            coords={"member": ["m0", "m1"], "time": [0, 1]},
        )
        reduced = ResultAggregator.aggregate(data, method="weighted_mean", weights=np.array([0.25, 0.75]))
        np.testing.assert_allclose(reduced.values, np.array([4.0, 6.0]))

    def test_weighted_std_matches_manual_computation(self):
        data = xr.DataArray(
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=float),
            dims=("member", "time"),
            coords={"member": ["m0", "m1"], "time": [0, 1]},
        )
        reduced = ResultAggregator.aggregate(data, method="std", weights=np.array([0.25, 0.75]))
        expected_var = np.array([3.0, 3.0], dtype=float)
        np.testing.assert_allclose(reduced.values, np.sqrt(expected_var))

    def test_quantile_with_weights_is_not_supported(self):
        data = xr.DataArray(
            np.array([[1.0], [5.0]], dtype=float),
            dims=("member", "time"),
            coords={"member": ["m0", "m1"], "time": [0]},
        )
        with self.assertRaises(NotImplementedError):
            ResultAggregator.aggregate(data, method="quantile_0.5", weights=np.array([0.5, 0.5]))

    def test_weighted_mean_supports_dataset_input(self):
        ds = xr.Dataset(
            {
                "SSF_pred": (("member", "time"), np.array([[1.0, 3.0], [5.0, 7.0]], dtype=float)),
                "A_channel": (("member", "time"), np.array([[2.0, 4.0], [6.0, 8.0]], dtype=float)),
            },
            coords={"member": ["m0", "m1"], "time": [0, 1]},
        )
        reduced = ResultAggregator.aggregate(ds, method="weighted_mean", weights=np.array([0.25, 0.75]))
        np.testing.assert_allclose(reduced["SSF_pred"].values, np.array([4.0, 6.0]))
        np.testing.assert_allclose(reduced["A_channel"].values, np.array([5.0, 7.0]))


if __name__ == "__main__":
    unittest.main()

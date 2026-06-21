from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from CRSEM._model_core import snowmelt_accumulation_core
from CRSEM._model_crsem import CRSEMModel
from CRSEM._model_rusle import RUSLEModel
from CRSEM.calibrator import Calibrator
from CRSEM.contracts import RunContext
from CRSEM.preparation import prepare_inputs
from CRSEM.parameters import CRSEMParameters, RUSLEParameters
from tests.helpers import make_grid_context, make_parameter_batch, make_point_context, make_point_driver


class PointModelTests(unittest.TestCase):
    def test_prepare_inputs_marks_point_mode(self):
        prepared = prepare_inputs(make_point_context())
        self.assertTrue(prepared.is_point_mode)
        self.assertEqual(prepared.dynamic_dims, ("time",))
        self.assertEqual(prepared.T.ndim, 1)

    def test_snowmelt_core_batched_mode_matches_per_cell_1d(self):
        p_snow = np.array([[10.0, 0.0], [5.0, 3.0], [0.0, 4.0]], dtype=np.float32)
        temperature = np.array([[-5.0, -2.0], [1.0, 3.0], [6.0, 8.0]], dtype=np.float32)
        days = np.array([31.0, 28.0, 31.0], dtype=np.float32)

        batched = snowmelt_accumulation_core(p_snow, temperature, days, 2.0, 2.0)
        expected = np.column_stack(
            [
                snowmelt_accumulation_core(p_snow[:, idx], temperature[:, idx], days, 2.0, 2.0)
                for idx in range(p_snow.shape[1])
            ]
        )
        np.testing.assert_allclose(batched, expected)

    def test_prepare_inputs_flattens_gridded_mode_to_cells(self):
        prepared = prepare_inputs(make_grid_context())
        self.assertFalse(prepared.is_point_mode)
        self.assertEqual(prepared.dynamic_dims, ("time", "cell"))
        self.assertEqual(prepared.T.ndim, 2)
        self.assertEqual(prepared.T.shape[1], 6)

    def test_prepare_inputs_rejects_dynamic_time_mismatch(self):
        context = make_grid_context()
        bad_pre = context.inputs.Pre.assign_coords(time=pd.date_range("2001-01-31", periods=context.inputs.Pre.sizes["time"], freq="ME"))
        bad_context = RunContext(
            inputs=type(context.inputs)(
                T=context.inputs.T,
                Pre=bad_pre,
                NDVI=context.inputs.NDVI,
                LS=context.inputs.LS,
                P_f=context.inputs.P_f,
                IC=context.inputs.IC,
                K=context.inputs.K,
            ),
            q=context.q,
            ssf_obs=context.ssf_obs,
            s_area=context.s_area,
            metadata=context.metadata,
        )
        with self.assertRaises(ValueError):
            prepare_inputs(bad_context)

    def test_prepare_inputs_rejects_static_grid_mismatch(self):
        context = make_grid_context()
        bad_ls = xr.DataArray(
            np.full((context.inputs.LS.sizes["latitude"], context.inputs.LS.sizes["longitude"]), 2.5, dtype=np.float32),
            coords={
                "latitude": context.inputs.LS.coords["latitude"].values + 1.0,
                "longitude": context.inputs.LS.coords["longitude"].values,
            },
            dims=("latitude", "longitude"),
            name="LS",
        )
        bad_context = RunContext(
            inputs=type(context.inputs)(
                T=context.inputs.T,
                Pre=context.inputs.Pre,
                NDVI=context.inputs.NDVI,
                LS=bad_ls,
                P_f=context.inputs.P_f,
                IC=context.inputs.IC,
                K=context.inputs.K,
            ),
            q=context.q,
            ssf_obs=context.ssf_obs,
            s_area=context.s_area,
            metadata=context.metadata,
        )
        with self.assertRaises(ValueError):
            prepare_inputs(bad_context)

    def test_prepare_inputs_rejects_q_length_mismatch(self):
        context = make_point_context()
        bad_q = context.q.iloc[:-1]
        bad_context = RunContext(
            inputs=context.inputs,
            q=bad_q,
            ssf_obs=context.ssf_obs,
            s_area=context.s_area,
            metadata=context.metadata,
        )
        with self.assertRaises(ValueError):
            prepare_inputs(bad_context)

    def test_rusle_point_run_batch_has_member_dimension(self):
        context = make_point_context()
        model = RUSLEModel(RUSLEParameters.from_default())
        batch = make_parameter_batch("rusle", n_members=2)
        result = model.run_batch(context, params_batch=batch)
        ds = result.to_dataset()
        self.assertEqual(ds["SSF_pred"].dims, ("member", "time"))
        self.assertEqual(result.n_members, 2)

    def test_crsem_point_run_batch_has_member_dimension(self):
        context = make_point_context()
        model = CRSEMModel(CRSEMParameters.from_default())
        batch = make_parameter_batch("crsem", n_members=2)
        result = model.run_batch(context, params_batch=batch)
        ds = result.to_dataset()
        self.assertEqual(ds["SSF_pred"].dims, ("member", "time"))
        self.assertIn("E_hillslope_rain", ds.data_vars)
        self.assertIn("E_hillslope_melt", ds.data_vars)

    def test_rusle_gridded_hillslope_restores_spatial_dims(self):
        context = make_grid_context()
        model = RUSLEModel(RUSLEParameters.from_default())
        outputs = model.run_hillslope(context)
        self.assertEqual(outputs.E_hillslope.dims, ("time", "latitude", "longitude"))
        self.assertEqual(outputs.C_factor.dims, ("time", "latitude", "longitude"))

    def test_crsem_gridded_hillslope_restores_spatial_dims(self):
        context = make_grid_context()
        model = CRSEMModel(CRSEMParameters.from_default())
        outputs = model.run_hillslope(context)
        self.assertEqual(outputs.E_hillslope.dims, ("time", "latitude", "longitude"))
        self.assertEqual(outputs.R_rain.dims, ("time", "latitude", "longitude"))
        self.assertEqual(outputs.R_melt.dims, ("time", "latitude", "longitude"))

    def test_gridded_river_run_returns_time_series_flux_and_spatial_factors(self):
        context = make_grid_context()
        model = CRSEMModel(CRSEMParameters.from_default())
        outputs = model.run_hillslope_river(context)
        self.assertEqual(outputs.SSF_pred.dims, ("time",))
        self.assertEqual(outputs.A_channel.dims, ("time",))
        self.assertEqual(outputs.E_hillslope_rain.dims, ("time",))
        self.assertEqual(outputs.C_factor.dims, ("time",))

    def test_calibrator_can_rerun_selected_batch(self):
        driver = make_point_driver()
        calibrator = Calibrator(driver=driver, model_type="rusle", plot_progress=False, ensemble_para=False)
        calibrator.selected_batch = make_parameter_batch("rusle", n_members=1)
        result = calibrator.run_selected_batch()
        self.assertEqual(result.n_members, 1)
        self.assertEqual(result.to_dataset()["SSF_pred"].dims, ("member", "time"))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from CRSEM._model_core import (
    c_factor_ndvi_core,
    c_factor_simple_core,
    channel_erosion_core,
    hillslope_erosion_core,
    k_factor_freeze_thaw_core,
    partition_precipitation_core,
    r_factor_melt_core,
    r_factor_rain_core,
    sdr_dynamic_core,
    sdr_static_core,
    snowmelt_accumulation_core,
    total_sediment_flux_core,
)


class ModelCoreTests(unittest.TestCase):
    def test_r_factor_rain_respects_threshold(self):
        rainfall = np.array([5.0, 10.0, 15.0], dtype=np.float32)
        result = r_factor_rain_core(rainfall, a_rain=2.0, threshold=10.0)
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 10.0], dtype=np.float32))

    def test_r_factor_melt_respects_threshold(self):
        melt = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        result = r_factor_melt_core(melt, a_melt=3.0, threshold=2.0)
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 6.0], dtype=np.float32))

    def test_partition_precipitation_splits_rain_and_snow(self):
        temperature = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
        precipitation = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        p_rain, p_snow = partition_precipitation_core(temperature, precipitation, T_threshold=0.0)
        np.testing.assert_allclose(p_rain, np.array([0.0, 0.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(p_snow, np.array([10.0, 20.0, 0.0], dtype=np.float32))

    def test_snowmelt_accumulation_1d_preserves_snowpack_limit(self):
        p_snow = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        temperature = np.array([-5.0, 5.0, 10.0], dtype=np.float32)
        days = np.array([31.0, 30.0, 31.0], dtype=np.float32)
        melt = snowmelt_accumulation_core(p_snow, temperature, days, k_melt=1.0, T_melt_start=0.0)
        self.assertEqual(melt.dtype, np.float32)
        np.testing.assert_allclose(melt, np.array([0.0, 10.0, 0.0], dtype=np.float32))

    def test_snowmelt_accumulation_rejects_shape_mismatch(self):
        p_snow = np.array([10.0, 0.0], dtype=np.float32)
        temperature = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        days = np.array([31.0, 28.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            snowmelt_accumulation_core(p_snow, temperature, days, k_melt=1.0)

    def test_snowmelt_accumulation_rejects_non_1d_days(self):
        p_snow = np.array([10.0, 0.0], dtype=np.float32)
        temperature = np.array([1.0, 3.0], dtype=np.float32)
        days = np.array([[31.0, 28.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            snowmelt_accumulation_core(p_snow, temperature, days, k_melt=1.0)

    def test_k_factor_freeze_thaw_is_clipped(self):
        temperature = np.array([-20.0, 0.0, 20.0], dtype=np.float32)
        result = k_factor_freeze_thaw_core(
            temperature,
            K_base=0.03,
            alpha_K=10.0,
            K_min_ratio=0.8,
            K_max_ratio=1.2,
            T_0=0.0,
            sigma_K=1.0,
        )
        np.testing.assert_allclose(result, np.array([0.03, 0.036, 0.03], dtype=np.float32), rtol=1e-5)

    def test_c_factor_ndvi_clips_extreme_values(self):
        ndvi = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = c_factor_ndvi_core(ndvi, alpha_C=2.0, NDVI_min=0.1, NDVI_max=0.9)
        expected_ndvi = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        expected = np.exp(-2.0 * expected_ndvi / (1.0 - expected_ndvi))
        np.testing.assert_allclose(result, expected.astype(np.float32))

    def test_c_factor_simple_matches_exponential_form(self):
        ndvi = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = c_factor_simple_core(ndvi, alpha_C=2.0)
        np.testing.assert_allclose(result, np.exp(-2.0 * ndvi).astype(np.float32))

    def test_sdr_static_core_returns_expected_sigmoid_value(self):
        result = sdr_static_core(IC=0.5, ic0=0.5, k=0.1)
        self.assertAlmostEqual(float(result), 0.4, places=6)

    def test_sdr_dynamic_core_returns_base_when_beta_zero(self):
        p_rain = np.array([10.0, 20.0], dtype=np.float32)
        melt = np.array([0.0, 5.0], dtype=np.float32)
        result = sdr_dynamic_core(IC=0.5, P_rain=p_rain, Melt=melt, ic0=0.5, k=0.1, beta_sdr=0.0)
        np.testing.assert_allclose(result, np.full_like(p_rain, 0.4, dtype=np.float32))

    def test_sdr_dynamic_core_is_capped_at_one(self):
        p_rain = np.array([500.0, 800.0], dtype=np.float32)
        melt = np.array([200.0, 100.0], dtype=np.float32)
        result = sdr_dynamic_core(IC=5.0, P_rain=p_rain, Melt=melt, ic0=0.0, k=0.1, beta_sdr=10.0)
        self.assertTrue(np.all(result <= 1.0))

    def test_channel_erosion_core_handles_erosion_and_deposition(self):
        discharge = np.array([1.0, 2.0], dtype=np.float32)
        sediment_in = np.array([5.0, 1.0], dtype=np.float32)
        result = channel_erosion_core(discharge, sediment_in, c_base=2.0, n_chan=1.0, K_chan=0.5)
        np.testing.assert_allclose(result, np.array([-3.0, 1.5], dtype=np.float32))

    def test_hillslope_erosion_and_total_flux_core_match_manual_calculation(self):
        r_factor = np.array([2.0, 3.0], dtype=np.float32)
        k_factor = np.array([0.5, 0.25], dtype=np.float32)
        c_factor = np.array([1.0, 0.5], dtype=np.float32)
        erosion = hillslope_erosion_core(r_factor, k_factor, LS=2.0, C=c_factor, P_factor=0.5)
        np.testing.assert_allclose(erosion, np.array([1.0, 0.375], dtype=np.float32))

        sdr = np.array([0.2, 0.4], dtype=np.float32)
        a_channel = np.array([1.0, -1.0], dtype=np.float32)
        flux = total_sediment_flux_core(erosion, sdr, s_area=100.0, A_channel=a_channel)
        np.testing.assert_allclose(flux, np.array([21.0, 14.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

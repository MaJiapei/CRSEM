from __future__ import annotations

import numpy as np

from CRSEM._model_base import BaseModel
from CRSEM._model_core import (
    c_factor_ndvi_core,
    hillslope_erosion_core,
    k_factor_freeze_thaw_core,
    partition_precipitation_core,
    r_factor_melt_core,
    r_factor_rain_core,
    sdr_dynamic_core,
    snowmelt_accumulation_core,
)
from CRSEM.model import ModelOutputs
from CRSEM.parameters import CRSEMParameters


class CRSEMModel(BaseModel):
    """Cold Region Soil Erosion Model (CRSEM)."""

    PARAM_NAMES = tuple(CRSEMParameters.DEFAULT_PARAMS.keys())
    PARAM_BOUNDS = tuple(CRSEMParameters.PARAM_BOUNDS[name] for name in PARAM_NAMES)

    def _init_params(self, params: CRSEMParameters) -> None:
        self.params = params
        self.a_rain = params.a_rain
        self.r_th = params.r_th
        self.a_melt = params.a_melt
        self.m_th = params.m_th
        self.k_melt_factor = params.k_melt
        self.alpha_K = params.alpha_K
        self.K_min_ratio = params.K_min_r
        self.K_max_ratio = params.K_max_r
        self.alpha_C = params.alpha_C
        self.ic0 = params.ic0
        self.k = params.k
        self.beta_sdr = params.beta_sdr
        self.c_base = params.c_base
        self.n_chan = params.n_chan
        self.K_chan = params.K_chan
        self.T_threshold = np.float32(0.0)
        self.T_melt_start = np.float32(2.0)
        self.T_0 = np.float32(0.0)
        self.sigma_K = np.float32(2.5)

    def _days_in_month(self, context) -> np.ndarray:
        if context is None or context.inputs["T"] is None:
            raise ValueError("RunContext with temperature input is required for CRSEM execution.")
        return context.inputs["T"].time.dt.daysinmonth.values.astype(np.float32)

    def _run_prepared_hillslope_numpy(self, prepared, *, context=None):
        T, P, NDVI, LS, K_base, IC, P_factor = self._prepared_fields(
            prepared, "T", "Pre", "NDVI", "LS", "K", "IC", "P_f"
        )
        days_in_month = self._days_in_month(context)

        P_rain, P_snow = partition_precipitation_core(T, P, self.T_threshold)
        Melt = snowmelt_accumulation_core(P_snow, T, days_in_month, self.k_melt_factor, self.T_melt_start)
        R_rain = r_factor_rain_core(P_rain, self.a_rain, self.r_th)
        R_melt = r_factor_melt_core(Melt, self.a_melt, self.m_th)
        K_factor = k_factor_freeze_thaw_core(
            T,
            K_base,
            self.alpha_K,
            self.K_min_ratio,
            self.K_max_ratio,
            self.T_0,
            self.sigma_K,
        )
        C_factor = c_factor_ndvi_core(NDVI, self.alpha_C, NDVI_min=self.NDVI_MIN, NDVI_max=self.NDVI_MAX)
        SDR = sdr_dynamic_core(IC, P_rain, Melt, self.ic0, self.k, self.beta_sdr)
        A_rain = hillslope_erosion_core(R_rain, K_factor, LS, C_factor, P_factor)
        A_melt = hillslope_erosion_core(R_melt, K_factor, LS, C_factor, P_factor)
        E_hillslope_rain = (A_rain * SDR).astype(np.float32)
        E_hillslope_melt = (A_melt * SDR).astype(np.float32)
        E_hillslope = (E_hillslope_rain + E_hillslope_melt).astype(np.float32)

        return {
            "E_hillslope": E_hillslope,
            "E_hillslope_rain": E_hillslope_rain,
            "E_hillslope_melt": E_hillslope_melt,
            "R_rain": R_rain,
            "R_melt": R_melt,
            "K_factor": K_factor,
            "C_factor": C_factor,
            "SDR": SDR,
        }

    def _run_prepared_hillslope_river_numpy(self, prepared, *, context=None, output_mode: str = "full"):
        if prepared.q is None or context is None or context.s_area is None:
            raise ValueError("RunContext must provide q and s_area for river routing.")
        output_mode = output_mode.lower()
        if output_mode not in {"full", "compact"}:
            raise ValueError(f"Unsupported output_mode: {output_mode}. Expected 'full' or 'compact'.")

        hillslope = self._run_prepared_hillslope_numpy(prepared, context=context)
        A_hillslope_rain = self._basin_flux_from_hillslope(hillslope["E_hillslope_rain"], context.s_area)
        A_hillslope_melt = self._basin_flux_from_hillslope(hillslope["E_hillslope_melt"], context.s_area)
        A_hillslope = (A_hillslope_rain + A_hillslope_melt).astype(np.float32)
        A_channel = self._channel_erosion_numpy(prepared.q, A_hillslope)
        SSF_pred = (A_hillslope + A_channel).astype(np.float32)

        # For grid mode, average intermediate variables over space for diagnostics
        if prepared.is_point_mode:
            # Point mode: variables are already 1D (time,)
            payload = {
                "E_hillslope": A_hillslope,
                "E_hillslope_rain": A_hillslope_rain,
                "E_hillslope_melt": A_hillslope_melt,
                "A_channel": A_channel,
                "SSF_pred": SSF_pred,
            }
            if output_mode == "full":
                # K/C/SDR already in hillslope from _run_prepared_hillslope_numpy
                pass
            else:
                payload["K_factor"] = None
                payload["C_factor"] = None
                payload["SDR"] = None
            hillslope.update(payload)
        else:
            # Grid mode: preserve spatial hillslope fields for downstream analysis.
            payload = {
                "E_hillslope": hillslope["E_hillslope"],
                "E_hillslope_rain": hillslope["E_hillslope_rain"],
                "E_hillslope_melt": hillslope["E_hillslope_melt"],
                "A_channel": A_channel,
                "SSF_pred": SSF_pred,
                "R_rain": hillslope["R_rain"],
                "R_melt": hillslope["R_melt"],
            }
            if output_mode == "full":
                payload["K_factor"] = hillslope["K_factor"]
                payload["C_factor"] = hillslope["C_factor"]
                payload["SDR"] = hillslope["SDR"]
            else:
                payload["K_factor"] = None
                payload["C_factor"] = None
                payload["SDR"] = None
            hillslope.update(payload)
        return hillslope

    def run_hillslope(self, source) -> ModelOutputs:
        context, prepared = self._prepare_context(source)
        outputs = self._run_prepared_hillslope_numpy(prepared, context=context)
        return self._prepared_outputs_to_model_outputs(context, prepared, outputs)

    def run_hillslope_river(self, source) -> ModelOutputs:
        context, prepared = self._prepare_context(source)
        outputs = self._run_prepared_hillslope_river_numpy(prepared, context=context)
        return self._prepared_outputs_to_model_outputs(context, prepared, outputs)


if __name__ == "__main__":
    raise SystemExit("_model_crsem.py no longer embeds local example data paths. Use RuntimeConfig/DataPaths from external code.")

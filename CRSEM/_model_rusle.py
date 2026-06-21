from __future__ import annotations

import numpy as np

from CRSEM._model_base import BaseModel
from CRSEM._model_core import c_factor_ndvi_core, hillslope_erosion_core, r_factor_rain_core, sdr_static_core
from CRSEM.model import ModelOutputs
from CRSEM.parameters import RUSLEParameters


class RUSLEModel(BaseModel):
    """Basic RUSLE implementation using the shared model contract."""

    PARAM_NAMES = tuple(RUSLEParameters.DEFAULT_PARAMS.keys())
    PARAM_BOUNDS = tuple(RUSLEParameters.PARAM_BOUNDS[name] for name in PARAM_NAMES)

    def _init_params(self, params: RUSLEParameters) -> None:
        self.params = params
        self.a_rain = params.a_rain
        self.b_rain = params.b_rain
        self.c_base = params.c_base
        self.n_chan = params.n_chan
        self.K_chan = params.K_chan
        self.ic0 = params.ic0
        self.k = params.k
        self.alpha_C = 2.0

    def _run_prepared_hillslope_numpy(self, prepared, *, context=None):
        P, NDVI, LS, K_base, IC, P_factor = self._prepared_fields(
            prepared, "Pre", "NDVI", "LS", "K", "IC", "P_f"
        )

        R_factor = r_factor_rain_core(P, self.a_rain, self.b_rain)
        K_factor = np.broadcast_to(K_base, R_factor.shape).astype(np.float32)
        C_factor = c_factor_ndvi_core(NDVI, self.alpha_C, NDVI_min=self.NDVI_MIN, NDVI_max=self.NDVI_MAX)
        SDR = np.broadcast_to(np.asarray(sdr_static_core(IC, self.ic0, self.k), dtype=np.float32), R_factor.shape).astype(np.float32)
        E_hillslope = hillslope_erosion_core(R_factor, K_factor, LS, C_factor, P_factor)
        E_hillslope = (E_hillslope * SDR).astype(np.float32)

        return {
            "E_hillslope": E_hillslope,
            "R_rain": R_factor,
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
        A_hillslope = self._basin_flux_from_hillslope(hillslope["E_hillslope"], context.s_area)
        A_channel = self._channel_erosion_numpy(prepared.q, A_hillslope)
        SSF_pred = (A_hillslope + A_channel).astype(np.float32)

        # For grid mode, average intermediate variables over space for diagnostics
        if prepared.is_point_mode:
            # Point mode: variables are already 1D (time,)
            payload = {
                "E_hillslope": A_hillslope,
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
                "A_channel": A_channel,
                "SSF_pred": SSF_pred,
                "R_rain": hillslope["R_rain"],
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

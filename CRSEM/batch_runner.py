from __future__ import annotations

from CRSEM.contracts import BatchRunResult
from CRSEM.model import ModelFactory


def run_parameter_batch(model_type: str, source, params, *, run_method: str = "run_hillslope_river") -> BatchRunResult:
    """Run a single-member or multi-member parameter batch through the canonical model path."""
    model, batch = ModelFactory.create_execution(model_type, params)
    return model.run_batch(source, params_batch=batch, run_method=run_method)

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from CRSEM.batch_runner import run_parameter_batch
from CRSEM.calibration_api import refine_parameters
from CRSEM.contracts import ParameterBatch
from CRSEM.driver import BasinDriver
from CRSEM.model import ModelFactory
from CRSEM.result_aggregator import ResultAggregator
from scripts.run_model import infer_model_type


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = PROJECT_ROOT / "example" / "zhimenda_sample" / "drivers"


def _has_example_data() -> bool:
    required = (
        EXAMPLE_DIR / "static.nc",
        EXAMPLE_DIR / "dynamic.nc",
        EXAMPLE_DIR / "observations.nc",
    )
    return all(path.exists() for path in required)


@pytest.fixture(scope="module")
def driver() -> BasinDriver:
    if not _has_example_data():
        pytest.skip("Bundled real-case example data is not available.")
    return BasinDriver.from_nc_files(
        static_nc=EXAMPLE_DIR / "static.nc",
        dynamic_nc=EXAMPLE_DIR / "dynamic.nc",
        observations_nc=EXAMPLE_DIR / "observations.nc",
        station_name="zhimenda",
    ).collapse_ndvi_members()


def test_basin_driver_nc_mode(driver: BasinDriver):
    assert driver.model_inputs is not None
    assert driver.model_inputs.K is not None
    assert driver.model_inputs.LS is not None
    assert driver.model_inputs.T is not None
    assert driver.model_inputs.Pre is not None
    assert driver.model_inputs.NDVI is not None

    ctx = driver.to_run_context()
    assert ctx.inputs is not None
    assert ctx.q is not None
    assert ctx.ssf_obs is not None
    assert ctx.s_area > 0


def test_single_model_run(driver: BasinDriver):
    default_params = ModelFactory.get_parameter_template("crsem")
    result = run_parameter_batch(
        model_type="crsem",
        source=driver,
        params=default_params,
    )
    ds = result.to_dataset()

    assert result.n_members == 1
    assert ds["SSF_pred"].dims == ("member", "time")
    assert ds.sizes["time"] == len(driver.SSF)


def test_ensemble_run(driver: BasinDriver):
    default_params = ModelFactory.get_parameter_template("crsem")
    base = ModelFactory.coerce_parameter_batch("crsem", default_params).values[0]
    ensemble_params = [base, base * 1.02, base * 0.98]

    result = run_parameter_batch(
        model_type="crsem",
        source=driver,
        params=ensemble_params,
    )
    ds = result.to_dataset()
    agg_mean = ResultAggregator.aggregate(ds["SSF_pred"], method="mean")
    agg_std = ResultAggregator.aggregate(ds["SSF_pred"], method="std")

    assert result.n_members == 3
    assert ds["SSF_pred"].dims == ("member", "time")
    assert agg_mean.dims == ("time",)
    assert agg_std.dims == ("time",)


def test_point_mode(driver: BasinDriver):
    point_driver = driver.to_point_driver(keep_rivers=True)
    result = run_parameter_batch(
        model_type="crsem",
        source=point_driver,
        params=ModelFactory.get_parameter_template("crsem"),
    )
    ds = result.to_dataset()

    assert point_driver.model_inputs.T.dims == ("time",)
    assert ds["SSF_pred"].dims == ("member", "time")


def test_lightweight_calibration(driver: BasinDriver):
    selected_batch, metrics = refine_parameters(
        driver=driver,
        model_type="crsem",
        maxiter=1,
        plot_progress=False,
        objective_method="nse",
        selector_name="best_only",
        ensemble_para=False,
        calibration_output_mode="compact",
        optimizer_kwargs={"popsize": 4, "polish": False},
    )

    assert selected_batch.n_members == 1
    assert "NSE" in metrics
    assert "selected_n_members" in metrics


def test_lightweight_ensemble_calibration(driver: BasinDriver):
    selected_batch, metrics = refine_parameters(
        driver=driver.to_point_driver(keep_rivers=True),
        model_type="crsem",
        maxiter=1,
        plot_progress=False,
        objective_method="nse",
        selector_name="aic",
        ensemble_para=True,
        calibration_output_mode="compact",
        optimizer_kwargs={"popsize": 4, "polish": False},
    )

    assert selected_batch.n_members >= 1
    assert metrics["selected_n_members"] == selected_batch.n_members
    if selected_batch.n_members > 1:
        assert selected_batch.weights is not None
        assert pytest.approx(float(selected_batch.weights.sum()), rel=1e-6) == 1.0


def test_lightweight_glue_calibration(driver: BasinDriver):
    selected_batch, metrics = refine_parameters(
        driver=driver.to_point_driver(keep_rivers=True),
        model_type="crsem",
        optimizer="glue",
        maxiter=8,
        plot_progress=False,
        objective_method="nse",
        selector_name="glue",
        ensemble_para=True,
        calibration_output_mode="compact",
        optimizer_kwargs={"sampler": "lhs", "n_samples": 8},
    )

    assert selected_batch.n_members >= 1
    assert metrics["optimizer"] == "glue"
    assert metrics["n_archived_candidates"] >= 8
    if selected_batch.n_members > 1:
        assert selected_batch.weights is not None
        assert pytest.approx(float(selected_batch.weights.sum()), rel=1e-6) == 1.0


def test_run_model_style_multi_member_parameter_file_round_trip(driver: BasinDriver):
    default_params = ModelFactory.get_parameter_template("crsem")
    base = ModelFactory.coerce_parameter_batch("crsem", default_params).values[0]
    batch = ParameterBatch(
        values=[base, base * 1.02],
        param_names=tuple(ModelFactory.get_parameter_info("crsem")[0]),
        weights=[0.25, 0.75],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = Path(tmpdir) / "params.json"
        batch.to_file(params_path, metrics={"model_type": "crsem"})
        restored_batch, saved_metrics = ParameterBatch.from_file(params_path)

    model_type = infer_model_type(restored_batch, saved_metrics)
    result = run_parameter_batch(
        model_type=model_type,
        source=driver,
        params=restored_batch,
        run_method="run_hillslope_river",
    )
    dataset = result.to_dataset()
    aggregated = ResultAggregator.aggregate(dataset, method="weighted_mean", weights=result.weights)

    assert result.n_members == 2
    assert result.weights is not None
    assert dataset["SSF_pred"].dims == ("member", "time")
    assert aggregated["SSF_pred"].dims == ("time",)
    expected = (
        dataset["SSF_pred"].isel(member=0).values * result.weights[0]
        + dataset["SSF_pred"].isel(member=1).values * result.weights[1]
    )
    assert pytest.approx(float(result.weights.sum()), rel=1e-6) == 1.0
    assert model_type == "crsem"
    assert aggregated["SSF_pred"].shape == dataset["SSF_pred"].isel(member=0).shape
    np.testing.assert_allclose(aggregated["SSF_pred"].values, expected)

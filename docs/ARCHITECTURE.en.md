# CRSEM Architecture

## 1. Overview

CRSEM is a distributed soil erosion model built on the RUSLE framework and extended for cold-region processes such as snowmelt erosion and freeze-thaw effects.

Core workflow:

```text
Data preparation -> Parameter calibration -> Model run -> Result analysis
```

Main data flow:

```text
NetCDF files -> BasinDriver -> RunContext -> PreparedInputs -> BaseModel.run_batch() -> BatchRunResult
```

## 2. Core Principles

### 2.1 xarray at the boundary, numpy in the core

- I/O and context layers keep `xarray`
- internal model execution uses `numpy`
- outputs are wrapped back to `xarray` at the boundary

### 2.2 Unified single-member and ensemble execution

- single-model run: `ParameterBatch.n_members == 1`
- ensemble run: `ParameterBatch.n_members > 1`
- there is no separate ensemble model class

### 2.3 Data preparation is decoupled from model execution

- preparation logic is independent from execution
- standard NetCDF files are the exchange format
- `BasinDriver` loads prepared NC files only

## 3. Module Structure

```text
CRSEM/
├── __init__.py
├── driver.py
├── contracts.py
├── model.py
├── parameters.py
├── _model_base.py
├── _model_core.py
├── _model_crsem.py
├── _model_rusle.py
├── calibration_evaluation.py
├── batch_runner.py
├── preparation.py
├── result_aggregator.py
├── calibrator.py
├── calibration_api.py
├── calibration_optimizer.py
├── calibration_result.py
├── calibration_reporting.py
├── data_preparation/
│   ├── spatial.py
│   ├── builders.py
│   ├── quality.py
│   └── obs_preprocessing.py
├── config.py
├── parameter_config.py
└── sensitivity.py

scripts/
├── prepare_basin_drivers.py
├── calibrate_parameters.py
├── run_model.py
├── attribution_analysis.py        # driver attribution, always hillslope-only
├── plot_ndvi_attribution_analysis.py
└── test_full_workflow.py
```

## 4. Layered Architecture

```text
Data Preparation Layer
  data_preparation/spatial.py, builders.py, io_legacy.py
  outputs: static.nc, dynamic.nc, observations.nc

Driver Layer
  BasinDriver.from_nc_files() -> ModelInputs

Context Layer
  RunContext, PreparedInputs

Execution Layer
  BaseModel.run_batch() -> CRSEMModel / RUSLEModel

Result Layer
  BatchRunResult -> ResultAggregator

Calibration Layer
  Calibrator, Selector, ObjectiveEvaluator
```

## 5. Data Flow

### 5.1 Data preparation

```text
Raw data (NetCDF/GeoTIFF/CSV)
    -> data_preparation/spatial.py
    -> data_preparation/builders.py
    -> static.nc / dynamic.nc / observations.nc
```

### 5.2 Model run

```text
BasinDriver.from_nc_files(...)
    -> ModelInputs
    -> RunContext
    -> prepare_inputs()
    -> PreparedInputs
    -> BaseModel.run_batch(...)
    -> BatchRunResult
    -> xr.Dataset
```

### 5.3 Calibration flow

```text
CalibrationModelRunner(source, output_mode)
    -> prepare_inputs()  # cached once per calibration runner
    -> ModelFactory.create_model(...)
    -> model._run_prepared_hillslope_river_numpy(...)
    -> CandidateEvaluation
```

`output_mode`:

| Mode | Output payload | Memory use | Typical use |
|------|------|------|------|
| `full` | `SSF_pred`, `A_channel`, `R_rain`, `R_melt`, `K/C/SDR` | higher | detailed diagnostics |
| `compact` | `SSF_pred`, `A_channel`, `R_rain`, `R_melt` | lower | faster calibration |

In gridded river runs, intermediate variables such as `R_rain` and `R_melt` are spatially averaged with `np.nanmean` before being exposed to reporting or penalty logic.

## 6. Core Classes

### 6.1 BasinDriver

Loads prepared NC files and provides basin metadata.

```python
driver = BasinDriver.from_nc_files(
    static_nc="example/tuotuohe_1990_2000/static.nc",
    dynamic_nc="example/tuotuohe_1990_2000/dynamic.nc",
    observations_nc="example/tuotuohe_1990_2000/observations.nc",
    station_name="Tuotuohe",
)
```

Responsibilities:

- load static and dynamic inputs
- load observations
- expose basin metadata such as area and time range

### 6.2 ModelInputs

Container for xarray-based model inputs.

```python
@dataclass
class ModelInputs:
    K: Optional[xr.DataArray]
    LS: Optional[xr.DataArray]
    IC: Optional[xr.DataArray]
    P_f: Optional[xr.DataArray]
    T: Optional[xr.DataArray]
    Pre: Optional[xr.DataArray]
    NDVI: Optional[xr.DataArray]
```

### 6.3 RunContext

Execution context holding all model inputs.

```python
@dataclass
class RunContext:
    inputs: ModelInputs
    q: pd.Series
    ssf_obs: pd.Series
    s_area: float
    metadata: dict
```

### 6.4 ParameterBatch

Single-member and ensemble parameter container.

```python
batch = ParameterBatch(
    values=np.array([[p1, p2, ...]]),
    param_names=("a_rain", "r_th", ...),
    weights=[0.5, 0.3, 0.2],
)
```

### 6.5 BatchRunResult

Stores model outputs with explicit member semantics.

```python
result = run_parameter_batch("crsem", driver, params)
ssf_pred = result.variables["SSF_pred"]
ds = result.to_dataset()
```

## 7. Model Parameters

### 7.1 CRSEM parameters

| Parameter | Meaning | Default | Range |
|------|------|------|------|
| a_rain | rainfall erosivity coefficient | 0.5 | [0.5, 1.0] |
| r_th | rainfall threshold | 10.0 | [1, 20] |
| a_melt | snowmelt erosivity coefficient | 0.5 | [0.1, 1.0] |
| m_th | snowmelt threshold | 5.0 | [0, 10] |
| k_melt | melt coefficient | 2.0 | [1, 5] |
| alpha_K | freeze-thaw `K` modifier | 0.5 | [0.1, 0.8] |
| K_min_r | minimum `K` ratio | 0.5 | [0.4, 1.0] |
| K_max_r | maximum `K` ratio | 1.5 | [1.0, 2.0] |
| alpha_C | `C` factor coefficient | 3.0 | [1, 5] |
| ic0 | SDR inflection point | 0.5 | [0.1, 1.0] |
| k | SDR slope parameter | 1.5 | [0.5, 4.0] |
| beta_sdr | SDR exponent | 0.5 | [0.3, 1.0] |
| c_base | base channel erosion coefficient | 5.0 | [0.1, 20] |
| n_chan | channel exponent | 1.5 | [1.0, 2.0] |
| K_chan | channel erodibility coefficient | 0.5 | [0.1, 1.0] |

## 8. Unit Conventions

| Variable | Unit |
|------|------|
| s_area | hectare (`ha`) |
| SSF | `t/month` |
| Q | `m^3/s` |
| Pre | `mm/month` |
| T | `degC` |
| E_hillslope | `t/ha/month` |
| R_rain`, `R_melt` | `MJ*mm/(ha*h*month)` |

## 9. Maintainer Guide

### 9.1 Placement rules

- path handling, file reading, coordinate alignment -> I/O or `data_preparation`
- formulas and array physics -> execution layer (`_model_*.py`)
- NSE, KGE, RMSE, penalties -> calibration layer
- top-k, AIC, means, quantiles -> selector or aggregator layer
- plotting, logging, progress display -> reporting layer

### 9.2 Testing requirements

- every new feature must include unit tests
- run `pytest tests/` before merging
- keep practical test coverage high

### 9.3 Documentation updates

- API changes require updates to `docs/USER_GUIDE.md` and `docs/USER_GUIDE.en.md`
- architecture changes require updates to both architecture documents

## 10. Change History

| Version | Change |
|------|------|
| v1.0 | initial version with config-based data loading |
| v2.0 | added `data_preparation` and NC-file workflow |
| v2.1 | `BasinDriver` loads NC files only |
| v2.2 | removed `dhesm`, removed `run/`, unified scripts under `scripts/` |
| v2.3 | added `output_mode`, optimized `prepare_inputs` reuse, updated CLI behavior around run mode and calibration reporting |

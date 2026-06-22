# CRSEM User Guide

## 1. Overview

CRSEM is a semi-distributed cold-region soil erosion model operating at monthly time steps and 1 km spatial resolution. The hillslope component follows the RUSLE framework and adds snowmelt erosion and freeze-thaw effects. A river transport-capacity component then routes hillslope sediment to the basin outlet so that observed discharge `Q` and suspended sediment flux `SSF` can be used for calibration.

### 1.1 Physical Processes

#### 1.1.1 Hillslope erosion

The hillslope module follows the RUSLE structure and estimates how much sediment is detached from the land surface in each month. It combines climate forcing, soil susceptibility, terrain, vegetation cover, and conservation effects into one erosion term.

```text
A = R x K x LS x C x P_f
```

- `A`: hillslope erosion (`t/ha/month`)
- `R`: rainfall or snowmelt erosivity
- `K`: soil erodibility
- `LS`: topographic factor
- `C`: vegetation cover factor
- `P_f`: support practice factor

#### 1.1.2 Rainfall and snowmelt erosivity

Cold-region basins cannot treat all precipitation the same way. The model first separates rainfall and snowfall using temperature, then computes erosivity from rainfall runoff potential and from actual snowmelt so that warm-season storms and thaw-season melt events can contribute through different pathways.

```text
P_rain = P    when T > T_threshold
P_snow = P    when T <= T_threshold

R_rain = max(0, a_rain x (P_rain - r_th))
R_melt = max(0, a_melt x (M_actual - m_th))
```

#### 1.1.3 Degree-day snowmelt

Snow storage is updated month by month and released through a degree-day formulation. This gives the model a simple but physically interpretable way to represent accumulation in cold months and meltwater release when temperature rises above the melt threshold.

```text
S_pack = S_pack + P_snow
M_potential = k_melt x max(0, T - T_melt) x N_days
M_actual = min(S_pack, M_potential)
S_pack = S_pack - M_actual
```

#### 1.1.4 Freeze-thaw effect on `K`

Freeze-thaw cycles can weaken soil structure and increase erodibility even when rainfall is not extreme. The model represents this by amplifying the baseline `K` factor near a characteristic temperature window around freezing, where repeated thawing and refreezing are most active.

```text
K = K_base x (1 + alpha_K x F_i)
F_i = exp(-(T - T_0)^2 / (2 x sigma_K^2))
```

#### 1.1.5 NDVI-based `C`

Vegetation cover reduces sediment detachment by shielding soil and weakening raindrop impact. Instead of prescribing `C` directly, the model derives it from NDVI so that seasonal vegetation dynamics are reflected in monthly erosion estimates.

```text
C = exp(-alpha_C x NDVI / (1 - NDVI))
```

#### 1.1.6 Dynamic SDR

Not all eroded sediment reaches the outlet. The sediment delivery ratio links hillslope erosion to delivered sediment by accounting for basin connectivity and by allowing wetter months to transport a larger fraction of available material downstream.

```text
SDR_base = 0.8 / (1 + exp((ic0 - IC) / k))
f_dyn = clip(1 + beta_sdr x (P_total / P_mean), 1, 3)
SDR = min(1, SDR_base x f_dyn)
```

#### 1.1.7 Channel erosion and deposition

After hillslope sediment enters the river network, the channel component determines whether the river can carry it, store it, or erode additional material from the bed and banks. This is controlled by transport capacity, which increases with discharge and is compared against incoming sediment supply.

```text
T_cap = c_base x Q^n_chan
E_potential = T_cap - S_in

If E_potential > 0: A_channel = E_potential x K_chan
If E_potential <= 0: A_channel = E_potential
```

#### 1.1.8 Basin outlet sediment flux

The final outlet flux combines sediment delivered from hillslopes and the net channel contribution. This allows the model to match observed outlet sediment flux while still separating where sediment was produced and how it was modified during routing.

```text
SSF_pred = (E_hillslope x SDR x S_area) + A_channel
```

#### 1.1.9 Main calibration parameters

| Parameter | Default range | Meaning |
|------|------|------|
| a_rain | [0.5, 1.0] | Rainfall erosivity coefficient |
| r_th | [1, 20] | Rainfall threshold |
| a_melt | [0.1, 1.0] | Snowmelt erosivity coefficient |
| m_th | [0, 10] | Snowmelt threshold |
| k_melt | [1, 5] | Degree-day melt factor |
| alpha_K | [0.1, 0.8] | Freeze-thaw enhancement factor |
| K_min_r | [0.4, 1.0] | Minimum `K` ratio |
| K_max_r | [1.0, 2.0] | Maximum `K` ratio |
| alpha_C | [1, 5] | NDVI-to-`C` coefficient |
| ic0 | [0.1, 1.0] | SDR inflection point |
| k | [0.5, 4.0] | SDR slope parameter |
| beta_sdr | [0.3, 1.0] | Dynamic SDR factor |
| c_base | [0.1, 20] | Base channel transport coefficient |
| n_chan | [1.0, 2.0] | Channel flow exponent |
| K_chan | [0.1, 1.0] | Channel erodibility coefficient |

Core workflow:

```text
Data preparation -> Parameter calibration -> Model run -> Result analysis
```

Main scripts:

| Script | Location | Purpose |
|------|------|------|
| Data preparation | `scripts/prepare_basin_drivers.py` | Generic basin driver preparation |
| Calibration | `scripts/calibrate_parameters.py` | Parameter calibration |
| Model run | `scripts/run_model.py` | Ensemble simulation |
| Plotting | `scripts/plot_ssf_comparison.py` | Simulated vs observed plotting |

## 2. Data Preparation

### 2.1 Required inputs

| File | Variables | Description |
|------|------|------|
| `static.nc` | K, LS, IC, P_f, mask | Static spatial factors |
| `dynamic.nc` | T, Pre, NDVI | Time-varying forcing |
| `observations.nc` | Q, SSF | Observations for calibration |

### 2.2 Generic preparation script

Minimum command:

```bash
python scripts/prepare_basin_drivers.py \
  --config config/basin_data_sources.tuotuohe.yml \
  --basin tuotuohe \
  --years 1990 2000 \
  --output example/tuotuohe_1990_2000
```

With quality reporting:

```bash
python scripts/prepare_basin_drivers.py \
  --config config/basin_data_sources.tuotuohe.yml \
  --basin tuotuohe \
  --years 1990 2000 \
  --output example/tuotuohe_1990_2000 \
  --quality-report
```

Parameter reference:

| Argument | Meaning | Default | Required | Notes |
|------|------|------|------|------|
| `--config` | Path to the data-source YAML file | None | Yes | Defines where basin inputs are read from |
| `--basin` | Basin name | None | Yes | Must match a basin key in the config file |
| `--years START END` | Start and end year | `1990 2000` | No | Two integers, both included in the processed range |
| `--output` | Output directory | None | Yes | Writes `drivers/`, metadata, and optional reports |
| `--quality-report` | Generate and print a quality report | `False` | No | Also writes `quality_report.json` |
| `--verbose`, `-v` | Print detailed progress | `False` | No | Useful when diagnosing data-preparation issues |

The prepared example bundled in this repository is under [example/tuotuohe_1990_2000](/mnt/d/code/sediment/example/tuotuohe_1990_2000).

## 3. Parameter Calibration

### 3.1 Basic usage

Start from the minimum command:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --maxiter 100
```

Then add common options:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "Tuotuohe" \
  --run-mode point \
  --selector best_only \
  --popsize 8 \
  --maxiter 100
```

Parameter reference:

| Argument | Meaning | Default | Required | Notes |
|------|------|------|------|------|
| `--static-nc` | Path to the static input file | None | Yes | `static.nc` |
| `--dynamic-nc` | Path to the dynamic forcing file | None | Yes | `dynamic.nc` |
| `--observations-nc` | Path to the observations file | None | Yes | `observations.nc` |
| `--station-name` | Station name metadata | `"unknown"` | No | Written into output metadata |
| `--model-type` | Model type to calibrate | `crsem` | No | `crsem` or `rusle` |
| `--run-mode` | Calibration input mode | `point` | No | `point` is faster; `gridded` keeps spatial heterogeneity |
| `--optimizer` | Optimizer name | `differential_evolution` | No | `differential_evolution` or `glue` |
| `--objective-method` | Objective metric name | `nse` | No | Passed through to the calibration API; built-in options are `nse`, `nse_pbias`, `kge`, `kge_pbias`, `rmse`, `mae`, and `r2` |
| `--config` | Path to parameter-config YAML | None | No | Overrides built-in defaults, bounds, and penalties |
| `--selector` | Post-calibration member selector | Auto | No | Defaults to `aic` for differential evolution and `glue` for the sampling workflow |
| `--aic-numbers` | Force an exact ensemble size | None | No | Only used with `--selector aic` |
| `--aic-max-numbers` | Maximum ensemble size for automatic AIC selection | None | No | Only used with `--selector aic` |
| `--aic-delta-threshold` | Delta-AIC threshold | None | No | Only used with `--selector aic` |
| `--aic-cum-weight` | Cumulative AIC weight threshold | None | No | Only used with `--selector aic` |
| `--maxiter` | Maximum optimizer iterations | `100` | No | DE iteration cap; GLUE uses it as the fallback sample count |
| `--popsize` | Differential-evolution population multiplier | Optimizer default | No | Overrides the optimizer default only when set |
| `--workers` | Number of parallel workers | Serial | No | Supported only in `gridded` mode; `-1` uses all available CPUs |
| `--polish` | Enable final local search | `False` | No | Turns on L-BFGS-B polishing, which costs more time |
| `--n-samples` | GLUE sample count | None | No | Only used with `--optimizer glue`; falls back to `--maxiter` when omitted |
| `--sampling-method` | GLUE sampling scheme | `sobol` | No | `sobol`, `lhs`, or `random` |
| `--seed` | Random seed | Optimizer default | No | Controls reproducibility for differential evolution and scrambled GLUE sampling |
| `--glue-threshold` | GLUE behavioral threshold | None | No | Lower bound for NSE/KGE/R², upper bound for RMSE/MAE |
| `--glue-top-fraction` | GLUE fallback retention fraction | None | No | Used when no explicit or built-in threshold is available |
| `--glue-max-members` | GLUE maximum behavioral members | None | No | Caps the size of the behavioral set |
| `--glue-channel-ratio-lower` | GLUE lower channel-ratio bound | None | No | `channel_ratio = net channel contribution / hillslope sediment` |
| `--glue-channel-ratio-upper` | GLUE upper channel-ratio bound | None | No | `channel_ratio = net channel contribution / hillslope sediment` |
| `--save [PATH]` | Save calibration results | Off | No | Bare `--save` writes beside the input data; `PATH` may be a directory or `.json` file |
| `--plot-progress` | Plot calibration progress | `False` | No | Supported only in `point` mode |

Run mode options:

| Option | Description | Use case |
|------|------|------|
| `--run-mode point` | Basin-mean point inputs, default | Faster calibration; plotting supported |
| `--run-mode gridded` | Original gridded inputs | Finer calibration; supports `--workers`, plotting not supported |

Approximate runtime on the bundled Tuotuohe case:

| Mode | Grid size | Time length | Average runtime |
|------|------|------|------|
| Grid | 175 x 256 (23,354 cells) | 132 months | ~415 ms |
| Point | 1 | 132 months | ~2.6 ms |

### 3.1.1 Diagnostic output and plotting

Default behavior:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "Tuotuohe"
```

Point-mode plotting:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --run-mode point \
  --plot-progress
```

Gridded calibration:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --run-mode gridded
```

Internal output modes:

| Mode | Internal diagnostics | Use case |
|------|------|------|
| Default, no `--plot-progress` | `compact`: `SSF_pred`, `A_channel`, `R_rain`, `R_melt` | Faster calibration |
| `point + --plot-progress` | `full`: adds `K/C/SDR` diagnostics | Interactive inspection |

Restriction:

- `gridded` calibration does not support `--plot-progress`

Calibration output is saved next to `static.nc` by default.
In the lower-left progress panel, `Loss` is the actual objective value, while `Conv` is the optimizer convergence indicator; they are different quantities.

### 3.2 Configuration-based parameter management

Use a YAML file to control parameter defaults, bounds, and penalties:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --config config/parameter_config.custom.yml \
  --maxiter 100
```

### 3.3 Ensemble calibration mode

Minimum AIC-based ensemble example:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --selector aic
```

Control ensemble size explicitly:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --selector aic \
  --aic-numbers 3
```

### 3.4 Calibration outputs

The calibration summary includes:

- performance metrics
- selected parameter table
- ensemble information when `selector=aic`
- file paths and metadata

**Calibration result example:**

Tuotuohe basin 1990-2000 calibration result:

![SSF comparison](ssf_comparison.png)

The figure shows simulated vs observed sediment flux, including time series, scatter plot, monthly climatology, and annual totals.

## 4. Model Run

### 4.1 Point-mode run

Start from the minimum run command:

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/drivers/params.json \
  --start-year 1990 \
  --end-year 2000
```

Parameter reference:

| Argument | Meaning | Default | Required | Notes |
|------|------|------|------|------|
| `--static-nc` | Path to the static input file | None | Yes | `static.nc` |
| `--dynamic-nc` | Path to the dynamic forcing file | None | Yes | `dynamic.nc` |
| `--observations-nc` | Path to the observations file | None | Yes | `observations.nc` |
| `--station-name` | Station name metadata | `"unknown"` | No | Written into output metadata |
| `--start-year` | Simulation start year | Driver start year | No | Crops the driver time axis before execution, inclusive |
| `--end-year` | Simulation end year | Driver end year | No | Crops the driver time axis before execution, inclusive |
| `--params-file` | Path to the saved calibration parameters | None | Yes | `params.json` |
| `--run-method` | Model execution entry point | `run_hillslope_river` | No | `run_hillslope` or `run_hillslope_river` |
| `--run-mode` | Execution input mode | `gridded` | No | `point` uses basin-mean inputs; `gridded` uses original grids |
| `--aggregate` | Ensemble aggregation method | `none` | No | `none` keeps all parameter members; other values are handled by the aggregator |
| `--output-file [PATH]` | Output NetCDF path | Off | No | Bare `--output-file` writes beside `static.nc`; a directory path writes `PATH/model_output.nc`; a `.nc` path writes to that file |

Point-mode sediment prediction:

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/drivers/params.json \
  --start-year 1990 \
  --end-year 2000 \
  --run-mode point \
  --run-method run_hillslope_river
```

Notes:

- `run_model.py` defaults to `gridded`
- model type is inferred from `params.json`
- use `--start-year/--end-year` to crop the simulation period before execution
- if `NDVI` includes a `member` dimension, the standard run path collapses NDVI members to their mean before execution
- add `--output-file` to write `model_output.nc` beside `static.nc`

Point-mode output variables:

- `SSF_pred`
- `A_channel`
- `E_hillslope`

Dimensions: `(member, time)`

### 4.2 Gridded run

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/drivers/params.json \
  --run-mode gridded \
  --run-method run_hillslope_river
```

### 4.3 Ensemble aggregation

Weighted mean:

```bash
python scripts/run_model.py \
  ... \
  --aggregate weighted_mean
```

Keep all members:

```bash
python scripts/run_model.py \
  ... \
  --aggregate none
```

Use `--output-file` to write `model_output.nc`, `--output-file DIR` to write `DIR/model_output.nc`, or `--output-file FILE.nc` for an explicit file.

### 4.4 Output attributes

`model_output.nc` stores metadata such as:

| Attribute | Meaning |
|------|------|
| calibration_start_year | calibration start year |
| calibration_end_year | calibration end year |
| calibration_NSE | calibration NSE |
| calibration_KGE | calibration KGE |
| calibration_R2 | calibration R2 |
| station_name | station name |

## 5. Visualization

### 5.1 SSF comparison plot

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000/model_output.nc \
  --observed example/tuotuohe_1990_2000/observations.nc \
  --output figures/ssf_comparison.png
```

Parameter reference:

| Argument | Meaning | Default | Required | Notes |
|------|------|------|------|------|
| `--simulated` | Path to the simulated NetCDF file | None | Yes | Usually `model_output.nc` |
| `--observed` | Path to the observed NetCDF file | None | Yes | Usually `observations.nc` |
| `--calibration-start` | Calibration start year | Read from simulation metadata | No | Falls back to NetCDF attributes when omitted |
| `--calibration-end` | Calibration end year | Read from simulation metadata | No | Falls back to NetCDF attributes when omitted |
| `--output` | Output figure path | `ssf_comparison.png` beside the simulated file | No | Any writable PNG path |
| `--title` | Figure title | Auto-generated | No | Generated from metadata if omitted |
| `--member` | Ensemble selection mode | `auto` | No | Ignored for single-member files; ensemble files support `auto`/`mean`, a zero-based index, or a member label |
| `--force-split` | Force calibration/validation splitting | `False` | No | Splits periods even when the simulation window matches calibration |

Plot contents:

- time series comparison with calibration shading
- scatter plot with 1:1 line and `R^2`
- monthly climatology comparison
- annual total comparison

Manual calibration period:

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000_1985_2015/model_output.nc \
  --observed example/tuotuohe_1990_2000_1985_2015/observations.nc \
  --calibration-start 1990 \
  --calibration-end 2000 \
  --output figures/ssf_comparison.png
```

Forced split:

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000/model_output.nc \
  --observed example/tuotuohe_1990_2000/observations.nc \
  --calibration-start 1990 \
  --calibration-end 1995 \
  --force-split \
  --output figures/ssf_comparison_split.png
```
## 6. Vegetation Greening Erosion-Reduction Simulation

Vegetation greening erosion-reduction simulation answers the question: compared with a counterfactual NDVI sequence without sustained greening, how much hillslope erosion did the actual NDVI change reduce? This is different from [`CRSEM/sensitivity.py`](../CRSEM/sensitivity.py): sensitivity analysis reports variable importance, while this simulation reports cumulative contribution.

Basic idea:

- real scenario: run the hillslope erosion module with the original `NDVI` sequence
- counterfactual scenario: replace `NDVI` with a baseline monthly climatology, for example the mean NDVI for each calendar month over 1982-2000
- erosion reduction definition: `reduction = E_counterfactual - E_real`
- if `reduction > 0`, the actual vegetation state reduced erosion relative to the baseline climatology
- use `run_hillslope` so that the result isolates hillslope erosion and does not mix vegetation effects with river routing or discharge `Q`

### 6.1 Zhimenda Lightweight Example

The example below uses the bundled [`example/zhimenda_sample`](../example/zhimenda_sample) dataset to estimate the erosion reduction from actual NDVI during 2001-2019 relative to the 1982-2000 monthly NDVI climatology. Note that this `drivers/` directory is not a full two-dimensional spatial grid. It is a basin-mean sample aggregated from the full gridded Zhimenda drivers; the NetCDF files keep the minimum spatial dimensions `y=1, x=1`, but the data are effectively one-dimensional time-series drivers. This is suitable for demonstrating the workflow and checking that the code runs. Formal spatial analysis should use the full gridded driver data.

```python
from pathlib import Path

import xarray as xr

from CRSEM.batch_runner import run_parameter_batch
from CRSEM.contracts import ParameterBatch
from CRSEM.driver import BasinDriver

base = Path("example/zhimenda_sample")

# 1. Load the Zhimenda sample and collapse multi-source NDVI members.
#    example/zhimenda_sample/drivers is a 1x1 basin-mean sample.
#    It is effectively a one-dimensional time-series driver, not a full 2D grid driver.
driver = BasinDriver.from_nc_files(
    static_nc=base / "drivers/static.nc",
    dynamic_nc=base / "drivers/dynamic.nc",
    observations_nc=base / "drivers/observations.nc",
    station_name="zhimenda",
).collapse_ndvi_members()

# 2. Convert to basin-mean point mode. The erosion-reduction analysis uses
#    only hillslope processes, so Q/SSF do not need to be retained.
driver = driver.to_point_driver(keep_rivers=False)

# 3. Build the real and counterfactual scenarios.
#    The counterfactual replaces 2001-2019 NDVI with the 1982-2000
#    monthly climatology.
real_driver = driver.crop_time_range(2001, 2019, align_to_obs=False)
cf_driver = driver.to_cf_driver(
    "NDVI",
    baseline_start=1982,
    baseline_end=2000,
).crop_time_range(2001, 2019, align_to_obs=False)

# 4. Load the calibrated parameter ensemble and run both scenarios.
params, _ = ParameterBatch.from_file(
    base / "params_1982_2000_kge_pbias_m120.json"
)
real_result = run_parameter_batch(
    "crsem",
    real_driver,
    params,
    run_method="run_hillslope",
)
cf_result = run_parameter_batch(
    "crsem",
    cf_driver,
    params,
    run_method="run_hillslope",
)

real = real_result.to_dataset()
cf = cf_result.to_dataset()

# 5. Calculate monthly, annual, and cumulative erosion reduction.
#    E_hillslope is in t ha-1 month-1.
#    driver.s_area is basin area in ha.
monthly_reduction = cf["E_hillslope"] - real["E_hillslope"]
annual_reduction = monthly_reduction.groupby("time.year").sum("time")

if real_result.weights is not None:
    weights = xr.DataArray(
        real_result.weights,
        dims=("member",),
        coords={"member": annual_reduction.member},
    )
else:
    weights = xr.ones_like(annual_reduction.isel(year=0)) / annual_reduction.sizes["member"]

weighted_annual = (annual_reduction * weights).sum("member")
cumulative_reduction_t = (weighted_annual * float(driver.s_area)).cumsum("year")

print("Mean annual reduction (t ha-1 yr-1):", float(weighted_annual.mean()))
print("Cumulative reduction (t):", float(cumulative_reduction_t.isel(year=-1)))
```

For the current lightweight example, the script returns a positive cumulative erosion reduction, meaning that actual NDVI during 2001-2019 reduced hillslope erosion relative to the 1982-2000 NDVI climatology. Because this is a 1x1 basin-mean sample, the values are intended mainly to demonstrate the method. Formal studies should use full spatial drivers and quantify uncertainty from NDVI products, parameter ensembles, and baseline-period selection.

### 6.2 Interpretation

| Quantity | Meaning | Units |
|------|------|------|
| `monthly_reduction` | monthly hillslope erosion-modulus reduction, `E_cf - E_real` | `t ha-1 month-1` |
| `annual_reduction` | annual hillslope erosion-modulus reduction | `t ha-1 yr-1` |
| `weighted_annual` | annual reduction after parameter-ensemble weighting | `t ha-1 yr-1` |
| `cumulative_reduction_t` | cumulative total reduction after multiplying annual reduction by basin area | `t` |

Interpretation notes:

- `weighted_annual > 0` means actual vegetation reduced erosion relative to the baseline climatology.
- `weighted_annual < 0` means actual vegetation increased erosion relative to the baseline climatology.
- If `dynamic.nc` contains multiple NDVI members, the standard example uses `collapse_ndvi_members()` to average them first. For formal uncertainty analysis, repeat the workflow for each NDVI member.
- The baseline period controls the counterfactual meaning. For example, `1982-2000` means "later NDVI compared with the early-period monthly average vegetation state."

## 7. Run Mode Comparison

| run_method | run_mode | Use case | Output shape |
|------|------|------|------|
| run_hillslope | point | hillslope process analysis | `(member, time)` |
| run_hillslope | gridded | spatial erosion pattern | `(member, time, y, x)` |
| run_hillslope_river | point | calibration and sediment prediction | `(member, time)` |
| run_hillslope_river | gridded | integrated analysis | `SSF_pred` / `A_channel`: `(member, time)`; hillslope fields: `(member, time, y, x)` |

Recommended usage:

- parameter calibration: `run_hillslope_river + point`
- sediment prediction: `run_hillslope_river + point`
- spatial analysis: `run_hillslope + gridded`

## 8. Python API

### 8.1 BasinDriver

```python
from CRSEM.driver import BasinDriver

driver = BasinDriver.from_nc_files(
    static_nc="example/tuotuohe_1990_2000/static.nc",
    dynamic_nc="example/tuotuohe_1990_2000/dynamic.nc",
    observations_nc="example/tuotuohe_1990_2000/observations.nc",
    station_name="Tuotuohe",
)

ctx = driver.to_run_context()
print(driver.s_area)
print(driver.Q)
print(driver.SSF)
```

### 8.2 Run the model

```python
from CRSEM.batch_runner import run_parameter_batch
from CRSEM.contracts import ParameterBatch
from scripts.run_model import infer_model_type

params, metrics = ParameterBatch.from_file("calibration_results/params.json")
model_type = infer_model_type(params, metrics)

point_driver = driver.to_point_driver(keep_rivers=True)
result = run_parameter_batch(
    model_type=model_type,
    source=point_driver,
    params=params,
    run_method="run_hillslope_river",
)

ds = result.to_dataset()
ssf_pred = ds["SSF_pred"]
```

## 9. Units

| Variable | Unit |
|------|------|
| s_area | hectare (`ha`) |
| SSF | `t/month` |
| Q | `m^3/s` |
| Pre | `mm/month` |
| T | `degC` |
| E_hillslope | `t/ha/month` |
| R_rain`, `R_melt` | `MJ*mm/(ha*h*month)` |

## 10. FAQ

### Q1: Why is NSE negative?

Possible causes:

- initial parameter ranges are poor
- iteration count is too small
- forcing and observation periods are not aligned

Typical fixes:

- increase `--maxiter`
- check the time ranges of all input files

### Q2: Why is predicted SSF at the wrong scale?

Check:

- `s_area` is in hectares
- observed `SSF` is in `t/month`
- `Pre` is in `mm/month`

### Q3: What if gridded mode runs out of memory?

Try:

- reducing the spatial or temporal domain
- running in chunks
- using a machine with more memory

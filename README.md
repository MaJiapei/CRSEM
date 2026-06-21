# CRSEM

CRSEM is a cold-region soil erosion and sediment delivery modeling workflow built around monthly gridded drivers. It supports prepared NetCDF inputs, parameter calibration, basin-scale simulation, and vegetation attribution workflows for CRSEM/RUSLE-style model runs.

Chinese documentation is available in [README.zh-CN.md](README.zh-CN.md). Detailed guides are in [docs/USER_GUIDE.en.md](docs/USER_GUIDE.en.md), [docs/USER_GUIDE.md](docs/USER_GUIDE.md), [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md), and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repository Contents

- Python package: [CRSEM](CRSEM)
- Command-line workflows: [scripts](scripts)
- Tests: [tests](tests)
- Example configuration: [config](config)
- Bundled lightweight real-case example: [example/zhimenda_sample](example/zhimenda_sample)

## Bundled Zhimenda Example

The repository includes a lightweight Zhimenda basin example derived from prepared gridded drivers. It is stored as a 1x1 basin-mean sample so the example can be kept in GitHub while still using real Zhimenda forcing, observations, and calibrated parameters.

- [static.nc](example/zhimenda_sample/drivers/static.nc)
- [dynamic.nc](example/zhimenda_sample/drivers/dynamic.nc)
- [observations.nc](example/zhimenda_sample/drivers/observations.nc)
- [params_1982_2000_kge_pbias_m120.json](example/zhimenda_sample/params_1982_2000_kge_pbias_m120.json)

The full gridded Zhimenda `dynamic.nc` is about 1.2 GB and is intentionally not tracked. Large gridded drivers, DEM products, intermediate GIS files, and model output NetCDF files should be published through release assets or a data archive rather than committed to this repository.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional dependency:

- `rioxarray` is only needed when spatial reprojection is required during data preparation.

## Quick Start

Run the bundled Zhimenda example:

```bash
python scripts/run_model.py \
  --static-nc example/zhimenda_sample/drivers/static.nc \
  --dynamic-nc example/zhimenda_sample/drivers/dynamic.nc \
  --observations-nc example/zhimenda_sample/drivers/observations.nc \
  --params-file example/zhimenda_sample/params_1982_2000_kge_pbias_m120.json \
  --station-name zhimenda \
  --run-mode point \
  --start-year 1982 \
  --end-year 2019
```

Add `--output-file example/zhimenda_sample/output/model_output.nc` if you want to save the model output locally. Output files are ignored by Git.

Run a short calibration against the same example:

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/zhimenda_sample/drivers/static.nc \
  --dynamic-nc example/zhimenda_sample/drivers/dynamic.nc \
  --observations-nc example/zhimenda_sample/drivers/observations.nc \
  --station-name zhimenda \
  --calibration-start 1982 \
  --calibration-end 2000 \
  --maxiter 3
```

Calibration defaults to `point` mode. If you switch to `--run-mode gridded`, progress plotting is disabled and `--workers` becomes available.

## Tests

```bash
python -m pytest tests -q
```

For a fast smoke test on the bundled real dataset:

```bash
python -m pytest tests/test_real_data_smoke.py -q
```

## Publishing Notes

Before a public release, confirm:

- the project license
- whether any bundled data can be redistributed publicly
- whether a release tag and citation metadata are needed

Without a `LICENSE` file, other users can view the code but do not have clear reuse rights.

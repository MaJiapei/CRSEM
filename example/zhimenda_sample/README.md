# Zhimenda Lightweight Example

This directory contains a lightweight Zhimenda basin example for CRSEM/CRSEM.

The files are derived from the prepared gridded Zhimenda drivers by averaging valid basin cells into a 1x1 basin-mean sample. This keeps the example small enough for GitHub while preserving real monthly forcing, observations, NDVI ensemble members, and calibrated CRSEM parameters.

## Files

- `drivers/static.nc`: basin-mean static factors
- `drivers/dynamic.nc`: monthly `T`, `Pre`, and four-member `NDVI` series for 1982-01 to 2019-11
- `drivers/observations.nc`: monthly discharge and suspended sediment flux observations for 1982-01 to 2019-11
- `params_1982_2000_kge_pbias_m120.json`: calibrated CRSEM parameter ensemble
- `metadata.json`: source and sample metadata

## Run

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

Use `--output-file example/zhimenda_sample/output/model_output.nc` to save the result locally. The output directory is ignored by Git.

## Full Gridded Data

The full Zhimenda gridded `dynamic.nc` is about 1.2 GB and is intentionally excluded from this repository. Publish full gridded drivers through a release asset or external data archive when needed.

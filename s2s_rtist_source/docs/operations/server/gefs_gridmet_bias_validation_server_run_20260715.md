# GEFS-gridMET Bias Validation Server Run

Scope: five confirmed sites, ten 2024 decision cycles, local D through D+6, GEFS `geavg` versus gridMET. This run performs weather-data validation only. It does not train a surrogate model or generate SWAP labels.

## Files

- `src/s2s_rtist/weather/gefs_gridmet_bias.py` (library; original: `gefs_gridmet_bias_validation_v1.py`)
- `scripts/diagnostics/run_gefs_gridmet_bias_validation_v1.py` (CLI id: `gefs-gridmet-bias`)
- `requirements/requirements_gefs_gridmet_bias_validation_v1.txt`
- `tests/test_gefs_gridmet_bias_validation_v1.py`

Run with:

```bash
python3 project_cli.py run gefs-gridmet-bias -- <args>
```

## Expected workload

- Downloads six complete 2024 gridMET files, approximately 0.8 GB total, only when the existing files do not cover the required August dates.
- Downloads selected GRIB messages from ten GEFS `geavg` cycles. Mini-GRIB files are deleted after point extraction unless `--keep-grib` is supplied.
- Writes resumable point CSVs under the run cache. Reusing the same run ID resumes completed cycle/lead pairs.

## Output contract

The run directory must contain:

- `gefs_download_manifest.csv`
- `gefs_point_records.csv`
- `gefs_daily_weather.csv`
- `gridmet_reference_daily_long.csv`
- `gefs_gridmet_paired_daily.csv`
- `bias_metrics_overall.csv`
- `bias_metrics_by_lead_day.csv`
- `bias_metrics_by_site.csv`
- `bias_metrics_by_reference_condition.csv`
- `precipitation_event_metrics.csv`
- `precipitation_event_metrics_by_lead_day.csv`
- `gefs_gridmet_bias_validation_v1.md`
- `run_metadata.json`

Expected paired rows: `10 cycles x 5 sites x 7 days x 6 variables = 2100`.

## Interpretation limits

- gridMET is a gridded reference, not station-observed truth.
- The experiment evaluates official `geavg`, not individual ensemble members or spread calibration.
- TMAX/TMIN intervals crossing local midnight are assigned by interval midpoint.
- Precipitation and radiation intervals crossing local midnight are allocated uniformly by overlap duration.

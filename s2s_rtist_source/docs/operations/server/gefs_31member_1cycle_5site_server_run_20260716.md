# GEFS 31-Member One-Cycle Five-Site Smoke Run

## Scope

- Cycle: 2024-07-16 00 UTC
- Members: gec00 and gep01-gep30
- Sites: P1, P2, P3, P4, P15
- Local forecast window: D through D+6
- Reference: existing complete 2024 gridMET files
- Metrics: ensemble-mean error, spread, CRPS, P10-P90 coverage, min-max coverage, and precipitation Brier score

This run does not train a surrogate and does not generate SWAP labels.

## Expected Counts

- GEFS member products: 31
- Lead files per member: 60 (f003-f180)
- Download manifest rows: 1860
- Member daily weather rows: 1085 (31 x 5 x 7)
- Paired member-variable rows: 6510 (1085 x 6)
- Ensemble observation rows: 210 (5 x 7 x 6)

## Server Command

Set the project root and dependencies:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source

export PYTHONPATH="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.gefs_bias_deps_20260715:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
```

Run the focused tests:

```bash
python3 -m unittest \
tests.test_gefs_member_gridmet_validation_v1 \
tests.test_gefs_gridmet_bias_validation_v1 \
-v
```

Start the smoke run:

```bash
nohup python3 \
scripts/diagnostics/run_gefs_member_gridmet_validation_v1.py \
--run-id gefs_31member_1cycle_5site_20260716_v1 \
--decision-dates 2024-07-16 \
--gridmet-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_gridmet_bias_validation_v1/gefs_gridmet_bias_10cycle_5site_20260715_v1/gridmet_complete_2024 \
--workers 6 \
--timeout 120 \
--retries 5 \
> /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_31member_1cycle_5site_20260716_v1.nohup.log \
2>&1 &

echo "PID=$!"
```

Monitor progress:

```bash
tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_31member_1cycle_5site_20260716_v1.nohup.log
```

## Completion Check

```bash
RUN_DIR="/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_member_gridmet_validation_v1/gefs_31member_1cycle_5site_20260716_v1"

cat "$RUN_DIR/run_metadata.json"

wc -l \
"$RUN_DIR/gefs_member_daily_weather.csv" \
"$RUN_DIR/paired_members.csv" \
"$RUN_DIR/ensemble_observations.csv" \
"$RUN_DIR/gefs_member_download_manifest.csv"
```

CSV line counts include one header row, so the expected values are 1086, 6511, 211, and 1861.

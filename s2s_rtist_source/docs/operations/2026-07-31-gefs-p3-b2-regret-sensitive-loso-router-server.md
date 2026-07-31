# P3 B2 Regret-Sensitive LOSO Router Server Commands

Project root:

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
```

This screen reads only P2/P4 rows from 2015-2019. It must not be pointed at
P3, 2020, 2021, or 2024 evaluation artifacts.

## 1. Verify And Test The Package

Place the ZIP in the project root with the GUI file-transfer workflow, then
run:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
printf '%s  %s\n' 'DE68A9D5D4319D6669F26A879B433166330876F9E66A3EDB656FE9CB1127C081' 'gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.zip' | sha256sum -c -
test ! -e /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1
mkdir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1
unzip -q /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.zip -d /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1
env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1/src python3 -m unittest tests.test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1
```

Install only the two new B2 files into the existing project. The reusable
modules in the ZIP are for isolated package testing and are not copied over
the server versions:

```bash
install -m 0644 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1/scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py
install -m 0644 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_b2_20260731_v1/tests/test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/tests/test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -m unittest tests.test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1
```

## 2. Launch In The Background

The runner fails closed if the output directory already exists.

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py --dataset-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_exact_schedule_surrogate_dataset_v1 --protocol-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2021_dual_gate_five_member_protocol_v1 --development-screen-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_strict_loso_cross_year_gate_screen_v1 --final-model-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2021_final_full_history_refit_v1r2 --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1 --device cuda > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.log 2>&1 & echo $! | tee /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.pid
```

## 3. Monitor

```bash
ps -fp "$(cat /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.pid)"
tail -f /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.log
nvidia-smi
grep -E '^b2_fold=' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.log | tail -n 20
grep -nE 'Traceback|RuntimeError|ValueError|CUDA out of memory|Killed' /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.log || true
tail -n 80 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.log
find /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1 -maxdepth 1 -type f -printf '%f\n' | sort
```

## 4. Read The Result

Audit status, leakage boundary, and all nine gates:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
python3 - <<'PY'
import json
from pathlib import Path

path = Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1/gefs_p3_b2_regret_sensitive_router_screen_audit_v1.json')
audit = json.loads(path.read_text(encoding='utf-8'))
print('STATUS:', audit['status'])
print('DEVELOPMENT/FINAL:', audit['development_gate_passed'], audit['final_gate_written'])
print('RETAINED P3/2020/2021/2024:', audit['P3_rows_retained'], audit['year_2020_rows_retained'], audit['year_2021_rows_retained'], audit['year_2024_rows_retained'])
print('SCOPE FILTER:', json.dumps(audit['scope_filter'], indent=2, sort_keys=True))
print('NINE GATES:', json.dumps(audit['development_conditions'], indent=2, sort_keys=True))
print('FALLBACKS:', json.dumps(audit['fold_fallback_reasons'], indent=2, sort_keys=True))
PY
```

Full development-gate row, eight-fold metrics, and per-site metrics:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1')
print('=== DEVELOPMENT GATE ===')
print(pd.read_csv(root / 'gefs_p3_b2_regret_sensitive_development_gate_v1.csv').T.to_string(header=False))
print('\n=== EIGHT FOLDS: B2 VS ALWAYS-P15 ===')
folds = pd.read_csv(root / 'gefs_p3_b2_regret_sensitive_per_fold_metrics_v1.csv')
print(folds.loc[folds['policy'].isin(['p3_b2_regret_sensitive_router', 'always_P15']), ['protocol_fold', 'held_out_site', 'policy', 'cycle_count', 'mean_regret_7d', 'maximum_regret_7d', 'false_positive_count', 'predicted_60mm_count', 'effective_P1_route_count']].sort_values(['protocol_fold', 'policy']).to_string(index=False))
print('\n=== PER SITE ===')
print(pd.read_csv(root / 'gefs_p3_b2_regret_sensitive_per_site_metrics_v1.csv').to_string(index=False))
PY
```

Routing counts and the worst selected cycles:

```bash
python3 - <<'PY'
from pathlib import Path
import pandas as pd

path = Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1/gefs_p3_b2_regret_sensitive_validation_decisions_v1.csv')
rows = pd.read_csv(path)
b2 = rows.loc[rows['policy'].eq('p3_b2_regret_sensitive_router')].copy()
print('=== ROUTES BY FOLD ===')
print(b2.groupby(['protocol_fold', 'held_out_site', 'selected_source_site_id']).size().rename('cycles').reset_index().to_string(index=False))
print('\n=== WORST B2 CYCLES ===')
print(b2.nlargest(20, 'regret_7d')[['protocol_fold', 'held_out_site', 'decision_date', 'selected_source_site_id', 'gate_probability_p1', 'gate_fallback', 'p1_selected_irrigation_mm', 'p15_selected_irrigation_mm', 'true_best_irrigation_mm', 'selected_irrigation_mm', 'regret_7d', 'false_positive', 'predicted_60mm']].to_string(index=False))
PY
```

Weight normalization and manifest integrity:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
import pandas as pd

root = Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1')
samples = pd.read_csv(root / 'gefs_p3_b2_regret_sensitive_training_samples_v1.csv')
usable = samples.loc[samples['usable_for_gate'].astype(bool)]
print('=== ACTIONABLE WEIGHT AUDIT ===')
print(usable.groupby(['protocol_fold', 'training_site'])['normalized_regret_weight'].agg(['count', 'mean', 'min', 'max']).to_string())
print('\n=== MANIFEST HASH AUDIT ===')
manifest = pd.read_csv(root / 'gefs_p3_b2_regret_sensitive_router_screen_manifest_v1.csv')
for row in manifest.loc[manifest['role'].str.startswith('output_')].itertuples(index=False):
    path = root / str(row.path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(row.role, digest == str(row.sha256), path.name)
PY
```

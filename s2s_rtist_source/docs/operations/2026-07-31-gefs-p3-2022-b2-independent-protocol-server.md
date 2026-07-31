# P3 2022 B2 Independent Protocol Server Commands

Project root:

```text
/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
```

This package freezes the protocol only. It does not download ERA5 or GEFS,
read 2022 labels, run inference, generate recommendations, or run SWAP. Do
not authorize the next data stage until the returned five protocol artifacts
have been verified locally.

## 1. Verify And Test The ZIP

Place the ZIP in the project root with the GUI transfer workflow, then run:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
printf '%s  %s\n' '82215AB1EA78FDA7CF193BEBEC93817785BD28977219B68644648DABAA655E1F6' 'gefs_p3_2022_b2_independent_protocol_freeze_20260731_v1.zip' | sha256sum -c -
test ! -e /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1
mkdir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1
unzip -q /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_2022_b2_independent_protocol_freeze_20260731_v1.zip -d /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1
env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1/src python3 -m unittest -v tests.test_freeze_gefs_p3_2022_b2_independent_protocol_v1
```

Install only the two code files into the existing project:

```bash
install -m 0644 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1/scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py
install -m 0644 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/.package_verify_gefs_p3_2022_b2_independent_protocol_20260731_v1/tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
env PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -m unittest -v tests.test_freeze_gefs_p3_2022_b2_independent_protocol_v1
```

## 2. Freeze The Protocol

Run synchronously. The output directory must not already exist:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
python3 /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py \
  --b2-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_b2_regret_sensitive_loso_router_screen_v1 \
  --dual-protocol-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2021_dual_gate_five_member_protocol_v1 \
  --final-model-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2021_final_full_history_refit_v1r2 \
  --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2022_b2_independent_protocol_v1 \
  | tee /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_2022_b2_independent_protocol_freeze_20260731_v1.log
```

## 3. Read And Verify The Result

These are the standard result-viewing commands for this experiment:

```bash
ROOT=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2022_b2_independent_protocol_v1
python3 - <<'PY'
import csv, hashlib, json
from pathlib import Path

root = Path('/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_2022_b2_independent_protocol_v1')
audit = json.loads((root / 'gefs_p3_2022_b2_independent_protocol_audit_v1.json').read_text())
print('STATUS:', audit['status'])
print('MANDATORY_GATE:', audit['mandatory_gate_passed'])
print('TARGET:', audit['target_site'], audit['target_year'])
print('ZERO_ACCESS:', {k: v for k, v in audit.items() if k.endswith('_rows_read') or k.endswith('_rows_or_labels_read') or k.endswith('_network_requests') or k.endswith('_model_inference_rows') or k.endswith('_recommendation_rows')})
protocol = json.loads((root / 'gefs_p3_2022_b2_independent_protocol_v1.json').read_text())
print('CANDIDATE:', protocol['candidate_policy'])
print('MEMBERS:', protocol['operational_weather']['members'])
print('GRID_MM:', protocol['evaluation']['irrigation_candidates_mm'])
print('FEATURES:', protocol['routing_contract']['feature_names'])
print('PROMOTION_CONDITIONS:', len(protocol['evaluation']['independent_promotion_conditions']))
stages = list(csv.DictReader((root / 'gefs_p3_2022_b2_stage_registry_v1.csv').open(newline='')))
print('STAGES:', [(row['stage_order'], row['stage_id'], row['eligible_now']) for row in stages])
components = list(csv.DictReader((root / 'gefs_p3_2022_b2_component_registry_v1.csv').open(newline='')))
print('COMPONENTS:', [(row['component_role'], row['component_id'], row['sha256']) for row in components])
manifest = list(csv.DictReader((root / 'gefs_p3_2022_b2_independent_protocol_manifest_v1.csv').open(newline='')))
for row in manifest:
    if row['role'].startswith('output_'):
        path = root / row['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print('MANIFEST:', row['role'], digest == row['sha256'], path.name)
PY
```

The required result is status
`p3_2022_b2_independent_protocol_frozen_before_target_data_access`, all
zero-access fields equal to zero, stage 1 eligible and stages 2-8 ineligible,
five output files present, and every output hash matching the manifest.

## 4. Package The Five Returned Artifacts

After the local GUI transfer of the output directory back to Windows, keep the
server-side archive as a reproducibility handoff:

```bash
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source
tar -czf gefs_p3_2022_b2_independent_protocol_outputs_20260731_v1.tar.gz \
  -C site_general_surrogate_eval gefs_p3_2022_b2_independent_protocol_v1
sha256sum gefs_p3_2022_b2_independent_protocol_outputs_20260731_v1.tar.gz
tar -tzf gefs_p3_2022_b2_independent_protocol_outputs_20260731_v1.tar.gz
```

Do not begin the 2022 ERA5 acquisition, GEFS acquisition, schedule
materialization, recommendation, or shared SWAP stage until the returned
protocol JSON, component registry, stage registry, audit, and manifest pass
the same checks locally.

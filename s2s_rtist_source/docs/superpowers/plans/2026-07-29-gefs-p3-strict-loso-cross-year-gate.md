# P3 Strict-LOSO Cross-Year Expert Gate Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this
> plan task by task. Keep the checkbox state current and run each stated test
> before proceeding.

**Goal:** Produce a leakage-audited four-fold P3 routing screen between
independently retrained P1 and P15 composite experts, then decide whether the
method earns access to a new 2021 independent test year.

**Architecture:** Add one target-free protocol freezer and one resumable
training/evaluation runner. Each rolling fold trains two independent v4-style
four-output source experts using only earlier P1 or P15 rows, freezes their
composite checkpoints, produces historical P3 routing labels from fixed-list
SWAP outcomes, fits one deterministic NumPy logistic gate, and evaluates the
next P3 year. The runner compares the gate with `always_P15`, reports all four
years, and applies the pre-registered macro safety gate without touching 2021
or 2024.

**Tech stack:** Python 3.10, standard library, NumPy, pandas, PyTorch, existing
teacher-aligned model and loss modules, `unittest`. Do not add scikit-learn.

---

## Locked Decisions

- Source sites: P1 and P15 only.
- Target site: P3 only.
- Source architecture: existing
  `GefsTeacherAlignedDecoupledSiteExpert`, instantiated independently per
  source and fold.
- Source output heads: net gain, AET, seven-day fixed 0--100 cm VWC, and
  residual water-balance flux.
- Source checkpoint track: composite only.
- `lambda_balance=1.0`; no weight grid.
- Outer folds: validate 2016, 2017, 2018, and 2019 using only earlier years.
- Inner source checkpoint split: last
  `max(1, ceil(0.20 * N))` date-sorted historical source cycles.
- Gate: deterministic NumPy Newton/IRLS L2 logistic regression; fixed threshold
  `0.5`; exact tie and any audited failure fall back to P15.
- Baseline: `always_P15`. `always_P1` and label-using Oracle are diagnostics.
- Formal decisions use the eight frozen SWAP candidates only. No interpolation
  and no continuous optimization occur in this screen.
- 2019 is retrospective development validation, not a blind test.
- 2020 is excluded, 2021 remains unopened pending the development gate, and
  2024 remains reserved for TTA.

## File Map

- Create:
  `scripts/evaluation/freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py`
- Create:
  `tests/test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py`
- Create:
  `scripts/training/run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`
- Create:
  `tests/test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`
- Package after tests:
  `gefs_p3_strict_loso_cross_year_gate_20260729_v1.zip`

Do not modify the existing v5r2, continuous-optimization, SWAP-evaluation, or
2024 TTA artifacts. Do not overwrite the earlier rolling weak-site protocol.

---

### Task 1: Freeze The Target-Free Four-Fold Protocol

**Files:**

- Create:
  `tests/test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py`
- Create:
  `scripts/evaluation/freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py`

- [ ] **Step 1: Write failing fold and isolation tests**

Cover these invariants with a synthetic five-site, five-year, eight-candidate
fixture:

- exactly four forward folds;
- each validation year is strictly later than every training year;
- P1 registry rows contain only P1 source scope;
- P15 registry rows contain only P15 source scope;
- P3 is target-only and never source training data;
- source checkpoint track is exactly `composite` and
  `lambda_balance == 1.0`;
- assignments contain no formal target column;
- the freeze stage reports zero 2020, 2021, and 2024 rows read.

Run and confirm RED:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1 -v
```

Expected: import failure because the freezer does not exist yet.

- [ ] **Step 2: Implement protocol construction**

Reuse source-column and dataset-contract validation from
`freeze_gefs_teacher_aligned_rolling_temporal_protocol_v1.py`, but define the
four approved folds and P1/P15 source registry locally. The freezer may read
2015--2019 row identifiers, years, sites, dates, and irrigation amounts. It
must not load target values.

Required outputs:

```text
gefs_p3_strict_loso_fold_assignments_v1.csv
gefs_p3_strict_loso_fold_summary_v1.csv
gefs_p3_strict_loso_source_expert_registry_v1.csv
gefs_p3_strict_loso_gate_feature_contract_v1.json
gefs_p3_strict_loso_protocol_v1.json
gefs_p3_strict_loso_protocol_audit_v1.json
gefs_p3_strict_loso_protocol_manifest_v1.json
```

The feature contract must resolve the exact dataset columns for DVS, root
depth, fixed-layer VWC, and the six seven-day GEFS aggregates. Fail if a
required field is missing instead of silently changing the feature set.

- [ ] **Step 3: Run the protocol tests and existing regression tests**

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1 tests.test_freeze_gefs_teacher_aligned_rolling_temporal_protocol_v1 tests.test_audit_gefs_composite_expert_cross_site_complementarity_v1 -v
```

Expected: all PASS.

### Task 2: Add Fold-Local Independent Composite Expert Training

**Files:**

- Modify:
  `tests/test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`
- Create:
  `scripts/training/run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`

- [ ] **Step 1: Test chronological inner splitting**

Add tests for `chronological_source_split(frame, fraction=0.20)`:

- split by distinct decision cycle, never by candidate row;
- retain all eight candidates in both partitions;
- use exactly `max(1, ceil(0.20 * N))` latest cycles for selection;
- fitting cycles are strictly earlier;
- reject fewer than two total cycles.

- [ ] **Step 2: Test physical fold loading**

Implement and test a loader that first reads only row identity/year columns,
then physically skips CSV rows later than the current outer validation year
when loading features and targets. Assert that a 2016 fold never materializes
2017--2019 target rows. The audit must distinguish lightweight year-index reads
from target-bearing row reads.

- [ ] **Step 3: Implement one composite-only source training primitive**

Reuse:

- `fit_teacher_preprocessing` and `transform_teacher`;
- `GefsTeacherAlignedDecoupledSiteExpert`;
- `teacher_aligned_pretraining_loss`;
- `validation_metrics` and `selection_score`.

Do not call the v4 four-track/four-weight sweep. The new primitive trains one
model with `lambda_balance=1.0`, saves only the best composite checkpoint, and
early-stops only on composite staleness. Fixed defaults:

```text
seed=20260729
epochs=3000
minimum_epochs=300
patience=300
batch_size=128
learning_rate=1e-3
weight_decay=1e-4
composite_min_delta=1e-6
```

P1 and P15 preprocessing must be fitted independently on their own fitting
partitions. Checkpoint metadata must record the exact fitting and selection
years/dates, source hashes, architecture, four outputs, and zero P3 rows.

- [ ] **Step 4: Test source isolation and checkpoint selection**

Use mocked short training loops to prove:

- changing P3 values cannot change a source checkpoint;
- P1 and P15 weights and preprocessing are distinct objects;
- no future year enters fitting or selection;
- a worse composite epoch cannot replace the incumbent;
- profit/direct/ranking checkpoints are never written;
- deterministic reruns produce identical checkpoint metadata and predictions.

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1 -v
```

### Task 3: Build P3 Routing Samples And Deterministic Gate

**Files:**

- Modify:
  `scripts/training/run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`
- Modify:
  `tests/test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`

- [ ] **Step 1: Test source-routed P3 inference**

Apply P3 rows through each source expert's source-only preprocessing. The
independent model ignores site embeddings, but the test must still reject any
attempt to refit scaling on P3. Verify each P3 cycle has exactly eight ordered
candidates and one recommendation per expert.

- [ ] **Step 2: Test actionable labels**

For historical P3 cycles:

- identical expert recommendations produce no gate-fitting row;
- unequal recommendations compare SWAP-realized target net gains at the two
  selected candidates;
- strict P1/P15 wins become binary labels;
- equal realized gain is excluded;
- target values never appear in the returned gate feature matrix.

- [ ] **Step 3: Implement the fixed physical feature summary**

Build one candidate-invariant row per cycle using exactly:

```text
predecision_dvs
predecision_crop_root_depth_cm
predecision_soil_vwc_0_100cm
sum(weather_precipitation_mm_day01..day07)
mean(weather_temperature_min_c_day01..day07)
mean(weather_temperature_max_c_day01..day07)
mean(weather_actual_vapor_pressure_kpa_day01..day07)
mean(weather_wind_speed_m_s_day01..day07)
mean(weather_solar_kj_m2_day_day01..day07)
```

Assert candidate invariance before taking the zero-irrigation row as the cycle
representative. Fit feature means and standard deviations on historical P3
actionable training rows only.

- [ ] **Step 4: Implement and test NumPy Newton/IRLS logistic regression**

Use stable sigmoid/log-loss calculations and minimize:

```text
sum(binary_cross_entropy) + 0.5 * sum(non_intercept_weights ** 2)
```

Lock `max_iter=100`, parameter tolerance `1e-10`, threshold `0.5`, and P15 on
exact tie. Tests must compare coefficients/probabilities with a small known
fixture, prove determinism, and exercise singular/non-finite failures.

- [ ] **Step 5: Implement audited fallback**

Return `always_P15` with an explicit reason when there are fewer than eight
actionable cycles, only one label class, non-finite features, a singular solve,
or non-convergence. Never substitute another model or tune the threshold.

### Task 4: Complete The Four-Fold Screen And Development Gate

**Files:**

- Modify:
  `scripts/training/run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`
- Modify:
  `tests/test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py`

- [ ] **Step 1: Implement fold orchestration and resume**

Run folds in chronological order and sources in fixed `P1`, `P15` order.
Write each completed source checkpoint and fold gate atomically. `--resume` may
reuse an artifact only after hashes and completion audits pass; an existing
incomplete artifact must be recomputed or fail with a precise message. A normal
run must refuse to overwrite an existing output directory.

- [ ] **Step 2: Produce per-cycle policies**

For every validation cycle, write recommendations and realized gains for:

- `always_P1`;
- `always_P15`;
- `p3_logistic_gate`;
- `two_expert_label_oracle` diagnostic.

Report regret against the true best of the same eight SWAP candidates. Preserve
false positives, missed beneficial decisions, 60 mm counts, routed expert,
gate probability, fallback state, and expert disagreement.

- [ ] **Step 3: Implement the frozen macro gate**

Compare `p3_logistic_gate` with `always_P15` and require all six approved
conditions: macro regret improvement by at least `1e-6`, non-worse regret in at
least three of four folds, non-worse worst-fold maximum regret, non-worse total
false positives, non-worse total 60 mm count, and at least one P1 route.

Required outputs:

```text
gefs_p3_strict_loso_source_checkpoint_summary_v1.csv
gefs_p3_strict_loso_source_predictions_v1.csv
gefs_p3_strict_loso_gate_training_samples_v1.csv
gefs_p3_strict_loso_validation_decisions_v1.csv
gefs_p3_strict_loso_per_fold_metrics_v1.csv
gefs_p3_strict_loso_development_gate_v1.csv
gefs_p3_strict_loso_screen_audit_v1.json
gefs_p3_strict_loso_screen_manifest_v1.csv
```

The audit status must end in either
`p3_strict_loso_cross_year_gate_passed_pending_2021_freeze` or
`p3_strict_loso_cross_year_gate_failed_frozen_negative`. It must never download
or read 2021/2024 data.

- [ ] **Step 4: Add a CPU end-to-end fixture test**

Patch the expensive source trainer with deterministic tiny models while keeping
real fold, feature, label, metric, fallback, manifest, and gate code active.
Assert all output counts and both pass/fail macro-gate paths.

### Task 5: Verification And Regression

- [ ] **Step 1: Run syntax and focused tests**

```powershell
python -m py_compile scripts\evaluation\freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py scripts\training\run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py tests\test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1.py tests\test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_freeze_gefs_p3_strict_loso_cross_year_gate_protocol_v1 tests.test_run_gefs_p3_strict_loso_cross_year_gate_screen_v1 -v
```

- [ ] **Step 2: Run related regression tests**

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_gefs_teacher_aligned_decoupled_site_experts_v4 tests.test_gefs_teacher_aligned_three_output_pretraining_v2 tests.test_freeze_gefs_teacher_aligned_rolling_temporal_protocol_v1 tests.test_run_gefs_teacher_aligned_rolling_temporal_branch_screen_v1 tests.test_audit_gefs_composite_expert_cross_site_complementarity_v1 -v
```

- [ ] **Step 3: Run a local no-training protocol smoke**

Use the extracted frozen dataset copy. Inspect the JSON audit and verify
`training_performed=false`, `target_columns_loaded=[]`, four folds, two source
experts, composite-only registry, and zero 2020/2021/2024 rows read.

### Task 6: Package And Verify The Server Bundle

- [ ] **Step 1: Build a minimal ZIP**

Include the two new scripts, two new tests, required reusable source/model
modules, and any imported tests/helpers needed for unpacked execution. Preserve
Linux `/` member paths and exclude datasets, checkpoints, logs, PID files, and
earlier output directories.

- [ ] **Step 2: Verify archive safety and an unpacked test run**

Reject absolute members, drive-prefixed members, `..` traversal, backslashes,
duplicates, and case-insensitive collisions. Extract to a new temporary
directory and run both new test modules there.

- [ ] **Step 3: Record ZIP path and SHA256**

Expected local name:

```text
D:\study\s2s_rtist_source\gefs_p3_strict_loso_cross_year_gate_20260729_v1.zip
```

### Task 7: Deliver Independent Server Commands

- [ ] **Step 1: Give extraction and test commands**

Every command must use the complete server path and `python3`. Do not define
`ROOT`, `OUT`, `LOG`, or other temporary shell variables.

- [ ] **Step 2: Give the no-training protocol command and audit reader**

The first server action freezes the protocol only. Provide a Python heredoc
that prints audit status, target columns loaded, fold summary, and source
registry before any GPU run.

- [ ] **Step 3: Give the resumable GPU training command**

Use this operational shape with final paths filled explicitly:

```bash
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH=/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source:/media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/src python3 -u /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/scripts/training/run_gefs_p3_strict_loso_cross_year_gate_screen_v1.py --dataset-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_exact_schedule_surrogate_dataset_v1 --protocol-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_strict_loso_cross_year_gate_protocol_v1 --output-dir /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/site_general_surrogate_eval/gefs_p3_strict_loso_cross_year_gate_screen_v1 --device cuda > /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/gefs_p3_strict_loso_cross_year_gate_screen_v1.stdout.log 2>&1 &
```

Provide a separate `--resume` form using the same absolute paths.

- [ ] **Step 4: Give monitoring and final-result commands**

Include PID capture, `ps`, `tail -f`, GPU utilization, a search for
`early_stop|termination|Traceback`, output listing, and a final Python reader
that prints:

- audit and development-gate status;
- all four fold metrics against `always_P15`;
- macro gate conditions;
- source checkpoint epochs and composite scores;
- fallback reasons and route counts;
- three worst cycles per policy;
- the explicit `2021_release_allowed` value.

Do not provide a 2021 download or evaluation command unless the completed
four-fold audit passes.


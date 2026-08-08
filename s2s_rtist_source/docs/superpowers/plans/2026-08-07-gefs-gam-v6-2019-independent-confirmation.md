# GEFS GAM v6 2019 Independent Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage, hash-separated 2019 confirmation workflow for the frozen v6 confidence-protected GAM calibration rule.

**Architecture:** Stage 1 refits the unchanged GAM/B-spline expert family and v4 source-only gate on 2015-2018 rows, generates 2019 decisions without reading target columns, applies the frozen v5/v6 rules, and writes a recommendation manifest. Stage 2 verifies that manifest before loading 2019 targets, evaluates peak distance, runs or reuses only exact changed-cycle SWAP endpoints, expands results to a 69-cycle paired ledger, and applies the joint confirmation gate.

**Tech Stack:** Python 3, NumPy, pandas, existing hierarchical GAM/B-spline helpers, existing checkpoint-isolated SWAP runner, `unittest`, JSON and SHA256 manifests.

**Repository rule:** Do not stage or commit unless the user explicitly requests it. Preserve all unrelated worktree changes. The user explicitly authorized committing the necessary files on 2026-08-08.

---

### Task 1: Freeze the two-stage protocol

**Files:**
- Create: `docs/superpowers/specs/2026-08-07-teacher-guided-gam-v6-2019-independent-confirmation-v1.json`
- Test: `tests/test_run_gefs_gam_v6_2019_recommendation_freeze_v1.py`
- Test: `tests/test_run_gefs_gam_v6_2019_paired_swap_qualification_v1.py`

- [x] **Step 1: Write failing protocol tests**

Assert the exact protocol ID, 69-cycle/5-site boundary, 2015-2018-only fitting, `rolling_to_2018` penalty inheritance, frozen v4 gate constants, 2.5 mm trust radius, 5.0 mm limited-candidate LCB threshold, no search, no interpolation, no 2024 access, and all mandatory performance gates.

- [x] **Step 2: Run the focused protocol tests and verify failure**

Run:

```powershell
python -m unittest tests.test_run_gefs_gam_v6_2019_recommendation_freeze_v1 tests.test_run_gefs_gam_v6_2019_paired_swap_qualification_v1 -v
```

Expected: import or missing-protocol failures.

- [x] **Step 3: Add the frozen JSON protocol**

Define `stage_1_recommendation_freeze` and `stage_2_exact_swap_confirmation` sections. Freeze the source years, target year, site list, expected counts, expert penalty inheritance, gate constants, v5/v6 rules, fallback offsets, exact-reuse tolerance, all execution/performance conditions, and pass/fail actions from the approved design.

- [x] **Step 4: Re-run protocol tests**

Expected: protocol assertions pass while runner imports remain red.

### Task 2: Implement target-blind 2019 row loading and deployment expert fitting

**Files:**
- Create: `scripts/training/run_gefs_gam_v6_2019_recommendation_freeze_v1.py`
- Test: `tests/test_run_gefs_gam_v6_2019_recommendation_freeze_v1.py`

- [x] **Step 1: Add failing unit tests**

Cover these interfaces:

```python
validate_protocol(protocol)
load_feature_rows_without_targets(dataset_path, index, selector)
inherit_rolling_2018_penalties(summary_path, target_site)
apply_frozen_v5_v6_rule(frame)
```

The loader test places sentinel strings in every target column and proves the 2019 feature loader never parses them. The penalty test requires exactly one `GAM-VC` row for `holdout_<site>_rolling_to_2018`. The rule test covers unchanged, unlimited, limited-pass, and limited-fallback decisions.

- [x] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
python -m unittest tests.test_run_gefs_gam_v6_2019_recommendation_freeze_v1 -v
```

Expected: missing functions.

- [x] **Step 3: Implement protocol validation and target-blind loading**

Read only cycle keys plus the frozen GAM input and 19 gate feature source columns for selected 2019 rows. Reject post-2019 rows before parsing features and reject any requested target column. Validate complete eight-row cycles from the index without materializing target values.

- [x] **Step 4: Implement five deployment folds**

For each target site, fit the four-source baseline and four delete-one-source GAM-VC experts on other-site 2015-2018 rows. Reuse existing robust-development fitting helpers and inherit the target site's `rolling_to_2018` penalty row. Save coefficient files and model audits under `folds/holdout_<site>_rolling_to_2019/`.

- [x] **Step 5: Run tests**

Expected: loader, penalty, and deployment-fit tests pass.

### Task 3: Fit the frozen source-only gate and freeze recommendations

**Files:**
- Modify: `scripts/training/run_gefs_gam_v6_2019_recommendation_freeze_v1.py`
- Test: `tests/test_run_gefs_gam_v6_2019_recommendation_freeze_v1.py`

- [x] **Step 1: Add failing recommendation tests**

Use small synthetic four-source cycles to verify source-only anchor selection, positive-protected group weights, leave-one-source-site residual/support calibration, exact v4 expert selection, v5 clipping, and v6 endpoint selection. Assert that changing site/year identifiers cannot change the final v6 rule result.

- [x] **Step 2: Implement gate fitting and decision generation**

Reuse the frozen v4 feature and model functions. Fit on all eligible 2015-2018 source cycles, generate anchor and candidate recommendations for target-site 2019 feature rows, apply the 2.5 mm v5 trust region, then call the v6 rule before any target materialization.

- [x] **Step 3: Implement atomic Stage 1 outputs**

Write:

```text
gefs_gam_v6_2019_frozen_recommendations_v1.csv
gefs_gam_v6_2019_recommendation_model_audit_v1.json
gefs_gam_v6_2019_recommendation_freeze_gate_v1.json
gefs_gam_v6_2019_recommendation_freeze_audit_v1.json
gefs_gam_v6_2019_recommendation_freeze_manifest_v1.json
```

The gate requires 69 unique cycles, five sites, finite decision fields, complete source-site holds, unchanged v6 constants, zero 2019 target rows read, and zero 2024 rows read. Refuse to overwrite an existing output directory.

- [x] **Step 4: Run Stage 1 tests**

Expected: all Stage 1 tests pass and a synthetic end-to-end fixture produces a hash-verifiable manifest.

### Task 4: Implement exact changed-cycle 2019 SWAP qualification

**Files:**
- Create: `scripts/simulation/run_gefs_gam_v6_2019_paired_swap_qualification_v1.py`
- Test: `tests/test_run_gefs_gam_v6_2019_paired_swap_qualification_v1.py`

- [x] **Step 1: Add failing Stage 2 tests**

Test manifest-tamper rejection, 69-cycle inventory validation, delayed target loading, exact endpoint reuse, unchanged-cycle zero expansion, no interpolation, fallback offsets, site summaries, positive/zero strata, peak summaries, fixed-eight regret, and changed-cycle maximum adaptive regret.

- [x] **Step 2: Implement input verification and delayed scoring**

Verify the Stage 1 pass status and every manifest hash first. Only then load complete 2019 target rows, calculate the fixed-grid oracle and peak distances, and construct the changed-cycle exact endpoint plan.

- [x] **Step 3: Reuse the checkpoint-isolated SWAP machinery**

Use `run_swap_stage`, `evaluate_cycle`, and the same zero-irrigation reference handling validated by v5. Reuse an existing candidate only on exact site/date/irrigation match within `1e-6`; otherwise run the endpoint with frozen 0.1/0.2/0.3 mm downward numerical fallbacks. Record logical and simulated amounts separately.

- [x] **Step 4: Build the joint confirmation gate**

Write the 69-cycle ledger and require all approved execution and performance conditions. A failed performance gate still writes all evidence, marks 2019 confirmation negative/inconclusive, and raises `RuntimeError`; it never deletes or overwrites evidence.

- [x] **Step 5: Run Stage 2 tests**

Expected: pass fixtures pass, each individual scientific condition can force failure, and tampered Stage 1 artifacts are rejected before target loading.

### Task 5: Verify and package

**Files:**
- Verify all files above plus related v4/v5/v6 tests.
- Create one root-level `.tar.gz` containing only the two protocol/runner pairs and their tests; do not include the design or plan documents.

- [x] **Step 1: Run compilation and related tests**

Run:

```powershell
python -m py_compile scripts/training/run_gefs_gam_v6_2019_recommendation_freeze_v1.py scripts/simulation/run_gefs_gam_v6_2019_paired_swap_qualification_v1.py
python -m unittest tests.test_run_gefs_gam_v6_2019_recommendation_freeze_v1 tests.test_run_gefs_gam_v6_2019_paired_swap_qualification_v1 tests.test_run_gefs_gam_v6_confidence_protected_anchor_replay_v1 tests.test_run_gefs_gam_v5_changed_cycle_paired_swap_qualification_v1 -v
```

Expected: compilation succeeds and all tests report `OK`.

- [x] **Step 2: Verify CLI help and forbidden-field scans**

Confirm both runners expose absolute-path-compatible CLIs and that the Stage 1 selection function does not access target, oracle, SWAP, site, or year fields.

- [x] **Step 3: Build and inspect one valuable archive**

Create `gefs_gam_v6_2019_independent_confirmation_code_20260808_v1.tar.gz`, list its members, reject absolute or parent-traversal paths, and calculate SHA256. Do not commit or upload the archive itself.

- [x] **Step 4: Provide server commands in chat**

Give explicit absolute-path `tar -x -z -f`, Stage 1 foreground run, Stage 1 JSON inspection, Stage 2 `nohup` run with the required `LD_LIBRARY_PATH`, `tail -f`, and final gate/audit inspection commands. Use no shell variables.

# GEFS GAM v8 Unlimited-Preserving Confidence Shrinkage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the approved v8 peak replay and its hash-frozen 2015--2018 exact paired-SWAP qualification workflow without accessing 2019 targets or 2024 data.

**Architecture:** Stage A is a standalone training runner that replays the frozen v5 inventory, preserves unlimited v5 endpoints, shrinks only limited movements, and applies the existing peak gate. Stage B is a separate simulation runner that verifies Stage A and the returned v5 paired-SWAP evidence, freezes a 24-row anchor/v8 plan, reuses exact endpoints, and delegates only the eight missing intermediate endpoints to the existing isolated SWAP stage.

**Tech Stack:** Python 3.10+, pandas, NumPy, unittest, JSON/CSV/SHA256 artifact contracts, existing SWAP checkpoint runner.

**Scope:** This plan implements development Stages A and B only. A passing Stage B may authorize a separate 2019 protocol design; this plan does not read or run 2019 and does not touch 2024 or TTA.

---

### Task 1: Freeze the Stage A protocol and selection equation

**Files:**
- Create: `docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1.json`
- Create: `tests/test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py`
- Create: `scripts/training/run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py`

- [ ] **Step 1: Write the failing protocol and equation tests**

Create tests that import `validate_protocol` and
`apply_unlimited_preserving_confidence_shrinkage`. Freeze `5.0`, `[0, 1]`, 196
cycles, 15 folds, five sites, at least 10 changed cycles, at least three changed
sites, 12 nonworse folds, four nonworse sites, and closed 2019/2024 boundaries.

Use this equation fixture:

```python
frame = pd.DataFrame({
    "anchor_recommendation_mm": [10.0] * 5,
    "recommendation_mm": [12.5, 12.5, 7.5, 12.5, 10.0],
    "v5_movement_from_anchor_mm": [2.5, 2.5, -2.5, 2.5, 0.0],
    "selected_lcb_mm": [-1.0, 2.5, 5.0, 10.0, 4.0],
    "trust_region_limited": [False, True, True, True, False],
    "true_peak_mm": [13.0, 11.0, 8.0, 13.0, 10.0],
})
replay = apply_unlimited_preserving_confidence_shrinkage(frame)
np.testing.assert_allclose(replay["v8_confidence_scale"], [1.0, 0.5, 1.0, 1.0, 0.0])
np.testing.assert_allclose(replay["v8_recommendation_mm"], [12.5, 11.25, 7.5, 12.5, 10.0])
```

Also mutate the protocol to lower the 5 mm reference or enable 2019 access and
assert `ValueError`. Change site, year, truth, and oracle fields while retaining
the four allowed selection inputs and assert identical recommendations.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
$env:PYTHONPATH='D:\study\s2s_rtist_source;D:\study\s2s_rtist_source\src'
python -m unittest tests.test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1 -v
```

Expected: `ModuleNotFoundError` for the v8 runner.

- [ ] **Step 3: Add the frozen JSON and minimal implementation**

Define:

```python
PROTOCOL_ID = "teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1"
CONFIDENCE_REFERENCE_LCB_MM = 5.0
EXPECTED_CYCLES = 196
EXPECTED_FOLDS = 15
EXPECTED_SITES = 5
TOLERANCE = 1.0e-6
```

Implement strict bool parsing, finite numeric checks, endpoint consistency, and:

```python
changed = ~np.isclose(v5, anchor, rtol=0.0, atol=TOLERANCE)
limited_scale = np.clip(lcb / CONFIDENCE_REFERENCE_LCB_MM, 0.0, 1.0)
scale = np.where(~changed, 0.0, np.where(limited, limited_scale, 1.0))
recommendation = anchor + (v5 - anchor) * scale
```

Expose frozen v5 audit columns plus `v8_confidence_scale`,
`v8_movement_from_anchor_mm`, `v8_recommendation_mm`, updated
`recommendation_mm`, `selected_peak_distance_mm`, and `irrigation_changed`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: protocol, formula, invariance, bounds, and inconsistent-movement tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1.json tests/test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py scripts/training/run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py
git commit -m "feat: add GAM v8 hybrid shrinkage rule"
```

### Task 2: Complete Stage A upstream validation, outputs, and peak gate

**Files:**
- Modify: `tests/test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py`
- Modify: `scripts/training/run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py`

- [ ] **Step 1: Write failing end-to-end tests**

Build 196-row v5 fixtures with 15 folds and five sites plus v5, v6, and v7
gate/audit/manifest fixtures. Require v5 and v6 passed states, the frozen v7
failed state, zero upstream 2019 target/SWAP access, and zero 2024 access.

Assert these outputs exist:

```text
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_decisions_v1.csv
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_fold_metrics_v1.csv
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_site_metrics_v1.csv
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_gate_v1.json
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_audit_v1.json
gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_manifest_v1.json
```

Add tests that mutate an upstream artifact after manifest creation and that
replace the required v7 failed status with a passed status; both must be rejected.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: failures for missing upstream loader and `run()` output behavior.

- [ ] **Step 3: Implement upstream verification and `run()`**

Reuse SHA256 helpers and `build_performance_gate` patterns from v7. Write all
outputs before raising when the peak performance gate fails. The passing status
is:

```text
gam_v8_unlimited_preserving_confidence_shrinkage_replay_passed_pending_changed_cycle_exact_swap_design
```

The audit must disclose the post-v7 iteration, prior review of 2019 covariates,
zero 2019 targets/SWAP outcomes read, zero 2024 rows, zero model fitting, zero
threshold search, and no automatic promotion.

- [ ] **Step 4: Run Stage A against the real 196-row local v5 inventory**

Use a new directory under `.codex_tmp`. Verify the expected diagnostic values:

```text
changed cycles: 12
changed sites: 4
overall selected mean: 11.69857816 mm
positive selected mean: 10.60176389 mm
zero selected mean: 12.28126074 mm
nonworse folds: 14
nonworse sites: 4
```

- [ ] **Step 5: Run focused and v5/v6/v7 regression tests**

```powershell
python -m unittest tests.test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1 tests.test_run_gefs_gam_v7_continuous_confidence_shrinkage_replay_v1 tests.test_run_gefs_gam_v6_confidence_protected_anchor_replay_v1 tests.test_run_gefs_gam_anchor_trust_region_mixture_replay_v5 -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- tests/test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py scripts/training/run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py
git commit -m "feat: complete GAM v8 peak replay"
```

### Task 3: Freeze the Stage B plan and exact-reuse contract

**Files:**
- Create: `docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-changed-cycle-paired-swap-qualification-v1.json`
- Create: `tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py`
- Create: `scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py`

- [ ] **Step 1: Write failing protocol, plan, and reuse tests**

Freeze 12 changed cycles, 24 plan rows, 196 ledger rows, four exact selected
endpoint reuses, eight maximum new selected endpoint runs, 48 restart print days,
0.01-day SWAP dtmax, and fallback offsets `[0.1, 0.2, 0.3]`.

Require `build_changed_plan(stage_a_decisions)` to emit `anchor` and `v8` roles
per changed cycle. Require exact identity at `1e-6` and reject a nearby endpoint:

```python
prior = pd.DataFrame({"ir": [8.0, 10.0]})
reused, missing = split_exact_reuse(prior, [8.000002, 10.0])
self.assertEqual(reused, [10.0])
self.assertEqual(missing, [8.000002])
```

Assert a real-data planning fixture yields exactly 16 reused plan rows and eight
missing selected endpoints: 12 anchors plus the four v8 endpoints identical to
v5 endpoints.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: `ModuleNotFoundError` for the Stage B runner.

- [ ] **Step 3: Implement protocol validation and pure plan helpers**

Adapt the v5 qualification runner, renaming `V5_ROLE` to `V8_ROLE` and reading
the Stage A decisions. Preserve logical and simulated irrigation fields. Reject
anything other than the frozen 196/12/24/4/8 structure.

- [ ] **Step 4: Run the focused pure-function tests and verify GREEN**

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-changed-cycle-paired-swap-qualification-v1.json tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py
git commit -m "feat: freeze GAM v8 paired SWAP plan"
```

### Task 4: Implement the Stage B ledger, gate, and isolated execution wrapper

**Files:**
- Modify: `tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py`
- Modify: `scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py`

- [ ] **Step 1: Write failing ledger and gate tests**

Require unchanged cycles to receive exact zero paired differences. Require the
gate to fail independently for negative pooled, positive-oracle, zero-oracle,
fixed-regret, maximum-adaptive-regret, fold, site, or retained Stage A peak
conditions. Require all 196 rows, 12 changed rows, and finite numeric fields.

- [ ] **Step 2: Run the new tests and verify RED**

- [ ] **Step 3: Implement changed comparison, full ledger, summaries, and gate**

Adapt the v5 pure functions and rename metric columns from `v5_minus_anchor_*`
to `v8_minus_anchor_*`. Use 10/15 SWAP-nonworse folds, 4/5 sites, both oracle
strata, pooled mean gain, fixed regret, changed-cycle maximum adaptive regret,
and every Stage A peak condition.

- [ ] **Step 4: Write failing upstream and execution-boundary tests**

Reject changed Stage A hashes, changed returned v5 SWAP hashes, failed prior
execution audits, incomplete fixed-eight candidate grids, recommendation changes
after plan freeze, interpolation, changed numerical controls, 2019 paths, and
2024 paths.

- [ ] **Step 5: Implement `run()` by delegating missing values only**

Reuse the existing `unit_resources`, `run_swap_stage`, and `evaluate_cycle`
entry points. For each changed cycle, combine the returned prior candidate grid
with only the missing v8 endpoint. Record exact reuses and new stage roots.
Write the frozen plan before executing any missing endpoint.

Write decisions, changed comparison, 196-row ledger, fold/site summaries, reuse
audit, cycle oracles, candidates, gate, audit, and manifest before raising on a
performance failure. Execution failure and performance failure must use distinct
status strings and stop before 2019.

- [ ] **Step 6: Run focused tests and syntax checks**

```powershell
python -m unittest tests.test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1 tests.test_run_gefs_gam_v5_changed_cycle_paired_swap_qualification_v1 tests.test_gefs_controlled_continuous_irrigation_v1 -v
python -m py_compile scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 7: Commit Task 4**

```powershell
git add -- tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py
git commit -m "feat: add GAM v8 exact SWAP qualification"
```

### Task 5: Verify, package, and hand off the server run

**Files:**
- Verify: both v8 protocols, runners, and focused tests
- Create locally only: `gefs_gam_v8_unlimited_preserving_confidence_shrinkage_code_20260808_v1.tar.gz`

- [ ] **Step 1: Run the full related regression suite**

Run both v8 tests plus v5, v6, v7, existing v5 SWAP qualification, and controlled
continuous irrigation tests. Require zero failures and errors.

- [ ] **Step 2: Verify the real Stage A artifacts and Stage B frozen plan**

Hash-check every Stage A output. Generate the Stage B plan without executing
SWAP and verify 196/12/24 structure, four reusable selected endpoints, eight
missing selected endpoints, zero 2019/2024 access, and no interpolation.

- [ ] **Step 3: Build the minimal archive**

Include only:

```text
docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-unlimited-preserving-confidence-shrinkage-replay-v1.json
docs/superpowers/specs/2026-08-08-teacher-guided-gam-v8-changed-cycle-paired-swap-qualification-v1.json
scripts/training/run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py
scripts/simulation/run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py
tests/test_run_gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_v1.py
tests/test_run_gefs_gam_v8_changed_cycle_paired_swap_qualification_v1.py
```

List every member, reject absolute and parent-traversal names, extract into a
fresh verification directory, compile both runners, run both focused tests, and
record archive SHA256.

- [ ] **Step 4: Provide server commands in project style**

Use absolute `/media/data_hot/...` paths, `python3`, explicit `PYTHONPATH`, a new
output directory, and `nohup ... > log 2>&1 &` with PID, `tail -f`, and final
`python3 -m json.tool` gate/audit inspection commands. Stage A must pass before
the Stage B command is issued.

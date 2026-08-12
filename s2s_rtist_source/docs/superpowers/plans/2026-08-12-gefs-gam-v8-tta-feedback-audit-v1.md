# GEFS GAM v8 TTA Feedback Audit V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only audit that proves which 2024 post-decision feedback signals are causally available for later GAM v8 TTA updates without changing any frozen v8 artifacts.

**Architecture:** A diagnostics script loads the frozen 68-cycle schedule and recommendations, joins the post-freeze exact SWAP candidate rows only to the realized frozen irrigation action, and emits one feedback ledger row per site-cycle/signal. It applies explicit chronological eligibility rules, separates physics-only from delayed-VWC availability, excludes outcome/oracle fields from update eligibility, and writes a new versioned audit directory with hashes and a manifest.

**Tech Stack:** Python 3, pandas, NumPy, stdlib `argparse`/`json`/`hashlib`, existing project test conventions (`unittest`).

---

### Task 1: Define the audit contract in tests

**Files:**
- Create: `tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py`
- Create: `scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py`

- [ ] **Step 1: Write the failing tests**

Add a synthetic two-cycle, one-site fixture with consecutive seven-day dates. Include one exact-SWAP candidate row per selected action, complete physics fields, daily VWC fields, and forbidden outcome fields (`target_value`, `net_gain_7d`, `best_ir_for_date`). Assert the public module exposes `build_feedback_ledger` and `build_audit`.

Add these tests:

```python
def test_feedback_from_cycle_zero_can_update_only_cycle_one():
    ledger = module.build_feedback_ledger(schedule, recommendations, candidates)
    cycle_zero = ledger[(ledger["feedback_cycle_index"] == 0) & (ledger["signal_type"] == "physics")].iloc[0]
    assert cycle_zero["eligible_update_cycle_index"] == 1
    assert cycle_zero["causal_order_valid"] is True
    assert not (ledger["feedback_cycle_index"] == ledger["eligible_update_cycle_index"]).any()

def test_missing_vwc_is_explicit_but_physics_can_still_be_available():
    candidates.loc[:, "soil_vwc_0_100cm_day07"] = np.nan
    ledger = module.build_feedback_ledger(schedule, recommendations, candidates)
    row = ledger[ledger["signal_type"] == "delayed_vwc"].iloc[0]
    assert row["availability_status"] == "feedback_missing"
    physics = ledger[ledger["signal_type"] == "physics"].iloc[0]
    assert physics["availability_status"] == "physics_only_available"

def test_forbidden_outcome_columns_are_never_online_features():
    ledger = module.build_feedback_ledger(schedule, recommendations, candidates)
    assert not {"target_value", "net_gain_7d", "best_ir_for_date"}.intersection(ledger.columns)
    assert ledger["eligibility_basis"].str.contains("target_value|best_ir_for_date", regex=True).sum() == 0

def test_audit_requires_exact_cycle_coverage_and_records_hashes():
    audit = module.build_audit(schedule, ledger, input_hashes={"schedule": "abc"})
    assert audit["cycle_count"] == 2
    assert audit["site_count"] == 1
    assert audit["tta_performed"] is False
    assert audit["model_training_performed"] is False
    with pytest.raises(ValueError, match="duplicate cycle"):
        module.build_feedback_ledger(pd.concat([schedule, schedule.iloc[[0]]]), recommendations, candidates)
```

Use the repository's `unittest` style rather than introducing pytest-only fixtures; the snippet above describes the assertions and should be adapted to `unittest.TestCase` and `self.assertRaisesRegex`.

- [ ] **Step 2: Run the focused test file and verify the expected RED failure**

Run:

```powershell
python -m unittest tests.test_audit_gefs_gam_v8_tta_feedback_availability_v1 -v
```

Expected result: import or attribute failures because the audit module does not yet exist.

### Task 2: Implement deterministic feedback extraction

**Files:**
- Modify: `scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py`
- Test: `tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py`

- [ ] **Step 1: Implement the public contract and constants**

Define `CYCLE_KEYS = ("site_id", "target_year", "decision_date")`, `FORBIDDEN_ONLINE_COLUMNS = {"target_value", "net_gain_7d", "best_ir_for_date", "best_target_for_date", "is_best_ir"}`, and explicit required field groups:

```python
PHYSICS_COLUMNS = (
    "rain_7d_mm", "irrigation_7d_mm", "aet_7d_mm",
    "predecision_soil_storage_0_100cm_mm",
    "final_soil_storage_0_100cm_mm",
    "residual_flux_7d_mm", "water_balance_residual_0_100cm_7d_mm",
)
VWC_COLUMNS = tuple(f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8))
```

Implement `load_csv_checked(path, required_columns, label)` with finite numeric checks only on fields required for the selected signal. Implement `sha256_file(path)` and `write_json_atomic(path, value)` using the existing project pattern.

- [ ] **Step 2: Implement frozen-cycle validation**

Implement `validate_schedule(schedule)` to require exactly the expected 2024 site set (`P1`, `P15`, `P2`, `P3`, `P4`), unique `CYCLE_KEYS`, chronological `cycle_index` per site, seven-day horizons, and no manual post-outcome filtering. Implement `validate_recommendations(recommendations)` to require one row per schedule key and finite `selected_irrigation_mm`.

- [ ] **Step 3: Implement selected-action candidate matching**

Implement `match_shadow_candidate(recommendation, candidates)` by selecting the same site/date row whose `requested_ir_mm` or `simulated_ir_mm` equals `selected_irrigation_mm` within `1e-6` mm. If no unique row matches, return a `feedback_missing` reason `selected_action_shadow_row_missing_or_ambiguous`; never choose a candidate using `target_value`, `best_ir_for_date`, or `is_best_ir`.

- [ ] **Step 4: Implement causal ledger construction**

Implement `build_feedback_ledger(schedule, recommendations, candidates)` returning a DataFrame with stable columns:

```text
site_id,target_year,decision_date,cycle_index,feedback_cycle_index,
signal_type,availability_status,feedback_source,feedback_available_date,
eligible_update_cycle_index,causal_order_valid,eligibility_basis,
missing_required_columns,physics_field_count,vwc_field_count
```

Emit two signal rows per cycle (`physics`, `delayed_vwc`). Set `feedback_available_date = horizon_end_date + 1 day`. Set `eligible_update_cycle_index` to the next chronological cycle for the same site, or null for the final cycle. Mark `physics_only_available` when all `PHYSICS_COLUMNS` are finite. Mark `delayed_vwc_available` only when all `VWC_COLUMNS` are finite and a later same-site cycle exists. Preserve missing rows instead of dropping them.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the same `unittest` command from Task 1. Expected result: all contract, coverage, causal-order, and forbidden-column tests pass.

### Task 3: Add the audit summary, output files, and command entrypoint

**Files:**
- Modify: `scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py`
- Modify: `tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py`

- [ ] **Step 1: Write failing summary/output tests**

Add a test that runs `run(...)` against temporary CSV inputs and asserts it writes exactly:

```text
gefs_gam_v8_tta_feedback_audit_v1.csv
gefs_gam_v8_tta_feedback_audit_v1.json
gefs_gam_v8_tta_feedback_audit_manifest_v1.csv
```

Assert the JSON contains `status`, `mandatory_gate_passed`, `cycle_count`, `site_count`, `physics_only_available_count`, `delayed_vwc_available_count`, `feedback_missing_count`, `tta_performed: false`, and `model_training_performed: false`.

- [ ] **Step 2: Implement `build_audit` and output manifest**

`build_audit(schedule, ledger, input_hashes)` must fail if any causal gate fails. It must count signal statuses, record exact input hashes, include `forbidden_online_columns_excluded: true`, and set status to `gefs_gam_v8_tta_feedback_audit_v1_passed` only when all gates pass. Write the ledger with `index=False`, write JSON atomically, then write one manifest row per output with bytes and SHA-256.

- [ ] **Step 3: Implement CLI argument parsing and resumeless refusal**

Add required absolute-path arguments `--schedule`, `--recommendations`, `--shadow-candidates`, and `--output-dir`; optional `--target-year` defaults to `2024`. Refuse to overwrite a nonempty output directory unless its existing manifest hashes validate exactly. Print only final output paths, not full ledger rows.

- [ ] **Step 4: Run focused tests and the module help smoke test**

Run:

```powershell
python -m unittest tests.test_audit_gefs_gam_v8_tta_feedback_availability_v1 -v
python scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py --help
```

Expected result: all tests pass and help exits with code 0.

### Task 4: Run the read-only 2024 feedback audit

**Files:**
- Create: `site_general_surrogate_eval/gefs_gam_v8_tta_feedback_audit_v1/`
- Test: `tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py`

- [ ] **Step 1: Execute the audit with frozen absolute inputs**

Run the script using the existing 68-cycle schedule, frozen recommendation CSV, exact SWAP candidate CSV, and a new output directory. Do not pass exact SWAP scored decisions as an update input.

- [ ] **Step 2: Validate the generated audit**

Read the JSON with `python -m json.tool` and assert:

```text
status == gefs_gam_v8_tta_feedback_audit_v1_passed
cycle_count == 68
site_count == 5
tta_performed == false
model_training_performed == false
```

Confirm the manifest contains only the three new output files and the frozen input hashes match the source files used.

- [ ] **Step 3: Run regression tests**

Run the focused audit tests plus the existing frozen-feature and recommendation-freeze tests:

```powershell
python -m unittest tests.test_audit_gefs_gam_v8_tta_feedback_availability_v1 tests.test_materialize_gefs_gam_v8_2024_target_blind_features_v2 tests.test_run_gefs_gam_v8_2024_teacher_aligned_recommendation_freeze_v2 -v
```

Expected result: all tests pass; no frozen v8 file hashes change.

### Task 5: Commit the implementation and hand off to TTA replay planning

**Files:**
- Add: `scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py`
- Add: `tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py`
- Add: generated audit outputs under `site_general_surrogate_eval/gefs_gam_v8_tta_feedback_audit_v1/`

- [ ] **Step 1: Check the diff scope**

Run `git status --short` and verify only the new audit script, its tests, and its versioned output directory are included. Do not stage shared documentation, frozen v8 directories, or unrelated worktree changes.

- [ ] **Step 2: Commit the implementation**

```powershell
git add scripts/diagnostics/audit_gefs_gam_v8_tta_feedback_availability_v1.py tests/test_audit_gefs_gam_v8_tta_feedback_availability_v1.py site_general_surrogate_eval/gefs_gam_v8_tta_feedback_audit_v1
git commit -m "feat: audit causal feedback for GAM v8 TTA"
```

- [ ] **Step 3: Stop before model adaptation**

Do not implement coefficient updates, gate updates, loss weights, or 2024 performance selection in this plan. The next plan must use the audit output and fixed 2019-derived TTA settings.

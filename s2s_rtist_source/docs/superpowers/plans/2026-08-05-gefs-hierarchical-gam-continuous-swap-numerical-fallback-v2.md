# Hierarchical GAM Continuous SWAP Numerical Fallback v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将诊断确认的 `-0.1、-0.2、-0.3 mm` 顺序数值回退固化为分层 GAM SWAP v2 正式协议，并在不改变冻结推荐和性能门的前提下生成可独立全量重跑的服务器包。

**Architecture:** 保持通用连续 SWAP 运行器对旧实验的单值 `-0.2 mm` 默认行为，在调用边界新增显式回退偏移参数。新建 GAM v2 协议和包装入口，由包装入口验证固定三阶回退并传入通用运行器；v1 协议、入口和失败输出均不修改。

**Tech Stack:** Python 3.10、pandas、NumPy、unittest、JSON 协议、SWAP 4.0.1、tar.gz 服务器交付包。

---

### Task 1: Parameterize the shared numerical fallback policy

**Files:**
- Modify: `scripts/simulation/run_gefs_controlled_continuous_swap_evaluation_v1.py`
- Modify: `tests/test_gefs_controlled_continuous_irrigation_v1.py`

- [ ] **Step 1: Write failing tests for legacy and ordered policies**

Add tests asserting that the default remains a scalar legacy fallback while explicit offsets produce an ordered tuple:

```python
legacy = fallback_policy([0.0, 0.1, 25.0])
self.assertEqual(legacy, {0.1: 0.0, 25.0: 24.8})

ordered = fallback_policy([0.0, 0.1, 25.0], offsets_mm=(0.1, 0.2, 0.3))
self.assertEqual(ordered[0.1], (0.0,))
self.assertEqual(ordered[25.0], (24.9, 24.8, 24.7))
```

Add rejection tests for empty, non-positive, duplicated and non-increasing offsets.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m unittest tests.test_gefs_controlled_continuous_irrigation_v1.SwapEvaluationTests.test_grid_and_fallback_controls_stay_bounded -v
```

Expected: failure because `fallback_policy` does not accept `offsets_mm`.

- [ ] **Step 3: Implement normalized fallback offsets**

Add a validator and preserve scalar output for the one-offset legacy path:

```python
FallbackValue = float | tuple[float, ...]

def normalize_fallback_offsets(offsets_mm: tuple[float, ...]) -> tuple[float, ...]:
    offsets = tuple(float(value) for value in offsets_mm)
    if not offsets or any(value <= 0.0 for value in offsets):
        raise ValueError("fallback offsets must be positive")
    if tuple(sorted(set(offsets))) != offsets:
        raise ValueError("fallback offsets must be unique and increasing")
    return offsets

def fallback_policy(
    values: list[float],
    *,
    offsets_mm: tuple[float, ...] = (0.2,),
) -> dict[float, FallbackValue]:
    offsets = normalize_fallback_offsets(offsets_mm)
    policy: dict[float, FallbackValue] = {}
    for requested_raw in values:
        requested = float(requested_raw)
        if requested <= 0.0:
            continue
        candidates = tuple(
            dict.fromkeys(
                round(max(0.0, requested - offset), 1)
                for offset in offsets
                if round(max(0.0, requested - offset), 1) < requested
            )
        )
        if candidates:
            policy[requested] = candidates[0] if len(offsets) == 1 else candidates
    return policy
```

Thread `fallback_offsets_mm` through `run_swap_stage()` and `run()`. Both coarse and refinement stages must receive the same normalized tuple. Add the offsets, maximum adjustment and target-independent selection statement to each stage audit and the base audit.

- [ ] **Step 4: Run shared-runner tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_gefs_controlled_continuous_irrigation_v1 tests.test_run_gefs_checkpoint_one_date_eight_ir_smoke_v1 tests.test_restart_decision_endpoint_fallback_v1 -v
```

Expected: all tests pass; existing single-value callers remain unchanged.

### Task 2: Freeze and validate the GAM v2 protocol

**Files:**
- Create: `docs/superpowers/specs/2026-08-05-teacher-guided-hierarchical-gam-continuous-swap-qualification-v2.json`
- Create: `scripts/simulation/run_gefs_hierarchical_gam_continuous_swap_qualification_v2.py`
- Create: `tests/test_gefs_hierarchical_gam_continuous_swap_qualification_v2.py`

- [ ] **Step 1: Write failing protocol tests**

Tests must load the v2 JSON and assert:

```python
validate_protocol_v2(protocol)
self.assertEqual(
    protocol["swap_validation"]["numerical_fallback_offsets_mm"],
    [0.1, 0.2, 0.3],
)
```

Mutations to the protocol ID, ordered offsets, maximum adjustment, target-independent selection, new output directory and the existing SWAP controls must raise `ValueError`.

- [ ] **Step 2: Run the v2 test and verify RED**

Run:

```bash
python3 -m unittest tests.test_gefs_hierarchical_gam_continuous_swap_qualification_v2 -v
```

Expected: import or file-not-found failure because v2 artifacts do not exist.

- [ ] **Step 3: Create the frozen v2 protocol**

Copy all v1 model, data, optimizer and performance-gate fields unchanged. Change only the protocol identity and add:

```json
"numerical_fallback_offsets_mm": [0.1, 0.2, 0.3],
"maximum_numerical_fallback_adjustment_mm": 0.3,
"fallback_only_after_exact_request_failure": true,
"fallback_selection_uses_target_or_surrogate": false,
"fallback_exhaustion_fails_stage": true,
"reuse_v1_passed_stages": false
```

Set the formal output identity to v2 and record that `P2/2019-05-27/25 mm` diagnostic evidence motivated the amendment without treating the diagnostic result as formal input.

- [ ] **Step 4: Implement the v2 wrapper**

Reuse v1 frozen-recommendation validation, paired comparison and unchanged performance gate. Implement `validate_protocol_v2()` by validating all inherited v1 fields plus the exact v2 numerical policy. Call:

```python
base_outputs = run_base(
    args,
    artifact_version="hierarchical_gam_vc_strict_loso_2019_v2",
    recommendation_plan_name=SWAP_PLAN_NAME,
    expected_site_cycles=expected_counts,
    fallback_offsets_mm=(0.1, 0.2, 0.3),
    additional_input_paths={
        "continuous_protocol_v2": protocol_path,
        **frozen_paths,
    },
    audit_context={
        "source_model_family": "hierarchical_gam_vc_five_strict_loso_target_models_shared_only",
        "recommendations_frozen_before_swap": True,
        "recommendation_changes_after_swap": False,
        "v1_stage_results_reused": False,
    },
)
```

Use v2 output filenames and status values. Reject a v1 protocol path and reject the v1 output directory.

- [ ] **Step 5: Run v2 tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_gefs_hierarchical_gam_continuous_swap_qualification_v2 tests.test_gefs_hierarchical_gam_continuous_qualification_v1 -v
```

Expected: all v1 and v2 qualification tests pass.

### Task 3: Add fallback audit summaries and invariant tests

**Files:**
- Modify: `scripts/simulation/run_gefs_hierarchical_gam_continuous_swap_qualification_v2.py`
- Modify: `tests/test_gefs_hierarchical_gam_continuous_swap_qualification_v2.py`

- [ ] **Step 1: Write failing summary tests**

Build a candidate frame containing exact, second-level and third-level successes. Assert counts, affected sites/dates and maximum adjustment:

```python
summary = summarize_fallback_usage(frame)
self.assertEqual(summary["fallback_count"], 2)
self.assertEqual(summary["second_level_fallback_count"], 1)
self.assertEqual(summary["third_level_fallback_count"], 1)
self.assertEqual(summary["maximum_absolute_adjustment_mm"], 0.3)
```

Missing required columns or an adjustment larger than `0.3 mm` must fail.

- [ ] **Step 2: Run summary tests and verify RED**

Run the new test class and expect an import failure for `summarize_fallback_usage`.

- [ ] **Step 3: Implement formal summary and manifest fields**

Summarize from the final candidate CSV, write the summary into the qualification audit, and include protocol, recommendation manifest and base SWAP manifest hashes. Require:

```python
audit["numerical_fallback_offsets_mm"] == [0.1, 0.2, 0.3]
audit["maximum_numerical_fallback_adjustment_mm"] <= 0.3 + 1.0e-9
audit["fallback_selection_used_target_or_surrogate"] is False
audit["v1_stage_results_reused"] is False
```

- [ ] **Step 4: Run all relevant regression tests**

Run:

```bash
python3 -m unittest tests.test_gefs_controlled_continuous_irrigation_v1 tests.test_run_gefs_checkpoint_one_date_eight_ir_smoke_v1 tests.test_restart_decision_endpoint_fallback_v1 tests.test_gefs_hierarchical_gam_continuous_qualification_v1 tests.test_gefs_hierarchical_gam_continuous_swap_qualification_v2 -v
```

Expected: all tests pass with no warning or error output.

### Task 4: Server operations and clean package verification

**Files:**
- Create: `docs/operations/2026-08-05-gefs-hierarchical-gam-continuous-swap-v2-server.md`
- Modify: `scripts/script_catalog.csv` only if the touched scripts already have catalog rows
- Create artifact: `gefs_hierarchical_gam_continuous_swap_20260805_v2.tar.gz`

- [ ] **Step 1: Write the server runbook**

Document absolute-path commands for archive inspection, extraction, focused tests, a fresh v2 `nohup` launch, PID capture, `tail -f`, cycle/stage counts, error search, v2-only resume, final JSON/CSV viewing and key-result packaging. The launch must include the project-local `libgfortran.so.5` directory in `LD_LIBRARY_PATH`.

- [ ] **Step 2: Run syntax and regression verification**

Run `py_compile` for modified/new Python files and the full relevant unittest command from Task 3.

- [ ] **Step 3: Build the tar archive**

Include the shared runner, checkpoint runner, v1/v2 GAM wrappers, v2 protocol, design, plan, operations document and all directly relevant tests. Verify all archive paths are relative and contain neither `..` nor absolute prefixes.

- [ ] **Step 4: Extract into a new clean package-check directory**

Run the package tests from the extracted directory with only the extracted root and `src` on `PYTHONPATH`. Directly assert:

```python
fallback_policy([25.0], offsets_mm=(0.1, 0.2, 0.3))[25.0] == (24.9, 24.8, 24.7)
```

- [ ] **Step 5: Record SHA256 and update project memory**

Record the package hash, test counts, v1 failure boundary, v2 fresh-output requirement and the exact server next step in `C:\Users\LEGION\.codex\memories\extensions\projects\s2s-rtist-source.md`.

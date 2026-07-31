# P3 B2 Regret-Sensitive Leave-Site-Out Router Implementation Plan

> Execute against the frozen design in
> `docs/superpowers/specs/2026-07-31-gefs-p3-b2-regret-sensitive-loso-router-design.md`.

**Goal:** Implement and package one leakage-audited B2 development screen that
uses fold-local regret-sensitive logistic weights and eight rolling-year
leave-site-out P2/P4 validation folds without reading P3, 2020, 2021, or 2024.

**Architecture:** Add one independent B2 runner and one focused test module.
Reuse B1 historical loading, source-checkpoint inference, physical gate
features, policy rows, and policy summaries. Keep the B1 unweighted logistic
function unchanged. Define weighted IRLS, eight-fold orchestration, strict
development gating, output audits, and final-gate eligibility in the B2 runner.

---

## Task 1: Lock Fold And Weight Primitives With Tests

**Files:**

- Create: `tests/test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`
- Create: `scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`

1. Add synthetic complete-cycle fixtures for P2/P4 across 2015-2019.
2. Add tests requiring exactly eight folds, the opposite training site, and
   strictly earlier training years.
3. Add tests for physical row-scope filtering before target materialization.
4. Add tests for exact `abs(gain_delta) / mean(abs(gain_delta))` weights and
   mean-one normalization.
5. Run the new module and confirm the tests fail because implementation is
   missing.
6. Implement fold definitions, scope validation, and training-weight creation.
7. Re-run the focused tests.

## Task 2: Implement Deterministic Weighted IRLS

**Files:**

- Modify: `scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`
- Modify: `tests/test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`

1. Add deterministic weighted logistic tests with a known fixture.
2. Prove changing validation targets cannot change a training gate hash.
3. Add fallback tests for insufficient rows, one class, zero/non-finite
   weights, singular Hessian, and non-convergence.
4. Implement fold-local unweighted feature standardization and weighted
   gradient/Hessian Newton/IRLS with fixed L2, threshold, and tie behavior.
5. Record training deltas, raw weights, normalized weights, and model metadata.
6. Run focused tests.
7. Run B1 gate regression tests to prove existing unweighted behavior is
   unchanged.

## Task 3: Implement Eight-Fold Orchestration And Strict Gate

**Files:**

- Modify: `scripts/training/run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`
- Modify: `tests/test_run_gefs_p3_b2_regret_sensitive_loso_router_screen_v1.py`

1. Add tests for the nine frozen pass conditions and one failure test per
   condition.
2. Add a mocked CPU end-to-end fixture covering all eight folds, output counts,
   final pass/fail states, final-gate pass-only behavior, and manifest hashes.
3. Reuse the existing fold-local P1/P15 source checkpoints and inference.
4. Implement equal-fold macro, per-site summaries, non-worse-fold counts,
   worst-regret, false-positive, 60 mm, effective-route, and non-fallback-site
   gates.
5. Fail closed on output overwrite and any scope/audit mismatch.
6. Write all frozen CSV/JSON/manifest outputs.
7. Run focused tests.

## Task 4: Verification And Regression

1. Run `python -m py_compile` for the new runner and test module.
2. Run the complete new B2 test module with `PYTHONPATH` set to project and
   `src`.
3. Run related B1, strict-LOSO, dual-gate, recommendation-freeze, and shared
   SWAP evaluation tests.
4. Inspect `git diff --check` and ensure no unrelated files are modified.

## Task 5: Package And Verify A Minimal Server ZIP

1. Build
   `gefs_p3_b2_regret_sensitive_loso_router_screen_20260731_v1.zip` in the
   project root.
2. Include the B2 runner/test and every imported reusable module required by a
   clean extraction.
3. Exclude datasets, checkpoints, outputs, 2021 results, logs, PID files, and
   caches.
4. Verify POSIX members, no absolute/drive/traversal/backslash paths, no
   duplicates or case-insensitive collisions.
5. Extract to a fresh temporary directory and run the focused B2 tests there.
6. Record ZIP SHA256 and member count.

## Task 6: Deliver Complete Server Commands

Provide complete absolute-path commands for:

1. SHA verification, extraction, and unpacked tests;
2. the B2 background run with `python3`, explicit `CUDA_VISIBLE_DEVICES=0`,
   `PYTHONUNBUFFERED=1`, `PYTHONPATH`, log, and PID;
3. `ps`, `tail -f`, `nvidia-smi`, fold progress, and exception search;
4. output listing;
5. final audit, nine gate conditions, eight-fold metrics, per-site metrics,
   fallback reasons, route counts, and worst-cycle result reading.

The commands must not evaluate P3, 2021, or 2024 and must not define reusable
temporary shell variables.

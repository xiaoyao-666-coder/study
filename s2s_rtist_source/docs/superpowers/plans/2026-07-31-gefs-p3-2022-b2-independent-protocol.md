# P3 2022 B2 Independent Protocol Implementation Plan

> Execute against the frozen design in
> `docs/superpowers/specs/2026-07-31-gefs-p3-2022-b2-independent-protocol-design.md`.

**Goal:** Freeze a hash-bound, zero-2022-access protocol for the first new
independent evaluation of the final B2 router without modifying any completed
P3 2021 artifact or script.

**Architecture:** Add one dedicated protocol-freezer script and one focused
test module. The freezer validates the B2 development release, recomputes the
final gate payload hash, verifies the P1/P15 final source files and the frozen
dual feature contract, then writes protocol, component registry, stage
registry, audit, and manifest outputs. It performs no acquisition, inference,
or simulation.

---

## Task 1: Lock Contract Builders With Tests

**Files:**

- Create: `tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py`
- Create: `scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py`

1. Add tests for target identity P3/2022 and sealed P3/2023 plus 2024/TTA.
2. Add tests for the exact five GEFS members, 00 UTC, D through D+6, and fixed
   eight-candidate grid.
3. Add tests for the ERA5-DVS schedule rule and the prohibition on manual,
   GEFS-based, or SWAP-label-based date selection.
4. Add tests for the exact ordered nine B2 gate features, threshold `0.5`, and
   P15 tie/fallback behavior.
5. Add tests for all ten independent promotion conditions and both fixed
   baselines.
6. Run the focused test module and confirm the expected import failure.
7. Implement pure contract and registry builders, then rerun the tests.

## Task 2: Implement Hash-Bound Input Validation

**Files:**

- Modify: `scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py`
- Modify: `tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py`

1. Build compact fixture directories for B2 outputs, the dual protocol, final
   policy, and two source checkpoint byte files.
2. Test exact B2 audit status, development pass, final-gate presence,
   `formal_promotion=false`, and zero forbidden rows/search actions.
3. Test final gate file SHA256 and recomputed payload SHA256 independently.
4. Test B2 manifest bindings for the final policy and both source checkpoints.
5. Test the actual final policy and checkpoint bytes against the frozen hashes.
6. Test the dual protocol's feature names and definitions against the B2 gate.
7. Add one fail-closed test for each changed status, leakage count, fallback,
   feature order, model hash, policy hash, and checkpoint hash.
8. Implement validation helpers without importing or materializing any target
   dataset.

## Task 3: Implement Protocol Output And Audit

**Files:**

- Modify: `scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py`
- Modify: `tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py`

1. Add an end-to-end fixture that runs entirely on CPU and local temporary
   files.
2. Verify the exact five output filenames and output directory overwrite
   refusal.
3. Implement the protocol JSON, component registry, eight-stage registry,
   zero-access audit, and SHA256 manifest.
4. Require stage order 1-8 and make only stage 1 eligible at protocol-freeze
   time.
5. Record zero 2022 ERA5/GEFS/inference/recommendation/SWAP/network access and
   zero 2023/2024 access.
6. Verify every non-manifest output hash from the generated manifest.
7. Verify no output contains an unbound or absolute local checkpoint path as a
   policy identity; component identities use frozen hashes and canonical roles.

## Task 4: Regression Verification

1. Run `python -m py_compile` for the new freezer and test module.
2. Run the focused protocol-freezer tests with `PYTHONPATH=.;src`.
3. Run the B2 development tests, dual-gate protocol tests, original P3 2021
   independent-protocol tests, ERA5 amendment tests, recommendation-freeze
   tests, and shared-SWAP tests.
4. Run `git diff --check` and inspect the scoped status.
5. Commit only the new freezer and test module.

## Task 5: Minimal Package And Clean Extraction

1. Build
   `gefs_p3_2022_b2_independent_protocol_freeze_20260731_v1.zip` in the project
   root.
2. Include the new freezer/test and only their transitive local dependencies.
3. Exclude all weather, datasets, checkpoints, B2 result files, 2021 results,
   2022/2023/2024 content, logs, PID files, and caches.
4. Reject absolute paths, drive prefixes, traversal, backslashes, links,
   duplicates, and case-insensitive collisions.
5. Extract into a fresh temporary directory and run the focused tests there.
6. Record archive member count, byte size, and SHA256.

## Task 6: Server Freeze And Result Commands

Create
`docs/operations/2026-07-31-gefs-p3-2022-b2-independent-protocol-server.md`
with full absolute-path commands for:

1. ZIP hash verification, clean extraction, and focused tests;
2. installing only the new freezer/test into the existing server project;
3. running the protocol freezer with `python3`, explicit B2, dual-protocol,
   final-model, and output directories;
4. listing outputs and reading audit status, zero-access fields, component
   hashes, stage order, promotion conditions, and manifest integrity;
5. packaging the five protocol outputs for GUI transfer back to local.

This protocol freeze is short and deterministic, so it does not use `nohup`,
GPU, PID, or SWAP runtime libraries. The handoff must explicitly state that no
2022 download or server simulation is authorized until the returned protocol
artifacts are verified.

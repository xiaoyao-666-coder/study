# GEFS GAM v8 TTA Feedback Audit V1

## Purpose

Create a read-only, leakage-focused audit that determines which feedback signals can be used by a causal test-time adaptation (TTA) replay for the frozen GAM v8 pipeline. This audit is exploratory infrastructure for a later TTA run; it does not update model coefficients, recommendations, gates, or frozen v8 artifacts.

## Scope

The audit covers the 68 teacher-aligned 2024 cycles and the already validated 2019 time-out confirmation cycles when available. It preserves the existing five-site, seven-day decision schedule and records feedback availability per site and decision cycle.

The audit distinguishes:

- `physics_only_available`: all quantities needed to calculate the model water-balance residual are available from information that is causally observable after the current decision horizon. This signal may supervise a later cycle, never the current cycle.
- `delayed_vwc_available`: a post-horizon VWC truth is available and can be used with the delayed supervised loss for a later cycle.
- `feedback_missing`: one or more required fields or their provenance is absent, ambiguous, or not causally ordered.

## Inputs

All inputs are read-only and hash recorded:

- Frozen 2024 schedule: `site_general_surrogate_eval/gefs_gam_v8_2024_season_trunk_schedule_v2_server_results/gefs_gam_v8_2024_teacher_aligned_cycle_schedule_v2.csv`
- Frozen 2024 target-blind features: `site_general_surrogate_eval/gefs_gam_v8_2024_target_blind_feature_materialization_v2/gefs_gam_v8_2024_target_blind_features_v2.csv`
- Frozen 2024 recommendations: `site_general_surrogate_eval/gefs_gam_v8_2024_teacher_aligned_recommendation_freeze_v2/gefs_gam_v8_2024_teacher_aligned_frozen_recommendations_v2.csv`
- Post-freeze 2024 exact SWAP candidate table, used only as delayed shadow feedback: `site_general_surrogate_eval/gefs_gam_v8_2024_teacher_aligned_exact_swap_execution_v2/gefs_gam_v8_2024_teacher_aligned_exact_swap_candidates_v2.csv`
- 2024 exact SWAP scored decisions, used only for post-hoc comparison and not for feedback eligibility: `site_general_surrogate_eval/gefs_gam_v8_2024_teacher_aligned_exact_swap_results_v2/gefs_gam_v8_2024_teacher_aligned_exact_swap_decisions_v2.csv`

The 2024 candidate table contains the fixed 0-100 cm water-balance fields, including precipitation, irrigation, AET, initial/final storage, daily VWC, residual flux, and water-balance residual. The audit treats these fields as unavailable until the candidate horizon has ended.

## Causal Timing Rules

For a decision cycle `t` with horizon `[horizon_start_date, horizon_end_date]`:

1. The recommendation at `t` may use only the frozen decision features and frozen model state available at `decision_date`.
2. Feedback derived from the horizon is eligible only after `horizon_end_date`.
3. A feedback row from cycle `t` may update only a later cycle `u` where `u > t` in the same site's chronological order.
4. No target gain, oracle irrigation, best candidate, or future-cycle outcome may be used to compute an online update.
5. Exact SWAP fields are labeled `shadow_feedback` because they are post-hoc research artifacts, not deploy-time observations.

## Outputs

The future implementation will write a new versioned directory, without touching frozen v8 directories:

- `gefs_gam_v8_tta_feedback_audit_v1.csv`: one row per site-cycle and feedback signal, with availability, source, first-available timestamp, eligible update-cycle index, and causal checks.
- `gefs_gam_v8_tta_feedback_audit_v1.json`: input hashes, row counts, signal counts, missing-field counts, and mandatory gate status.
- `gefs_gam_v8_tta_feedback_audit_manifest_v1.csv`: output hashes and provenance.

## Mandatory Gates

The audit passes only when:

- all five sites and all 68 2024 cycles are represented exactly once;
- frozen input hashes match the recorded values;
- every numeric feedback value is finite when present;
- no feedback row is eligible for its own decision cycle;
- no target gain, oracle irrigation, or best-candidate column is used in the eligibility calculation;
- every `physics_only_available` row has complete water-balance fields;
- every `delayed_vwc_available` row has a valid post-horizon VWC source and a later eligible cycle;
- missing or ambiguous feedback is explicitly retained and counted;
- the audit reports `tta_performed=false` and `model_training_performed=false`.

## Non-goals

This version does not choose TTA learning rates, loss weights, adapter parameterization, gate-update rules, or performance thresholds. Those must be fixed from 2015-2019 validation before a 2024 TTA replay is run.

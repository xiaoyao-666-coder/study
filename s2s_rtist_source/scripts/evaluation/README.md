# Evaluation

## Purpose
Scripts that evaluate policies, summarize model results, finalize tables, and apply learned decision rules.

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
| `apply-lstm-twostage-irrigation-policy-v1` | active | `apply_lstm_twostage_irrigation_policy_v1.py` | apply lstm twostage irrigation policy v1 |
| `apply-safe-threshold-policy-v1` | active | `apply_safe_threshold_policy_v1.py` | apply safe threshold policy v1 |
| `collect-decision-smoke-results` | active | `collect_decision_smoke_results.py` | collect decision smoke results |
| `evaluate-binary-trigger-fewshot-site-calibration-v1` | active | `evaluate_binary_trigger_fewshot_site_calibration_v1.py` | evaluate binary trigger fewshot site calibration v1 |
| `evaluate-binary-trigger-loso-calibration-policies-v1` | active | `evaluate_binary_trigger_loso_calibration_policies_v1.py` | evaluate binary trigger loso calibration policies v1 |
| `evaluate-binary-trigger-nested-calibration-selector-v1` | active | `evaluate_binary_trigger_nested_calibration_selector_v1.py` | evaluate binary trigger nested calibration selector v1 |
| `evaluate-binary-trigger-site-rate-policy-v1` | active | `evaluate_binary_trigger_site_rate_policy_v1.py` | evaluate binary trigger site rate policy v1 |
| `evaluate-binary-trigger-threshold-transfer-v1` | active | `evaluate_binary_trigger_threshold_transfer_v1.py` | evaluate binary trigger threshold transfer v1 |
| `evaluate-confirmed-5site-collapse-guard-v1` | active | `evaluate_confirmed_5site_collapse_guard_v1.py` | evaluate confirmed 5site collapse guard v1 |
| `evaluate-confirmed-5site-surrogate-policy-v1` | active | `evaluate_confirmed_5site_surrogate_policy_v1.py` | evaluate confirmed 5site surrogate policy v1 |
| `evaluate-gefs-member-ensemble-policy-v1` | formal | `evaluate_gefs_member_ensemble_policy_v1.py` | aggregate member-level GEFS profit predictions and select irrigation by ensemble-mean profit |
| `evaluate-expanded-plateau-amount-policy-v1` | active | `evaluate_expanded_plateau_amount_policy_v1.py` | evaluate expanded plateau amount policy v1 |
| `evaluate-expanded-stage-cap-policy-v1` | active | `evaluate_expanded_stage_cap_policy_v1.py` | evaluate expanded stage cap policy v1 |
| `evaluate-fixed-list-local-surrogate-refinement-v1` | active | `evaluate_fixed_list_local_surrogate_refinement_v1.py` | evaluate fixed list local surrogate refinement v1 |
| `evaluate-learned-trigger-curve-policy-expanded-v1` | active | `evaluate_learned_trigger_curve_policy_expanded_v1.py` | evaluate learned trigger curve policy expanded v1 |
| `evaluate-learned-trigger-curve-policy-v1` | active | `evaluate_learned_trigger_curve_policy_v1.py` | evaluate learned trigger curve policy v1 |
| `evaluate-learned-trigger-curve-policy-v2` | active | `evaluate_learned_trigger_curve_policy_v2.py` | evaluate learned trigger curve policy v2 |
| `evaluate-persite-curve-ranker-guard-policies-v1` | active | `evaluate_persite_curve_ranker_guard_policies_v1.py` | evaluate persite curve ranker guard policies v1 |
| `evaluate-persite-tinyforest-fewshot-dates-v1` | active | `evaluate_persite_tinyforest_fewshot_dates_v1.py` | evaluate persite tinyforest fewshot dates v1 |
| `evaluate-threshold-curve-policy-fast-v1` | active | `evaluate_threshold_curve_policy_fast_v1.py` | evaluate threshold curve policy fast v1 |
| `evaluate-threshold-curve-policy-v1` | active | `evaluate_threshold_curve_policy_v1.py` | evaluate threshold curve policy v1 |
| `evaluate-tta-base-plus-rolling-calibration-v1` | active | `evaluate_tta_base_plus_rolling_calibration_v1.py` | evaluate tta base plus rolling calibration v1 |
| `evaluate-tta-deployable-guard-from-lightweight-calibration-v1` | active | `evaluate_tta_deployable_guard_from_lightweight_calibration_v1.py` | evaluate tta deployable guard from lightweight calibration v1 |
| `evaluate-tta-lightweight-output-calibration-v1` | active | `evaluate_tta_lightweight_output_calibration_v1.py` | evaluate tta lightweight output calibration v1 |
| `evaluate-tta-rolling-date-coverage-v1` | active | `evaluate_tta_rolling_date_coverage_v1.py` | evaluate tta rolling date coverage v1 |
| `finalize-shortterm-surrogate-v1` | active | `finalize_shortterm_surrogate_v1.py` | finalize shortterm surrogate v1 |
| `summarize-binary-trigger-calibration-selector-data-need-v1` | active | `summarize_binary_trigger_calibration_selector_data_need_v1.py` | summarize binary trigger calibration selector data need v1 |
| `summarize-binary-trigger-mainline-v1` | active | `summarize_binary_trigger_mainline_v1.py` | summarize binary trigger mainline v1 |
| `summarize-continuous-ir-surrogate-models-v1` | active | `summarize_continuous_ir_surrogate_models_v1.py` | summarize continuous ir surrogate models v1 |
| `summarize-fixed-list-local-refinement-mainline-v1` | active | `summarize_fixed_list_local_refinement_mainline_v1.py` | summarize fixed list local refinement mainline v1 |
| `summarize-paper-fixed-list-era5-single-scenario-v1` | active | `summarize_paper_fixed_list_era5_single_scenario_v1.py` | summarize paper fixed list era5 single scenario v1 |
| `summarize-restart-dataset` | active | `summarize_restart_dataset.py` | summarize restart dataset |

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one.

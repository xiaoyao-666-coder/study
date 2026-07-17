# Diagnostics

## Purpose
Scripts that audit inputs, compare policies, diagnose failures, and run formal validation diagnostics.

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
| `analyze-discrete-vs-lstm-guard-v1` | active | `analyze_discrete_vs_lstm_guard_v1.py` | analyze discrete vs lstm guard v1 |
| `audit-confirmed-5site-input-generation-v1` | active | `audit_confirmed_5site_input_generation_v1.py` | audit confirmed 5site input generation v1 |
| `audit-confirmed-5site-multidate-training-readiness-v1` | active | `audit_confirmed_5site_multidate_training_readiness_v1.py` | audit confirmed 5site multidate training readiness v1 |
| `audit-confirmed-5site-restart-curves-v1` | active | `audit_confirmed_5site_restart_curves_v1.py` | audit confirmed 5site restart curves v1 |
| `audit-multisite-inputs-v1` | active | `audit_multisite_inputs_v1.py` | audit multisite inputs v1 |
| `audit-uploaded-base-data` | active | `audit_uploaded_base_data.py` | audit uploaded base data |
| `compare-discrete-vs-continuous-ir-optimization-v1` | active | `compare_discrete_vs_continuous_ir_optimization_v1.py` | compare discrete vs continuous ir optimization v1 |
| `compare-expanded-policy-results-v3` | active | `compare_expanded_policy_results_v3.py` | compare expanded policy results v3 |
| `compare-restart-to-author-ensemble` | active | `compare_restart_to_author_ensemble.py` | compare restart to author ensemble |
| `diagnose-binary-trigger-failures-v1` | active | `diagnose_binary_trigger_failures_v1.py` | diagnose binary trigger failures v1 |
| `diagnose-binary-trigger-loso-calibration-gap-v1` | active | `diagnose_binary_trigger_loso_calibration_gap_v1.py` | diagnose binary trigger loso calibration gap v1 |
| `diagnose-binary-trigger-site-threshold-oracle-v1` | active | `diagnose_binary_trigger_site_threshold_oracle_v1.py` | diagnose binary trigger site threshold oracle v1 |
| `diagnose-continuous-ir-site-generalization-v1` | active | `diagnose_continuous_ir_site_generalization_v1.py` | diagnose continuous ir site generalization v1 |
| `diagnose-curve-top-failure-from-decisions-v1` | active | `diagnose_curve_top_failure_from_decisions_v1.py` | diagnose curve top failure from decisions v1 |
| `diagnose-expanded-date-feature-state-v1` | active | `diagnose_expanded_date_feature_state_v1.py` | diagnose expanded date feature state v1 |
| `diagnose-expanded-policy-worst-dates-v1` | active | `diagnose_expanded_policy_worst_dates_v1.py` | diagnose expanded policy worst dates v1 |
| `diagnose-failure-case-environment-from-public-data-v1` | active | `diagnose_failure_case_environment_from_public_data_v1.py` | diagnose failure case environment from public data v1 |
| `diagnose-fixed-list-local-refinement-headroom-v1` | active | `diagnose_fixed_list_local_refinement_headroom_v1.py` | diagnose fixed list local refinement headroom v1 |
| `diagnose-lstm-ranking-vs-discrete-policy-v1` | active | `diagnose_lstm_ranking_vs_discrete_policy_v1.py` | diagnose lstm ranking vs discrete policy v1 |
| `diagnose-persite-curve-ranker-cv-failures-v1` | active | `diagnose_persite_curve_ranker_cv_failures_v1.py` | diagnose persite curve ranker cv failures v1 |
| `diagnose-persite-curve-top-ranker-cv-failures-v1` | active | `diagnose_persite_curve_top_ranker_cv_failures_v1.py` | diagnose persite curve top ranker cv failures v1 |
| `diagnose-tree-surrogate-curves-v1` | active | `diagnose_tree_surrogate_curves_v1.py` | diagnose tree surrogate curves v1 |
| `gefs-gridmet-bias` | formal | `run_gefs_gridmet_bias_validation_v1.py` | run gefs gridmet bias validation v1 |
| `gefs-member-gridmet-validation` | formal | `run_gefs_member_gridmet_validation_v1.py` | validate 31 GEFS members against gridMET with CRPS, coverage, and precipitation probability metrics |
| `map-code-sites-to-paper-fig4-v1` | active | `map_code_sites_to_paper_fig4_v1.py` | map code sites to paper fig4 v1 |
| `restart-raw-audit` | formal | `restart_raw_audit_v1.py` | restart raw audit v1 |
| `rootzone-frequency` | formal | `run_rootzone_flux_frequency_validation_v1.py` | run rootzone flux frequency validation v1 |

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one.

# Data Preparation

## Purpose
Scripts that prepare workspaces, features, weather/soil inputs, and surrogate tables before training or evaluation.

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
| `apply-confirmed-5site-gridmet-weather-inputs-v1` | active | `apply_confirmed_5site_gridmet_weather_inputs_v1.py` | apply confirmed 5site gridmet weather inputs v1 |
| `apply-confirmed-5site-polaris-soil-inputs-v1` | active | `apply_confirmed_5site_polaris_soil_inputs_v1.py` | apply confirmed 5site polaris soil inputs v1 |
| `apply-confirmed-5site-static-swp-inputs-v1` | active | `apply_confirmed_5site_static_swp_inputs_v1.py` | apply confirmed 5site static swp inputs v1 |
| `apply-continuous-ir-12site-gridmet-weather-inputs-v1` | active | `apply_continuous_ir_12site_gridmet_weather_inputs_v1.py` | apply continuous ir 12site gridmet weather inputs v1 |
| `apply-continuous-ir-12site-static-polaris-inputs-v1` | active | `apply_continuous_ir_12site_static_polaris_inputs_v1.py` | apply continuous ir 12site static polaris inputs v1 |
| `build-binary-trigger-calibration-selector-supervision-v1` | active | `build_binary_trigger_calibration_selector_supervision_v1.py` | build binary trigger calibration selector supervision v1 |
| `build-confirmed-5site-true-input-surrogate-features-v1` | active | `build_confirmed_5site_true_input_surrogate_features_v1.py` | build confirmed 5site true input surrogate features v1 |
| `build-confirmed-5site-true-input-surrogate-table-v1` | active | `build_confirmed_5site_true_input_surrogate_table_v1.py` | build confirmed 5site true input surrogate table v1 |
| `build-continuous-ir-12site-surrogate-features-v1` | active | `build_continuous_ir_12site_surrogate_features_v1.py` | build continuous ir 12site surrogate features v1 |
| `build-continuous-ir-sequence-wide-features-v1` | active | `build_continuous_ir_sequence_wide_features_v1.py` | build continuous ir sequence wide features v1 |
| `build-expanded-formal-evaluation-table-v1` | active | `build_expanded_formal_evaluation_table_v1.py` | build expanded formal evaluation table v1 |
| `build-shortterm-surrogate-expanded-v1` | active | `build_shortterm_surrogate_expanded_v1.py` | build shortterm surrogate expanded v1 |
| `extract-current-state-v1` | active | `extract_current_state_v1.py` | extract current state v1 |
| `extract-soil-pressure-state-v1` | active | `extract_soil_pressure_state_v1.py` | extract soil pressure state v1 |
| `extract-true-predecision-state-v1` | active | `extract_true_predecision_state_v1.py` | extract true predecision state v1 |
| `extract-weather-sequences-v1` | active | `extract_weather_sequences_v1.py` | extract weather sequences v1 |
| `extract-weather-sequences-v2` | active | `extract_weather_sequences_v2.py` | extract weather sequences v2 |
| `merge-current-state-into-shortterm-v1` | active | `merge_current_state_into_shortterm_v1.py` | merge current state into shortterm v1 |
| `merge-soil-pressure-into-shortterm-v1` | active | `merge_soil_pressure_into_shortterm_v1.py` | merge soil pressure into shortterm v1 |
| `merge-true-state-into-shortterm-v1` | active | `merge_true_state_into_shortterm_v1.py` | merge true state into shortterm v1 |
| `plan-continuous-irrigation-sampling-v1` | active | `plan_continuous_irrigation_sampling_v1.py` | plan continuous irrigation sampling v1 |
| `plan-old-year-paper-schedule-doy-aligned-v1` | active | `plan_old_year_paper_schedule_doy_aligned_v1.py` | plan old year paper schedule doy aligned v1 |
| `plan-tta-date-coverage-sampling-v1` | active | `plan_tta_date_coverage_sampling_v1.py` | plan tta date coverage sampling v1 |
| `prepare-author-ensemble-label-table` | active | `prepare_author_ensemble_label_table.py` | prepare author ensemble label table |
| `prepare-confirmed-5site-workspaces-v1` | active | `prepare_confirmed_5site_workspaces_v1.py` | prepare confirmed 5site workspaces v1 |
| `prepare-continuous-ir-12site-workspaces-v1` | active | `prepare_continuous_ir_12site_workspaces_v1.py` | prepare continuous ir 12site workspaces v1 |
| `prepare-restart-surrogate-table` | active | `prepare_restart_surrogate_table.py` | prepare restart surrogate table |
| `prepare-shortterm-surrogate-v1` | active | `prepare_shortterm_surrogate_v1.py` | prepare shortterm surrogate v1 |

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one.

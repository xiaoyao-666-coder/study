# Script Version Notes

This file records explicit replacement relationships. Do not infer a replacement solely from a higher version number.

## Confirmed Replacements

| Older script | Replaced by | Evidence |
|---|---|---|
| `compare_expanded_policy_results_v1.py` | `compare_expanded_policy_results_v3.py` | Locked migration rule: v3 is the active comparator and v1/v2 are archived as superseded. |
| `compare_expanded_policy_results_v2.py` | `compare_expanded_policy_results_v3.py` | Locked migration rule: v3 is the active comparator and v1/v2 are archived as superseded. |

## Historical Without Confirmed Replacement

| Script | Notes |
|---|---|
| `extract_weather_sequences_v1.py` | Coexists with `extract_weather_sequences_v2.py`. Keep both historical/active as cataloged; do not set `replaced_by` without import or result evidence. |
| `evaluate_learned_trigger_curve_policy_v1.py` | Coexists with `evaluate_learned_trigger_curve_policy_v2.py` and the expanded evaluator. Keep both until results/docs confirm a single replacement. |

## Original Application Entry Points

| Script | Status | Reason preserved |
|---|---|---|
| `Main_win.py` | historical | Original SWAP/RTIST application entry point retained for reference and reproducibility. |
| `Main_win_ensemble_mean.py` | historical | Original ensemble-mean application entry point retained for reference and reproducibility. |

Both live under `scripts/archive/original_application/` and are not formal package APIs.

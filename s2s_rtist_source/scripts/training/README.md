# Training

## Purpose
Scripts that train, calibrate, sweep, optimize, or smooth surrogate and ranking models.

## Current Scripts
| ID | Status | Script | Purpose |
|---|---|---|---|
| `calibrate-continuous-irrigation-twohead-threshold-v1` | active | `calibrate_continuous_irrigation_twohead_threshold_v1.py` | calibrate continuous irrigation twohead threshold v1 |
| `optimize-continuous-irrigation-lstm-surrogate-v1` | active | `optimize_continuous_irrigation_lstm_surrogate_v1.py` | optimize continuous irrigation lstm surrogate v1 |
| `resweep-binary-trigger-thresholds-v1` | active | `resweep_binary_trigger_thresholds_v1.py` | resweep binary trigger thresholds v1 |
| `smooth-tree-surrogate-candidates-v1` | active | `smooth_tree_surrogate_candidates_v1.py` | smooth tree surrogate candidates v1 |
| `sweep-continuous-ir-twohead-threshold-v1` | active | `sweep_continuous_ir_twohead_threshold_v1.py` | sweep continuous ir twohead threshold v1 |
| `train-confirmed-5site-true-input-surrogate-baseline-v1` | active | `train_confirmed_5site_true_input_surrogate_baseline_v1.py` | train confirmed 5site true input surrogate baseline v1 |
| `train-continuous-irrigation-binary-trigger-lstm-v1` | active | `train_continuous_irrigation_binary_trigger_lstm_v1.py` | train continuous irrigation binary trigger lstm v1 |
| `train-continuous-irrigation-binary-trigger-weighted-lstm-v1` | active | `train_continuous_irrigation_binary_trigger_weighted_lstm_v1.py` | train continuous irrigation binary trigger weighted lstm v1 |
| `train-continuous-irrigation-surrogate-baseline-v1` | active | `train_continuous_irrigation_surrogate_baseline_v1.py` | train continuous irrigation surrogate baseline v1 |
| `train-continuous-irrigation-surrogate-lstm-ranker-v1` | active | `train_continuous_irrigation_surrogate_lstm_ranker_v1.py` | train continuous irrigation surrogate lstm ranker v1 |
| `train-continuous-irrigation-surrogate-lstm-v1` | active | `train_continuous_irrigation_surrogate_lstm_v1.py` | train continuous irrigation surrogate lstm v1 |
| `train-continuous-irrigation-surrogate-lstm-weighted-v1` | active | `train_continuous_irrigation_surrogate_lstm_weighted_v1.py` | train continuous irrigation surrogate lstm weighted v1 |
| `train-continuous-irrigation-surrogate-mlp-nosklearn-v1` | active | `train_continuous_irrigation_surrogate_mlp_nosklearn_v1.py` | train continuous irrigation surrogate mlp nosklearn v1 |
| `train-continuous-irrigation-surrogate-persite-tree-v1` | active | `train_continuous_irrigation_surrogate_persite_tree_v1.py` | train continuous irrigation surrogate persite tree v1 |
| `train-continuous-irrigation-surrogate-tree-nosklearn-v1` | active | `train_continuous_irrigation_surrogate_tree_nosklearn_v1.py` | train continuous irrigation surrogate tree nosklearn v1 |
| `train-continuous-irrigation-surrogate-twohead-v1` | active | `train_continuous_irrigation_surrogate_twohead_v1.py` | train continuous irrigation surrogate twohead v1 |
| `train-fixed-list-local-refinement-tree-v1` | active | `train_fixed_list_local_refinement_tree_v1.py` | train fixed list local refinement tree v1 |
| `train-persite-curve-dual-score-tinyforest-ranker-v1` | active | `train_persite_curve_dual_score_tinyforest_ranker_v1.py` | train persite curve dual score tinyforest ranker v1 |
| `train-persite-curve-mlp-ranker-v1` | active | `train_persite_curve_mlp_ranker_v1.py` | train persite curve mlp ranker v1 |
| `train-persite-curve-top-tinyforest-ranker-v1` | active | `train_persite_curve_top_tinyforest_ranker_v1.py` | train persite curve top tinyforest ranker v1 |
| `train-persite-curve-zero-margin-tinyforest-ranker-v1` | active | `train_persite_curve_zero_margin_tinyforest_ranker_v1.py` | train persite curve zero margin tinyforest ranker v1 |
| `train-persite-fixedlist-first-tinyforest-ranker-v1` | active | `train_persite_fixedlist_first_tinyforest_ranker_v1.py` | train persite fixedlist first tinyforest ranker v1 |
| `train-persite-lstm-profit-surrogate-v1` | active | `train_persite_lstm_profit_surrogate_v1.py` | train persite lstm profit surrogate v1 |
| `train-persite-tinyforest-profit-surrogate-v1` | active | `train_persite_tinyforest_profit_surrogate_v1.py` | train persite tinyforest profit surrogate v1 |
| `train-shortterm-surrogate-baseline-v1` | active | `train_shortterm_surrogate_baseline_v1.py` | train shortterm surrogate baseline v1 |
| `train-shortterm-surrogate-ranker-v1` | active | `train_shortterm_surrogate_ranker_v1.py` | train shortterm surrogate ranker v1 |
| `train-shortterm-surrogate-tree-nosklearn-expanded-v1` | active | `train_shortterm_surrogate_tree_nosklearn_expanded_v1.py` | train shortterm surrogate tree nosklearn expanded v1 |
| `train-shortterm-surrogate-tree-nosklearn-v1` | active | `train_shortterm_surrogate_tree_nosklearn_v1.py` | train shortterm surrogate tree nosklearn v1 |
| `train-shortterm-surrogate-tree-v1` | active | `train_shortterm_surrogate_tree_v1.py` | train shortterm surrogate tree v1 |

## Usage
`python project_cli.py show <id>`
`python project_cli.py run <id> -- <args>`

## Version Notes
Formal package modules live under `src/s2s_rtist/`. See `scripts/archive/VERSIONS.md` when a higher-version script replaces an older one.

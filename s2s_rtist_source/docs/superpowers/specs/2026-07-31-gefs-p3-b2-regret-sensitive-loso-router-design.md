# P3 B2 Regret-Sensitive Leave-Site-Out Router Design

## Status

Frozen design approved on 2026-07-31. This document defines one minimal B2
development screen. It does not authorize reuse of P3 2021 as an independent
test and does not modify the frozen B1 router.

## Objective

Test whether the existing P1/P15 source-only router can improve unseen-site
generalization by changing exactly two mechanisms:

1. train the logistic gate with fold-local regret-sensitive sample weights;
2. validate with rolling-year, leave-site-out folds across P2 and P4.

The B2 screen must not read P3, 2020, 2021, or 2024 feature or target rows.
P3 2021 is already consumed evidence and is not part of model selection,
development gating, or independent evaluation for B2.

## Frozen Components

B2 retains all of the following from B1 without search or modification:

- P1 and P15 composite-only source experts;
- `lambda_balance=1.0`;
- the existing nine decision-time physical gate features;
- the fixed irrigation grid `[0, 10, 15, 20, 25, 30, 40, 60] mm`;
- feature standardization by unweighted training-fold mean and standard deviation;
- logistic probability threshold `0.5`;
- P15 routing on an exact threshold tie;
- always-P15 fallback behavior;
- downstream SWAP metrics and definitions.

B2 does not add expert-prediction features, tune a threshold, cap 60 mm,
change source checkpoints, retrain source experts, or introduce a second gate
model.

## Eight Development Folds

The validation years are 2016, 2017, 2018, and 2019. Each year is crossed
with held-out sites P2 and P4 for exactly eight folds.

For a fold `(held_out_site, validation_year)`:

- the gate training site is the other member of `{P2, P4}`;
- gate training years are all years from 2015 through
  `validation_year - 1`;
- gate validation uses only `held_out_site` in `validation_year`;
- the source experts are the existing fold-local P1/P15 checkpoints whose
  source training scope ends before `validation_year`;
- validation labels are read only after the gate model is frozen.

Example: `holdout_P2_to_2018` trains the gate on P4 2015-2017 and validates
on P2 2018 using the existing source checkpoints for the 2018 rolling fold.

The loader must first inspect only row identity, site, and year fields, then
physically skip P3 and disallowed years before materializing features or
targets. A fold fails closed if it contains incomplete eight-candidate cycles,
the training and validation sites overlap, or a future year survives loading.

## Regret-Sensitive Logistic Gate

Only actionable training cycles are used: P1 and P15 must recommend different
irrigation amounts and their realized gains must not tie. For actionable cycle
`i`:

```text
gain_delta_i = realized_gain_P1_i - realized_gain_P15_i
label_i = 1 if gain_delta_i > 0 else 0
raw_weight_i = abs(gain_delta_i)
sample_weight_i = raw_weight_i / mean(raw_weight_training_fold)
```

Weights are computed only from the current gate-training fold. They are not
clipped and no weighting hyperparameter is searched. Their training-fold mean
must be one within floating-point tolerance.

The fixed objective is:

```text
sum(sample_weight_i * binary_cross_entropy_i)
+ 0.5 * sum(non_intercept_coefficients ** 2)
```

The NumPy Newton/IRLS implementation applies sample weights in both the
gradient and Hessian. The intercept remains unregularized. Feature
standardization remains unweighted so regret weighting is the only optimizer
change.

The gate falls back to always-P15 when any of the following holds:

- fewer than eight actionable training cycles;
- only one winner class;
- non-finite features, gain deltas, weights, coefficients, or solver steps;
- mean absolute gain delta is at most `1e-12`;
- a singular Hessian;
- failure to converge in 100 iterations at parameter tolerance `1e-10`.

Every training sample output records realized gain delta, raw absolute weight,
normalized sample weight, fold identity, training site, and training years.

## Development Pass Gate

The sole formal baseline is always-P15. Metrics are first computed per fold,
then macro-aggregated with equal fold weight. B2 passes only if every condition
holds:

1. eight-fold macro mean regret is at least `1e-6` lower than always-P15;
2. P2 four-fold mean regret is no worse than its baseline by more than `1e-6`;
3. P4 four-fold mean regret is no worse than its baseline by more than `1e-6`;
4. at least six of eight folds have mean regret no worse than baseline by more
   than `1e-6`;
5. the worst fold maximum regret is no greater than baseline by more than
   `1e-6`;
6. total false-positive irrigation count is no greater;
7. total predicted-60-mm count is no greater;
8. at least one P1 route changes the always-P15 irrigation decision;
9. each held-out site has at least one fold with a successfully fitted,
   non-fallback gate.

Failure of any condition freezes B2 as a negative result. The criteria may not
be weakened after results are observed.

## Final Gate Boundary

Only after all development conditions pass may the runner fit one final B2
gate using P2 and P4 from 2015-2019 with the existing final full-history P1/P15
experts. That gate is marked
`pending_new_independent_freeze`; it is not evaluated on P3, 2021, or 2024.

If development fails, no promotable final gate or policy is written. Diagnostic
fold artifacts remain available for review.

## Outputs

The new runner writes to a separate B2 directory and never overwrites B1:

```text
gefs_p3_b2_regret_sensitive_training_samples_v1.csv
gefs_p3_b2_regret_sensitive_validation_decisions_v1.csv
gefs_p3_b2_regret_sensitive_per_fold_metrics_v1.csv
gefs_p3_b2_regret_sensitive_per_site_metrics_v1.csv
gefs_p3_b2_regret_sensitive_development_gate_v1.csv
gefs_p3_b2_regret_sensitive_fold_models_v1.json
gefs_p3_b2_regret_sensitive_final_gate_v1.json          # pass only
gefs_p3_b2_regret_sensitive_router_screen_audit_v1.json
gefs_p3_b2_regret_sensitive_router_screen_manifest_v1.csv
```

The only final audit states are:

```text
b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze
b2_regret_sensitive_loso_development_failed_frozen_negative
```

The audit records zero P3/2020/2021/2024 target rows, zero source checkpoint
retraining or reselection, zero threshold search, all fold scopes, gate
fallback reasons, every pass condition, and hashes of all inputs and outputs.

## Testing

Tests must cover:

- exactly eight rolling-year leave-site-out folds;
- opposite training and validation sites and strictly earlier training years;
- physical target-row skipping before numeric loading;
- complete eight-candidate cycles;
- exact fold-local weight arithmetic and mean-one normalization;
- invariance of gate weights and hashes to validation-target changes;
- deterministic weighted IRLS and all fallback paths;
- each development condition in pass and fail states;
- output cardinality, audit state, and manifest integrity in an end-to-end CPU
  fixture;
- unchanged behavior of the existing unweighted B1 logistic gate.

## Packaging And Execution

The server ZIP contains only the new runner, its tests, and required reusable
modules. It excludes datasets, checkpoints, logs, PID files, 2021 results, and
all server outputs. Archive verification rejects unsafe paths, backslashes,
duplicates, and case-insensitive collisions, then runs the B2 tests from a
clean extraction directory.

Server delivery must use full absolute paths and `python3`. The final handoff
includes extraction, tests, background launch, PID, `ps`, `tail -f`, GPU
status, progress, exception search, output listing, and final audit/CSV readers
in the same response.

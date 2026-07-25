# GEFS Exact-Schedule OOF Irrigation Gate

## Purpose

Stage46 is the frozen no-TTA curve-model baseline.  Its validation-selected
checkpoint improves 2019 mean regret, maximum regret, P3 regret, and
nonzero-irrigation recall relative to stage44, but it retains seven
false-positive irrigation decisions.  Those seven decisions account for 171.0
of the 328.8 total 2019 regret.

Stage47 isolates the question of whether a training-only out-of-fold (OOF)
gate can suppress unsafe nonzero recommendations without changing the frozen
stage46 curve model or using 2019 to fit the gate.

## Scope And Frozen Inputs

- The final stage46 checkpoint is frozen at SHA256
  `e580be6fa60e4508a896ddcc90e54c972f938e8bdfaa46eab94372991fc5bf51`.
- The stage46 feature schema, preprocessing, curve architecture, curve-aware
  loss, optimizer, seed, candidate set, and fixed training duration of 929
  epochs are unchanged for all OOF mirror models.
- No physics loss, TTA, additional SWAP simulation, feature download, or
  data generation is permitted.
- The formal data split remains 2015-2018 for gate development, 2019 for one
  post-lock validation run, and 2024 untouched.

The OOF mirror models are not candidates for promotion.  Their sole purpose is
to generate a prediction distribution on each training year that was produced
without fitting that year's labels.

## OOF Curve Features

For each held-out year in 2015, 2016, 2017, and 2018:

1. Fit preprocessing on the remaining three years only.
2. Train one stage46-identical curve model on those three years for exactly
   929 epochs.  Do not inspect or select a checkpoint using the held-out year.
3. Predict all eight candidates for every held-out cycle.
4. Record the original nonzero or zero recommendation, the full predicted
   curve, and the actual SWAP curve for audit only.

The gate sees only information available at decision time:

- site identifier;
- the candidate-invariant features from the `0 mm` row, excluding target
  columns and irrigation amount;
- predicted selected irrigation amount;
- selected predicted net gain, the gap to `0 mm`, the gap to the second-best
  candidate, and curve summaries derived solely from the eight predictions.

It never receives a target column or a realised irrigation label as an input.

## Gate Target And Model

The gate acts only when the frozen curve policy's recommended candidate is
nonzero.  Its binary target is whether that recommended nonzero candidate has
strictly positive realised seven-day net gain relative to the anchored `0 mm`
candidate.  A negative or zero realised gain is an unsafe recommendation and
should be blocked.

Use a regularized binary logistic gate with site one-hot encoding and
standardized numeric features.  The small, linear model is deliberate: there
are 269 training cycles, so a deeper classifier would make a favorable 2019
result difficult to attribute and audit.  Class weighting is fitted from each
training partition only.

## Gate Calibration

The gate probabilities used to choose a threshold must themselves be
cross-fitted.  For each training year, train the gate on OOF curve features
from the other three years and predict the held-out year's features.  The
result is one out-of-gate probability for every 2015-2018 cycle.

Choose the threshold only from those out-of-gate probabilities.  For every
unique probability threshold, apply this policy to the OOF curve decisions:

- retain the curve model's recommendation when it is `0 mm` or the gate
  probability is at least the threshold;
- otherwise override the decision to `0 mm`.

Eligible thresholds must keep OOF nonzero-oracle recall at least 0.90 and not
increase OOF maximum regret over the unguarded OOF decisions.  Among eligible
thresholds, select lexicographically by lowest OOF mean regret, then lowest
false-positive count, then highest exact-match rate, then highest threshold.
The final gate is fitted on all 2015-2018 OOF-feature rows using the locked
feature schema and selected threshold.

## Final 2019 Evaluation And Promotion Gates

At inference, compute gate features from the frozen stage46 2019 prediction
curves.  The gate may only override a nonzero stage46 recommendation to
`0 mm`; it cannot choose a different nonzero amount.  This keeps the change
strictly attributable to the irrigation/no-irrigation decision.

Compare the guarded policy with the frozen stage46 decisions.  Promotion
requires all of the following prelocked gates:

1. mean 7-day regret is strictly below `4.765217391304349`;
2. maximum 7-day regret is at most `39.2`;
3. exact irrigation match rate is at least `0.6176811594202898`;
4. nonzero-irrigation recall is at least `0.95`;
5. P3 mean 7-day regret is at most `5.733333333333333`;
6. false-positive irrigation count is below `7`.

The 2019 set has already selected the stage46 checkpoint.  It is therefore a
validation/model-selection set, not an untouched test set; it must not be
reused to revise the gate target, features, threshold, or gates after this
run.  A passing candidate is a research baseline only and still requires a
separate 2024 evaluation before any deployment claim.

## Artifacts And Tests

The training script will write a gate checkpoint, final policy, OOF curve and
gate audit tables, threshold sweep, 2019 predictions and decisions, metrics,
stage46 comparison, provenance manifest, and SHA256 hashes.  It must refuse to
overwrite an existing output directory and verify the frozen stage46 checkpoint
hash before use.

Tests will cover OOF year separation, exclusion of targets from gate features,
zero-only override behavior, deterministic threshold selection, frozen
checkpoint-hash verification, and a CPU smoke run that writes all declared
artifacts.  The server training command will explicitly set
`CUDA_VISIBLE_DEVICES=0` and pass `--device cuda`.

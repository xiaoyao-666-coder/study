# GEFS Exact-Schedule Joint Benefit Head

## Purpose

Stage46 remains the current no-TTA research baseline.  Stage47 showed that a
post-hoc linear gate over frozen stage46 predictions cannot reduce the seven
2019 false-positive irrigation decisions while preserving recall and maximum
regret.  The OOF gate contained limited classification signal, but its safe
threshold collapsed to an effectively non-intervening policy.

Stage48 tests one controlled hypothesis: sharing the curve model's learned
representation with a directly supervised cycle-level irrigation-benefit head
can improve the irrigation/no-irrigation boundary without abandoning the
formal curve-regression task.

## Mainline Boundaries

- Inputs remain the frozen, leakage-audited six-variable corrected GEFS
  features and verified continuous-season checkpoint state.
- Targets remain SWAP-generated model labels, not observations.
- Training and all design decisions use 2015-2018 only.  The 2019 data remain
  a validation/model-selection set, and 2024 remains untouched.
- The eight label-generation candidates remain
  `[0, 10, 15, 20, 25, 30, 40, 60] mm`.
- No new weather download, SWAP simulation, feature, physics loss, TTA,
  hyperparameter sweep, or continuous-irrigation optimizer is introduced.

Stage48 is a no-TTA pretraining ablation on the path to the teacher-approved
three-output and later physics/TTA workflow.  A passing result is not a 2024
test result or a deployment claim.

## Fixed Auxiliary Target

For each complete eight-candidate cycle, define one binary target:

```text
beneficial_irrigation =
    max(target_net_gain_7d for irrigation_mm > 0) > 1e-6
```

The `0 mm` net gain is exactly anchored at zero.  The tolerance prevents
floating-point noise around zero from changing the class.  This target answers
whether any nonzero irrigation is beneficial.  It does not depend on the
model's current selected dose and therefore does not move during training.

## Architecture

Keep the stage46 site embedding, two-layer MLP trunk, zero-anchored net-gain
head, nonnegative AET head, and bounded seven-day VWC head.  Add one linear
benefit-logit head to the shared 64-dimensional trunk representation.

The benefit head is evaluated once per cycle using only the hidden
representation of the `0 mm` row.  All candidate-invariant state, date, site,
and forecast fields are available, while the proposed irrigation amount is
zero.  The head cannot read a target column, realised SWAP output, or another
candidate's irrigation amount.

The resulting policy has two outputs with separate responsibilities:

1. the benefit head decides whether nonzero irrigation is allowed;
2. the existing curve head chooses the amount by hard argmax when irrigation
   is allowed.

The benefit head can only override a nonzero curve recommendation to `0 mm`.
It cannot choose or modify a nonzero amount.

## Joint Loss

Retain the complete stage46 curve-aware supervised loss, including the
zero-anchor, pointwise net-gain error, pairwise ranking, soft regret, AET, and
VWC terms.  Add unweighted binary cross-entropy on the cycle-level benefit
logit:

```text
joint_loss =
    0.80 * stage46_curve_aware_loss
  + 0.20 * binary_cross_entropy_with_logits
```

The internal stage46 curve-task ratios remain unchanged.  Class weighting,
focal loss, threshold sweeps, and loss-weight sweeps are prohibited in this
stage.  The fixed decision threshold is `sigmoid(logit) >= 0.5`.

## Training And OOF Gate

Use the stage46 seed `20260725`, AdamW optimizer, learning rate `1e-3`, weight
decay `1e-4`, cycle batch size `16`, gradient clipping at `5.0`, and exactly
`929` epochs.  Train from scratch; stage46 weights are a comparison baseline,
not an initialization.  There is no early stopping or 2019 checkpoint
selection.

Before training a final model, run four leave-one-year-out folds over
2015-2018.  In every fold:

1. fit preprocessing on the other three years only;
2. train the joint model for exactly 929 epochs without inspecting the held-out
   year;
3. produce both unguarded curve decisions and fixed-threshold guarded
   decisions for the held-out year.

Aggregate the four held-out years.  Stage48 proceeds to final training only if
the guarded OOF policy, relative to the same joint model without the benefit
override, satisfies all of these locked conditions:

- false-positive irrigation count is strictly lower;
- mean seven-day regret is strictly lower;
- maximum seven-day regret is not higher;
- nonzero-irrigation recall is at least `0.90`.

If the OOF gate fails, write the fold predictions, decisions, metrics, audit,
and manifest with a failed status, then stop before training the all-years
model or evaluating 2019.

## Final 2019 Evaluation

If the OOF gate passes, train one final joint model on all 2015-2018 rows for
exactly 929 epochs and evaluate 2019 once.  Report three policies side by side:

1. frozen stage46 checkpoint;
2. stage48 joint model using unguarded curve argmax;
3. stage48 joint model using the fixed `0.5` benefit gate.

This ablation separates changes caused by retraining the shared curve model
from changes caused by the benefit decision itself.

Promotion of the guarded stage48 policy over frozen stage46 requires every
prelocked gate:

1. mean seven-day regret is strictly below `4.765217391304349`;
2. maximum seven-day regret is at most `39.2`;
3. exact irrigation match rate is at least `0.6176811594202898`;
4. nonzero-irrigation recall is at least `0.95`;
5. P3 mean seven-day regret is at most `5.733333333333333`;
6. false-positive irrigation count is below `7`.

The gates cannot be changed after observing stage48 OOF or 2019 results.

## Observability And Artifacts

Print a start record for every fold and progress every 50 epochs containing
fold identity, epoch, mean joint loss, curve loss, benefit loss, benefit
accuracy, elapsed time, and device.  Print the final OOF gate decision before
starting the all-years model.

Write:

- the joint checkpoint and full policy;
- OOF candidate predictions and cycle decisions for guarded and unguarded
  policies;
- OOF fold histories and aggregate comparison;
- final 2019 candidate predictions and the three-policy comparison when the
  OOF gate passes;
- audit and provenance manifests with SHA256 hashes.

Refuse to overwrite an existing output directory.  Verify the frozen stage46
checkpoint hash before comparison, and record that physics loss and TTA were
not used.

## Tests

Tests must cover:

- fixed auxiliary-label construction and the `1e-6` boundary;
- one benefit logit per complete cycle from the `0 mm` hidden state;
- exact zero-irrigation net-gain anchoring;
- joint-loss weighting and finite gradients;
- benefit policy overriding only nonzero recommendations to zero;
- OOF preprocessing and held-out-year separation;
- stopping before final training when the OOF gate fails;
- the three-policy 2019 output when the OOF gate passes;
- frozen stage46 SHA256 verification and no-overwrite behavior.

The server command must explicitly set `CUDA_VISIBLE_DEVICES=0` and pass
`--device cuda`.

# GEFS Three-Output Physics Pretraining Design

## Status And Scope

This specification supersedes the unexecuted joint-benefit-head proposal. It
starts the teacher-aligned formal surrogate mainline from the frozen stage43
dataset. Stages 44-47 remain no-physics decision diagnostics and stage46 stays
the frozen discrete diagnostic baseline; they are not the formal final model.

No 2024 data may be read. Training uses 2015-2018, while 2019 is used only for
epoch and hyperparameter selection. TTA is explicitly out of scope until this
pretraining gate passes.

## Locked Model Outputs And Control Volume

The model emits:

1. future 7-day net gain;
2. future 7-day cumulative actual evapotranspiration;
3. day01-day07 mean VWC for the fixed 0-100 cm soil layer;
4. an internal nonnegative 7-day residual-flux head.

The residual flux is directly supervised by the integrated SWAP label
`runoff + drainage_0_100cm + outflow_at_100cm`. The physical closure is

```text
balance_residual_mm =
    corrected_GEFS_precipitation_7d_mm + requested_irrigation_mm
  - predicted_AET_7d_mm
  - (1000 * predicted_VWC_day07 - predecision_storage_0_100cm_mm)
  - predicted_residual_flux_7d_mm
```

The factor 1000 converts fixed-depth mean volumetric water content to
millimetres of storage. The logical requested irrigation is used, never the
numerical SWAP fallback amount. The stage43 rain identity must pass within
0.01 mm before training.

## Formal Training Loss

The formal loss contains exactly three terms:

```text
L = L_profit + lambda_balance * L_balance + lambda_flux * L_flux
```

- `L_profit`: MSE of 7-day net gain, divided by the squared training-only net
  gain standard deviation.
- `L_balance`: mean squared physical closure residual, divided by the squared
  training-only standard deviation of `precipitation_7d + irrigation`.
- `L_flux`: MSE of the residual-flux head, divided by the squared training-only
  residual-flux standard deviation.

Every scale is fitted on 2015-2018 only. The net-gain scale has a `1e-6`
floor; the water-throughput and residual-flux scales have a physical `1 mm`
floor. Direct AET loss, direct VWC loss,
benefit classification, ranking loss, soft regret, exact-match loss, and
decision-gate loss are prohibited. The zero-irrigation net-gain anchor remains
an architectural constraint, not an extra loss.

## Predeclared Weight Exploration

Keep the profit coefficient fixed at 1. Search the Cartesian grid

```text
lambda_balance in {0.1, 1.0, 10.0}
lambda_flux    in {0.1, 1.0, 10.0}
```

for nine configurations. The search space is written to the audit before any
result is evaluated and cannot be expanded after viewing 2019. Each run uses
the same seed, architecture, optimizer, maximum epoch count, and patience.

## 2019 Selection And Reporting

For every checkpoint, compute five teacher-aligned validation groups:

1. net-gain normalized RMSE;
2. cumulative-AET normalized RMSE;
3. seven-day VWC-curve normalized RMSE, averaged across days;
4. residual-flux normalized RMSE;
5. physical balance normalized RMSE to zero.

Normalization denominators are training-only scales. The checkpoint score is
the unweighted mean of the five groups, so AET and VWC remain evaluation and
selection metrics without becoming formal training losses. Select the lowest
score; ties within `1e-12` prefer lower balance RMSE, then lower net-gain RMSE,
then smaller `(lambda_balance, lambda_flux)`. Early stopping patience is 200
epochs after a 100-epoch minimum, with at most 2000 epochs.

Also report raw MAE/RMSE/R2 for net gain and AET, daily and aggregate VWC
MAE/RMSE, residual-flux MAE/RMSE, balance bias/MAE/RMSE/max absolute residual,
and the existing discrete decision diagnostics. Those decision diagnostics
must not affect loss, checkpoint selection, or weight selection.

AET and VWC are only indirectly identified by the balance equation. Their
validation errors must be reported even when poor; adding direct supervision
to improve them is forbidden without a later teacher decision.

## Continuous Irrigation Requirement

The selected checkpoint must support differentiable irrigation in `[0,60]`
mm. Emit continuous 2019 recommendations from projected gradient ascent and
retain the eight-point discrete recommendation only as a diagnostic. The
formal model is not promoted solely because it improves exact discrete match
or discrete regret.

## Gates And Artifacts

Training must refuse missing/nonfinite physics fields, future-feature leakage,
non-2015-2019 rows, a non-CUDA device when `--device cuda` is requested, or an
existing output directory. It must record zero 2019 preprocessing rows and
zero 2024 rows read.

Write the predeclared grid, per-configuration histories and metrics, selected
checkpoint and preprocessing state, candidate predictions, discrete and
continuous decisions, comparison to frozen stage46 as a diagnostic, audit,
and SHA256 manifest. The audit must explicitly state:

- physics loss used: true;
- residual-flux supervision used: true;
- direct AET supervision used: false;
- direct VWC supervision used: false;
- benefit classification used: false;
- TTA performed: false.

## Tests

Tests cover the exact fixed-control-volume formula and gradients; loss-term
membership; training-only scale fitting; nine fixed weight combinations;
teacher-aligned checkpoint scoring and tie breaks; no AET/VWC target access in
the formal loss; split and forbidden-feature gates; projected `[0,60]` bounds;
no-overwrite behavior; and audit declarations.

The server command must explicitly set `CUDA_VISIBLE_DEVICES=0`, pass
`--device cuda`, use unbuffered output, write an absolute log path, and save the
background PID.

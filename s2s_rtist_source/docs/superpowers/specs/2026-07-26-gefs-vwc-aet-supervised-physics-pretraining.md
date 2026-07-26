# GEFS VWC+AET-Supervised Physics Pretraining Design

## Decision

Stage49 remains the frozen VWC-supervised incumbent. Stage50 established that
more epochs do not improve its runner-up, and Stage51 established that local
balance-weight refinement does not repair the remaining Stage48 balance gate.
Pure epoch and weight refinement therefore stop.

Stage52 makes exactly one objective change: retain the seven-day VWC curve
supervision introduced in Stage49 and add direct supervision for cumulative
seven-day AET. Removing VWC supervision is prohibited because it would again
leave the daily storage trajectory underidentified.

No model architecture, input feature, data split, preprocessing rule,
optimizer setting, existing loss weight, selection metric, or promotion gate
changes.

## Formal Loss

```text
L = L_profit
  + 1.0 * L_balance
  + 0.1 * L_flux
  + 10.0 * L_vwc
  + lambda_aet * L_aet
```

`L_profit`, `L_balance`, `L_flux`, and `L_vwc` retain their Stage49
definitions. `L_aet` is mean squared error for cumulative seven-day AET after
dividing the error by the 2015-2018 training-only AET standard deviation.
Validation and 2024 data cannot contribute to this scale.

The VWC curve remains one seven-day supervised term. AET is one cumulative
supervised term. Benefit classification, regret loss, ranking loss, and daily
water-balance losses are not added.

## Minimal Weight Search

Train exactly three configurations from scratch:

```text
lambda_aet in {0.1, 1.0, 10.0}
```

For every configuration, keep `lambda_balance=1.0`, `lambda_flux=0.1`, and
`lambda_vwc=10.0`. Do not rerun the previous 27-configuration Stage49 grid or
the Stage51 local balance grid.

Every Stage52 configuration uses seed `20260726`, maximum 4000 epochs, minimum
100 epochs, patience 200, batch size 256, AdamW learning rate `1e-3`, and
weight decay `1e-4`. Training starts from a fresh initialization. Warm-start
or fine-tuning results cannot enter the formal comparison.

## Selection And Gates

Checkpoint selection remains the Stage49 unweighted mean of validation NRMSE
for net gain, cumulative AET, seven-day VWC, residual flux, and physical
balance. Adding AET to the training objective does not add a sixth validation
selection group or otherwise double-count AET in selection.

The formal winner across the three Stage52 candidates and frozen Stage49 uses
the unchanged lexicographic key:

```text
selection_score, balance_nrmse, net_gain_nrmse,
lambda_balance, lambda_flux, lambda_vwc, lambda_aet
```

The final `lambda_aet` component only resolves an exact tie between Stage52
candidates; the frozen Stage49 incumbent is represented with
`lambda_aet=0.0`.

Each candidate is evaluated against all prelocked Stage48 repair gates:

1. AET RMSE and aggregate VWC RMSE improve;
2. all seven daily VWC R2 values improve and remain finite;
3. balance RMSE and maximum absolute balance residual improve;
4. net-gain RMSE remains within 5%;
5. mean and maximum discrete regret do not worsen;
6. continuous recommendations remain finite and within `[0,60]` mm.

Gate results do not change checkpoint selection. Discrete irrigation metrics
remain diagnostic and cannot select a checkpoint or weight configuration.
Passing Stage48 gates still requires a separate review against frozen Stage46
before any TTA.

## Stop Rule

Stage52 succeeds only if its best candidate both beats the frozen Stage49
selection key and passes all Stage48 repair gates. Otherwise Stage49 remains
frozen, Stage52 is recorded as a failed supervision ablation, and no TTA is
allowed. Do not expand `lambda_aet` after observing these three results without
a new predeclared design.

## Data And Audit Boundaries

Training remains 2015-2018 and validation remains 2019. No 2024 row may be
read. Preprocessing and every normalization scale use training rows only.
Logical requested irrigation remains the model input.

The audit records:

- termination details for all three configurations;
- frozen Stage46, Stage48, and Stage49 checkpoint hashes;
- direct VWC supervision used: true;
- direct AET supervision used: true;
- residual-flux supervision used: true;
- benefit classification used: false;
- discrete decision metrics used for selection: false;
- TTA performed: false;
- candidate gate outcomes, formal winner, and stop-rule result.

## Operation And Verification

The server command explicitly sets `CUDA_VISIBLE_DEVICES=0`, passes
`--device cuda`, uses absolute paths and unbuffered output, and writes absolute
log and PID paths. The delivery includes commands for PID/process state,
`tail -f`, GPU utilization, completion/error markers, output listing, weight
summary, and final audit/comparison JSON files.

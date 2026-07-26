# GEFS VWC Local Balance-Weight Refinement Design

## Decision

Stage49 and Stage50 are frozen. Stage50 confirmed that extending the capped
runner-up did not improve its Stage49 checkpoint, so epoch extension stops.
Stage51 tests whether the sole failed Stage48 repair gate, balance RMSE, can be
repaired by a minimal local change to the balance weight around the Stage49
incumbent.

The model, four-term loss, data split, preprocessing, optimizer, seed,
checkpoint-selection score, and all Stage48 gates remain unchanged. Only three
previously untested balance weights are added:

```text
lambda_balance in {2.0, 3.0, 5.0}
lambda_flux    = 0.1
lambda_vwc     = 10.0
```

This interpolates between the Stage49 incumbent at balance weight 1 and the
coarse-grid point at balance weight 10. No other weight is searched.

## Training And Selection

Each configuration trains from scratch with seed `20260726`, maximum 4000
epochs, minimum 100 epochs, and patience 200. Stage50 established that the
6000-epoch extension is unnecessary.

Checkpoint selection within each run remains the unweighted mean of validation
NRMSE for net gain, cumulative AET, seven-day VWC, residual flux, and physical
balance. The formal winner across the three new candidates and the frozen
Stage49 incumbent uses the unchanged lexicographic Stage49 selection key:

```text
selection_score, balance_nrmse, net_gain_nrmse,
lambda_balance, lambda_flux, lambda_vwc
```

Discrete irrigation metrics are diagnostic and cannot select a checkpoint or
weight configuration.

## Gates And Stop Rule

Every new candidate is evaluated against the unchanged Stage48 repair gates.
The report records whether any candidate repairs balance RMSE and whether all
gates pass, but gate outcomes do not replace the frozen selection key.

If none of the three candidates both improves the Stage49 selection key and
passes all Stage48 repair gates, pure balance-weight refinement stops. Do not
add more balance-weight points after observing Stage51. Any later change must
be separately designed as a teacher-aligned supervision change.

Passing Stage48 repair gates still does not authorize TTA. The selected model
must undergo a separate diagnostic review against frozen Stage46.

## Data And Audit Boundaries

Training remains 2015-2018 and validation remains 2019. No 2024 row may be
read. Preprocessing and loss scales use training rows only. The formal loss
remains:

```text
profit + lambda_balance * balance + lambda_flux * residual_flux
       + lambda_vwc * soil_vwc_curve
```

Direct VWC and residual-flux supervision remain enabled. Direct AET
supervision, benefit classification, TTA, and discrete-decision checkpoint
selection remain disabled.

The audit records termination details for every configuration, frozen
Stage46/48/49 checkpoint hashes, candidate gate outcomes, the formal winner,
and the predeclared stop-rule result.

## Operation

The server command explicitly sets `CUDA_VISIBLE_DEVICES=0`, passes
`--device cuda`, uses absolute paths and unbuffered output, and is accompanied
by PID, process, live-log, GPU, error/completion, output-listing, and final JSON
inspection commands.

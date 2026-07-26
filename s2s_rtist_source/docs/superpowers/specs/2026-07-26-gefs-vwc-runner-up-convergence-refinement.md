# GEFS VWC Runner-Up Convergence Refinement Design

## Decision

Stage49 is frozen. Its selected configuration
`(lambda_balance=1.0, lambda_flux=0.1, lambda_vwc=10.0)` remains the incumbent.
The only Stage49 configuration that both approached the selected validation
score and reached the 4000-epoch cap was configuration 6:
`(lambda_balance=0.1, lambda_flux=1.0, lambda_vwc=10.0)`. Stage50 retrains only
that configuration from scratch with the same seed and a 6000-epoch cap.

No loss term, loss weight, model architecture, preprocessing rule, split,
optimizer setting, checkpoint-selection metric, or promotion gate changes.
The other 26 configurations are not rerun.

## Training And Termination

Use the Stage49 training routine with:

```text
lambda_balance = 0.1
lambda_flux    = 1.0
lambda_vwc     = 10.0
seed           = 20260726
maximum epochs = 6000
minimum epochs = 100
patience       = 200
```

Training starts from a fresh initialization. Resuming is not allowed because
Stage49 did not preserve configuration 6 optimizer state as a standalone
checkpoint. The same seed preserves the intended controlled comparison.

The audit records `best_epoch`, `last_epoch`, `final_stale`, `early_stopped`,
and `hit_epoch_cap`. For this fixed-weight run, `final_stale` is
`last_epoch - best_epoch`. `early_stopped` is true only when training ended
before the epoch cap after reaching patience; `hit_epoch_cap` is true only when
`last_epoch == maximum_epochs`.

## Frozen Inputs And Selection

Before training, verify the Stage46, Stage48, and Stage49 checkpoint SHA256
values against their audits. Also verify that Stage49 selected
`(1.0, 0.1, 10.0)` and used the same seed and training controls expected by
this refinement.

Compare the refined candidate with the frozen Stage49 incumbent using the
unchanged Stage49 selection key: validation selection score, balance NRMSE,
net-gain NRMSE, then the three weights. The candidate replaces the incumbent
only if its complete selection key is strictly smaller. Discrete irrigation
metrics remain diagnostic and cannot select the winner.

The refined candidate is also evaluated against the prelocked Stage48 repair
gates without changing any threshold. Passing Stage48 gates does not by itself
authorize TTA; a separate Stage46 review remains necessary.

## Data And Audit Boundaries

Training remains 2015-2018 and validation remains 2019. Preprocessing and loss
scales use training rows only. No 2024 row may be read. The formal loss remains:

```text
profit + lambda_balance * balance + lambda_flux * residual_flux
       + lambda_vwc * soil_vwc_curve
```

Direct VWC and residual-flux supervision remain enabled. Direct AET
supervision, benefit classification, physics-informed TTA, and discrete
decision checkpoint selection remain disabled.

## Operation

The server command explicitly sets `CUDA_VISIBLE_DEVICES=0`, passes
`--device cuda`, uses unbuffered output, writes absolute output/log/PID paths,
and does not depend on temporary shell variables.

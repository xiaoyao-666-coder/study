# GEFS VWC-Supervised Physics Pretraining Design

## Decision

Stage48 is frozen as a failed identifiability ablation. Its teacher-aligned
profit, cumulative balance, and directly supervised residual-flux objective
left day01-day06 VWC unconstrained and allowed cumulative AET to compensate
for day07 storage. It must not enter TTA.

Stage49 makes exactly one supervised-objective change: add direct supervision
for the seven-day fixed 0-100 cm VWC curve. Cumulative AET remains indirectly
identified through the physical balance. No direct AET loss, benefit loss,
ranking loss, regret loss, or daily-balance loss is permitted.

## Formal Loss

```text
L = L_profit
  + lambda_balance * L_balance
  + lambda_flux * L_flux
  + lambda_vwc * L_vwc
```

`L_profit`, `L_balance`, and `L_flux` retain the stage48 definitions and
training-only normalization. `L_vwc` is the mean squared error of all seven
daily VWC outputs after dividing each day by its own 2015-2018 training-only
standard deviation. The seven days form one curve-level supervised term; they
are not seven separately weighted objectives.

The loss API and audit must state that VWC supervision is used and AET
supervision is not used.

## Weight Search

Search the predeclared Cartesian grid:

```text
lambda_balance in {0.1, 1.0, 10.0}
lambda_flux    in {0.1, 1.0, 10.0}
lambda_vwc     in {0.1, 1.0, 10.0}
```

This gives 27 configurations. Do not freeze the stage48-selected balance and
flux weights because adding VWC supervision changes the optimization surface.
Every configuration uses the same initialization seed, optimizer, maximum
epochs, minimum epochs, patience, and training data.

## Selection And Gates

Keep the stage48 checkpoint-selection score: the unweighted mean of 2019
normalized RMSE for net gain, cumulative AET, seven-day VWC, residual flux,
and physical balance. Tie breaks prefer lower balance NRMSE, then lower net
gain NRMSE, then smaller `(lambda_balance, lambda_flux, lambda_vwc)`.

Stage49 cannot enter TTA unless all of the following hold on 2019:

1. AET RMSE and aggregate VWC RMSE are both strictly below stage48;
2. day01-day07 VWC R2 values are all finite and each exceeds its stage48 value;
3. balance RMSE and maximum absolute residual are both below stage48;
4. net-gain RMSE is not more than 5% above stage48;
5. mean and maximum discrete regret are not above stage48;
6. continuous recommendations are finite and remain in `[0,60]` mm.

These gates compare against the failed stage48 model only. Passing them means
the identifiability repair worked; it does not by itself establish superiority
to stage46 or authorize TTA. A separate review against stage46 is still
required.

## Data And Audit Boundaries

Training remains 2015-2018; 2019 remains validation and hyperparameter
selection; 2024 rows read must remain zero. Preprocessing and all loss scales
must use 2015-2018 only. Logical requested irrigation remains the model input.

The audit must record:

- physics loss used: true;
- residual-flux supervision used: true;
- direct VWC supervision used: true;
- direct AET supervision used: false;
- benefit classification used: false;
- discrete decision metrics used for selection: false;
- TTA performed: false.

The stage48 checkpoint and metrics hashes must be recorded before comparison.

## Tests And Operation

Tests cover the single added curve term, exact four-term weighting, absence of
an AET target from the loss API, the 27-combination grid, selection tie breaks,
stage48 comparison gates, split isolation, bounded continuous optimization,
and audit declarations.

The server command explicitly sets `CUDA_VISIBLE_DEVICES=0`, passes
`--device cuda`, uses unbuffered output, writes an absolute log path, and saves
the background PID.

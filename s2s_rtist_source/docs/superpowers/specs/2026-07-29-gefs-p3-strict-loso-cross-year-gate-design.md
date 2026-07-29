# P3 Strict-LOSO Cross-Year Expert Gate Design

## Status

Frozen design approved on 2026-07-29. This document defines an exploratory
P3-only gate between independently retrained P1 and P15 composite experts. It
does not authorize a general five-site mixture-of-experts model and does not
start training.

## Objective

Test whether decision-time P3 state and weather features can route between P1
and P15 source-site experts more reliably than the pre-registered `always_P15`
baseline. The experiment must separate cross-site transfer from target-site
training: no P3 row may enter source-expert fitting, preprocessing, checkpoint
selection, or early stopping.

The source experts retain all four outputs:

1. seven-day net gain;
2. seven-day cumulative AET;
3. seven daily fixed 0--100 cm VWC values;
4. seven-day residual water-balance flux.

Every source-expert checkpoint is selected only by the existing four-output
`composite` criterion. Historical `profit`, `direct`, and `ranking` tracks are
not candidates in this protocol. The balance-loss weight is fixed at the
formal incumbent value `lambda_balance=1.0`; it is not searched by fold.

## Evidence Boundary

The existing P1/P15 complementarity audit used 2019 P3 SWAP outcomes and
therefore consumed 2019 as a blind test. The protocol treats 2019 as the fourth
retrospective development-validation fold. It must not describe 2019 as an
untouched test year.

The resulting evidence is a strict cross-site, forward-chaining retrospective
validation. A separate independent test claim requires a later year whose
labels were not inspected. Subject to the development gate below, 2021 is the
preferred independent year because the full growing season uses operational
GEFSv12. The year 2020 is excluded from the formal path because its growing
season predates the 23 September 2020 GEFSv12 operational transition. The year
2024 remains reserved for TTA.

## Rolling Outer Folds

The four outer folds are fixed:

| Fold | Historical years available to source experts and gate | P3 validation year |
|---|---|---|
| `rolling_to_2016` | 2015 | 2016 |
| `rolling_to_2017` | 2015--2016 | 2017 |
| `rolling_to_2018` | 2015--2017 | 2018 |
| `rolling_to_2019` | 2015--2018 | 2019 |

No row later than an outer fold's validation year may be loaded while that fold
is trained or evaluated. All preprocessing statistics are fitted again inside
each fold. Results must be reported for every fold and macro-aggregated across
the four validation years; no year may be omitted after its result is seen.

## Fold-Local Source Experts

Each outer fold trains two independent source models from scratch:

- the P1 expert reads only historical P1 rows;
- the P15 expert reads only historical P15 rows.

The models may share an architecture but may not share fitted weights,
preprocessing statistics, batches, or gradients. In particular, they do not
reuse the current v5r2 shared encoder because that encoder saw P3 and other
sites. P2, P3, and P4 rows are forbidden from source-expert training.

The source training routine must use a deterministic chronological inner
selection split made only from the fold's historical source-site cycles. The
last `max(1, ceil(0.20 * N))` of the `N` date-sorted distinct decision cycles
are the source checkpoint-selection partition; earlier cycles are the fitting
partition. Candidate checkpoint selection uses only the four-output composite
score on that source-site selection partition. When a fold contains only one
historical year, the split remains chronological within that year.

No fold may initialize from a checkpoint selected with 2019 or from a model
trained on any target-site P3 row.

## Gate Samples And Labels

After the two fold-local source experts are frozen, both are applied to the
eight fixed irrigation candidates `[0, 10, 15, 20, 25, 30, 40, 60]` for each
historical P3 decision cycle. Each expert independently recommends the
candidate with maximum predicted seven-day net gain.

Only cycles on which the experts recommend different irrigation amounts are
actionable gate samples. The gate label is P1 when the SWAP-realized net gain
at P1's recommendation is strictly greater than the realized net gain at
P15's recommendation; it is P15 for the converse. Equal realized gains are
non-actionable and excluded from fitting. SWAP outcomes are used only to form
historical routing labels and evaluate routing decisions, never as gate
features.

The outer validation year contributes no gate-fitting label. Its labels are
read only after that fold's experts, preprocessing, and gate are frozen.

## Gate Features

One cycle-level feature row is constructed before candidate irrigation is
applied. Irrigation amount, surrogate predictions, candidate ordering, SWAP
targets, future observations, and post-decision states are forbidden.

The feature contract contains a compact physical summary rather than all daily
columns:

- decision-time DVS;
- decision-time fixed 0--100 cm VWC or equivalent initial root-zone storage;
- decision-time root depth when present in the frozen dataset;
- seven-day GEFS precipitation sum;
- seven-day mean minimum temperature;
- seven-day mean maximum temperature;
- seven-day mean actual vapor pressure;
- seven-day mean wind speed;
- seven-day mean solar radiation.

Missing optional state fields are recorded in the protocol audit and omitted
globally; they are not imputed from validation data. Static site attributes are
not used because they are constant in a P3-only gate. Feature standardization
is fitted on historical P3 gate-training cycles within each outer fold.

## Gate Model And Fallback

The only candidate gate is deterministic L2-regularized logistic regression.
To avoid adding a scikit-learn dependency, it is solved with a NumPy
Newton/IRLS implementation minimizing summed binary cross-entropy plus
`0.5 * ||weights||^2`; the intercept is not regularized. The solver uses
`max_iter=100`, parameter tolerance `1e-10`, a fixed decision threshold of
`0.5`, and no hyperparameter sweep. An exact probability tie routes to P15.
The earlier proposed threshold of `0.75` is rejected because it was suggested
after inspecting 2019 results.

If an outer fold has fewer than eight actionable gate-training cycles, contains
only one winner class, has non-finite features, or fails to converge, that fold
must emit an audited `always_P15` fallback. The runner must not tune a threshold,
change features, oversample labels, or substitute another classifier in
response to a failed fold.

## Baselines And Metrics

The pre-registered primary baseline is `always_P15`. `always_P1` and the
label-using two-expert oracle are diagnostic bounds only. The oracle cannot be
promoted or used to choose a deployable configuration.

Source-expert checkpoint selection continues to use only composite. Policy
evaluation uses independent fixed-list SWAP outcomes and reports:

- mean seven-day regret;
- maximum seven-day regret;
- false-positive irrigation count;
- missed-beneficial-irrigation count;
- predicted 60 mm count;
- P1/P15 routing counts and actionable-cycle accuracy.

This is not a second proxy-checkpoint standard: composite selects proxy
checkpoints, while SWAP regret evaluates the downstream irrigation policy as
required by the formal decision protocol.

## Development Pass Gate

The P3 gate passes only when all conditions hold against `always_P15` across
the four outer validation folds:

1. candidate macro mean regret is at most baseline macro mean regret minus
   `1e-6`;
2. candidate mean regret is at most baseline mean regret plus `1e-6` in at
   least three of four folds;
3. worst-fold maximum regret is no greater;
4. total false-positive count is no greater;
5. total predicted-60-mm count is no greater;
6. at least one validation cycle is actually routed to P1.

Failure of any condition freezes the conclusion that this P1/P15 pool does not
support a deployable P3 gate. The criteria cannot be weakened after results are
observed.

## Independent-Year Release Gate

No 2021 download, SWAP label generation, or model evaluation is part of the
four-fold screen. The 2021 stage may begin only after the complete development
audit records `development_gate_passed=true` and the code, feature contract,
checkpoint rule, logistic parameters, and evaluation metrics are frozen.

If released, the final source experts and gate are refitted once using
2015--2019 under the same rules, then evaluated once on 2021. The precipitation
and non-precipitation corrections remain the already frozen transformations;
they are not refitted on 2021. Any necessary change limited to locating the
operational GEFSv12 archive must be audited as an input-adapter change, not a
new model selection opportunity.

## Required Protocol Artifacts

The no-training freeze stage must produce:

- a row-level outer-fold assignment CSV without target columns;
- a fold/source expert registry proving P1-only and P15-only training scopes;
- a gate feature contract;
- a machine-readable protocol JSON;
- an audit JSON recording zero 2020, 2021, and 2024 rows read and zero target
  columns loaded by the freeze stage;
- a concise fold summary and commands for audit, later training, monitoring,
  and result inspection.

The later screen must preserve per-fold checkpoints, preprocessing, gate
training samples, validation decisions, per-fold metrics, macro gates, fallback
reasons, and a manifest containing input and output hashes.

## Failure Handling And Tests

The protocol and runner must fail closed when a future-year row appears in a
fold, P3 enters source-expert fitting, a non-composite source checkpoint is
registered, the irrigation grid changes, a target column enters gate features,
an output directory would be overwritten without an explicit resume contract,
or a required source audit has not passed.

Tests must cover fold chronology, strict source-site isolation, target-free
protocol reading, composite-only checkpoint registration, deterministic inner
cycle splitting, actionable-label construction, gate fallback behavior,
development-gate arithmetic, absence of 2020/2021/2024 data, and unpacked ZIP
execution on Linux-style paths.

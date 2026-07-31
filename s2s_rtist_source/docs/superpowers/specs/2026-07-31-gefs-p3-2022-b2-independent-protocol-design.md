# P3 2022 B2 Independent Evaluation Protocol Design

## Status

Design approved on 2026-07-31. This specification covers only the protocol
freeze and its data-access barrier. It does not authorize 2022 acquisition,
model inference, SWAP execution, or label access until the protocol freeze
artifacts pass their mandatory audit.

## Objective

Freeze P3 2022 as the first new independent evaluation of the already fitted
B2 regret-sensitive source-only router. The protocol must bind every model,
feature, routing, weather, schedule, and evaluation choice before any 2022
weather content or SWAP candidate label is accessed.

P3 2023 remains sealed as a replication reserve. The entire 2024 workflow
remains assigned to the separate TTA question and is not part of this
evaluation.

## Why A Dedicated Protocol

The frozen P3 2021 pipeline must remain unchanged. The new implementation
adds a dedicated 2022 protocol freezer rather than parameterizing or editing
the 2021 scripts. This preserves the provenance of the completed 2021 A/B
evaluation and keeps failures in the new experiment isolated.

A general multi-year framework is deferred. It would expand the blast radius
before 2022 archive and simulation availability have passed even a smoke
check.

## Frozen Candidate And Components

The only candidate policy is the final B2 regret-sensitive router. Experiment
A and the old B1 router are not evaluated again.

The protocol freezer must verify and bind these values:

```text
B2 development status:
b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze

B2 final gate file SHA256:
CFD509A41141F64ABE7CB8D51A42A53D9E9A3B89893530377D30044E99052056

B2 model payload SHA256:
6d6c4bd4e3ab9b5a7038a4e13b066eced13880829d149be5d584bf76c7bb503f

B2 audit SHA256:
D88480395A83E5C12CC2D42EA02FE89F6C38CD1C02BF84055F4BD9B0A7F8A06C

B2 manifest SHA256:
3DDB602B8D26D67F8875097520AE7F9FF72B9E06DC6A4FA0ACAA99D048D4998B

P1 final source checkpoint SHA256:
61172cbdee69c1e43c66570baa209287a1ef73ed658c48fd144ca035d7fc683b

P15 final source checkpoint SHA256:
b2ad3cd245d0dd88689f316a29037ceb1fd0b4d59ef915d2a903155dbec15363

Final source policy SHA256:
d9a2b2e85063c68e2812b7cef374b2c87c21efc3443c35e2ab99358f5825d726
```

The freezer must also recompute the B2 model payload hash after excluding its
`model_sha256` field. It may not trust only the recorded value.

The B2 audit must report `development_gate_passed=true`,
`final_gate_written=true`, `formal_promotion=false`, zero retained P3/2020/
2021/2024 rows, and no checkpoint, feature, weight, or threshold search.

## Frozen Routing Contract

The source experts remain P1 and P15 full-history composite checkpoints fit on
2015-2019 with `lambda_balance=1.0`. The gate retains exactly the existing nine
decision-time physical features:

1. predecision DVS;
2. predecision crop root depth;
3. predecision 0-100 cm soil VWC;
4. seven-day precipitation sum;
5. seven-day mean minimum temperature;
6. seven-day mean maximum temperature;
7. seven-day mean actual vapor pressure;
8. seven-day mean wind speed;
9. seven-day mean solar radiation.

The probability threshold remains `0.5`. An exact threshold tie routes to P15.
Any audited gate failure routes to P15. No feature, coefficient, sample weight,
threshold, source checkpoint, or preprocessing statistic may be refit after
the protocol is frozen.

## Target And Schedule Contract

The target is P3 (`N3`, latitude `46.321`, longitude `-96.877`, timezone
`America/Chicago`) in 2022.

Decision dates are not copied from 2021 and are never manually selected. The
protocol freezes the same deterministic rule used by the corrected 2021
evaluation:

- build one continuous P3 2022 zero-irrigation season using ERA5 daily
  weather and the historical ERA5-to-SWAP conversion;
- use sowing month-day `04-26` and the existing P3 crop/workspace contract;
- select the first decision one day after checkpoint DVS first reaches `0.1`;
- continue every seven days;
- retain every eligible pre-maturity cycle with a complete D through D+6
  horizon;
- require at least eight decision cycles;
- freeze the resolved dates and checkpoint hashes before any operational GEFS
  acquisition or candidate SWAP label generation.

ERA5 may determine crop state and dates only through this preregistered rule.
GEFS values and candidate SWAP outcomes may not select, drop, or reorder dates.

## Operational Weather Contract

The future branch uses the five operational members
`gec00/gep01/gep02/gep03/gep04`, initialized at 00 UTC, with local decision day
D through D+6 coverage. The variables, grid, temporal aggregation, and local
timezone conversion remain identical to the corrected P3 2021 five-member
pipeline.

The frozen precipitation correction remains the pre-target-year weekly
two-stage site-factor model fit through 2019 with shrinkage alpha `0.75`. The
nonprecipitation branch remains the 2015-2019 fitted affine/solar correction
with its already frozen alphas. No 2022 reference weather may fit, select, or
modify either correction.

External ERA5 and GEFS acquisition runs only on the local Windows workflow.
The server does not download external weather data.

## Evaluation Contract

The shared candidate grid is fixed at:

```text
[0, 10, 15, 20, 25, 30, 40, 60] mm
```

The three frozen policies evaluated on the same SWAP label table are:

- `B2_regret_sensitive_router`;
- `always_P1`;
- `always_P15`.

The two-expert label oracle may be computed only after the shared labels exist
and is diagnostic only. It is not a frozen online policy and cannot affect the
promotion decision.

Formal metrics use actual fixed-candidate SWAP runs with no interpolation and
no continuous optimization. The report includes cycle count, mean and maximum
regret, false-positive irrigation count, missed-beneficial count, 60 mm count,
P1/P15 route counts, effective route count, and every per-cycle decision.

## Independent Promotion Gate

B2 passes the independent gate only if every condition holds:

1. B2 mean regret is at least `1e-6` lower than always-P1.
2. B2 mean regret is at least `1e-6` lower than always-P15.
3. B2 maximum regret is no greater than always-P1 plus `1e-6`.
4. B2 maximum regret is no greater than always-P15 plus `1e-6`.
5. B2 false-positive count is no greater than either fixed expert.
6. B2 60 mm count is no greater than either fixed expert.
7. At least one effective route changes the selected irrigation relative to a
   fixed expert on a cycle where P1 and P15 recommendations differ.
8. Every scheduled cycle has all eight successful candidate SWAP labels.
9. No numerical endpoint fallback, irrigation mismatch, non-finite metric, or
   gate fallback occurs.
10. All zero-access, frozen-hash, recommendation-before-label, and shared-label
    audit conditions pass.

Failure of any condition freezes the result as negative. The criteria cannot
be weakened after recommendations, weather outcomes, or SWAP labels are seen.
Even a passing result remains scoped to P3 2022 until the sealed P3 2023
replication is separately authorized.

## Ordered Data-Access Stages

The protocol stage registry is fixed as follows:

1. `protocol_freeze`: read only historical contracts and frozen hashes.
2. `local_ERA5_2022_acquisition`: allowed only after stage 1 passes.
3. `server_ERA5_zero_irrigation_trunk_and_schedule`: no GEFS or candidate
   labels.
4. `local_operational_GEFSv12_five_member_acquisition`: only frozen dates.
5. `frozen_weather_correction`: no model inference or labels.
6. `B2_and_fixed_expert_recommendation_freeze`: no SWAP candidate labels.
7. `shared_fixed_eight_SWAP_generation`: one table for all policies.
8. `single_independent_evaluation`: apply the ten frozen conditions once.

Each stage binds its inputs and outputs by SHA256. A later stage must reject a
missing, changed, incomplete, or ineligible predecessor.

## Protocol-Freezer Implementation Scope

The first implementation creates only:

```text
scripts/evaluation/freeze_gefs_p3_2022_b2_independent_protocol_v1.py
tests/test_freeze_gefs_p3_2022_b2_independent_protocol_v1.py
```

The freezer reads the B2 result directory, the final source-model directory,
and the frozen dual-protocol directory. It writes a new output directory and
refuses overwrite:

```text
gefs_p3_2022_b2_independent_protocol_v1.json
gefs_p3_2022_b2_component_registry_v1.csv
gefs_p3_2022_b2_stage_registry_v1.csv
gefs_p3_2022_b2_independent_protocol_audit_v1.json
gefs_p3_2022_b2_independent_protocol_manifest_v1.csv
```

The success audit status is:

```text
p3_2022_b2_independent_protocol_frozen_before_target_data_access
```

It records zero 2022 ERA5 rows, GEFS rows, model inference rows, recommendation
rows, SWAP candidate labels, and network requests. It also records zero 2023
and 2024 access.

No acquisition, trunk, recommendation, or SWAP runner is part of this first
implementation. Those stages receive separate implementation plans only after
the protocol output is generated and verified.

## Failure Handling

The freezer fails closed before creating its output directory when an input
directory or required file is missing, an expected hash differs, a manifest
binding fails, a B2 status or leakage field differs, the final gate is a
fallback, source checkpoints are absent or changed, feature names/order differ,
or any 2022/2023/2024 access flag is nonzero.

Once output creation begins, every output is included in the manifest except
the manifest itself. No partial protocol is eligible for the next stage.

## Testing And Packaging

Tests cover exact frozen hashes, recomputed model payload hash, nine ordered
features, all ten promotion conditions, stage order, zero target-year access,
source checkpoint verification, output cardinality, overwrite refusal, and
manifest integrity. Related B2 and P3 2021 protocol tests must remain unchanged
and pass.

The server ZIP contains the new freezer, its tests, and the smallest required
reusable dependency set. It excludes weather, datasets, checkpoints, results,
logs, PID files, and all 2022/2023/2024 content. A clean extraction must pass
archive path validation and the focused tests before server installation.

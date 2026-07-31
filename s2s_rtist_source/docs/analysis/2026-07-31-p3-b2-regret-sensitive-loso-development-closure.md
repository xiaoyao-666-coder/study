# P3 B2 Regret-Sensitive LOSO Development Closure

## Frozen Status

The B2 development screen completed with status:

```text
b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze
```

All nine preregistered development conditions passed. The final gate was fit
successfully and is not a fallback. This is a valid target-label-isolated
development result, not a formal independent-test success and not a promotion
to a reliable or deployable router.

The screen retained zero P3, 2020, 2021, and 2024 rows. It used only P2/P4
2015-2019 labels in eight rolling-year, leave-site-out folds. Validation truth
was materialized only after each fold gate was frozen.

## Development Evidence

The sole preregistered formal baseline was always-P15.

| Metric | always-P15 | B2 | Change |
|---|---:|---:|---:|
| Eight-fold macro mean regret | 19.507830 | 12.823535 | -34.3% |
| P2 mean regret | 16.513462 | 11.698718 | -29.2% |
| P4 mean regret | 22.502198 | 13.948352 | -38.0% |
| Total false positives | 12 | 8 | -4 |
| Total 60 mm decisions | 10 | 3 | -7 |
| Worst fold maximum regret | 208.0 | 208.0 | unchanged |

Seven of eight folds were non-worse than always-P15. There were 27 effective
P1 routes. `holdout_P4_to_2019` fell back to always-P15 because only two
actionable training cycles were available; the preregistered gate required at
least one non-fallback fold per held-out site, not a non-fallback fit in every
fold.

The main adverse fold was `holdout_P2_to_2019`: mean regret increased from
`6.738462` to `12.707692`. The worst regret also remained `208.0`, so B2 did
not solve the tail-risk problem.

## Diagnostic Baseline Boundary

The eight-fold always-P1 macro mean regret was approximately `10.718315`,
which is lower than B2 by `2.105220`. The two-expert label-oracle macro mean
regret was approximately `4.274703`.

Therefore B2 demonstrates a substantial improvement over the preregistered
always-P15 baseline, but it does not yet demonstrate that routing is better
than the strongest fixed expert. A future independent promotion gate must
report both fixed experts and cannot rely only on always-P15.

## Final Gate Integrity

The final P2/P4 2015-2019 gate contains 79 actionable cycles: 57 P1 wins and
22 P15 wins. Its normalized sample-weight mean is
`0.9999999999999998`; the raw absolute gain-delta mean is
`64.44810126582279`.

```text
model payload SHA256:
6d6c4bd4e3ab9b5a7038a4e13b066eced13880829d149be5d584bf76c7bb503f

final gate file SHA256:
CFD509A41141F64ABE7CB8D51A42A53D9E9A3B89893530377D30044E99052056
```

The recorded model payload hash was recomputed locally and matched. All eight
output files covered by the manifest existed and matched their recorded
SHA256 values.

## Independent-Test Provenance Audit

No currently available labeled site/year qualifies as a new independent B2
test:

- P2 and P4 2015-2019 were used for B2 development.
- P3 2015-2019 was consumed by prior target-supervised and expert development.
- P3 2021 was consumed by the completed A/B shared-SWAP evaluation.
- P1 and P15 are the source-expert training sites.
- 2024 is reserved by the project protocol for the separate TTA question. It
  has also been accessed by weather-bias diagnostics, restart/SWAP smoke runs,
  and cached schedule diagnostics, so it must not be silently relabeled as a
  pristine B2 independent test.

Consequently, B2 is frozen without further feature, weight, threshold, or
checkpoint changes. No 2024 router evaluation is authorized by this result.

A valid next independent test requires either a newly acquired site/year or a
separately approved protocol change with full provenance disclosure. Before
any new labels are opened, the protocol must bind the final gate hash, source
checkpoint hashes, nine features, threshold `0.5`, P15 tie/fallback rule,
fixed eight-irrigation grid, weather semantics, and evaluation dates. The
independent report must compare B2 with both always-P1 and always-P15 and must
report mean regret, maximum regret, false positives, 60 mm decisions, route
counts, and per-cycle outcomes.

## Local Result Archive

```text
D:\study\s2s_rtist_source\gefs_p3_b2_regret_sensitive_loso_router_results_20260731_v1.tar.gz
SHA256: 52C211756AA1F859A4E5313297258E90C80F06612C353EF46FAF13813D909DF8
```

The archive contained 9 files under the expected result directory. Local
archive validation rejected absolute paths, traversal, links, backslashes,
duplicates, and case-insensitive collisions before extraction.

Selected extracted file hashes:

```text
development gate: 928504EC6ADB5E6FBD8463C60198EE1198CDC63413C3E64F3B8C4F3C8DF4F74B
fold models:      6761084FB4EA8795CB036B0DCB1D985FD6F6C874A44B3101F68F704689C2C1D0
per-fold metrics: 498A936961A26F9748A4E80B11513951DF5651F813A21E19290DF7DA183112B4
per-site metrics: B8180215A9267EDDE7D178AEADBADC8FBAC11E8A22B1F6F388619EB7417294CD
audit:            D88480395A83E5C12CC2D42EA02FE89F6C38CD1C02BF84055F4BD9B0A7F8A06C
manifest:         3DDB602B8D26D67F8875097520AE7F9FF72B9E06DC6A4FA0ACAA99D048D4998B
```

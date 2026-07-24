# GEFS QM v3 design exploration conclusion

## Scope

This exploration uses only the existing 2015-2018 local point cache and leave-one-year-out OOF evaluation. It does not use 2019 or 2024, perform network downloads, train the surrogate model, or change TTA.

The following three design questions were tested:

1. Whether a real seasonal grouping based on forecast initialization month (`init_month`) is more stable than a single global mapping.
2. Whether hierarchical shrinkage should use independent reference counts rather than repeated member rows.
3. Whether occurrence correction or the amount mapping is responsible for the ensemble coverage loss.

## Results

The first exploration is stored under `site_general_surrogate_eval/gefs_qm_v3_design_exploration_v1/`.

| Variant | 7-day MAE difference | CRPS difference | Brier difference | Heavy coverage |
|---|---:|---:|---:|---|
| `global_month_occurrence` | -0.2458 mm | +0.00010 mm | -0.00255 | failed |
| `site_month_occurrence` | +0.8265 mm | +0.10445 mm | +0.00138 | failed |
| `site_month_shrink_lambda18_occurrence` | +0.1504 mm | +0.04935 mm | -0.00166 | failed |
| `site_month_shrink_lambda36_occurrence` | -0.0768 mm | +0.02761 mm | -0.00150 | failed |
| `site_only_shrink_lambda18_occurrence` | -0.2412 mm | -0.02449 mm | -0.00200 | failed |
| `site_only_no_occurrence` | +0.4582 mm | +0.03330 mm | +0.00032 | failed |
| `site_only_shrink_lambda18_no_occurrence` | +0.4025 mm | +0.02747 mm | +0.00038 | failed |

The evidence indicates:

- Month grouping alone is not sufficient. `site_month` is too sparse and degrades all pooled primary metrics unless heavily shrunk.
- Shrinkage reduces fine-group overfitting. For example, `site_month` improves from +0.8265 mm MAE difference to -0.0768 mm with lambda 36, but CRPS remains worse and heavy coverage still fails.
- Occurrence correction is useful for mean/probabilistic scores: removing it turns `site_only_shrink_lambda18` from pooled improvement into pooled degradation. It is therefore not the sole cause of the coverage failure.
- All ordinary QM variants reduce heavy-event ensemble spread. Raw heavy-event mean spread is about 9.375 mm, while QM variants are about 7.95-8.41 mm.

## Spread-preservation diagnostic

A diagnostic then mixed the base QM anomalies with raw GEFS anomalies. The best pooled compromise was:

`site_only_shrink_lambda18_occurrence + 50% raw anomaly`

It produced:

- 7-day MAE difference: `-0.0853 mm`;
- CRPS difference: `-0.01245 mm`;
- mean Brier difference: `-0.001095`;
- heavy coverage gate: passed.

However, it produced `1566` negative values before nonnegative clipping. After explicit clipping, its year-wise results were:

| Validation year | MAE difference | CRPS difference | Brier difference | Heavy coverage |
|---|---:|---:|---:|---|
| 2015 | +1.2743 mm | +0.05235 mm | +0.00624 | passed |
| 2016 | -1.0480 mm | +0.00531 mm | -0.00167 | passed |
| 2017 | -0.0797 mm | -0.05415 mm | -0.00519 | passed |
| 2018 | -0.4879 mm | -0.05334 mm | -0.00376 | passed |

It fails the prelocked year-stability requirement for CRPS (`2/4` years not worse) and is not physically clean before clipping. This is a diagnostic signal, not a promotion candidate.

## Decision

No v3 QM candidate is ready for a formal holdout application. The unresolved issue is a real objective conflict: improving the ensemble mean and CRPS can narrow the five-member ensemble and damage heavy-event coverage.

The next design must be chosen with an explicit decision about whether probabilistic coverage is a hard requirement. If it is hard, the next v3 contract should specify a nonnegative, spread-aware calibration method rather than post-hoc raw-anomaly mixing. If mean accuracy has priority over coverage, that trade-off must be approved explicitly before relaxing the gate.

Until that decision is made, retain raw GEFS as the operational baseline and do not apply any exploratory v3 mapping to 2019 or 2024.

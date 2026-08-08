# Teacher-Guided GAM v6 Frozen 2019 Confirmation Design

## Status and evidence claim

This design is approved after the v6 confidence-protected anchor replay passed every predeclared 2015-2018 development gate. It freezes the complete v6 decision procedure before reading 2019 targets or SWAP outcomes.

The 2019 experiment is independent of v6 threshold and rule selection, but it is not claimed as a pristine final generalization test because earlier project work inspected 2019 context. The untouched 2024 boundary remains the final test and TTA boundary.

## Research question

When the frozen GAM/B-spline expert family, source-only gate, 2.5 mm anchor trust region, and 5.0 mm confidence threshold are transferred to 2019 without retuning, does the resulting decision remain nonworse than its frozen source-selected anchor in both peak location and exact continuous SWAP outcome?

## Stage 1: recommendation freeze

Stage 1 must finish and hash-freeze all recommendations before any 2019 target, oracle, gain, or regret column is read.

For each target site in P1, P2, P3, P4, and P15:

1. Exclude every row from the target site from all model and gate fitting.
2. Fit the four-source GAM-VC baseline and four delete-one-source experts on the other four sites' complete 2015-2018 cycles.
3. Reuse the target site's `rolling_to_2018` frozen GAM penalties without a new search.
4. Select the best-single anchor using only 2015-2018 source-site labels.
5. Refit the v4 positive-protected action-aware gate on the available 2015-2018 source cycles with the already frozen feature contract, L2 penalty, weighting, residual quantile, support quantile, and minimum action difference.
6. Generate the anchor and gate-selected endpoints for all target-site 2019 cycles using only decision-time features.
7. Apply the frozen v5 trust region: move at most 2.5 mm from the anchor toward the gate-selected endpoint.
8. Apply the frozen v6 rule: retain an unlimited v5 endpoint; retain a trust-region-limited endpoint only when `selected_lcb_mm >= 5.0`; otherwise use the anchor.

No site identifier, year, target, oracle, SWAP gain, or SWAP regret may enter the v6 selection function. Site and year remain identifiers and grouping fields only.

Stage 1 writes a 69-cycle recommendation inventory, fitted-artifact audit, gate audit, and SHA256 manifest. It records zero 2019 target rows read and zero 2024 rows read. Its only passing action is to authorize Stage 2 against the frozen manifest.

## Stage 2: exact paired confirmation

Stage 2 hash-verifies the frozen Stage 1 inventory before reading 2019 targets or SWAP results.

After verification it may read the 69 complete 2019 fixed-list cycles for scoring. For every cycle it computes anchor and selected peak distance as secondary confirmation evidence. It then evaluates exact continuous SWAP endpoints:

- unchanged recommendation: the selected-minus-anchor difference is exactly zero and no new SWAP is run;
- changed recommendation: run or exactly reuse the anchor and selected irrigation endpoints;
- an endpoint may reuse a prior result only when site, decision date, and irrigation amount match within `1e-6 mm`;
- interpolation is forbidden;
- numerical fallback may use only the frozen downward offsets 0.1, 0.2, and 0.3 mm after exact-run failure, with both logical and simulated amounts audited.

The full paired ledger always contains 69 cycles. SWAP evaluation uses the continuous seven-day outcome, not the eighth day alone.

## Confirmation gate

All execution conditions are mandatory:

1. All 69 cycles, all five sites, and both exact endpoints for every changed cycle are complete and finite.
2. Stage 1 hashes are unchanged.
3. All model and gate fitting rows are from 2015-2018 source sites only.
4. No 2019 target was read before the recommendation manifest was frozen.
5. No 2024 row was read.
6. No threshold, feature, expert family, penalty, trust radius, or gate condition was selected using 2019.

All performance conditions are also mandatory:

1. At least two irrigation decisions differ from anchor.
2. Changed decisions cover at least two target sites.
3. At least four of five sites have nonworse mean selected-minus-anchor SWAP gain.
4. Pooled mean SWAP gain is nonworse.
5. Positive-oracle mean SWAP gain is nonworse.
6. Zero-oracle mean SWAP gain is nonworse.
7. Overall fixed-eight regret is nonworse.
8. Changed-cycle maximum adaptive regret is nonworse.
9. Overall, positive-oracle, zero-oracle, and global-maximum peak distance are all nonworse.

The numerical tolerance is `1e-6`. This is a predeclared engineering confirmation gate, not a significance test.

## Outcomes

If the joint gate passes, the only authorized next action is to design a 2015-2019 final refit and frozen 2024 test protocol. No automatic model promotion, final refit, 2024 access, or TTA occurs in this experiment.

If any condition fails, freeze v6 2019 confirmation as negative or inconclusive. The same 2019 cycles cannot be used to tune v6 and then be relabeled as independent confirmation.

## Required outputs

Stage 1 must write recommendation, model-audit, execution-audit, and manifest artifacts. Stage 2 must write the full 69-cycle ledger, changed-cycle comparison, site summary, peak summary, performance gate, execution audit, and manifest. All source protocols, datasets, model artifacts, recommendation inventories, and result files are SHA256-audited.

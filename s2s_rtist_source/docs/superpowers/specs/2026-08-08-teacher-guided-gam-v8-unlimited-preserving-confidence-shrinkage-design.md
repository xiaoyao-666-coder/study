# Teacher-Guided GAM v8 Unlimited-Preserving Confidence Shrinkage Design

## Status and disclosure

This design was approved after the v6 development replay passed, the v6 2019
target-blind freeze retained zero changes, and the v7 continuous confidence
shrinkage replay failed one positive-oracle peak-distance condition. It is a
post-v7 development iteration, not independent confirmation.

The 2015--2018 v5 exact paired-SWAP results and v7 development results were
reviewed before this design. The 2019 covariates and the range of the six
rejected 2019 confidence values were also reviewed. No 2019 target, oracle, or
SWAP outcome has been reviewed. The 2024 boundary remains sealed.

## Motivation

The v6 rule protects limited candidates with a hard 5 mm LCB threshold, but it
produces zero changed recommendations in 2019. Lowering that threshold cannot
solve the coverage problem safely: the maximum reviewed 2019 LCB is 3.2453 mm,
while the harmful P2/2017 development endpoint has an LCB of 3.3071096292 mm.
Any scalar threshold low enough to admit a 2019 candidate first admits the
harmful development endpoint.

The v7 rule applies continuous shrinkage to every v5 movement. This needlessly
shrinks the unlimited P4/2017 endpoint, which has positive peak and exact SWAP
evidence. The resulting loss of that positive movement causes v7 to fail the
positive-oracle mean peak-distance gate by 0.00578298 mm.

V8 preserves the complete v5 endpoint for unlimited candidates and applies
continuous confidence shrinkage only to trust-region-limited candidates. It
adds no feature, fitted model, site rule, year rule, or searched threshold.

## Research question

Does preserving unlimited v5 endpoints while continuously shrinking only
limited v5 movements produce a development policy that is nonworse than the
frozen anchor in both peak location and exact continuous SWAP outcomes, with
enough action coverage to justify a separately frozen 2019 time-out
validation?

## Frozen selection rule

For each cycle, define:

```text
movement_v5 = v5_recommendation_mm - anchor_recommendation_mm

if abs(movement_v5) <= 1e-6:
    scale = 0
elif trust_region_limited is false:
    scale = 1
else:
    scale = clip(selected_lcb_mm / 5.0, 0.0, 1.0)

v8_recommendation_mm = clip(
    anchor_recommendation_mm + scale * movement_v5,
    0.0,
    60.0,
)
```

The 5.0 mm confidence reference is inherited from v6. Threshold search,
feature search, site-specific rules, year-specific rules, recommendation
optimization, and model refitting are forbidden. Site and year may be retained
only as identifiers and grouping fields.

The selection function may read only the frozen anchor, frozen v5 endpoint,
frozen LCB, and frozen trust-region-limited flag. It may not read a target peak,
oracle stratum, SWAP gain, SWAP regret, or any 2019 or 2024 outcome.

## Stage A: 2015--2018 peak replay

Stage A hash-verifies the frozen v5 inventory and the relevant v5, v6, and v7
gate, audit, and manifest artifacts. It reconstructs the v8 recommendation for
all 196 development cycles without fitting a model or running SWAP.

The replay writes a complete decision inventory, fold summary, site summary,
performance gate, execution audit, and SHA256 manifest. Before implementation,
the read-only design diagnostic produced:

- 196 complete cycles, 15 outer folds, and five target sites;
- 12 changed recommendations across four sites;
- overall mean peak distance: 11.75400084 to 11.69857816 mm;
- positive-oracle mean peak distance: 10.62501519 to 10.60176389 mm;
- zero-oracle mean peak distance: 12.35377447 to 12.28126074 mm;
- unchanged global maximum peak distance: 42.693373 mm;
- 14 of 15 nonworse folds and four of five nonworse sites.

The formal Stage A gate requires all of the following:

1. Exactly 196 cycles, 15 outer folds, and five target sites are complete.
2. All numeric outputs are finite and all irrigation amounts are in [0, 60].
3. At least 10 recommendations change and at least three target sites change.
4. At least 12 of 15 folds and four of five sites have nonworse mean peak
   distance.
5. Overall, positive-oracle, and zero-oracle mean peak distance are nonworse.
6. Global maximum peak distance is nonworse.
7. No 2019 or 2024 row is read and no automatic model promotion occurs.

Failure freezes v8 as a negative development result before any new SWAP run.

## Stage B: frozen exact paired-SWAP development qualification

After Stage A passes, Stage B first writes and hash-freezes the complete
changed-cycle execution plan. No recommendation may change after this plan is
written.

The expected plan has 12 changed cycles:

- 184 unchanged cycles contribute exact zero selected-minus-anchor difference;
- four selected endpoints are identical to already completed v5 endpoints and
  are reused after identity and hash verification;
- eight intermediate v8 endpoints require one new selected-endpoint SWAP run;
- all anchor endpoints are reused from exact existing evidence.

An existing endpoint may be reused only when the site, decision date, logical
irrigation amount, weather, initial state, SWAP configuration, and result
contract match. Irrigation identity uses a tolerance of 1e-6 mm. Interpolation
and nearest-endpoint substitution are forbidden.

If an exact endpoint fails numerically, the runner may use only the previously
frozen downward fallbacks of 0.1, 0.2, and 0.3 mm. The logical and simulated
amounts, attempt order, logs, and raw-output hashes must be recorded. Failure of
all frozen attempts makes the execution gate fail; it does not authorize a new
fallback or a changed recommendation.

The full paired ledger always contains all 196 cycles. SWAP evaluation uses the
continuous seven-day outcome.

The Stage B execution gate requires:

1. Stage A and execution-plan hashes are unchanged.
2. All 196 paired differences and all 12 changed-cycle endpoints are complete
   and finite.
3. Every reuse and new SWAP execution audit passes.
4. No interpolation, recommendation change, 2019 access, or 2024 access occurs.

The Stage B performance gate requires:

1. Pooled mean selected-minus-anchor SWAP gain is nonworse.
2. Positive-oracle and zero-oracle mean SWAP gain are each nonworse.
3. Overall fixed-eight regret is nonworse.
4. Changed-cycle maximum adaptive regret is nonworse.
5. At least 10 of 15 folds and four of five sites have nonworse mean SWAP gain.
6. Every Stage A peak-distance condition remains satisfied.

The numerical tolerance is 1e-6. Any failure freezes v8 as negative or
inconclusive and stops before 2019.

## Stage C: separately frozen 2019 time-out validation

Stage C is authorized only if Stages A and B pass. It must be implemented as a
separate, hash-separated protocol.

Recommendation freeze occurs before reading any 2019 target, oracle, or SWAP
outcome. The frozen procedure is applied to all 69 cycles. Unchanged cycles
contribute exact zero difference. Only genuinely changed cycles receive exact
paired endpoint evaluation. Based on the reviewed target-blind candidate
inventory, no more than six new selected endpoints are expected; this is not a
passing condition and does not permit selection using 2019 outcomes.

The 2019 coverage gate requires at least two changed cycles across at least two
target sites. The final gate evaluates the full 69-cycle ledger, site coverage,
overall and oracle-stratified peak distance, exact SWAP gain, fixed regret, and
adaptive regret. Regardless of outcome, 2019 may not be used to revise v8 and
then be relabeled as independent confirmation.

The evidence claim is "frozen 2019 time-out validation with target and outcome
holdout," not a pristine independent final test. The 2024 test remains sealed.

## Outputs and audit contract

Each stage writes deterministic CSV and JSON artifacts plus a SHA256 manifest.
The audits record input hashes, output hashes, protocol identity, years read,
target-read timing, recommendation counts, reuse sources, exact-run counts,
fallback attempts, interpolation status, and the mandatory stop or passing
action.

The implementation must refuse to overwrite a nonempty output directory unless
an explicit resume path verifies the frozen plan and every existing output.
Manifest changes, duplicate cycle keys, incomplete folds, non-finite values,
endpoint mismatches, or forbidden-year access are hard errors.

## Test strategy

Unit tests must cover:

1. Unchanged actions remain at anchor.
2. Unlimited actions retain the complete v5 endpoint regardless of LCB.
3. Limited actions use the clipped LCB/5 scale and preserve direction.
4. Site, year, target, oracle, and SWAP outcomes cannot affect selection.
5. Protocol constants and mandatory gates cannot be weakened.
6. Changed manifests, duplicate keys, endpoint mismatches, and interpolation
   attempts are rejected.
7. Existing exact endpoints are reused and only the eight expected intermediate
   endpoints enter the new-run plan.
8. The 196-cycle ledger, stratified summaries, hard gates, audits, and manifests
   are complete and deterministic.

Tests are written and observed failing before production implementation. The
v6, v7, and v5 paired-SWAP regression suites remain part of final verification.

## Explicitly unauthorized actions

This design does not authorize 2019 target access, 2019 SWAP execution, 2024
access, TTA, final refitting, interpolation, a new model family, a feature or
threshold search, or automatic promotion. Passing Stage B authorizes only a
separate frozen 2019 validation protocol.

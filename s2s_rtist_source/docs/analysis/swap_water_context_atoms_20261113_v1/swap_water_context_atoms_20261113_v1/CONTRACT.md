# Context-Weighted Paired Response Atoms

This is a fixed empirical screening candidate for one narrow question: does a
single mature seven-day early storage observation improve late irrigation-dose
response predictions after the prior has selected response atoms using public
predecision crop and forecast-weather context?

The related API-SWB-Mamba-MoE paper separates a water-balance prior from a
time-aware residual learner, then conditions expert routing on environmental
context (DOI: 10.3390/rs18183125). This package uses none of its neural
architecture or performance claims. The transferable design idea is to keep
physical/context prior information distinct from residual correction and test
transfer by held-out site.

## Fixed sample construction

For each outer site s, use the eight 2014-2015 station-year pairs from the
other four sites. Each pair is one atom. The atom stores:

- The eight-by-seven early storage residual matrix from M(s,h).
- The eight-by-seven late native crop-transpiration residual matrix from that
  same model identity.
- The atom's public early-decision context and donor site-year identity.

The model M(s,h) excludes both outer site s and donor site h. Target
branches use the exact same fitted model identity as their atom. An atom's
source residual is never applied to an unrelated target model branch.

## Context prior

Context contains exactly eleven public, predecision values: five crop state
values (cwdm, cwso, dvs, lai, root_cm) and six seven-day forecast
summaries (rain sum, and mean radiation, maximum temperature, minimum
temperature, vapor pressure, and wind). It excludes action dose, irrigation,
observed post-action state, initial water profile, and late-period labels.

Within each outer source set, standardize each feature using the eight source
contexts' population mean and standard deviation. A zero-variance feature is
inactive. The fixed Gaussian-kernel bandwidth is the median positive
pairwise standardized RMS distance among those eight contexts; if all
distances are zero, use one. The prior weight for atom i is proportional to
exp(-0.5 * d_i^2 / bandwidth^2), where d_i^2 is mean squared standardized
distance over active features. Features with population standard deviation at
most 1e-12 are inactive. All-inactive contexts produce uniform prior weights.
Target context never changes this scale.

## One-action feedback update

For executed early dose b, form the seven-day target residual under every
matched model branch: observed early storage minus that branch's forecast.
Compare this vector with atom i's saved source early-storage residual for the
same dose. Each day is standardized by the source-only population standard
deviation across the eight early residual atoms at that dose/day, floored at
0.05 mm. The atom likelihood is
exp(-0.5 * mean_day(standardized_mismatch^2)).

The posterior is prior times likelihood, computed and normalized in log space.
The late readout is the posterior-weighted sum of each atom's matched target
late base prediction plus that atom's late native uptake residual. No clipping,
rescaling, tuning, or second feedback is allowed. Off is the same readout
with context-prior weights and no feedback. Raw is the equal-weight mean of
the four target base-model branches.
For branch-level uptake checks, normalize the two atom weights within each
donor in log space and form its conditional corrected prediction; do not use
the donor's mixture contribution as its branch prediction.

## Scoring and limits

For each behavior b, the primary diagnostic scores only the seven doses
other than b, using all 21 unordered dose pairs and their seven-day
cumulative uptake differences. This avoids scoring the behavior used in the
likelihood update. Report daily paired-difference error, daily absolute uptake
error, sign repairs/new reversals, and minimum branch uptake as secondary
checks. Aggregate behavior versions within physical window, then years and
sites equally; also report the drop-largest-improvement-window result.

The fixed descriptive gates require positive macro seven-day improvement,
improvement at three of five sites and in both years, improvement after
dropping the largest improved physical window, no worsening of either daily
secondary error, no below -0.01 mm unobserved branch uptake, and no excess of
new wrong signs over repaired signs. These are screening gates, not
significance tests or adoption criteria. A positive improvement must exceed
1e-10 mm; daily non-worsening uses the same tolerance. Sign classification uses
a 0.01 mm zero band. Only ten late physical windows are
independent; dose pairs, behaviors, atoms, and fitted branches are dependent.

This package does not reconstruct a physical water profile, identify hydraulic
parameters, simulate the water/crop bridge, estimate yield or profit, rerun
SWAP, or alter the existing adoption pointer. A positive result permits a
separately designed legal crop/economic validation only. A negative result
closes this fixed candidate without target-informed tuning.

## Execution boundary

Local work is limited to projecting authenticated saved fields, artificial
fixtures, syntax/hash/ZIP checks, and package assembly. All real fitting,
prediction, and scoring run only after deployment to the specified Linux
server. The server package uses eight workers, BLAS thread count one, global
stage barriers, a 30-second heartbeat, 1800-second worker and 10800-second
total limits, a lifetime lock, duplicate-run protection, and automatic
numbered result ZIPs on success or failure.

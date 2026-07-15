# GEFS Three-Output Physics-Constrained TTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a site-general continuous-irrigation surrogate that predicts 7-day net gain, 7-day actual evapotranspiration, and 7 daily fixed 0-100 cm soil-moisture values, then performs label-free physics-only online TTA on 2024 GEFS inputs.

**Architecture:** Keep the repository's existing flat, versioned-script convention. Extend the restart-based SWAP label generator with reusable parsers for `result_restart.inc`, `result_restart.vap`, and `result_restart.crp`; construct year-separated sequence datasets; train a three-head PyTorch model with supervised and water-balance losses; then adapt the pretrained model sequentially on 2024 using only the physics loss. Irrigation is a differentiable scalar constrained to 0-60 mm and optimized by gradient ascent against predicted net gain, with final decisions evaluated against SWAP oracle curves.

**Tech Stack:** Python, pandas, NumPy, PyTorch, SWAP 4.0.1 outputs, NOAA GEFS GRIB2/xarray stack already available in the workspace runtime, pytest.

---

## Locked Requirements

- Output 1: future 7-day net gain.
- Output 2: future 7-day cumulative actual evapotranspiration.
- Output 3: future 7-day daily fixed 0-100 cm soil-layer mean volumetric water content.
- Current SWAP 4.0.1 `result.inc` contains `Tact`, `Eact`, and `Interc`; use `Tact + Eact + Interc` after confirming all three are water-depth increments in cm/day.
- Irrigation decision variable is continuous and constrained to `[0, 60]` mm.
- Forecast weather source for 2024 is GEFS.
- Data split: 2015-2018 training, 2019 validation/hyperparameter selection, 2024 final test with continuously applied online TTA.
- Pretraining loss: supervised output losses plus physics-consistency loss.
- TTA loss: physics-consistency loss only; no 2024 SWAP labels may participate in adaptation.

## Fixed 0-100 cm Control-Volume Update (2026-07-15)

The teacher selected a fixed control volume to avoid adding a moving-boundary output head. The follow-up diagnostic supports `0-100 cm`: all audited maize configurations use `RDC=100 cm`, the profile has an exact boundary at 100 cm, and the dynamic N2 sample closes at `-0.1590 mm` when recomputed over fixed 0-100 cm without a moving-boundary term.

This changes the meaning of Output 3. It is `soil_vwc_0_100cm_daily`, not dynamic-root-zone VWC. Daily root depth remains an input/state feature because early in the season part of the fixed layer is not yet accessible to the crop. Dynamic-root calculations and their moving-boundary term remain in the audit layer only. Detailed evidence is in `site_general_surrogate_eval/fixed_0_100cm_control_volume_validation_2026-07-15.md`.

## Root-Zone Water-Balance Audit Update (2026-07-13)

The following points are now confirmed and supersede earlier assumptions in this plan:

- SWAP native vertical-flux sign is positive upward and negative downward.
- `result_restart.vap` reports instantaneous `waterflux` and `drainage` rates in `cm/day`.
- The current `NPrintDay=1` extraction is a one-day rectangle-rule approximation, not a direct cumulative-flux output.
- Fixed-root samples currently audited match the root depth to an exact compartment-top boundary.
- Dynamic-root samples require an explicit moving-control-volume term, `10 * integral(theta(R(t), t), dR(t))`.
- The approved frequency diagnostic completed on 2026-07-14. At `NPrintDay=24`, the two fixed-root C2 residuals were `-0.1555 mm` and `-0.1393 mm`; the N2 dynamic-root residual changed from `-58.3264 mm` to `0.0986 mm` after adding a `58.4250 mm` moving-boundary term.

Detailed evidence and current values are recorded in `site_general_surrogate_eval/three_output_rootzone_water_balance_audit_2026-07-13.md`.

The completed bounded diagnostic used:

```text
code_C2 / 16-Jul-2024 / 30 mm
code_C2 / 16-Jul-2024 / 60 mm
code_N2 / 15-May-2024 / 30 mm
NPrintDay = 1, 4, 24
```

The two C2 samples tested fixed-root temporal integration. The N2 sample tested temporal integration plus the moving-root-boundary term. No model training or bulk data generation was performed. `NPrintDay=1` was inadequate, `NPrintDay=4` did not consistently converge, and the teacher formally adopted `NPrintDay=24` for production and the next multi-site smoke.

## Evidence-Supported Method Decisions Before Model Training

The water-balance method questions are resolved:

1. Use the directly integrated physical outflow as the supervised residual-flux target.
2. Use the fixed 0-100 cm control volume for formal model labels and physics loss; it has no moving-boundary term.
3. Calculate and store the moving-boundary term only in the retained dynamic-root audit branch.
4. Use production `NPrintDay=24`, actual-`Time` trapezoidal integration, native SWAP flux sign through integration, and `Dcum` aggregation for subdaily increments.

The GEFS feasibility review completed on 2026-07-14. The evidence-supported protocol uses a 06:00 local decision cutoff and same-date 00 UTC cycle; keeps all 31 members as separate surrogate scenarios; maps `D` through `D+6` using IANA site timezones; reconstructs precipitation from GRIB `startStep/endStep` rather than blind adjacent differencing; and falls back the entire ensemble, not individual members, by at most 24 hours. The mean-weather input is retained only as an ablation. A 5-site, 10-cycle `geavg` diagnostic completed on 2026-07-15: daily precipitation bias was `-0.0303 mm`, but 7-day precipitation MAE was `10.4145 mm`, with dry/light precipitation overestimated and moderate/heavy precipitation underestimated. This supports retaining member scenarios as the primary method. See `site_general_surrogate_eval/gefs_protocol_feasibility_evidence_2026-07-14.md` and `site_general_surrogate_eval/gefs_gridmet_bias_validation_analysis_2026-07-15.md`.

The formal `NPrintDay=24` multi-site smoke has passed. Do not begin bulk GEFS preparation or final model training until an automated preflight verifies every required 2024 date, all 31 members, required variables, and `f003-f180`, followed by a one-date GEFS extraction smoke.

## File Map

- Modify: `generate_restart_decision_dataset.py` - extract three-output and water-balance labels immediately after every SWAP candidate run.
- Create: `swap_three_output_labels_v1.py` - reusable SWAP output parsers, fixed 0-100 cm aggregation, and retained dynamic-root audit calculations.
- Create: `audit_swap_three_output_labels_v1.py` - one-site/one-date audit with independent balance checks.
- Create: `prepare_gefs_site_forecasts_v1.py` - GEFS download/index, unit conversion, site extraction, and daily windows.
- Create: `build_three_output_surrogate_dataset_v1.py` - merge SWAP labels, static features, history, actual-weather windows, and GEFS windows.
- Create: `three_output_physics_surrogate_v1.py` - model, preprocessing, supervised losses, physics loss, and checkpoint schema.
- Create: `train_three_output_physics_surrogate_v1.py` - 2015-2018 training and 2019 hyperparameter selection.
- Create: `run_2024_gefs_online_tta_v1.py` - chronological physics-only adaptation without label leakage.
- Create: `optimize_continuous_irrigation_gradient_v1.py` - projected gradient ascent over irrigation amount.
- Create: `evaluate_three_output_tta_v1.py` - output metrics, decision metrics, ablations, and stratified summaries.
- Create: `tests/test_swap_three_output_labels_v1.py`.
- Create: `tests/test_gefs_site_forecasts_v1.py`.
- Create: `tests/test_three_output_physics_surrogate_v1.py`.
- Create: `tests/test_online_tta_no_leakage_v1.py`.

### Task 1: Freeze The Experiment Contract

**Files:**
- Create: `site_general_surrogate_eval/three_output_gefs_tta_experiment_contract_v1.json`
- Create: `site_general_surrogate_eval/three_output_gefs_tta_experiment_contract_v1.md`

- [ ] **Step 1: Record output units and shapes**

Use the following schema after the blocking questions are resolved:

```json
{
  "horizon_days": 7,
  "irrigation_bounds_mm": [0.0, 60.0],
  "outputs": {
    "net_gain_7d": {"shape": [1], "unit": "USD/ha"},
    "aet_daily": {"shape": [7], "unit": "mm/day", "reported_as": "sum_7d"},
    "soil_vwc_0_100cm_daily": {"shape": [7], "unit": "cm3/cm3"}
  },
  "aet_components": ["Tact", "Eact", "Interc"],
  "split": {"train": [2015, 2016, 2017, 2018], "validation": [2019], "test_tta": [2024]},
  "tta_uses_labels": false
}
```

- [ ] **Step 2: Record the exact physics equation and GEFS protocol**

The contract must contain named terms, signs, units, daily/aggregate resolution, ensemble treatment, and missing-data behavior. Reject configurations that omit any required field.

- [ ] **Step 3: Add a contract validation test**

Run: `pytest tests/test_three_output_physics_surrogate_v1.py::test_experiment_contract_is_complete -v`

Expected: PASS only when the physics equation and GEFS protocol are explicit.

### Task 2: Audit And Parse SWAP Three-Output Labels

**Files:**
- Create: `swap_three_output_labels_v1.py`
- Create: `audit_swap_three_output_labels_v1.py`
- Test: `tests/test_swap_three_output_labels_v1.py`

- [x] **Step 1: Write parser tests using small committed text fixtures**

Test that `result.inc` parses daily `Tact`, `Eact`, `Interc`, storage change, runoff, drainage, bottom flux, and `baldev`; convert cm to mm exactly once.

- [x] **Step 2: Implement daily AET extraction**

```python
def actual_et_mm(frame: pd.DataFrame) -> pd.Series:
    return 10.0 * (frame["Tact"] + frame["Eact"] + frame["Interc"])
```

- [x] **Step 3: Implement fixed 0-100 cm VWC extraction**

For each of the seven days, select `result_restart.vap` compartments intersecting `[0, 100 cm]`, weight `wcontent` by intersected compartment thickness, and return one fixed-layer mean VWC per day. Keep daily root depth as an input/audit field. Retain the existing dynamic-root calculation only as an audit branch.

- [x] **Step 4: Verify parser consistency**

Run: `pytest tests/test_swap_three_output_labels_v1.py -v`

Expected: all parser, unit, partial-layer, and seven-day-shape tests PASS.

- [x] **Step 5: Run a one-site/one-date SWAP audit**

The audit CSV must show the seven daily ET values, seven fixed 0-100 cm VWC values, cumulative ET, fixed-layer start/end storage, each balance term, SWAP `baldev`, and independently reconstructed residual. A dynamic-root comparison may be retained as an audit-only table.

Acceptance: no missing day; VWC remains in physical bounds; reconstructed residual agrees with SWAP `baldev` within the documented tolerance.

Completed locally with the retained P1 0 mm and 30 mm raw outputs. Both candidates passed the fixed-schema validator; 131 fields were produced, no legacy `rootzone` or moving-boundary fields remained, and the maximum absolute balance residual was `0.1171495 mm` at the historical `NPrintDay=1` audit frequency.

### Task 3: Extend Restart Label Generation

**Files:**
- Modify: `generate_restart_decision_dataset.py:249`
- Test: `tests/test_swap_three_output_labels_v1.py`

- [x] **Step 1: Write a failing integration test for one candidate row**

Require these new fields:

```text
net_gain_7d
aet_7d_mm
aet_day01_mm ... aet_day07_mm
soil_vwc_0_100cm_day01 ... soil_vwc_0_100cm_day07
root_depth_day01_cm ... root_depth_day07_cm
water_balance_residual_day01_mm ... water_balance_residual_day07_mm
```

- [x] **Step 2: Parse outputs immediately after each candidate run**

At `generate_restart_decision_dataset.py:268`, parse `result_restart.inc`, `result_restart.vap`, and `result_restart.crp` before the next irrigation candidate overwrites them.

- [x] **Step 3: Preserve a small audit subset of raw SWAP files**

Keep raw outputs for at least one zero-irrigation and one nonzero-irrigation candidate per site/year. Store paths in the manifest; avoid copying every large profile file unless required for reproducibility.

- [x] **Step 4: Run a five-site fixed 0-100 cm smoke generation**

Acceptance: every candidate has exactly seven valid AET and VWC values, irrigation lies in `[0, 60]`, and all units are recorded in the manifest.

Completed on 2026-07-15 with 5 sites, 1 decision date, and 8 irrigation candidates per site. All 40 candidates passed the fixed-schema validator; all control depths were 100 cm, no legacy `rootzone` or moving-boundary formal fields remained, and the maximum absolute balance residual was `0.308267 mm`. The dynamic-root branch remains covered by the separate N2 diagnostic. See `site_general_surrogate_eval/three_output_fixed_0_100cm_npd24_5site_smoke_results_2026-07-15.md`.

### Task 4: Build GEFS 2024 Forecast Inputs

**Files:**
- Create: `prepare_gefs_site_forecasts_v1.py`
- Test: `tests/test_gefs_site_forecasts_v1.py`

- [ ] **Step 1: Test lead-time and accumulation-interval conversion**

Use synthetic and real-index fixtures to parse `startStep/endStep`, reconstruct non-overlapping precipitation intervals across six-hour reset boundaries, and reject blind adjacent differencing such as `f009-f006`.

- [ ] **Step 2: Implement forecast-cycle selection**

Set the decision timestamp to 06:00 in each site's IANA timezone. Select the same-date 00 UTC cycle only when all 31 members, required variables, and `f003-f180` are complete. If incomplete, fallback the entire ensemble to the previous 00 UTC cycle by at most 24 hours. Record initialization time, object timestamp, member, lead, valid time, source URL/key, checksum/ETag, and fallback reason.

- [ ] **Step 3: Map GEFS variables to the existing SWAP weather schema**

Produce daily minimum/maximum temperature, precipitation, shortwave radiation, wind speed, humidity/vapor-pressure variables, and any PET inputs required by the established SWAP preparation path. All conversions must be covered by unit tests.

- [ ] **Step 4: Extract the 12 site time series**

Write one tidy table keyed by `site_id`, `forecast_init`, `member`, and `valid_date`, plus a manifest reporting missing cycles and coverage.

- [ ] **Step 5: Compare GEFS with 2024 actual weather**

Report bias, MAE, RMSE, and correlation by variable, station, and lead day. This is the formal weather-domain-gap report and is separate from model performance.

### Task 5: Build Year-Separated Three-Output Datasets

**Files:**
- Create: `build_three_output_surrogate_dataset_v1.py`
- Test: `tests/test_three_output_physics_surrogate_v1.py`

- [ ] **Step 1: Define immutable split assignment by year**

```python
def split_for_year(year: int) -> str:
    if 2015 <= year <= 2018:
        return "train"
    if year == 2019:
        return "validation"
    if year == 2024:
        return "test_tta"
    raise ValueError(f"unsupported year: {year}")
```

- [ ] **Step 2: Add time and categorical encodings**

Store day-of-year sine/cosine explicitly. Retain raw categorical IDs for embedding layers. Fit all continuous-feature mean/std statistics using 2015-2018 rows only and serialize them with the model checkpoint.

- [ ] **Step 3: Produce actual-weather and GEFS variants**

Use identical sample keys and output labels where SWAP oracle evaluation is available. Mark weather source and forecast initialization explicitly so results cannot mix actual and forecast weather silently.

- [ ] **Step 4: Run leakage and shape checks**

Acceptance: no 2019 or 2024 row contributes to preprocessing statistics; each sample has 7 future weather days, 7 daily fixed 0-100 cm VWC targets, 7 daily ET targets, and one net-gain target.

### Task 6: Implement The Three-Output Physics-Constrained Model

**Files:**
- Create: `three_output_physics_surrogate_v1.py`
- Test: `tests/test_three_output_physics_surrogate_v1.py`

- [ ] **Step 1: Test model output shapes and bounds**

Expected model dictionary:

```python
{
    "net_gain_7d": tensor_of_shape_Bx1,
    "aet_daily": tensor_of_shape_Bx7,
    "soil_vwc_0_100cm_daily": tensor_of_shape_Bx7,
}
```

Use a nonnegative transform for daily AET and a bounded transform for VWC using site/layer physical limits where available.

- [ ] **Step 2: Implement embeddings, sequence encoder, and heads**

Use embeddings for site/categorical fields, standardized continuous static fields, and an LSTM or Transformer encoder for history/future sequences. Irrigation remains an unquantized continuous input.

- [ ] **Step 3: Implement supervised losses**

Use separately normalized losses for net gain, daily ET, and daily VWC so scale differences do not let one head dominate. Log every component separately.

- [ ] **Step 4: Implement physics loss from the locked contract**

The loss function must accept only model predictions, known weather/irrigation inputs, initial state, static soil information, and any explicitly modeled auxiliary fluxes. It must not read SWAP labels when `mode="tta"`.

- [ ] **Step 5: Test TTA loss independence from labels**

Change all target labels while keeping inputs fixed; `physics_loss(mode="tta")` must remain bitwise identical within floating-point tolerance.

### Task 7: Pretrain On 2015-2018 And Select On 2019

**Files:**
- Create: `train_three_output_physics_surrogate_v1.py`
- Test: `tests/test_three_output_physics_surrogate_v1.py`

- [ ] **Step 1: Implement deterministic training and checkpointing**

Checkpoint model weights, optimizer settings, preprocessing statistics, feature schema, output units, loss weights, random seed, and experiment-contract hash.

- [ ] **Step 2: Tune only against 2019 validation results**

Select architecture, learning rate, supervised-loss weights, physics-loss weight, and early stopping using 2019. Do not inspect 2024 decision metrics during selection.

- [ ] **Step 3: Report validation metrics**

Report net-gain MAE/RMSE/R2, cumulative and daily ET error, daily VWC error, balance residual, irrigation regret, nonzero recall, and irrigation-amount error by site and decision period.

- [ ] **Step 4: Lock the selected checkpoint**

Write a checksum and immutable configuration before any 2024 TTA experiment begins.

### Task 8: Implement Continuous Irrigation Gradient Optimization

**Files:**
- Create: `optimize_continuous_irrigation_gradient_v1.py`
- Test: `tests/test_three_output_physics_surrogate_v1.py`

- [ ] **Step 1: Test projected gradient ascent**

Use analytic toy profit curves with optima at 0, inside the interval, and 60 mm. Verify all returned values remain within `[0, 60]`.

- [ ] **Step 2: Optimize from multiple initial values**

Run projected Adam or gradient ascent from several initial irrigation amounts and retain the solution with highest predicted net gain. This reduces sensitivity to local maxima.

- [ ] **Step 3: Keep a dense-grid diagnostic only**

The reported method is gradient optimization. A dense grid may be retained solely to visualize the learned curve and diagnose failed optimization, not to generate the final recommendation.

### Task 9: Run Chronological Physics-Only TTA On 2024

**Files:**
- Create: `run_2024_gefs_online_tta_v1.py`
- Test: `tests/test_online_tta_no_leakage_v1.py`

- [ ] **Step 1: Test chronological ordering and label isolation**

The runner must process decision dates in order, start from the locked pretrained checkpoint, and reject any batch containing target columns in the adaptation function.

- [ ] **Step 2: Implement always-on TTA baseline**

At every 2024 decision date: construct the GEFS input, update the model for a small fixed number of steps using physics loss only, then optimize irrigation continuously. Preserve checkpoints before and after every date.

- [ ] **Step 3: Add stability guards**

Clip gradients, constrain learning rate and update steps, monitor parameter drift, and fall back to the previous checkpoint if physics loss becomes non-finite or exceeds the configured divergence threshold.

- [ ] **Step 4: Record a complete TTA trace**

Store date, GEFS initialization, pre/post physics loss, update count, parameter-drift norm, recommended irrigation, predicted outputs, optimizer convergence, and fallback status.

### Task 10: Evaluate Decisions And Required Ablations

**Files:**
- Create: `evaluate_three_output_tta_v1.py`

- [ ] **Step 1: Evaluate 2024 without using labels for adaptation**

After recommendations are frozen, use independent SWAP runs/curves only for evaluation: SWAP-oracle regret, nonzero recall, irrigation absolute error, and true SWAP gain at the recommended irrigation.

- [ ] **Step 2: Compare the minimum required baselines**

```text
pretrained model + actual 2024 weather, no TTA
pretrained model + GEFS, no TTA
pretrained model + GEFS, always-on physics TTA
paper fixed-list SWAP oracle reference
```

- [ ] **Step 3: Report output and physics metrics**

Report net-gain error, cumulative/daily ET error, daily fixed 0-100 cm VWC error, and water-balance residual. Stratify by site, date range, DVS/maturity state, root-depth band, GEFS lead day, and irrigation/non-irrigation class.

- [ ] **Step 4: Decide whether event-triggered TTA is justified**

Only after the always-on TTA baseline is complete, analyze whether adaptation helps during large weather-distribution shifts and harms stable periods. If supported, create a separate version that triggers TTA from an explicit drift statistic; do not fold this into the first baseline.

### Task 11: Update The Formal Teacher Report

**Files:**
- Modify: `reports/build_training_test_report.py`
- Create: `site_general_surrogate_eval/three_output_gefs_tta_stage_report_v1.md`

- [ ] **Step 1: Update dataset description**

Document years, sites, decision dates, candidate sampling, units, GEFS protocol, three label definitions, split policy, preprocessing, and missing-data rules.

- [ ] **Step 2: Update model input/output description**

State that daily ET is an internal output used for the cumulative reported value and physics loss. Describe embeddings, sine/cosine time encoding, training-only standardization, three heads, and continuous irrigation optimization.

- [ ] **Step 3: Update evaluation tables and figures**

Use black text, Song typeface for Chinese text, consistent font sizes, uniform page size, and restrained black-and-white tables. Keep actual-weather, GEFS-no-TTA, and GEFS-TTA results visibly separated.

## Execution Order

1. Resolve the four blocking method questions and freeze the contract.
2. Complete Tasks 2-3 and regenerate only a small smoke dataset.
3. Complete Task 4 and verify GEFS coverage before bulk SWAP generation.
4. Complete Tasks 5-7 and lock the 2019-selected pretrained model.
5. Complete Task 8 and verify continuous optimization independently.
6. Complete Tasks 9-10 for the first always-on 2024 TTA baseline.
7. Update the formal report; consider event-triggered TTA only after baseline analysis.

## Completion Criteria

- All output definitions, units, and balance terms are explicit and tested.
- SWAP-derived AET includes `Interc` for the current SWAP 4.0.1 outputs.
- Every sample contains one net-gain label and two seven-day sequences.
- GEFS inputs are reproducible from recorded initialization/member/lead metadata.
- Preprocessing statistics use 2015-2018 only.
- Hyperparameters are selected using 2019 only.
- 2024 TTA never consumes net-gain, ET, soil-moisture, irrigation-oracle, or SWAP-result labels.
- Final irrigation comes from bounded gradient optimization, not candidate enumeration.
- Final report includes weather-gap, no-TTA/TTA, decision-regret, nonzero-recall, and stratified results.

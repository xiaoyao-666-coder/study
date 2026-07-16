# S2S-RTIST First-Step Reproduction Notes

Date: 2026-05-29

## Goal

Run the first minimal reproduction step for the S2S climate forecast informed irrigation scheduling tool:

1. Download and unpack the author package.
2. Identify the executable workflow and SWAP input/output files.
3. Run or validate a minimal scheduling case.
4. Locate the first decision point for later optimization changes.

## Local Files

- Source package directory: `D:\study\s2s_rtist_source`
- Full archive: `D:\study\s2s_rtist_source\model3_opt_sto_upload.zip`
- Original unpacked package: `D:\study\s2s_rtist_source\model3_opt_sto_upload`
- Working copy: `D:\study\s2s_rtist_source\rtist_minimal_work`
- Main annotated script: `D:\study\s2s_rtist_source\Main_win_ensemble_mean.py`
- SWAP working directory: `D:\study\s2s_rtist_source\rtist_minimal_work\Maize`

## What Was Completed

- The Figshare package was downloaded successfully. The archive size is about 9.98 GB.
- The archive was unpacked. The top-level source directory is `model3_opt_sto_upload`.
- A lightweight working copy was created at `rtist_minimal_work` so the original package remains untouched.
- The code structure was checked:
  - `Main_win.py` and `Main_win_ensemble_mean.py` are the main entry scripts.
  - `Maize\ForecastStep.py` controls SWAP forecast runs.
  - `Maize\day_scheduled.csv` stores averaged irrigation recommendations.
  - `Maize\all_day_ir_var_results.csv` stores per-ensemble optimal irrigation choices.
  - `Maize\result_forec.*` stores SWAP forecast outputs.
- In the working copy, `ForecastStep.py` was adapted to call Windows `Swap.exe` instead of Linux `swap_test`.
- SWAP can start on Windows and writes forecast output files, so the executable and core input files are recognized.

## Current Windows SWAP Blocker

Windows `Swap.exe` starts normally but consistently stalls around the same point:

- Last crop output line reached: `2024-06-11`
- Number of crop output rows written: `103`
- The console/day-number log reaches `2024-06-10`

This happened with both:

- a shortened forecast end date around `2024-07-23`
- the original full-season end date `2024-12-01`

The Linux executables `swap_test`, `swap`, and `swap420` are ELF binaries. They cannot be run directly in the current Windows environment because WSL is installed as a command but no Linux distribution is configured.

Working interpretation: the author-provided results are usable, but direct Windows reruns are blocked by the packaged `Swap.exe` behavior on this case. The immediate research workflow can continue by reading the packaged SWAP results while the executable issue is treated separately.

## First Scheduling Date Reproduced From Packaged Results

The first decision date in the packaged results is `16-Jul-2024`.

Per-ensemble optimal irrigation choices:

| Irrigation (mm) | Count | Mean target value | Mean CWDM | Mean CWSO |
|---:|---:|---:|---:|---:|
| 20 | 1 | 77.00 | 5933.00 | 1211.00 |
| 25 | 6 | 89.23 | 6131.83 | 1386.00 |
| 30 | 2 | 165.10 | 6175.50 | 1401.00 |

The averaged recommendation in `day_scheduled.csv` is:

- `mean_ir = 25.555555555555557 mm`
- `mean_target_value = 104.73333333333332`

This confirms the decision layer is reachable: the original method chooses one optimal irrigation amount per ensemble member and averages those recommendations across ensemble runs.

## Decision Objective Located

The optimization loop is in:

- `D:\study\s2s_rtist_source\Main_win.py`
- `D:\study\s2s_rtist_source\Main_win_ensemble_mean.py`

Core logic:

1. Build irrigation decision dates every 4 days from the current date.
2. For each date, enumerate eight candidate irrigation depths:
   `0, 10, 15, 20, 25, 30, 40, 60 mm/ha`
3. For each candidate, update SWAP irrigation input files.
4. Run SWAP to simulate 7 days ahead.
5. Read `result_forec.crp` and extract `CWDM`, `CWSO`, `DVS`, and `Daynr`.
6. Use the no-irrigation candidate as the baseline.
7. Score each nonzero candidate with:

```text
target_value = (cwdm_value - cwdm_ir0) * yield_price_per_kg
               - ir * water_cost_per_ha_per_mm * weight_index
```

8. Select the candidate with maximum `target_value`.
9. In ensemble mode, repeat over ensemble members and average the selected irrigation recommendations.

This is the cleanest place to modify the research contribution: replace or augment the single-step discrete enumeration/objective while keeping SWAP as the crop-water simulator.

## Next Step

Use `all_day_ir_var_results.csv`, `day_scheduled.csv`, and the relevant sections of `Main_win_ensemble_mean.py` / `use_s2s.py` to isolate the decision objective. Then implement a small replacement decision experiment that changes only the optimization logic, while treating SWAP outputs as cached model evaluations until the Windows executable issue is solved.

# Project History

## Project
- Soil moisture forecasting paper reproduction in `D:\study`
- Main target paper: GCCL / "Combining graph neural network and convolutional LSTM network for ..."
- Auxiliary robustness paper: 2023 NC "Robust recurrent neural networks for time series forecasting"

## Current Status
- Core GCCL code has been refactored to align data processing more closely with the paper:
  - use `Hubei_Mask.tif`
  - fit normalization on training split only
  - normalize ERA5 per channel
  - build adjacency using valid pixels only
  - use 6-channel input but 1-channel SMAP output
- Kaggle training for the no-LSS version completed successfully.
- No-LSS reproduction metrics:
  - `1d`: RMSE `0.0258`, MAE `0.0202`, R2 `0.9067`
  - `3d`: RMSE `0.0323`, MAE `0.0248`, R2 `0.8540`
  - `5d`: RMSE `0.0358`, MAE `0.0272`, R2 `0.8204`
  - `7d`: RMSE `0.0382`, MAE `0.0292`, R2 `0.7951`
- Result interpretation:
  - `3d-7d` are already close to the paper's Table 1.
  - main gap is still at `1d`.

## LSS Work
- `loss.py` has been replaced with a more paper-aligned `J_MSE + lambda * J_LSS` implementation.
- Current practical implementation uses Halton low-discrepancy perturbations over time-channel dimensions and broadcasts spatially.
- A time-saving `train_nc2023.py` variant was created:
  - warm-start from no-LSS checkpoint
  - reduced `k_samples`
  - reduced epochs
  - early stopping
  - AMP disabled if sparse-mm half precision is incompatible

## Kaggle Notes
- Data paths differ across accounts:
  - one account used `/kaggle/input/datasets/yaoxiaoge/dataset`
  - another used `/kaggle/input/datasets/zhenxianliang/dataset`
- Warm-start checkpoint for LSS fine-tuning was loaded from:
  - `/kaggle/input/notebooks/zhenxianliang/notebookadd3394078/checkpoints/best_model.pth`

## LSS Fine-Tuning Outcome
- Fast LSS fine-tuning completed with early stopping.
- LSS metrics:
  - `1d`: RMSE `0.0247`, MAE `0.0189`, R2 `0.9148`
  - `3d`: RMSE `0.0322`, MAE `0.0248`, R2 `0.8548`
  - `5d`: RMSE `0.0360`, MAE `0.0278`, R2 `0.8181`
  - `7d`: RMSE `0.0386`, MAE `0.0300`, R2 `0.7901`
- Interpretation:
  - LSS improves short-term `1d`.
  - `3d` is nearly unchanged.
  - `5d-7d` slightly regress under the time-saving fine-tuning setup.
  - Recommended reporting stance: no-LSS is the main balanced result; LSS is supporting evidence for improved short-term robustness.

## Figures Reviewed
- Five result figures in `D:\study` were reviewed:
  - `pred_vs_target (2).png`
  - `time_series (2).png`
  - `spatial_comparison (2).png`
  - `error_distribution (2).png`
  - `lead_time_analysis (2).png`
- Main visual conclusion:
  - model is strong on global correlation and medium/long horizons
  - main remaining issue is over-smoothing and weak short-term peak response

## 2026-04-28 Context Sync
- Re-read long-term memory in `D:\Codex\CodexRules.md` and synced the current soil-moisture reproduction context back into the project history.
- Current reporting stance remains unchanged:
  - no-LSS is the main balanced reproduction result
  - fast LSS fine-tuning is supporting evidence for improved `1d` robustness
- Kaggle execution context worth remembering:
  - dataset mount roots differed across accounts
  - writable outputs must stay in `/kaggle/working`
  - the LSS fine-tuning warm-start checkpoint came from `/kaggle/input/notebooks/zhenxianliang/notebookadd3394078/checkpoints/best_model.pth`
- Near-term direction:
  - prioritize diagnosing the `1d` gap
  - avoid degrading `3d-7d` while testing any new robustness or short-horizon improvements

## 2026-04-29 OTW Paper Extraction
- Read and extracted the main ideas from `2023ICASSP OTW.pdf`.
- Core paper contribution:
  - proposes OTW / Optimal Transport Warping as a time-series distance
  - starts from optimal transport, then adapts it to time-series with unbalanced transport and locality control
  - aims to retain shape sensitivity while achieving linear time and space complexity
- Key technical properties emphasized in the paper:
  - computable in linear time and space
  - differentiable via a smooth approximation
  - GPU/TPU-friendly in principle
  - behaves as a proper metric under the smoothed formulation
- Main formulation points to remember for future implementation:
  - handle unequal total mass through an unbalanced OT sink construction
  - use a locality/window parameter similar in spirit to Sakoe-Chiba constraints
  - define a local OTW distance that interpolates between pointwise L1 comparison and global transport
  - use a smooth absolute-value surrogate for gradient-based learning
- Project implication:
  - teacher's requested first step should be interpreted as replacing the current prediction loss with an OTW-based shape-aware loss on the existing baseline before changing the model body
  - 7d horizon should be treated as the primary verification target, with 1d as a safety check and 3d/5d as minimal completeness if resources allow

## 2026-04-29 OTW Baseline Implementation
- Added an OTW-based loss path for the existing GCCL baseline without changing the model body.
- `loss.py` now includes `OTWLoss`, implemented around the ICASSP 2023 paper's main ingredients:
  - windowed cumulative sums for locality control
  - smooth L1 surrogate for differentiability
  - positive/negative split for handling arbitrary real-valued sequences
  - combined objective `J_MSE + lambda * J_OTW`
- Added `train_otw.py` as the first-step validation entry point:
  - keeps `7d` as the primary verification horizon
  - still evaluates and saves `1d/3d/5d/7d`
  - supports warm-start from baseline `best_model.pth`
  - disables AMP by default for sparse graph-op compatibility
- Added `compare_results.py` to print:
  - a `7d` primary table
  - an all-horizon summary table
  - delta vs baseline for follow-up reporting
- Smoke checks completed:
  - new scripts compile successfully
  - `OTWLoss` returns zero on identical sequences and increases on perturbed sequences

## 2026-04-29 Server And OTW Alignment
- Re-read the OTW paper more carefully around the actual formulation details:
  - `m` is the sink or waste cost
  - `s` is the locality window parameter
  - `s = 1` behaves like pointwise `L1`
  - `s = n` recovers the global OTW form
  - arbitrary signed sequences can be handled either directly or by positive/negative splitting
- Integrated the teacher-provided execution constraints into a separate note:
  - server-side project root should live under `/media/data_hot/lzx_projs/`
  - GPU should be chosen based on current availability
  - this stage should still prioritize `baseline + OTW`, `7d` first
- Confirmed that path handling is partially environment-configurable already through `runtime_paths.py`, so a server migration may not require immediate code edits if runtime environment variables are used correctly.
- Also noted a current blocker before actual server execution:
  - the exact dataset location on the server is still not confirmed

## 2026-04-29 Loss Split For Upload
- Split the loss implementations by experiment type so server upload can exclude LSS cleanly:
  - `loss.py` now keeps only the baseline `MSELoss`
  - `loss_lss.py` stores the NC2023 / LSS-related objective
  - `loss_otw.py` stores the OTW loss
- Updated dependent training entrypoints accordingly:
  - `train_nc2023.py` now imports from `loss_lss.py`
  - `train_otw.py` now imports from `loss_otw.py`
- Verified that the split files and both training scripts still compile successfully.

## 2026-04-30 Pure OTW Training Switch
- Adjusted the OTW experiment definition to match the teacher's stricter requirement:
  - default OTW training objective is now `J = lambda * J_OTW`
  - `MSE` is still computed and logged as a monitoring metric, but is no longer included in optimization by default
- Kept a compatibility switch in `train_otw.py`:
  - `--include-mse-term` restores the earlier hybrid objective `J_MSE + lambda * J_OTW` when needed for comparison
- Verified that `loss_otw.py` and `train_otw.py` still compile and that pure-OTW is now the default mode.

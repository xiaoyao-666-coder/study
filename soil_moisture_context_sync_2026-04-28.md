# Soil Moisture Reproduction Context Sync

## Sync Time
- 2026-04-28 20:24:32 +08:00

## Project Identity
- Project root: `D:\study`
- Topic: soil moisture forecasting paper reproduction
- Main paper: GCCL / "Combining graph neural network and convolutional LSTM network for ..."
- Auxiliary paper: 2023 NC "Robust recurrent neural networks for time series forecasting"

## Current Stage
- GCCL data pipeline has already been refactored toward paper alignment:
  - `Hubei_Mask.tif`
  - normalization fitted on training split only
  - ERA5 normalized per channel
  - adjacency built from valid pixels only
  - 6-channel input with 1-channel SMAP output
- No-LSS reproduction has finished and is currently the main balanced result.
- Fast LSS fine-tuning has finished and is best treated as supporting evidence for short-horizon robustness rather than the primary headline result.

## Key Metrics

### No-LSS
- `1d`: RMSE `0.0258`, MAE `0.0202`, R2 `0.9067`
- `3d`: RMSE `0.0323`, MAE `0.0248`, R2 `0.8540`
- `5d`: RMSE `0.0358`, MAE `0.0272`, R2 `0.8204`
- `7d`: RMSE `0.0382`, MAE `0.0292`, R2 `0.7951`

### Fast LSS Fine-Tuning
- `1d`: RMSE `0.0247`, MAE `0.0189`, R2 `0.9148`
- `3d`: RMSE `0.0322`, MAE `0.0248`, R2 `0.8548`
- `5d`: RMSE `0.0360`, MAE `0.0278`, R2 `0.8181`
- `7d`: RMSE `0.0386`, MAE `0.0300`, R2 `0.7901`

## Interpretation
- `3d-7d` no-LSS metrics are already close to the target paper table.
- The main remaining gap is `1d`.
- LSS helps short-term `1d`, barely changes `3d`, and slightly hurts `5d-7d` under the time-saving fine-tuning setup.

### Recommended Reporting Stance
- use no-LSS as the main balanced reproduction result
- use LSS as evidence that short-term robustness can be improved

## Important Files
- Training baseline: `D:\study\train.py`
- LSS fast fine-tuning: `D:\study\train_nc2023.py`
- Loss implementation: `D:\study\loss.py`
- Result analysis: `D:\study\analyze_results.py`
- LSS analysis: `D:\study\analyze_results_nc2023.py`
- Project history: `D:\study\history.md`

## Artifact Notes

### Reviewed Figures
- `pred_vs_target (2).png`
- `time_series (2).png`
- `spatial_comparison (2).png`
- `error_distribution (2).png`
- `lead_time_analysis (2).png`

### Main Visual Takeaway
- prediction tracks global structure well
- medium and long horizons are comparatively strong
- remaining weakness is over-smoothing and weak short-term peak response

## Kaggle Context
- Dataset root has differed across accounts:
  - `/kaggle/input/datasets/yaoxiaoge/dataset`
  - `/kaggle/input/datasets/zhenxianliang/dataset`
- Warm-start checkpoint used for LSS fine-tuning:
  - `/kaggle/input/notebooks/zhenxianliang/notebookadd3394078/checkpoints/best_model.pth`

## Next Useful Directions
- Prioritize diagnosing the `1d` gap without degrading `3d-7d`.
- Treat any future LSS or robustness tuning as a controlled comparison against the no-LSS baseline.
- If new results are produced, update `D:\study\history.md` first, then condense only durable lessons into `D:\Codex\CodexRules.md`.

## Pending Project Writeback
- `D:\study` is currently readable in this session, but write permission was not granted.
- Because of that, the intended 2026-04-28 sync note for `D:\study\history.md` is staged here first and should be copied back into project history once write access is available.

## Proposed history.md Sync Note

### 2026-04-28
- Re-read long-term memory in `D:\Codex\CodexRules.md` and project history in `D:\study\history.md`.
- Synced current project context into `D:\Codex\midTime\study\project_context.md` as a mid-term handoff snapshot.
- Reconfirmed current reporting stance: no-LSS remains the main balanced result, while fast LSS fine-tuning is supporting evidence for improved `1d` robustness.
- Pending action: write this sync back into `D:\study\history.md` after project-directory write permission is available.

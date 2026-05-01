# Server And OTW Alignment

## Sync Time
- 2026-04-29

## Purpose
- Consolidate the current OTW understanding and the teacher-provided server execution constraints.
- Do not modify training code yet.
- Treat this note as a pre-run alignment record before moving experiments to the server.

## OTW Paper: What Matters Most For This Project

### Paper Goal
- OTW is proposed to address the gap between:
  - DTW-like distances: shape-aware and alignment-aware, but usually quadratic and expensive
  - pointwise losses like MSE/L1: efficient and differentiable, but weak at sequence-shape comparison
- The paper's main target is not soil-moisture forecasting specifically, but a better time-series distance/loss/module.

### Core Construction
- Start from 1D optimal transport with transport cost `|i-j|`.
- Handle unequal total mass with an unbalanced OT construction using a sink point.
- Introduce a locality/window parameter `s` through windowed cumulative sums.
- Replace the absolute value by a smooth `L1` surrogate so the formulation can be optimized with gradients.

### Important Formula-Level Clarifications
- `m` in the paper is the sink or waste cost, not the locality window.
- `s` is the locality parameter and plays a role similar to the Sakoe-Chiba window in DTW.
- When `s = 1`, the local OTW behaves like pointwise `L1`.
- When `s = n`, it recovers the global OTW form.
- For arbitrary signed sequences, the paper suggests either:
  - applying the smoothed formula directly, or
  - splitting into positive and negative parts and summing the two OTW terms

### Why Teacher Wants 7d First
- OTW is meant to judge the shape of a whole forecast trajectory, not just pointwise amplitude.
- This makes it more meaningful on longer-horizon behavior like `7d` than on `1d`.
- `1d` should still be kept as a safety check to make sure shape-aware loss does not damage short-horizon accuracy.

### What The Paper Does Not Give Us Directly
- It does not provide a soil-moisture forecasting architecture to reproduce.
- It does not prove that OTW must improve every forecasting task.
- It mainly supports the argument that OTW is a reasonable shape-aware loss or module with better efficiency than DTW-like alternatives.

## Teacher-Provided Server Constraints

### Known Execution Environment
- Server should be accessed through SSH.
- Port is `4361`.
- Project code should be placed under:
  - `/media/data_hot/lzx_projs/`
- GPU should be selected based on which card is currently idle.
- Recommended workflow on the server:
  - inspect GPU availability first
  - explicitly pin `CUDA_VISIBLE_DEVICES`
  - use background training for long jobs

### Security Note
- SSH account credentials were provided separately in chat.
- They are intentionally not copied into project files to avoid spreading secrets inside the workspace.

## Current Codebase: Path Situation

### Important Observation
- The current path logic is not fully hardcoded.
- `runtime_paths.py` already supports environment-variable overrides for:
  - `PROJECT_DIR`
  - `WORK_DIR`
  - `DATASET_DIR`
  - `CHECKPOINT_DIR`
  - `CHECKPOINT_NC2023_DIR`
  - `CHECKPOINT_OTW_DIR`
  - `PLOTS_DIR`

### What Is Wrong Right Now
- The default values still assume a local project-root-style layout:
  - dataset defaults to `<work_dir>/dataset`
  - checkpoints default to subfolders under `<work_dir>`
- This means the code can probably be adapted to the server without code edits if:
  - the project is copied under `/media/data_hot/lzx_projs/...`
  - the dataset is placed somewhere known
  - the corresponding environment variables are set correctly at run time

### What Is Still Unknown
- The exact dataset directory on the server has not been confirmed yet.
- We know the current local workspace contains `Hubei_Mask.tif`, SMAP TIFFs, and ERA5 TIFFs, but we do not yet know:
  - whether the full dataset has already been uploaded to the server
  - where on the server it should live
  - whether baseline checkpoints also need to be copied there for OTW warm start

## Practical Implications Before Any Code Changes

### Good News
- There is no immediate need to rewrite path logic just because the server path is different.
- A first server run may be achievable through deployment and environment configuration alone.

### Cautions
- Sparse graph operations and AMP still need care in the server CUDA environment.
- OTW should be treated as a controlled loss replacement experiment, not yet as the final LGEM result.
- The first meaningful target remains:
  - baseline + OTW
  - `7d` primary
  - `1d` diagnostic
  - `3d/5d` added if resources allow

## Recommended Next Checks Before Running
- Confirm the exact dataset location on the server.
- Confirm whether a baseline checkpoint should be uploaded for warm start.
- Confirm whether the project directory name under `/media/data_hot/lzx_projs/` has a preferred convention.
- After that, prepare a server run command using environment variables instead of changing code immediately.

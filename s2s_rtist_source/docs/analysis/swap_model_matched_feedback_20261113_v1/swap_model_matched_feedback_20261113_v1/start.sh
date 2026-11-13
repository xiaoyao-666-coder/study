#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
cd /media/data_hot/lzx_projs/soil_moisture_otw/s2s_rtist_source/deploy/swap_model_matched_feedback_20261113_v1
exec python3 -u -B launch.py

#!/usr/bin/env python3
"""Run strict nested GAM peak-position interval calibration on 2015--2018 only."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import (
    CONTRACT_NAME,
    DATASET_NAME,
    EXPECTED_IRRIGATION_GRID,
    SOURCE_AUDIT_NAME,
    STATE_COLUMNS,
)
from scripts.evaluation.optimize_gefs_hierarchical_gam_continuous_v1 import optimize_cycle
from scripts.training.run_gefs_hierarchical_gam_bspline_development_v1 import (
    CYCLE_KEYS,
    _fit_model,
    add_states,
    validate_complete_cycles,
    validate_protocol as validate_gam_source_protocol,
)
from scripts.training.run_gefs_robust_envelope_gam_development_v1 import (
    build_jackknife_training_sets,
    read_frozen_penalties,
    true_fixed_grid_peak,
)
from s2s_rtist.models.gam_peak_interval_calibration_v1 import (
    FEATURE_COLUMNS,
    aggregate_peak_diagnostics,
    calibrated_interval,
    empirical_signed_quantiles,
    fit_weighted_ridge,
    select_hyperparameters,
)
from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import (
    GAM_VC,
    HierarchicalGamBSpline,
)
from s2s_rtist.models.robust_gam_envelope_v1 import gam_incremental_objective


PROTOCOL_ID = "teacher-guided-gam-peak-interval-calibration-development-v1"
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
EXPECTED_SITE_CYCLES = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
ROBUST_CYCLE_NAME = "gefs_robust_envelope_cycle_metrics_v1.csv"
ROBUST_FOLD_NAME = "gefs_robust_envelope_fold_metrics_v1.csv"
ROBUST_AUDIT_NAME = "gefs_robust_envelope_development_audit_v1.json"
ROBUST_GATE_NAME = "gefs_robust_envelope_development_gate_v1.json"
CALIBRATION_OUTPUT_NAMES = {
    "inner_samples": "gefs_gam_peak_calibration_inner_samples_v1.csv",
    "cycle_metrics": "gefs_gam_peak_calibration_cycle_metrics_v1.csv",
    "fold_metrics": "gefs_gam_peak_calibration_fold_metrics_v1.csv",
    "site_metrics": "gefs_gam_peak_calibration_site_metrics_v1.csv",
    "model_audit": "gefs_gam_peak_calibration_model_audit_v1.csv",
    "interval_audit": "gefs_gam_peak_calibration_interval_audit_v1.csv",
    "gate": "gefs_gam_peak_calibration_development_gate_v1.json",
    "audit": "gefs_gam_peak_calibration_development_audit_v1.json",
    "manifest": "gefs_gam_peak_calibration_development_manifest_v1.json",
}


@dataclass(frozen=True)
class InnerHoldoutUnit:
    held_out_site: str
    held_out_year: int
    group_id: str
    training: pd.DataFrame
    validation: pd.DataFrame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def validate_protocol(protocol: dict[str, Any]) -> None:
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "unexpected calibration protocol")
    data = protocol.get("data", {})
    _require(data.get("development_years") == [2015, 2016, 2017, 2018], "development years changed")
    _require(data.get("development_row_count") == 2152, "development row count changed")
    _require(data.get("development_cycle_count") == 269, "development cycle count changed")
    _require(data.get("reused_context_year") == 2019, "2019 context year changed")
    _require(data.get("sealed_final_test_year") == 2024, "2024 sealed year changed")
    _require(tuple(data.get("sites", ())) == EXPECTED_SITES, "site order changed")
    _require(data.get("candidate_count_per_cycle") == 8, "candidate count changed")
    _require(len(str(data.get("dataset_sha256", ""))) == 64, "dataset hash missing")
    _require(len(str(data.get("contract_sha256", ""))) == 64, "contract hash missing")
    source = protocol.get("source_robust_development", {})
    _require(source.get("protocol_id") == "teacher-guided-robust-envelope-gam-development-v1", "source robust protocol changed")
    _require(len(str(source.get("protocol_sha256", ""))) == 64, "source robust protocol hash missing")
    _require(source.get("expected_gate_passed") is False, "failed robust gate status changed")
    _require(source.get("reuse_preferred") is True, "source model reuse preference changed")
    _require(source.get("same_frozen_rows_refit_allowed_when_reuse_fails") is True, "source refit fallback changed")
    for name, digest in source.get("files", {}).items():
        _require(len(str(name)) > 0 and len(str(digest)) == 64, "source robust file hash missing")
    validation = protocol.get("validation", {})
    expected_folds = [
        {"train_years": [2015], "validation_year": 2016},
        {"train_years": [2015, 2016], "validation_year": 2017},
        {"train_years": [2015, 2016, 2017], "validation_year": 2018},
    ]
    _require(validation.get("rolling_year_folds") == expected_folds, "rolling folds changed")
    _require(validation.get("strict_outer_fold_count") == 15, "outer fold count changed")
    _require(validation.get("validation_cycle_count") == 196, "validation cycle count changed")
    _require(validation.get("site_cycle_counts") == EXPECTED_SITE_CYCLES, "site cycle counts changed")
    _require(tuple(protocol.get("features", {}).get("columns", ())) == FEATURE_COLUMNS, "calibration feature order changed")
    _require(protocol.get("features", {}).get("target_columns_allowed") is False, "target features enabled")
    calibrator = protocol.get("calibrator", {})
    _require(calibrator.get("gamma_grid") == [0.0, 2.0, 5.0], "gamma grid changed")
    _require(calibrator.get("lambda_grid") == [0.1, 1.0, 10.0], "lambda grid changed")
    _require(calibrator.get("high_risk_quantile") == 0.8, "high-risk quantile changed")
    interval = protocol.get("interval", {})
    _require(interval.get("signed_residual_quantiles") == [0.05, 0.95], "interval quantiles changed")
    _require(interval.get("empirical_quantile_method") == "inverse_cdf_order_statistic", "interval quantile method changed")
    _require(interval.get("range_mm") == [0.0, 60.0], "interval range changed")
    _require(interval.get("final_recommendation") == "clipped_interval_midpoint", "final recommendation changed")
    _require(interval.get("round_or_snap_to_fixed_grid") is False, "fixed-grid snapping enabled")
    gate = protocol.get("gate", {})
    _require(gate.get("tolerance") == 0.0, "gate tolerance changed")
    _require(gate.get("minimum_nonworse_outer_folds") == 10, "fold gate changed")
    _require(gate.get("minimum_interval_coverage") == 0.9, "coverage gate changed")
    forbidden = protocol.get("forbidden", {})
    for key in (
        "2019_rows_read",
        "2019_features_read",
        "2019_targets_read",
        "2019_swap_results_read",
        "2024_rows_read",
        "swap_rerun",
        "moe_training",
        "tta",
        "model_promotion",
        "second_stage_implementation",
    ):
        _require(forbidden.get(key) is True, f"{key.replace('_', ' ')} is not frozen as forbidden")


def build_inner_holdout_units(
    outer_train: pd.DataFrame,
    *,
    outer_target_site: str,
) -> list[InnerHoldoutUnit]:
    required = {"site_id", "target_year", "decision_date"}
    missing = sorted(required - set(outer_train.columns))
    if missing:
        raise ValueError(f"outer training frame is missing inner split columns: {missing}")
    sites = outer_train["site_id"].astype(str)
    years = pd.to_numeric(outer_train["target_year"], errors="raise").astype(int)
    if sites.eq(str(outer_target_site)).any():
        raise ValueError("outer target site leaked into inner sample source")
    source_sites = tuple(sorted(sites.unique()))
    if len(source_sites) < 3:
        raise ValueError("inner calibration requires at least three source sites")
    result: list[InnerHoldoutUnit] = []
    for held_out_year in sorted(years.unique()):
        for held_out_site in source_sites:
            validation_mask = sites.eq(held_out_site) & years.eq(held_out_year)
            if not validation_mask.any():
                continue
            training_mask = sites.ne(held_out_site) & years.le(held_out_year)
            training = outer_train.loc[training_mask].copy().reset_index(drop=True)
            validation = outer_train.loc[validation_mask].copy().reset_index(drop=True)
            if training.empty or validation.empty:
                raise ValueError("inner holdout produced an empty training or validation frame")
            if training["site_id"].astype(str).eq(held_out_site).any():
                raise RuntimeError("inner held-out site leaked into training")
            if pd.to_numeric(training["target_year"]).gt(held_out_year).any():
                raise RuntimeError("future year leaked into inner training")
            result.append(
                InnerHoldoutUnit(
                    held_out_site=held_out_site,
                    held_out_year=int(held_out_year),
                    group_id=f"{held_out_site}_{held_out_year}",
                    training=training,
                    validation=validation,
                )
            )
    if len(result) < 2:
        raise ValueError("inner calibration has fewer than two holdout groups")
    return result


def evaluate_development_gate(
    cycle_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    interval_audit: pd.DataFrame,
    *,
    execution_audit_passed: bool,
    multi_output_hashes_unchanged: bool,
) -> dict[str, Any]:
    required_cycle = {
        "fold_id",
        "true_oracle_is_positive",
        "baseline_peak_absolute_error_mm",
        "calibrated_peak_absolute_error_mm",
        "baseline_zero_oracle_false_positive",
        "calibrated_zero_oracle_false_positive",
    }
    missing = sorted(required_cycle - set(cycle_metrics.columns))
    if missing:
        raise ValueError(f"cycle metrics are missing gate columns: {missing}")
    if len(cycle_metrics) != 196 or len(fold_metrics) != 15 or len(interval_audit) != 196:
        raise ValueError("development gate requires 196 cycles, 15 folds, and 196 intervals")
    baseline = cycle_metrics["baseline_peak_absolute_error_mm"].to_numpy(dtype=np.float64)
    candidate = cycle_metrics["calibrated_peak_absolute_error_mm"].to_numpy(dtype=np.float64)
    positive = cycle_metrics["true_oracle_is_positive"].astype(bool).to_numpy()
    baseline_fold = fold_metrics["baseline_peak_mae_mm"].to_numpy(dtype=np.float64)
    candidate_fold = fold_metrics["calibrated_peak_mae_mm"].to_numpy(dtype=np.float64)
    coverage = float(interval_audit["interval_contains_true_peak"].mean())
    conditions = {
        "global_maximum_peak_distance_nonworse": float(candidate.max()) <= float(baseline.max()),
        "p95_peak_distance_nonworse": float(np.quantile(candidate, 0.95)) <= float(np.quantile(baseline, 0.95)),
        "positive_oracle_mean_peak_distance_nonworse": float(candidate[positive].mean()) <= float(baseline[positive].mean()) if positive.any() else True,
        "overall_mean_peak_distance_nonworse": float(candidate.mean()) <= float(baseline.mean()),
        "zero_oracle_false_positive_count_nonworse": int(cycle_metrics["calibrated_zero_oracle_false_positive"].sum()) <= int(cycle_metrics["baseline_zero_oracle_false_positive"].sum()),
        "at_least_10_of_15_fold_peak_distance_nonworse": int(np.sum(candidate_fold <= baseline_fold)) >= 10,
        "interval_coverage_at_least_0_90": coverage >= 0.90,
        "execution_isolation_finite_and_multi_output_hash_audits_passed": bool(execution_audit_passed and multi_output_hashes_unchanged),
    }
    return {
        "passed": bool(all(conditions.values())),
        "gate_type": "predeclared_paired_engineering_development_qualification_not_significance_test",
        "conditions": conditions,
        "observed": {
            "baseline_global_maximum_peak_distance_mm": float(baseline.max()),
            "candidate_global_maximum_peak_distance_mm": float(candidate.max()),
            "baseline_p95_peak_distance_mm": float(np.quantile(baseline, 0.95)),
            "candidate_p95_peak_distance_mm": float(np.quantile(candidate, 0.95)),
            "baseline_positive_oracle_mean_peak_distance_mm": float(baseline[positive].mean()) if positive.any() else 0.0,
            "candidate_positive_oracle_mean_peak_distance_mm": float(candidate[positive].mean()) if positive.any() else 0.0,
            "baseline_overall_mean_peak_distance_mm": float(baseline.mean()),
            "candidate_overall_mean_peak_distance_mm": float(candidate.mean()),
            "baseline_zero_oracle_false_positive_count": int(cycle_metrics["baseline_zero_oracle_false_positive"].sum()),
            "candidate_zero_oracle_false_positive_count": int(cycle_metrics["calibrated_zero_oracle_false_positive"].sum()),
            "paired_folds_nonworse_peak_distance": int(np.sum(candidate_fold <= baseline_fold)),
            "interval_coverage": coverage,
        },
        "passing_action": "authorize_separate_2015_2018_paired_continuous_swap_protocol_design_only",
        "failing_action": "freeze_peak_interval_calibration_as_negative_stop_before_swap_or_2019",
    }


def _write_fold_outputs(
    fold_dir: Path,
    *,
    protocol_sha256: str,
    inner_samples: pd.DataFrame,
    cycle_metrics: pd.DataFrame,
    model_audit: pd.DataFrame,
    interval_audit: pd.DataFrame,
    calibrator_arrays: dict[str, np.ndarray],
    selection: dict[str, Any],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "inner_samples": fold_dir / "inner_samples.csv",
        "cycle_metrics": fold_dir / "cycle_metrics.csv",
        "model_audit": fold_dir / "model_audit.csv",
        "interval_audit": fold_dir / "interval_audit.csv",
        "calibrator": fold_dir / "calibrator.npz",
        "selection": fold_dir / "selection.json",
    }
    for name, frame in (
        ("inner_samples", inner_samples),
        ("cycle_metrics", cycle_metrics),
        ("model_audit", model_audit),
        ("interval_audit", interval_audit),
    ):
        _atomic_write_csv(paths[name], frame)
    temporary_model = paths["calibrator"].with_suffix(".npz.tmp")
    with temporary_model.open("wb") as handle:
        np.savez_compressed(handle, **calibrator_arrays)
    temporary_model.replace(paths["calibrator"])
    _atomic_write_json(paths["selection"], selection)
    _atomic_write_json(
        fold_dir / "fold_complete.json",
        {
            "status": "gam_peak_interval_calibration_fold_complete",
            "protocol_sha256": protocol_sha256,
            "files": {name: sha256_file(path) for name, path in paths.items()},
        },
    )


def _completed_fold_outputs(
    fold_dir: Path,
    *,
    protocol_sha256: str,
) -> dict[str, Any] | None:
    marker_path = fold_dir / "fold_complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("protocol_sha256") != protocol_sha256:
        raise ValueError("completed fold protocol hash changed")
    paths = {
        "inner_samples": fold_dir / "inner_samples.csv",
        "cycle_metrics": fold_dir / "cycle_metrics.csv",
        "model_audit": fold_dir / "model_audit.csv",
        "interval_audit": fold_dir / "interval_audit.csv",
        "calibrator": fold_dir / "calibrator.npz",
        "selection": fold_dir / "selection.json",
    }
    expected_hashes = marker.get("files", {})
    for name, path in paths.items():
        if not path.is_file() or expected_hashes.get(name) != sha256_file(path):
            raise ValueError(f"completed fold file hash changed: {name}")
    model_audit = pd.read_csv(paths["model_audit"])
    if len(model_audit) != 1:
        raise ValueError("completed fold model audit row count changed")
    source_paths = str(model_audit.iloc[0]["source_model_paths"]).split("|")
    source_hashes = str(model_audit.iloc[0]["source_model_sha256"]).split("|")
    if len(source_paths) != len(source_hashes) or len(source_paths) != 5:
        raise ValueError("completed fold source model hash inventory changed")
    for source_path, expected_hash in zip(source_paths, source_hashes):
        path = Path(source_path)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"completed fold source model hash changed: {path.name}")
    return {
        "inner_samples": pd.read_csv(paths["inner_samples"]),
        "cycle_metrics": pd.read_csv(paths["cycle_metrics"]),
        "model_audit": model_audit,
        "interval_audit": pd.read_csv(paths["interval_audit"]),
        "selection": json.loads(paths["selection"].read_text(encoding="utf-8")),
    }


def _fold_id(target_site: str, validation_year: int) -> str:
    return f"holdout_{target_site}_rolling_to_{validation_year}"


def _curvature_abs(objective: Any, peak_mm: float, *, step_mm: float) -> float:
    peak = float(peak_mm)
    step = float(step_mm)
    if step <= 0.0:
        raise ValueError("curvature step must be positive")
    if peak <= step:
        x = np.asarray([0.0, step, 2.0 * step])
    elif peak >= 60.0 - step:
        x = np.asarray([60.0 - 2.0 * step, 60.0 - step, 60.0])
    else:
        x = np.asarray([peak - step, peak, peak + step])
    y = np.asarray(objective(x), dtype=np.float64)
    if y.shape != (3,) or not np.isfinite(y).all():
        raise ValueError("curvature objective is nonfinite or misaligned")
    return abs(float((y[0] - 2.0 * y[1] + y[2]) / step**2))


def _diagnostic_feature_rows(
    feature_rows: pd.DataFrame,
    baseline: HierarchicalGamBSpline,
    delete_one_models: Sequence[HierarchicalGamBSpline],
    *,
    optimization: dict[str, Any],
) -> pd.DataFrame:
    allowed = {"target_year", "site_id", "decision_date", "irrigation_mm", *STATE_COLUMNS}
    missing = sorted(allowed - set(feature_rows.columns))
    if missing:
        raise ValueError(f"feature rows are missing model input columns: {missing}")
    if len(delete_one_models) < 2:
        raise ValueError("at least two delete-one models are required")
    rows: list[dict[str, Any]] = []
    for keys, cycle in feature_rows.loc[:, sorted(allowed)].groupby(list(CYCLE_KEYS), sort=False):
        state = cycle.iloc[[0]].loc[:, ["site_id", "irrigation_mm", *STATE_COLUMNS]].copy()
        raw_result = optimize_cycle(
            baseline,
            state,
            fixed_grid=np.asarray(EXPECTED_IRRIGATION_GRID, dtype=np.float64),
            deployment_resolution_mm=float(optimization["deployment_resolution_mm"]),
            dense_step_mm=float(optimization["dense_diagnostic_step_mm"]),
        )
        raw_peak = float(raw_result["continuous_irrigation_mm"])
        raw_objective = gam_incremental_objective(baseline, state)
        raw_margin = float(raw_objective(np.asarray([raw_peak]))[0])
        delete_peaks: list[float] = []
        delete_gains: list[float] = []
        for model in delete_one_models:
            result = optimize_cycle(
                model,
                state,
                fixed_grid=np.asarray(EXPECTED_IRRIGATION_GRID, dtype=np.float64),
                deployment_resolution_mm=float(optimization["deployment_resolution_mm"]),
                dense_step_mm=float(optimization["dense_diagnostic_step_mm"]),
            )
            delete_peaks.append(float(result["continuous_irrigation_mm"]))
            delete_gains.append(float(gam_incremental_objective(model, state)(np.asarray([raw_peak]))[0]))
        diagnostics = aggregate_peak_diagnostics(
            raw_peak_mm=raw_peak,
            raw_gain_margin=raw_margin,
            curvature_abs=_curvature_abs(
                raw_objective,
                raw_peak,
                step_mm=float(optimization["curvature_step_mm"]),
            ),
            delete_one_peaks_mm=np.asarray(delete_peaks),
            delete_one_gains_at_raw=np.asarray(delete_gains),
        )
        rows.append(
            {
                "target_year": int(keys[0]),
                "site_id": str(keys[1]),
                "decision_date": str(keys[2]),
                **diagnostics,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result.duplicated(list(CYCLE_KEYS)).any():
        raise ValueError("diagnostic feature rows are empty or duplicated")
    numeric = result.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("diagnostic feature rows contain nonfinite values")
    return result


def _label_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cycle in frame.groupby(list(CYCLE_KEYS), sort=False):
        peak, gain = true_fixed_grid_peak(cycle)
        rows.append(
            {
                "target_year": int(keys[0]),
                "site_id": str(keys[1]),
                "decision_date": str(keys[2]),
                "true_fixed_list_irrigation_mm": float(peak),
                "true_fixed_list_net_gain_7d": float(gain),
            }
        )
    return pd.DataFrame(rows)


def _fit_inner_samples(
    outer_train: pd.DataFrame,
    *,
    outer_target_site: str,
    parameters: dict[str, float],
    source_protocol: dict[str, Any],
    optimization: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for unit in build_inner_holdout_units(outer_train, outer_target_site=outer_target_site):
        validate_complete_cycles(unit.training)
        validate_complete_cycles(unit.validation)
        baseline = _fit_model(unit.training, GAM_VC, source_protocol, parameters)
        source_sites = tuple(sorted(unit.training["site_id"].astype(str).unique()))
        delete_models = []
        for deleted_site in source_sites:
            selected = unit.training.loc[
                ~unit.training["site_id"].astype(str).eq(deleted_site)
            ].copy()
            validate_complete_cycles(selected)
            delete_models.append(_fit_model(selected, GAM_VC, source_protocol, parameters))
        feature_input = unit.validation.loc[
            :, ["target_year", "site_id", "decision_date", "irrigation_mm", *STATE_COLUMNS]
        ].copy()
        features = _diagnostic_feature_rows(
            feature_input,
            baseline,
            delete_models,
            optimization=optimization,
        )
        labels = _label_rows(unit.validation)
        samples = features.merge(labels, on=list(CYCLE_KEYS), how="inner", validate="one_to_one")
        samples["group_id"] = unit.group_id
        samples["outer_target_site"] = outer_target_site
        samples["inner_training_rows"] = len(unit.training)
        samples["inner_training_cycles"] = unit.training.groupby(list(CYCLE_KEYS)).ngroups
        samples["inner_validation_rows"] = len(unit.validation)
        samples["inner_validation_cycles"] = len(samples)
        samples["target_delta_mm"] = samples["true_fixed_list_irrigation_mm"] - samples["raw_peak_mm"]
        samples["baseline_peak_absolute_error_mm"] = samples["target_delta_mm"].abs()
        rows.append(samples)
        audits.append(
            {
                "group_id": unit.group_id,
                "held_out_site": unit.held_out_site,
                "held_out_year": unit.held_out_year,
                "training_sites": ",".join(sorted(unit.training["site_id"].astype(str).unique())),
                "training_years": ",".join(str(value) for value in sorted(unit.training["target_year"].astype(int).unique())),
                "training_rows": len(unit.training),
                "validation_rows": len(unit.validation),
                "validation_cycles": len(samples),
                "outer_target_training_rows": int(unit.training["site_id"].astype(str).eq(outer_target_site).sum()),
                "future_training_rows": int(unit.training["target_year"].astype(int).gt(unit.held_out_year).sum()),
                "baseline_model_count": 1,
                "delete_one_model_count": len(delete_models),
            }
        )
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(list(CYCLE_KEYS)).any():
        raise ValueError("inner calibration samples contain duplicate cycle keys")
    return result, audits


def _load_or_refit_outer_models(
    *,
    fold_dir: Path,
    outer_train: pd.DataFrame,
    target_site: str,
    parameters: dict[str, float],
    source_protocol: dict[str, Any],
    refit_dir: Path,
) -> tuple[HierarchicalGamBSpline, list[HierarchicalGamBSpline], dict[str, Any]]:
    source_sites = tuple(site for site in EXPECTED_SITES if site != target_site)
    baseline_path = fold_dir / "models" / "four_source_baseline.npz"
    delete_paths = [fold_dir / "models" / f"delete_{site}.npz" for site in source_sites]
    paths = [baseline_path, *delete_paths]
    mode = "reused_source_robust_models"
    reason = "all_source_model_files_loaded"
    try:
        marker_path = fold_dir / "fold_complete.json"
        if not marker_path.is_file():
            raise FileNotFoundError("source fold completion marker missing")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        for name, expected_hash in marker.get("files", {}).items():
            source_output = fold_dir / f"{name}.csv"
            if not source_output.is_file() or sha256_file(source_output) != expected_hash:
                raise ValueError(f"source fold output hash changed: {name}")
        baseline = HierarchicalGamBSpline.load_coefficients(baseline_path)
        delete_models = [HierarchicalGamBSpline.load_coefficients(path) for path in delete_paths]
        if tuple(baseline.site_ids_) != tuple(sorted(source_sites)):
            raise ValueError("source baseline site ownership changed")
        for deleted_site, model in zip(source_sites, delete_models):
            expected_sites = tuple(sorted(set(source_sites) - {deleted_site}))
            if tuple(model.site_ids_) != expected_sites:
                raise ValueError("source delete-one site ownership changed")
        for model in [baseline, *delete_models]:
            for name, value in parameters.items():
                if not np.isclose(model.penalty_weights_[name], value, rtol=0.0, atol=0.0):
                    raise ValueError("source model penalty weights changed")
    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        mode = "refit_on_same_frozen_outer_training_rows"
        reason = f"{type(error).__name__}: {error}"
        baseline = _fit_model(outer_train, GAM_VC, source_protocol, parameters)
        delete_models = [
            _fit_model(split.frame, GAM_VC, source_protocol, parameters)
            for split in build_jackknife_training_sets(
                outer_train,
                target_site=target_site,
                source_sites=source_sites,
            )
        ]
        baseline_path = refit_dir / "four_source_baseline.npz"
        delete_paths = [refit_dir / f"delete_{site}.npz" for site in source_sites]
        baseline.save_coefficients(baseline_path)
        for model, path in zip(delete_models, delete_paths):
            model.save_coefficients(path)
        paths = [baseline_path, *delete_paths]
    return baseline, delete_models, {
        "source_model_mode": mode,
        "source_model_reason": reason,
        "source_model_paths": [str(path) for path in paths],
        "source_model_sha256": [sha256_file(path) for path in paths],
    }


def _fit_calibration_fold(
    *,
    fold_id: str,
    target_site: str,
    validation_year: int,
    outer_train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict[str, float],
    source_protocol: dict[str, Any],
    robust_fold_dir: Path,
    calibration_fold_dir: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline, delete_models, source_audit = _load_or_refit_outer_models(
        fold_dir=robust_fold_dir,
        outer_train=outer_train,
        target_site=target_site,
        parameters=parameters,
        source_protocol=source_protocol,
        refit_dir=calibration_fold_dir / "refit_source_models",
    )
    inner_samples, inner_audits = _fit_inner_samples(
        outer_train,
        outer_target_site=target_site,
        parameters=parameters,
        source_protocol=source_protocol,
        optimization=protocol["continuous_optimization"],
    )
    x = inner_samples.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    delta = inner_samples["target_delta_mm"].to_numpy(dtype=np.float64)
    baseline_error = inner_samples["baseline_peak_absolute_error_mm"].to_numpy(dtype=np.float64)
    raw = inner_samples["raw_peak_mm"].to_numpy(dtype=np.float64)
    truth = inner_samples["true_fixed_list_irrigation_mm"].to_numpy(dtype=np.float64)
    selection = select_hyperparameters(
        x,
        delta,
        baseline_error,
        raw,
        truth,
        inner_samples["group_id"].astype(str).to_numpy(),
        gamma_grid=protocol["calibrator"]["gamma_grid"],
        lambda_grid=protocol["calibrator"]["lambda_grid"],
        feature_names=FEATURE_COLUMNS,
    )
    cross_residual = truth - selection.cross_fitted_peak_mm
    q05, q95 = empirical_signed_quantiles(cross_residual)
    final_model = fit_weighted_ridge(
        x,
        delta,
        baseline_error,
        gamma=selection.gamma,
        ridge_lambda=selection.ridge_lambda,
        feature_names=FEATURE_COLUMNS,
    )
    outer_feature_input = validation.loc[
        :, ["target_year", "site_id", "decision_date", "irrigation_mm", *STATE_COLUMNS]
    ].copy()
    outer_features = _diagnostic_feature_rows(
        outer_feature_input,
        baseline,
        delete_models,
        optimization=protocol["continuous_optimization"],
    )
    labels = _label_rows(validation)
    cycles = outer_features.merge(labels, on=list(CYCLE_KEYS), how="inner", validate="one_to_one")
    predicted_delta = final_model.predict(cycles.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    interval = calibrated_interval(
        raw_peak_mm=cycles["raw_peak_mm"].to_numpy(dtype=np.float64),
        predicted_delta_mm=predicted_delta,
        q05=q05,
        q95=q95,
    )
    cycles.insert(0, "fold_id", fold_id)
    cycles.insert(1, "target_site", target_site)
    cycles["predicted_delta_mm"] = predicted_delta
    cycles["corrected_point_peak_mm"] = interval.point_peak_mm
    cycles["residual_q05_mm"] = q05
    cycles["residual_q95_mm"] = q95
    cycles["interval_lower_mm"] = interval.lower_mm
    cycles["interval_upper_mm"] = interval.upper_mm
    cycles["interval_width_mm"] = interval.upper_mm - interval.lower_mm
    cycles["calibrated_peak_mm"] = interval.calibrated_peak_mm
    cycles["baseline_continuous_irrigation_mm"] = cycles["raw_peak_mm"]
    cycles["baseline_peak_absolute_error_mm"] = (
        cycles["raw_peak_mm"] - cycles["true_fixed_list_irrigation_mm"]
    ).abs()
    cycles["calibrated_peak_absolute_error_mm"] = (
        cycles["calibrated_peak_mm"] - cycles["true_fixed_list_irrigation_mm"]
    ).abs()
    threshold = float(protocol["gate"]["positive_irrigation_threshold_mm"])
    cycles["true_oracle_is_zero"] = cycles["true_fixed_list_irrigation_mm"].le(threshold).astype(int)
    cycles["true_oracle_is_positive"] = cycles["true_fixed_list_irrigation_mm"].gt(threshold).astype(int)
    cycles["baseline_zero_oracle_false_positive"] = (
        cycles["true_oracle_is_zero"].eq(1) & cycles["raw_peak_mm"].gt(threshold)
    ).astype(int)
    cycles["calibrated_zero_oracle_false_positive"] = (
        cycles["true_oracle_is_zero"].eq(1) & cycles["calibrated_peak_mm"].gt(threshold)
    ).astype(int)
    cycles["interval_contains_true_peak"] = (
        cycles["true_fixed_list_irrigation_mm"].ge(cycles["interval_lower_mm"])
        & cycles["true_fixed_list_irrigation_mm"].le(cycles["interval_upper_mm"])
    ).astype(int)
    interval_audit = cycles.loc[
        :,
        [
            "fold_id",
            "target_site",
            "target_year",
            "site_id",
            "decision_date",
            "true_fixed_list_irrigation_mm",
            "corrected_point_peak_mm",
            "residual_q05_mm",
            "residual_q95_mm",
            "interval_lower_mm",
            "interval_upper_mm",
            "interval_width_mm",
            "interval_contains_true_peak",
        ],
    ].copy()
    prediction_before = baseline.predict_shared(validation).to_numpy(dtype=np.float64)
    prediction_hash_before = _array_sha256(prediction_before)
    prediction_hash_after = _array_sha256(baseline.predict_shared(validation).to_numpy(dtype=np.float64))
    model_audit = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "target_site": target_site,
                "validation_year": validation_year,
                "outer_training_rows": len(outer_train),
                "outer_training_cycles": outer_train.groupby(list(CYCLE_KEYS)).ngroups,
                "outer_validation_rows": len(validation),
                "outer_validation_cycles": len(cycles),
                "inner_group_count": inner_samples["group_id"].nunique(),
                "inner_sample_count": len(inner_samples),
                "selected_gamma": selection.gamma,
                "selected_lambda": selection.ridge_lambda,
                "cross_fitted_residual_q05_mm": q05,
                "cross_fitted_residual_q95_mm": q95,
                "inner_cross_fitted_interval_coverage": float(
                    np.mean((truth >= selection.cross_fitted_peak_mm + q05) & (truth <= selection.cross_fitted_peak_mm + q95))
                ),
                "feature_order": ",".join(FEATURE_COLUMNS),
                "target_site_training_rows": int(outer_train["site_id"].astype(str).eq(target_site).sum()),
                "target_site_preprocessing_rows": 0,
                "future_outer_training_rows": int(outer_train["target_year"].astype(int).ge(validation_year).sum()),
                "all_inner_target_and_future_rows_zero": int(all(row["outer_target_training_rows"] == 0 and row["future_training_rows"] == 0 for row in inner_audits)),
                "multi_output_prediction_sha256_before": prediction_hash_before,
                "multi_output_prediction_sha256_after": prediction_hash_after,
                "multi_output_prediction_hash_unchanged": int(prediction_hash_before == prediction_hash_after),
                "source_model_mode": source_audit["source_model_mode"],
                "source_model_reason": source_audit["source_model_reason"],
                "source_model_paths": "|".join(source_audit["source_model_paths"]),
                "source_model_sha256": "|".join(source_audit["source_model_sha256"]),
                "fit_seconds": time.perf_counter() - started,
            }
        ]
    )
    inner_samples.insert(0, "fold_id", fold_id)
    selection_json = {
        "fold_id": fold_id,
        "selected_gamma": selection.gamma,
        "selected_lambda": selection.ridge_lambda,
        "selection_metrics": {
            "maximum_absolute_error_mm": selection.maximum_absolute_error_mm,
            "positive_mean_absolute_error_mm": selection.positive_mean_absolute_error_mm,
            "overall_mean_absolute_error_mm": selection.overall_mean_absolute_error_mm,
        },
        "cross_fitted_residual_q05_mm": q05,
        "cross_fitted_residual_q95_mm": q95,
        "search_rows": list(selection.search_rows),
        "inner_group_audit": inner_audits,
    }
    calibrator_arrays = {
        "feature_names": np.asarray(FEATURE_COLUMNS),
        "feature_means": final_model.feature_means,
        "feature_scales": final_model.feature_scales,
        "intercept": np.asarray([final_model.intercept]),
        "coefficients": final_model.coefficients,
        "gamma": np.asarray([final_model.gamma]),
        "ridge_lambda": np.asarray([final_model.ridge_lambda]),
        "q05": np.asarray([q05]),
        "q95": np.asarray([q95]),
    }
    return {
        "inner_samples": inner_samples,
        "cycle_metrics": cycles,
        "model_audit": model_audit,
        "interval_audit": interval_audit,
        "calibrator_arrays": calibrator_arrays,
        "selection": selection_json,
    }


def _summary_metrics(cycles: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, part in cycles.groupby(group_column, sort=False):
        positive = part["true_oracle_is_positive"].eq(1)
        rows.append(
            {
                group_column: value,
                "cycle_count": len(part),
                "baseline_peak_mae_mm": float(part["baseline_peak_absolute_error_mm"].mean()),
                "calibrated_peak_mae_mm": float(part["calibrated_peak_absolute_error_mm"].mean()),
                "baseline_positive_peak_mae_mm": float(part.loc[positive, "baseline_peak_absolute_error_mm"].mean()) if positive.any() else 0.0,
                "calibrated_positive_peak_mae_mm": float(part.loc[positive, "calibrated_peak_absolute_error_mm"].mean()) if positive.any() else 0.0,
                "baseline_zero_false_positive_count": int(part["baseline_zero_oracle_false_positive"].sum()),
                "calibrated_zero_false_positive_count": int(part["calibrated_zero_oracle_false_positive"].sum()),
                "baseline_maximum_peak_error_mm": float(part["baseline_peak_absolute_error_mm"].max()),
                "calibrated_maximum_peak_error_mm": float(part["calibrated_peak_absolute_error_mm"].max()),
                "interval_coverage": float(part["interval_contains_true_peak"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _verify_source_robust_inputs(
    robust_dir: Path,
    source_config: dict[str, Any],
) -> dict[str, Path]:
    paths = {name: robust_dir / name for name in source_config["files"]}
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != source_config["files"][name]:
            raise ValueError(f"frozen robust development file hash changed: {name}")
    gate = json.loads(paths[ROBUST_GATE_NAME].read_text(encoding="utf-8"))
    audit = json.loads(paths[ROBUST_AUDIT_NAME].read_text(encoding="utf-8"))
    if gate.get("passed") is not False or audit.get("status") != source_config["expected_status"]:
        raise ValueError("frozen robust negative result status changed")
    return paths


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = args.protocol.resolve()
    dataset_dir = args.dataset_dir.resolve()
    source_development_dir = args.source_development_dir.resolve()
    robust_development_dir = args.robust_development_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    protocol_sha = sha256_file(protocol_path)
    design_path = protocol_path.parent / protocol["design"]["filename"]
    if sha256_file(design_path) != protocol["design"]["sha256"]:
        raise ValueError("approved calibration design hash changed")
    robust_protocol_path = protocol_path.parent / protocol["source_robust_development"]["protocol_filename"]
    if sha256_file(robust_protocol_path) != protocol["source_robust_development"]["protocol_sha256"]:
        raise ValueError("frozen robust protocol hash changed")
    robust_protocol = json.loads(robust_protocol_path.read_text(encoding="utf-8"))
    source_protocol_path = protocol_path.parent / robust_protocol["source_development"]["protocol_filename"]
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    validate_gam_source_protocol(source_protocol)
    robust_inputs = _verify_source_robust_inputs(
        robust_development_dir,
        protocol["source_robust_development"],
    )
    dataset_path = dataset_dir / DATASET_NAME
    contract_path = dataset_dir / CONTRACT_NAME
    dataset_audit_path = dataset_dir / SOURCE_AUDIT_NAME
    if sha256_file(dataset_path) != protocol["data"]["dataset_sha256"]:
        raise ValueError("formal surrogate dataset hash changed")
    if sha256_file(contract_path) != protocol["data"]["contract_sha256"]:
        raise ValueError("formal surrogate contract hash changed")
    dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
    if dataset_audit.get("mandatory_gate_passed") is not True or dataset_audit.get("dataset_construction_gate_passed") is not True:
        raise ValueError("formal surrogate dataset construction gate did not pass")
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    development = pd.read_csv(dataset_path, nrows=int(protocol["data"]["development_row_count"]))
    if len(development) != int(protocol["data"]["development_row_count"]):
        raise ValueError("development-only read returned the wrong row count")
    development = add_states(development)
    years = pd.to_numeric(development["target_year"], errors="raise").astype(int)
    if sorted(years.unique()) != protocol["data"]["development_years"]:
        raise ValueError("development-only read crossed the frozen year boundary")
    validate_complete_cycles(development)
    if development.groupby(list(CYCLE_KEYS)).ngroups != int(protocol["data"]["development_cycle_count"]):
        raise ValueError("development cycle count changed")
    source_fold_metrics_path = source_development_dir / robust_protocol["source_development"]["fold_metrics_filename"]
    if sha256_file(source_fold_metrics_path) != robust_protocol["source_development"]["fold_metrics_sha256"]:
        raise ValueError("frozen GAM fold penalty summary hash changed")
    all_results: list[dict[str, Any]] = []
    completed = 0
    for target_site in EXPECTED_SITES:
        for fold in protocol["validation"]["rolling_year_folds"]:
            validation_year = int(fold["validation_year"])
            fold_id = _fold_id(target_site, validation_year)
            fold_dir = output_dir / "folds" / fold_id
            resumed = _completed_fold_outputs(fold_dir, protocol_sha256=protocol_sha) if args.resume else None
            if resumed is None:
                outer_train = development.loc[
                    years.lt(validation_year) & development["site_id"].astype(str).ne(target_site)
                ].copy().reset_index(drop=True)
                validation = development.loc[
                    years.eq(validation_year) & development["site_id"].astype(str).eq(target_site)
                ].copy().reset_index(drop=True)
                validate_complete_cycles(outer_train)
                validate_complete_cycles(validation)
                parameters = read_frozen_penalties(source_fold_metrics_path, expected_fold_id=fold_id)
                result = _fit_calibration_fold(
                    fold_id=fold_id,
                    target_site=target_site,
                    validation_year=validation_year,
                    outer_train=outer_train,
                    validation=validation,
                    parameters=parameters,
                    source_protocol=source_protocol,
                    robust_fold_dir=robust_development_dir / "folds" / fold_id,
                    calibration_fold_dir=fold_dir,
                    protocol=protocol,
                )
                _write_fold_outputs(
                    fold_dir,
                    protocol_sha256=protocol_sha,
                    inner_samples=result["inner_samples"],
                    cycle_metrics=result["cycle_metrics"],
                    model_audit=result["model_audit"],
                    interval_audit=result["interval_audit"],
                    calibrator_arrays=result["calibrator_arrays"],
                    selection=result["selection"],
                )
                resumed = {name: result[name] for name in ("inner_samples", "cycle_metrics", "model_audit", "interval_audit", "selection")}
                was_resumed = False
            else:
                was_resumed = True
            all_results.append(resumed)
            completed += 1
            print(f"fold={completed}/15 fold_id={fold_id} resumed={str(was_resumed).lower()}", flush=True)
    inner_samples = pd.concat([item["inner_samples"] for item in all_results], ignore_index=True)
    cycle_metrics = pd.concat([item["cycle_metrics"] for item in all_results], ignore_index=True)
    model_audit = pd.concat([item["model_audit"] for item in all_results], ignore_index=True)
    interval_audit = pd.concat([item["interval_audit"] for item in all_results], ignore_index=True)
    fold_metrics = _summary_metrics(cycle_metrics, "fold_id")
    site_metrics = _summary_metrics(cycle_metrics, "site_id")
    numeric_frames = [inner_samples, cycle_metrics, model_audit, interval_audit, fold_metrics, site_metrics]
    numeric_finite = all(
        np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)).all()
        for frame in numeric_frames
    )
    site_counts = cycle_metrics.groupby("site_id").size().astype(int).to_dict()
    keys_unique = not cycle_metrics.duplicated(list(CYCLE_KEYS)).any()
    intervals_valid = bool(
        interval_audit["interval_lower_mm"].le(interval_audit["interval_upper_mm"]).all()
        and interval_audit["interval_lower_mm"].between(0.0, 60.0).all()
        and interval_audit["interval_upper_mm"].between(0.0, 60.0).all()
        and interval_audit["interval_width_mm"].ge(0.0).all()
    )
    isolation = bool(
        model_audit["target_site_training_rows"].eq(0).all()
        and model_audit["target_site_preprocessing_rows"].eq(0).all()
        and model_audit["future_outer_training_rows"].eq(0).all()
        and model_audit["all_inner_target_and_future_rows_zero"].eq(1).all()
    )
    multi_output_unchanged = bool(model_audit["multi_output_prediction_hash_unchanged"].eq(1).all())
    execution = bool(
        len(cycle_metrics) == 196
        and len(fold_metrics) == 15
        and len(site_metrics) == 5
        and len(model_audit) == 15
        and len(interval_audit) == 196
        and site_counts == EXPECTED_SITE_CYCLES
        and keys_unique
        and numeric_finite
        and intervals_valid
        and isolation
    )
    gate = evaluate_development_gate(
        cycle_metrics,
        fold_metrics,
        interval_audit,
        execution_audit_passed=execution,
        multi_output_hashes_unchanged=multi_output_unchanged,
    )
    outputs = {name: output_dir / filename for name, filename in CALIBRATION_OUTPUT_NAMES.items()}
    for name, frame in (
        ("inner_samples", inner_samples),
        ("cycle_metrics", cycle_metrics),
        ("fold_metrics", fold_metrics),
        ("site_metrics", site_metrics),
        ("model_audit", model_audit),
        ("interval_audit", interval_audit),
    ):
        _atomic_write_csv(outputs[name], frame)
    _atomic_write_json(outputs["gate"], gate)
    audit = {
        "status": "gam_peak_interval_calibration_development_pregate_passed_pending_separate_swap_protocol" if gate["passed"] else "gam_peak_interval_calibration_development_failed_stop_before_swap_or_2019",
        "protocol_id": protocol["protocol_id"],
        "mandatory_execution_gate_passed": execution,
        "predeclared_performance_gate_passed": bool(gate["passed"]),
        "outer_fold_count": len(fold_metrics),
        "validation_cycle_count": len(cycle_metrics),
        "site_cycle_counts": site_counts,
        "inner_sample_count": len(inner_samples),
        "inner_group_count_sum": int(model_audit["inner_group_count"].sum()),
        "all_target_and_future_training_rows_zero": isolation,
        "all_multi_output_prediction_hashes_unchanged": multi_output_unchanged,
        "all_intervals_valid": intervals_valid,
        "all_numeric_outputs_finite": numeric_finite,
        "2019_rows_read": 0,
        "2019_feature_rows_read": 0,
        "2019_target_rows_read": 0,
        "2019_swap_results_read": 0,
        "2024_rows_read": 0,
        "swap_rerun_performed": False,
        "second_stage_implemented": False,
        "moe_training_performed": False,
        "tta_performed": False,
        "model_promotion_performed": False,
        "2019_prior_results_used_only_as_hypothesis_source": True,
        "next_gate": gate["passing_action"] if gate["passed"] else gate["failing_action"],
        "inputs": {
            "protocol_sha256": protocol_sha,
            "design_sha256": sha256_file(design_path),
            "dataset_sha256": sha256_file(dataset_path),
            "contract_sha256": sha256_file(contract_path),
            "source_protocol_sha256": sha256_file(source_protocol_path),
            **{f"robust_{name}": sha256_file(path) for name, path in robust_inputs.items()},
        },
    }
    _atomic_write_json(outputs["audit"], audit)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "inputs": audit["inputs"],
        "fold_source_model_hashes": {
            row.fold_id: str(row.source_model_sha256).split("|")
            for row in model_audit.itertuples(index=False)
        },
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    _atomic_write_json(outputs["manifest"], manifest)
    print(
        f"development_complete folds=15 cycles=196 gate_passed={str(gate['passed']).lower()} next_gate={audit['next_gate']}",
        flush=True,
    )
    if not gate["passed"]:
        raise RuntimeError(f"peak interval calibration gate failed; see {outputs['gate']}")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-development-dir", type=Path, required=True)
    parser.add_argument("--robust-development-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    outputs = run(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()

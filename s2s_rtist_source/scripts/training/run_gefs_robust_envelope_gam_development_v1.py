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


from scripts.training.run_gefs_hierarchical_gam_bspline_development_v1 import (
    CYCLE_KEYS,
    _fit_model,
    _dataset_index,
    _read_selected_rows,
    add_states,
    validate_protocol as validate_source_protocol,
    validate_complete_cycles,
)
from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import (
    CONTRACT_NAME,
    DATASET_NAME,
    EXPECTED_IRRIGATION_GRID,
    SOURCE_AUDIT_NAME,
    STATE_COLUMNS,
)
from scripts.evaluation.evaluate_gefs_hierarchical_gam_bspline_development_v1 import prediction_metrics
from scripts.evaluation.optimize_gefs_hierarchical_gam_continuous_v1 import optimize_cycle
from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import GAM_VC, HierarchicalGamBSpline
from s2s_rtist.models.robust_gam_envelope_v1 import (
    EnvelopeOptimizationResult,
    gam_incremental_objective,
    optimize_piecewise_cubic_lower_envelope,
)


PROTOCOL_ID = "teacher-guided-robust-envelope-gam-development-v1"
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
EXPECTED_SITE_CYCLES = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
EXPECTED_SOURCE_PROTOCOL = (
    "teacher-guided-hierarchical-gam-nested-shared-selection-development-v2"
)


@dataclass(frozen=True)
class JackknifeTrainingSet:
    deleted_source_site: str
    frame: pd.DataFrame
    target_site_training_rows: int
    target_site_preprocessing_rows: int
    deleted_source_training_rows: int
    deleted_source_preprocessing_rows: int


def build_jackknife_training_sets(
    outer_train: pd.DataFrame,
    *,
    target_site: str,
    source_sites: Sequence[str],
) -> list[JackknifeTrainingSet]:
    source_sites = tuple(str(site_id) for site_id in source_sites)
    if len(source_sites) != 4 or len(set(source_sites)) != 4:
        raise ValueError("jackknife source site list must contain four distinct sites")
    observed = set(outer_train["site_id"].astype(str))
    if str(target_site) in observed:
        raise ValueError("outer target site leaked into source training rows")
    if observed != set(source_sites):
        raise ValueError(
            "outer training sites do not match the frozen four-source site list: "
            f"observed={sorted(observed)} expected={sorted(source_sites)}"
        )
    if outer_train.empty:
        raise ValueError("outer source training frame is empty")
    result: list[JackknifeTrainingSet] = []
    for deleted_source_site in source_sites:
        selected = outer_train.loc[
            ~outer_train["site_id"].astype(str).eq(deleted_source_site)
        ].copy()
        included = set(selected["site_id"].astype(str))
        expected = set(source_sites) - {deleted_source_site}
        if included != expected or len(included) != 3:
            raise ValueError("jackknife submodel does not contain exactly three source sites")
        if str(target_site) in included or deleted_source_site in included:
            raise RuntimeError("target or deleted source site leaked into jackknife frame")
        validate_complete_cycles(selected)
        result.append(
            JackknifeTrainingSet(
                deleted_source_site=deleted_source_site,
                frame=selected.reset_index(drop=True),
                target_site_training_rows=0,
                target_site_preprocessing_rows=0,
                deleted_source_training_rows=0,
                deleted_source_preprocessing_rows=0,
            )
        )
    return result


def read_frozen_penalties(
    summary_path: Path,
    *,
    expected_fold_id: str | None = None,
) -> dict[str, float]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"frozen GAM-VC summary is missing: {summary_path}")
    frame = pd.read_csv(summary_path)
    if "variant" in frame.columns:
        frame = frame.loc[frame["variant"].astype(str).eq("GAM-VC")].copy()
    if expected_fold_id is not None and "fold_id" in frame.columns:
        frame = frame.loc[frame["fold_id"].astype(str).eq(expected_fold_id)].copy()
    if len(frame) != 1:
        raise ValueError("frozen GAM-VC summary must contain exactly one row")
    required = {"lambda_main", "lambda_site", "lambda_interaction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"frozen GAM-VC summary is missing penalties: {missing}")
    values = {
        name: float(frame.iloc[0][name])
        for name in sorted(required)
    }
    if not np.isfinite(np.asarray(list(values.values()), dtype=np.float64)).all():
        raise ValueError("frozen penalty summary contains nonfinite values")
    if any(value < 0.0 for value in values.values()):
        raise ValueError("frozen penalty summary contains negative values")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _water_scale(frame: pd.DataFrame) -> float:
    rain_columns = [column for column in frame.columns if column.startswith("weather_precipitation_mm_day")]
    precipitation = frame.loc[:, rain_columns].to_numpy(dtype=np.float64).sum(axis=1)
    irrigation = frame["irrigation_mm"].to_numpy(dtype=np.float64)
    return max(float(np.std(precipitation + irrigation)), 1.0)


def _prediction_frame(frame: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    output = frame.reset_index(drop=True).copy()
    values = predicted.reset_index(drop=True)
    for column in values.columns:
        output[column] = values[column]
    return output


def true_fixed_grid_peak(cycle: pd.DataFrame) -> tuple[float, float]:
    ordered = cycle.sort_values("irrigation_mm", kind="mergesort")
    values = ordered["target_net_gain_7d"].to_numpy(dtype=np.float64)
    if len(values) != 8 or not np.isfinite(values).all():
        raise ValueError("cycle does not contain eight finite target net gains")
    maximum = float(np.max(values))
    tied = np.isclose(values, maximum, rtol=1.0e-12, atol=1.0e-12)
    selected = ordered.loc[tied].sort_values("irrigation_mm", kind="mergesort").iloc[0]
    return float(selected["irrigation_mm"]), maximum


def evaluate_development_gate(
    cycle_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    prediction_metrics_frame: pd.DataFrame,
    *,
    execution_audit_passed: bool,
) -> dict[str, Any]:
    required_cycle = {
        "fold_id",
        "true_oracle_is_positive",
        "true_oracle_is_zero",
        "baseline_peak_absolute_error_mm",
        "robust_peak_absolute_error_mm",
        "baseline_zero_oracle_false_positive",
        "robust_zero_oracle_false_positive",
    }
    if len(cycle_metrics) != 196 or sorted(set(cycle_metrics["fold_id"])) != sorted(
        fold_metrics["fold_id"].astype(str).tolist()
    ):
        raise ValueError("development gate requires 196 cycles paired to 15 folds")
    missing = sorted(required_cycle - set(cycle_metrics.columns))
    if missing:
        raise ValueError(f"cycle metrics are missing gate columns: {missing}")
    if len(fold_metrics) != 15:
        raise ValueError("development gate requires exactly 15 fold rows")
    required_fold = {"fold_id", "baseline_peak_mae_mm", "robust_peak_mae_mm"}
    missing = sorted(required_fold - set(fold_metrics.columns))
    if missing:
        raise ValueError(f"fold metrics are missing gate columns: {missing}")
    if prediction_metrics_frame.empty:
        raise ValueError("prediction metrics are empty")
    required_roles = {"four_source_baseline", "jackknife_ensemble_mean"}
    if set(prediction_metrics_frame["model_role"].astype(str)) != required_roles:
        raise ValueError("prediction metrics do not contain the two frozen model roles")
    role_counts = prediction_metrics_frame.groupby("model_role")["fold_id"].nunique()
    if not role_counts.eq(15).all():
        raise ValueError("prediction metrics do not contain one row per role and fold")
    tolerance = 1.0e-12
    positive = cycle_metrics.loc[cycle_metrics["true_oracle_is_positive"].astype(bool)]
    baseline_folds = fold_metrics["baseline_peak_mae_mm"].to_numpy(dtype=np.float64)
    robust_folds = fold_metrics["robust_peak_mae_mm"].to_numpy(dtype=np.float64)
    baseline_predictions = prediction_metrics_frame.loc[
        prediction_metrics_frame["model_role"].eq("four_source_baseline"), "macro_composite"
    ]
    robust_predictions = prediction_metrics_frame.loc[
        prediction_metrics_frame["model_role"].eq("jackknife_ensemble_mean"), "macro_composite"
    ]
    conditions = {
        "overall_mean_peak_distance_nonworse": float(cycle_metrics["robust_peak_absolute_error_mm"].mean())
        <= float(cycle_metrics["baseline_peak_absolute_error_mm"].mean()) + tolerance,
        "positive_oracle_mean_peak_distance_nonworse": float(positive["robust_peak_absolute_error_mm"].mean())
        <= float(positive["baseline_peak_absolute_error_mm"].mean()) + tolerance
        if len(positive)
        else True,
        "zero_oracle_false_positive_count_nonworse": int(cycle_metrics["robust_zero_oracle_false_positive"].sum())
        <= int(cycle_metrics["baseline_zero_oracle_false_positive"].sum()),
        "at_least_10_of_15_fold_peak_distance_nonworse": int(
            np.sum(robust_folds <= baseline_folds + tolerance)
        )
        >= 10,
        "global_maximum_peak_distance_nonworse": float(cycle_metrics["robust_peak_absolute_error_mm"].max())
        <= float(cycle_metrics["baseline_peak_absolute_error_mm"].max()) + tolerance,
        "ensemble_mean_macro_composite_nonworse": float(robust_predictions.mean())
        <= float(baseline_predictions.mean()) + tolerance,
        "execution_and_audit_passed": bool(execution_audit_passed),
    }
    return {
        "gate_type": "predeclared_paired_engineering_development_qualification_not_significance_test",
        "formal_baseline": "four_source_gam_vc_continuous",
        "formal_candidate": "jackknife_robust_envelope_gam_vc_continuous",
        "conditions": {name: bool(value) for name, value in conditions.items()},
        "observed": {
            "cycle_count": int(len(cycle_metrics)),
            "fold_count": int(len(fold_metrics)),
            "baseline_overall_mean_peak_distance_mm": float(cycle_metrics["baseline_peak_absolute_error_mm"].mean()),
            "candidate_overall_mean_peak_distance_mm": float(cycle_metrics["robust_peak_absolute_error_mm"].mean()),
            "baseline_positive_oracle_mean_peak_distance_mm": float(positive["baseline_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
            "candidate_positive_oracle_mean_peak_distance_mm": float(positive["robust_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
            "baseline_zero_oracle_false_positive_count": int(cycle_metrics["baseline_zero_oracle_false_positive"].sum()),
            "candidate_zero_oracle_false_positive_count": int(cycle_metrics["robust_zero_oracle_false_positive"].sum()),
            "paired_folds_nonworse_peak_distance": int(np.sum(robust_folds <= baseline_folds + tolerance)),
            "baseline_global_maximum_peak_distance_mm": float(cycle_metrics["baseline_peak_absolute_error_mm"].max()),
            "candidate_global_maximum_peak_distance_mm": float(cycle_metrics["robust_peak_absolute_error_mm"].max()),
            "baseline_mean_macro_composite": float(baseline_predictions.mean()),
            "candidate_mean_macro_composite": float(robust_predictions.mean()),
        },
        "passed": bool(all(conditions.values())),
        "passing_action": "authorize_2015_2018_paired_continuous_swap_protocol_design_only",
        "failing_action": "stop_robust_envelope_route_before_swap_or_2019",
    }


def _robust_cycle(
    models: Sequence[HierarchicalGamBSpline],
    state: pd.DataFrame,
    *,
    deployment_resolution_mm: float,
    dense_step_mm: float,
    dense_gain_gap_tolerance: float,
) -> tuple[EnvelopeOptimizationResult, np.ndarray]:
    boundaries = np.unique(
        np.concatenate([model.irrigation_basis_.knots for model in models]).astype(np.float64)
    )
    objectives = tuple(gam_incremental_objective(model, state) for model in models)
    result = optimize_piecewise_cubic_lower_envelope(
        objectives=objectives,
        interval_boundaries=boundaries,
        deployment_resolution_mm=deployment_resolution_mm,
        dense_step_mm=dense_step_mm,
        dense_gain_gap_tolerance=dense_gain_gap_tolerance,
    )
    model_gains = np.asarray(
        [objective(np.asarray([result.irrigation_mm]))[0] for objective in objectives],
        dtype=np.float64,
    )
    if not np.isfinite(model_gains).all():
        raise ValueError("robust cycle model gains are nonfinite")
    return result, model_gains


def _fold_id(target_site: str, validation_year: int) -> str:
    return f"holdout_{target_site}_rolling_to_{validation_year}"


def _require_source_gate(
    source_development_dir: Path,
    source_config: dict[str, Any],
) -> tuple[Path, Path, Path]:
    gate_path = source_development_dir / "gam_development_gate_v1.json"
    audit_path = source_development_dir / "gam_development_audit_v1.json"
    fold_metrics_path = source_development_dir / source_config["fold_metrics_filename"]
    if not gate_path.is_file() or not audit_path.is_file() or not fold_metrics_path.is_file():
        raise FileNotFoundError("source nested GAM gate, audit, or fold metrics are missing")
    expected_hashes = {
        gate_path: source_config["gate_sha256"],
        audit_path: source_config["audit_sha256"],
        fold_metrics_path: source_config["fold_metrics_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen source development evidence hash changed: {path.name}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or audit.get("predeclared_performance_gate_passed") is not True:
        raise ValueError("frozen source nested GAM development gate did not pass")
    return gate_path, audit_path, fold_metrics_path


def _model_audit_row(
    *,
    fold_id: str,
    target_site: str,
    validation_year: int,
    role: str,
    deleted_source_site: str | None,
    model: HierarchicalGamBSpline,
    frame: pd.DataFrame,
    parameters: dict[str, float],
    fit_seconds: float,
) -> dict[str, Any]:
    included = sorted(set(frame["site_id"].astype(str)))
    if target_site in included or (deleted_source_site is not None and deleted_source_site in included):
        raise RuntimeError("model audit detected target or deleted source site")
    return {
        "fold_id": fold_id,
        "target_site": target_site,
        "validation_year": int(validation_year),
        "model_role": role,
        "deleted_source_site": deleted_source_site,
        "included_source_sites": ",".join(included),
        "training_rows": int(len(frame)),
        "training_cycles": int(frame.groupby(list(CYCLE_KEYS)).ngroups),
        "target_site_training_rows": 0,
        "target_site_preprocessing_rows": 0,
        "deleted_source_training_rows": 0,
        "deleted_source_preprocessing_rows": 0,
        "lambda_main": float(parameters["lambda_main"]),
        "lambda_site": float(parameters["lambda_site"]),
        "lambda_interaction": float(parameters["lambda_interaction"]),
        "parameter_count": int(model.coefficients_.size),
        "fit_seconds": float(fit_seconds),
    }


def _cycle_rows(
    validation: pd.DataFrame,
    baseline: HierarchicalGamBSpline,
    jackknife: Sequence[HierarchicalGamBSpline],
    *,
    deployment_resolution_mm: float,
    dense_step_mm: float,
    dense_gain_gap_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, float | int], dict[str, float | int]]:
    rows: list[dict[str, Any]] = []
    for keys, cycle in validation.groupby(list(CYCLE_KEYS), sort=False):
        cycle = cycle.sort_values("irrigation_mm", kind="mergesort").reset_index(drop=True)
        state = cycle.iloc[[0]].loc[:, ["site_id", "irrigation_mm", *STATE_COLUMNS]].copy()
        true_irrigation, true_gain = true_fixed_grid_peak(cycle)
        baseline_result = optimize_cycle(
            baseline,
            state,
            fixed_grid=np.asarray(EXPECTED_IRRIGATION_GRID, dtype=np.float64),
            deployment_resolution_mm=deployment_resolution_mm,
            dense_step_mm=dense_step_mm,
        )
        robust_result, model_gains = _robust_cycle(
            jackknife,
            state,
            deployment_resolution_mm=deployment_resolution_mm,
            dense_step_mm=dense_step_mm,
            dense_gain_gap_tolerance=dense_gain_gap_tolerance,
        )
        target_positive = true_irrigation > 0.0
        baseline_irrigation = float(baseline_result["continuous_irrigation_mm"])
        robust_irrigation = float(robust_result.irrigation_mm)
        rows.append(
            {
                "target_year": int(keys[0]),
                "site_id": str(keys[1]),
                "decision_date": str(keys[2]),
                "true_fixed_list_irrigation_mm": true_irrigation,
                "true_fixed_list_net_gain_7d": true_gain,
                "baseline_continuous_irrigation_mm": baseline_irrigation,
                "robust_continuous_irrigation_mm": robust_irrigation,
                "baseline_peak_absolute_error_mm": abs(baseline_irrigation - true_irrigation),
                "robust_peak_absolute_error_mm": abs(robust_irrigation - true_irrigation),
                "true_oracle_is_zero": int(not target_positive),
                "true_oracle_is_positive": int(target_positive),
                "baseline_zero_oracle_false_positive": int(not target_positive and baseline_irrigation > 0.0),
                "robust_zero_oracle_false_positive": int(not target_positive and robust_irrigation > 0.0),
                "baseline_global_predicted_gain_7d": float(baseline_result["predicted_net_gain_7d"]),
                "robust_min_incremental_gain_7d": float(robust_result.robust_incremental_gain),
                "robust_mean_incremental_gain_7d": float(robust_result.mean_incremental_gain),
                "robust_incremental_gain_std_7d": float(np.std(model_gains)),
                "robust_incremental_gain_positive_model_count": int(np.sum(model_gains > 0.0)),
                "robust_minimum_model_index": int(robust_result.minimum_model_index),
                "robust_candidate_count": int(robust_result.candidate_count),
                "robust_stationary_point_count": int(robust_result.stationary_point_count),
                "robust_pairwise_intersection_count": int(robust_result.pairwise_intersection_count),
                "robust_dense_diagnostic_irrigation_mm": float(robust_result.dense_diagnostic_irrigation_mm),
                "robust_dense_diagnostic_gain_gap": float(robust_result.dense_diagnostic_gain_gap),
                "baseline_continuous_is_boundary": int(baseline_result["raw_optimum_is_interval_boundary"]),
                "robust_continuous_is_boundary": int(
                    np.isclose(robust_result.raw_irrigation_mm, [0.0, 60.0], atol=1.0e-8, rtol=0.0).any()
                ),
            }
        )
    cycle_frame = pd.DataFrame(rows)
    if cycle_frame.empty:
        raise ValueError("outer validation has no cycles")
    positive = cycle_frame.loc[cycle_frame["true_oracle_is_positive"].eq(1)]
    baseline_summary = {
        "peak_mae_mm": float(cycle_frame["baseline_peak_absolute_error_mm"].mean()),
        "positive_peak_mae_mm": float(positive["baseline_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
        "zero_false_positive_count": int(cycle_frame["baseline_zero_oracle_false_positive"].sum()),
        "maximum_peak_error_mm": float(cycle_frame["baseline_peak_absolute_error_mm"].max()),
    }
    robust_summary = {
        "peak_mae_mm": float(cycle_frame["robust_peak_absolute_error_mm"].mean()),
        "positive_peak_mae_mm": float(positive["robust_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
        "zero_false_positive_count": int(cycle_frame["robust_zero_oracle_false_positive"].sum()),
        "maximum_peak_error_mm": float(cycle_frame["robust_peak_absolute_error_mm"].max()),
    }
    return cycle_frame, baseline_summary, robust_summary


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject any protocol that changes the frozen first-stage boundary."""
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "unexpected robust envelope protocol")
    data = protocol.get("data", {})
    _require(data.get("development_years") == [2015, 2016, 2017, 2018], "development years changed")
    _require(data.get("reused_context_year") == 2019, "2019 context year changed")
    _require(data.get("sealed_final_test_year") == 2024, "2024 sealed year changed")
    _require(tuple(data.get("sites", ())) == EXPECTED_SITES, "site order changed")
    _require(data.get("candidate_count_per_cycle") == 8, "candidate count changed")
    _require(len(str(data.get("dataset_sha256", ""))) == 64, "dataset hash missing")
    _require(len(str(data.get("contract_sha256", ""))) == 64, "contract hash missing")

    source = protocol.get("source_development", {})
    _require(source.get("protocol_id") == EXPECTED_SOURCE_PROTOCOL, "source protocol changed")
    _require(
        source.get("protocol_filename")
        == "2026-08-05-teacher-guided-hierarchical-gam-nested-shared-selection-development-v2.json",
        "source protocol filename changed",
    )
    _require(len(str(source.get("protocol_sha256", ""))) == 64, "source protocol hash missing")
    _require(len(str(source.get("gate_sha256", ""))) == 64, "source gate hash missing")
    _require(len(str(source.get("audit_sha256", ""))) == 64, "source audit hash missing")
    _require(source.get("fold_metrics_filename") == "gam_variant_fold_metrics_v1.csv", "source fold metric name changed")
    _require(len(str(source.get("fold_metrics_sha256", ""))) == 64, "source fold metric hash missing")
    _require(source.get("formal_variant") == "GAM-VC", "formal source variant changed")
    _require(source.get("must_have_passed_gate") is True, "source gate is not required")
    _require(source.get("reuse_selected_penalties_without_search") is True, "source penalties are not frozen")

    validation = protocol.get("validation", {})
    expected_year_folds = [
        {"train_years": [2015], "validation_year": 2016},
        {"train_years": [2015, 2016], "validation_year": 2017},
        {"train_years": [2015, 2016, 2017], "validation_year": 2018},
    ]
    _require(validation.get("rolling_year_folds") == expected_year_folds, "rolling folds changed")
    _require(validation.get("strict_outer_fold_count") == 15, "outer fold count changed")
    _require(validation.get("validation_cycle_count") == 196, "validation cycle count changed")
    _require(validation.get("site_cycle_counts") == EXPECTED_SITE_CYCLES, "site cycle counts changed")
    _require(validation.get("held_out_site_uses_shared_terms_only") is True, "held-out site uses deviations")

    ensemble = protocol.get("ensemble", {})
    _require(ensemble.get("jackknife_model_count_per_fold") == 4, "jackknife count changed")
    _require(ensemble.get("source_site_count_per_submodel") == 3, "jackknife source count changed")
    _require(
        ensemble.get("decision_target") == "minimum_incremental_net_gain_across_submodels",
        "decision target changed",
    )
    _require(ensemble.get("other_output_aggregation") == "arithmetic_mean", "output aggregation changed")

    optimization = protocol.get("continuous_optimization", {})
    _require(optimization.get("range_mm") == [0.0, 60.0], "irrigation range changed")
    _require(optimization.get("deployment_resolution_mm") == 0.000001, "deployment resolution changed")
    _require(optimization.get("dense_diagnostic_step_mm") == 0.25, "dense diagnostic step changed")
    _require(optimization.get("maximum_dense_gain_gap") == 0.05, "dense diagnostic tolerance changed")
    _require(
        optimization.get("candidate_sources")
        == ["interval_boundaries", "stationary_points", "pairwise_intersections"],
        "analytic candidate sources changed",
    )
    _require(optimization.get("dense_grid_used_for_recommendation") is False, "dense grid selects recommendations")
    _require(optimization.get("tie_break") == "smaller_irrigation", "tie break changed")

    penalties = protocol.get("penalties", {})
    _require(penalties.get("new_search_performed") is False, "new penalty search is forbidden")
    _require(penalties.get("reuse_outer_fold_selected_values") is True, "outer penalties are not frozen")

    forbidden = protocol.get("forbidden", {})
    for key, label in (
        ("2019_rows_read_for_training_or_selection", "2019 training/selection"),
        ("2019_features_read", "2019 features"),
        ("2019_targets_read", "2019 targets"),
        ("2024_rows_read", "2024 rows"),
        ("swap_rerun", "SWAP rerun"),
        ("moe_training", "MoE training"),
        ("tta", "TTA"),
        ("model_promotion", "model promotion"),
    ):
        _require(forbidden.get(key) is True, f"{label} is not frozen as forbidden")

    _require(
        protocol.get("passing_action")
        == "authorize_2015_2018_robust_envelope_development_swap_protocol_design_only",
        "passing action changed",
    )


def _completed_fold_outputs(fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    marker_path = fold_dir / "fold_complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "cycle_metrics": fold_dir / "cycle_metrics.csv",
        "fold_metrics": fold_dir / "fold_metrics.csv",
        "model_audit": fold_dir / "model_audit.csv",
        "prediction_metrics": fold_dir / "prediction_metrics.csv",
    }
    hashes = marker.get("files", {})
    for name, path in expected.items():
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            return None
    return tuple(pd.read_csv(path) for path in expected.values())  # type: ignore[return-value]


def _write_fold_outputs(
    fold_dir: Path,
    *,
    cycle_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    model_audit: pd.DataFrame,
    prediction_metrics_frame: pd.DataFrame,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cycle_metrics": fold_dir / "cycle_metrics.csv",
        "fold_metrics": fold_dir / "fold_metrics.csv",
        "model_audit": fold_dir / "model_audit.csv",
        "prediction_metrics": fold_dir / "prediction_metrics.csv",
    }
    frames = {
        "cycle_metrics": cycle_metrics,
        "fold_metrics": fold_metrics,
        "model_audit": model_audit,
        "prediction_metrics": prediction_metrics_frame,
    }
    for name, path in paths.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        frames[name].to_csv(temporary, index=False)
        temporary.replace(path)
    _atomic_write_json(
        fold_dir / "fold_complete.json",
        {
            "status": "robust_envelope_fold_complete",
            "files": {name: sha256_file(path) for name, path in paths.items()},
        },
    )


def _fit_fold(
    *,
    fold_id: str,
    target_site: str,
    validation_year: int,
    outer_train: pd.DataFrame,
    validation: pd.DataFrame,
    parameters: dict[str, float],
    source_protocol: dict[str, Any],
    fold_dir: Path,
    optimization: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_sites = tuple(site_id for site_id in EXPECTED_SITES if site_id != target_site)
    jackknife_sets = build_jackknife_training_sets(
        outer_train,
        target_site=target_site,
        source_sites=source_sites,
    )
    audit_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    baseline = _fit_model(outer_train, GAM_VC, source_protocol, parameters)
    baseline_seconds = time.perf_counter() - start
    baseline.save_coefficients(fold_dir / "models" / "four_source_baseline.npz")
    audit_rows.append(
        _model_audit_row(
            fold_id=fold_id,
            target_site=target_site,
            validation_year=validation_year,
            role="four_source_baseline",
            deleted_source_site=None,
            model=baseline,
            frame=outer_train,
            parameters=parameters,
            fit_seconds=baseline_seconds,
        )
    )
    jackknife_models: list[HierarchicalGamBSpline] = []
    for split in jackknife_sets:
        start = time.perf_counter()
        model = _fit_model(split.frame, GAM_VC, source_protocol, parameters)
        elapsed = time.perf_counter() - start
        model.save_coefficients(
            fold_dir / "models" / f"delete_{split.deleted_source_site}.npz"
        )
        jackknife_models.append(model)
        audit_rows.append(
            _model_audit_row(
                fold_id=fold_id,
                target_site=target_site,
                validation_year=validation_year,
                role="jackknife_submodel",
                deleted_source_site=split.deleted_source_site,
                model=model,
                frame=split.frame,
                parameters=parameters,
                fit_seconds=elapsed,
            )
        )

    cycle_metrics, baseline_summary, robust_summary = _cycle_rows(
        validation,
        baseline,
        jackknife_models,
        deployment_resolution_mm=float(optimization["deployment_resolution_mm"]),
        dense_step_mm=float(optimization["dense_diagnostic_step_mm"]),
        dense_gain_gap_tolerance=float(optimization["maximum_dense_gain_gap"]),
    )
    cycle_metrics.insert(0, "fold_id", fold_id)
    cycle_metrics.insert(1, "target_site", target_site)
    fold_metrics = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "target_site": target_site,
                "validation_year": validation_year,
                "cycle_count": int(len(cycle_metrics)),
                "baseline_peak_mae_mm": baseline_summary["peak_mae_mm"],
                "robust_peak_mae_mm": robust_summary["peak_mae_mm"],
                "baseline_positive_peak_mae_mm": baseline_summary["positive_peak_mae_mm"],
                "robust_positive_peak_mae_mm": robust_summary["positive_peak_mae_mm"],
                "baseline_zero_false_positive_count": baseline_summary["zero_false_positive_count"],
                "robust_zero_false_positive_count": robust_summary["zero_false_positive_count"],
                "baseline_maximum_peak_error_mm": baseline_summary["maximum_peak_error_mm"],
                "robust_maximum_peak_error_mm": robust_summary["maximum_peak_error_mm"],
            }
        ]
    )

    baseline_prediction = baseline.predict_shared(validation)
    jackknife_predictions = [model.predict_shared(validation) for model in jackknife_models]
    ensemble_prediction = sum(
        (prediction.to_numpy(dtype=np.float64) for prediction in jackknife_predictions),
        start=np.zeros_like(jackknife_predictions[0].to_numpy(dtype=np.float64)),
    ) / len(jackknife_predictions)
    ensemble_prediction = pd.DataFrame(
        ensemble_prediction,
        columns=jackknife_predictions[0].columns,
        index=validation.index,
    )
    scale = _water_scale(outer_train)
    baseline_metrics, _ = prediction_metrics(
        _prediction_frame(validation, baseline_prediction),
        target_scales=baseline.target_scales_,
        water_scale=scale,
    )
    ensemble_metrics, _ = prediction_metrics(
        _prediction_frame(validation, ensemble_prediction),
        target_scales=baseline.target_scales_,
        water_scale=scale,
    )
    prediction_rows = []
    for role, metrics in (
        ("four_source_baseline", baseline_metrics),
        ("jackknife_ensemble_mean", ensemble_metrics),
    ):
        prediction_rows.append(
            {
                "fold_id": fold_id,
                "target_site": target_site,
                "validation_year": validation_year,
                "model_role": role,
                **metrics,
            }
        )
    return cycle_metrics, fold_metrics, pd.DataFrame(audit_rows), pd.DataFrame(prediction_rows)


def _site_metrics(cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_id, part in cycles.groupby("site_id", sort=False):
        positive = part.loc[part["true_oracle_is_positive"].eq(1)]
        rows.append(
            {
                "site_id": site_id,
                "cycle_count": int(len(part)),
                "baseline_peak_mae_mm": float(part["baseline_peak_absolute_error_mm"].mean()),
                "robust_peak_mae_mm": float(part["robust_peak_absolute_error_mm"].mean()),
                "baseline_positive_peak_mae_mm": float(positive["baseline_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
                "robust_positive_peak_mae_mm": float(positive["robust_peak_absolute_error_mm"].mean()) if len(positive) else 0.0,
                "baseline_zero_false_positive_count": int(part["baseline_zero_oracle_false_positive"].sum()),
                "robust_zero_false_positive_count": int(part["robust_zero_oracle_false_positive"].sum()),
                "baseline_maximum_peak_error_mm": float(part["baseline_peak_absolute_error_mm"].max()),
                "robust_maximum_peak_error_mm": float(part["robust_peak_absolute_error_mm"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("site_id").reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = args.protocol.resolve()
    dataset_dir = args.dataset_dir.resolve()
    source_development_dir = args.source_development_dir.resolve()
    output_dir = args.output_dir.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    source_protocol_path = protocol_path.parent / protocol["source_development"]["protocol_filename"]
    if sha256_file(source_protocol_path) != protocol["source_development"]["protocol_sha256"]:
        raise ValueError("frozen source development protocol hash changed")
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    validate_source_protocol(source_protocol)
    source_gate_path, source_development_audit_path, source_fold_metrics_path = _require_source_gate(
        source_development_dir,
        protocol["source_development"],
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

    index = _dataset_index(dataset_path)
    years = pd.to_numeric(index["target_year"], errors="raise").astype(int)
    sites = index["site_id"].astype(str)
    all_cycles: list[pd.DataFrame] = []
    all_folds: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    completed = 0
    year_folds = [int(item["validation_year"]) for item in protocol["validation"]["rolling_year_folds"]]
    for target_site in EXPECTED_SITES:
        for validation_year in year_folds:
            fold_id = _fold_id(target_site, validation_year)
            fold_dir = output_dir / "folds" / fold_id
            resumed = _completed_fold_outputs(fold_dir) if args.resume else None
            if resumed is None:
                train_selector = years.lt(validation_year) & sites.ne(target_site)
                validation_selector = years.eq(validation_year) & sites.eq(target_site)
                outer_train = add_states(_read_selected_rows(dataset_path, index, train_selector))
                validation = add_states(_read_selected_rows(dataset_path, index, validation_selector))
                validate_complete_cycles(outer_train)
                validate_complete_cycles(validation)
                parameters = read_frozen_penalties(
                    source_fold_metrics_path,
                    expected_fold_id=fold_id,
                )
                resumed = _fit_fold(
                    fold_id=fold_id,
                    target_site=target_site,
                    validation_year=validation_year,
                    outer_train=outer_train,
                    validation=validation,
                    parameters=parameters,
                    source_protocol=source_protocol,
                    fold_dir=fold_dir,
                    optimization=protocol["continuous_optimization"],
                )
                _write_fold_outputs(
                    fold_dir,
                    cycle_metrics=resumed[0],
                    fold_metrics=resumed[1],
                    model_audit=resumed[2],
                    prediction_metrics_frame=resumed[3],
                )
                was_resumed = False
            else:
                was_resumed = True
            all_cycles.append(resumed[0])
            all_folds.append(resumed[1])
            all_audits.append(resumed[2])
            all_predictions.append(resumed[3])
            completed += 1
            print(
                f"fold={completed}/15 fold_id={fold_id} resumed={str(was_resumed).lower()}",
                flush=True,
            )

    cycle_metrics = pd.concat(all_cycles, ignore_index=True)
    fold_metrics = pd.concat(all_folds, ignore_index=True)
    model_audit = pd.concat(all_audits, ignore_index=True)
    prediction_metrics_frame = pd.concat(all_predictions, ignore_index=True)
    site_metrics = _site_metrics(cycle_metrics)
    numeric_columns = cycle_metrics.select_dtypes(include=[np.number]).columns
    numeric_finite = bool(np.isfinite(cycle_metrics.loc[:, numeric_columns].to_numpy(dtype=np.float64)).all())
    site_counts = cycle_metrics.groupby("site_id").size().astype(int).to_dict()
    isolation_passed = bool(
        model_audit["target_site_training_rows"].eq(0).all()
        and model_audit["target_site_preprocessing_rows"].eq(0).all()
        and model_audit["deleted_source_training_rows"].eq(0).all()
        and model_audit["deleted_source_preprocessing_rows"].eq(0).all()
    )
    execution_audit_passed = bool(
        len(cycle_metrics) == 196
        and len(fold_metrics) == 15
        and len(model_audit) == 75
        and model_audit["model_role"].eq("jackknife_submodel").sum() == 60
        and len(prediction_metrics_frame) == 30
        and site_counts == EXPECTED_SITE_CYCLES
        and isolation_passed
        and numeric_finite
        and cycle_metrics["robust_dense_diagnostic_gain_gap"].abs().le(
            float(protocol["continuous_optimization"]["maximum_dense_gain_gap"]) + 1.0e-12
        ).all()
    )
    gate = evaluate_development_gate(
        cycle_metrics,
        fold_metrics,
        prediction_metrics_frame,
        execution_audit_passed=execution_audit_passed,
    )

    outputs = {
        "cycle_metrics": output_dir / "gefs_robust_envelope_cycle_metrics_v1.csv",
        "fold_metrics": output_dir / "gefs_robust_envelope_fold_metrics_v1.csv",
        "site_metrics": output_dir / "gefs_robust_envelope_site_metrics_v1.csv",
        "model_audit": output_dir / "gefs_robust_envelope_model_audit_v1.csv",
        "prediction_metrics": output_dir / "gefs_robust_envelope_prediction_metrics_v1.csv",
        "gate": output_dir / "gefs_robust_envelope_development_gate_v1.json",
        "audit": output_dir / "gefs_robust_envelope_development_audit_v1.json",
        "manifest": output_dir / "gefs_robust_envelope_development_manifest_v1.json",
    }
    cycle_metrics.to_csv(outputs["cycle_metrics"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    site_metrics.to_csv(outputs["site_metrics"], index=False)
    model_audit.to_csv(outputs["model_audit"], index=False)
    prediction_metrics_frame.to_csv(outputs["prediction_metrics"], index=False)
    _atomic_write_json(outputs["gate"], gate)
    audit = {
        "status": "robust_envelope_gam_development_pregate_passed_pending_development_swap_protocol"
        if gate["passed"]
        else "robust_envelope_gam_development_failed_stop_before_swap_or_2019",
        "protocol_id": protocol["protocol_id"],
        "mandatory_execution_gate_passed": execution_audit_passed,
        "predeclared_performance_gate_passed": bool(gate["passed"]),
        "outer_fold_count": int(len(fold_metrics)),
        "validation_cycle_count": int(len(cycle_metrics)),
        "site_cycle_counts": site_counts,
        "baseline_model_count": int(model_audit["model_role"].eq("four_source_baseline").sum()),
        "jackknife_model_count": int(model_audit["model_role"].eq("jackknife_submodel").sum()),
        "all_target_and_deleted_source_training_rows_zero": isolation_passed,
        "new_penalty_search_count": 0,
        "2019_index_rows_read": int((years == 2019).sum()),
        "2019_rows_used_for_training_or_selection": 0,
        "2019_feature_rows_read": 0,
        "2019_target_rows_read": 0,
        "2024_rows_read": 0,
        "continuous_optimization_performed": True,
        "continuous_optimizer": "piecewise_cubic_lower_envelope_stationary_points_intersections_and_boundaries",
        "dense_grid_used_for_recommendation": False,
        "swap_rerun_performed": False,
        "moe_training_performed": False,
        "tta_performed": False,
        "model_promotion_performed": False,
        "2019_prior_results_used_only_as_hypothesis_source": True,
        "next_gate": gate["passing_action"] if gate["passed"] else gate["failing_action"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "source_protocol_sha256": sha256_file(source_protocol_path),
            "dataset_sha256": sha256_file(dataset_path),
            "contract_sha256": sha256_file(contract_path),
            "source_gate_sha256": sha256_file(source_gate_path),
            "source_development_audit_sha256": sha256_file(source_development_audit_path),
            "source_fold_metrics_sha256": sha256_file(source_fold_metrics_path),
        },
    }
    _atomic_write_json(outputs["audit"], audit)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    _atomic_write_json(outputs["manifest"], manifest)
    print(
        f"development_complete folds=15 cycles=196 gate_passed={str(gate['passed']).lower()} "
        f"next_gate={audit['next_gate']}",
        flush=True,
    )
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-development-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    outputs = run(parse_args())
    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()

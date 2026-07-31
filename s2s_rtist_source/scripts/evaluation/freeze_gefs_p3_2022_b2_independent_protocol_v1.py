#!/usr/bin/env python3
"""Freeze the P3 2022 B2 independent protocol before any target-year access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_B2_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_p3_b2_regret_sensitive_loso_router_screen_v1"
)
DEFAULT_DUAL_PROTOCOL_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_p3_2021_dual_gate_five_member_protocol_v1"
)
DEFAULT_FINAL_MODEL_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_p3_2021_final_full_history_refit_v1r2"
)
DEFAULT_OUTPUT_DIR = (
    Path("site_general_surrogate_eval")
    / "gefs_p3_2022_b2_independent_protocol_v1"
)

SUCCESS_STATUS = "p3_2022_b2_independent_protocol_frozen_before_target_data_access"
EXPECTED_B2_STATUS = (
    "b2_regret_sensitive_loso_development_passed_pending_new_independent_freeze"
)
EXPECTED_B2_FINAL_GATE_SHA256 = (
    "cfd509a41141f64abe7cb8d51a42a53d9e9a3b89893530377d30044e99052056"
)
EXPECTED_B2_MODEL_SHA256 = (
    "6d6c4bd4e3ab9b5a7038a4e13b066eced13880829d149be5d584bf76c7bb503f"
)
EXPECTED_B2_AUDIT_SHA256 = (
    "d88480395a83e5c12cc2d42ea02fe89f6c38cd1c02bf84055f4bd9b0a7f8a06c"
)
EXPECTED_B2_MANIFEST_SHA256 = (
    "3ddb602b8d26d67f8875097520ae7f9ff72b9e06dc6a4fa0acaa99d048d4998b"
)
EXPECTED_FINAL_POLICY_SHA256 = (
    "d9a2b2e85063c68e2812b7cef374b2c87c21efc3443c35e2ab99358f5825d726"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = {
    "P1": "61172cbdee69c1e43c66570baa209287a1ef73ed658c48fd144ca035d7fc683b",
    "P15": "b2ad3cd245d0dd88689f316a29037ceb1fd0b4d59ef915d2a903155dbec15363",
}

B2_FINAL_GATE_NAME = "gefs_p3_b2_regret_sensitive_final_gate_v1.json"
B2_AUDIT_NAME = "gefs_p3_b2_regret_sensitive_router_screen_audit_v1.json"
B2_MANIFEST_NAME = "gefs_p3_b2_regret_sensitive_router_screen_manifest_v1.csv"
FINAL_POLICY_NAME = "gefs_p3_2021_final_full_history_policy_v1r2.json"
DUAL_PROTOCOL_NAME = "gefs_p3_2021_dual_gate_protocol_v1.json"
DUAL_AUDIT_NAME = "gefs_p3_2021_dual_gate_five_member_protocol_audit_v1.json"
DUAL_MANIFEST_NAME = "gefs_p3_2021_dual_gate_five_member_protocol_manifest_v1.csv"

OUTPUT_NAMES = {
    "protocol": "gefs_p3_2022_b2_independent_protocol_v1.json",
    "components": "gefs_p3_2022_b2_component_registry_v1.csv",
    "stages": "gefs_p3_2022_b2_stage_registry_v1.csv",
    "audit": "gefs_p3_2022_b2_independent_protocol_audit_v1.json",
    "manifest": "gefs_p3_2022_b2_independent_protocol_manifest_v1.csv",
}

IRRIGATION_GRID = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
OPERATIONAL_MEMBERS = ("gec00", "gep01", "gep02", "gep03", "gep04")


def _weather_columns(variable: str) -> list[str]:
    return [f"weather_{variable}_day{day:02d}" for day in range(1, 8)]


GATE_FEATURE_DEFINITIONS = (
    ("predecision_dvs", "identity", ["predecision_dvs"]),
    (
        "predecision_crop_root_depth_cm",
        "identity",
        ["predecision_crop_root_depth_cm"],
    ),
    (
        "predecision_soil_vwc_0_100cm",
        "identity",
        ["predecision_soil_vwc_0_100cm"],
    ),
    (
        "weather_precipitation_mm_sum_7d",
        "sum",
        _weather_columns("precipitation_mm"),
    ),
    (
        "weather_temperature_min_c_mean_7d",
        "mean",
        _weather_columns("temperature_min_c"),
    ),
    (
        "weather_temperature_max_c_mean_7d",
        "mean",
        _weather_columns("temperature_max_c"),
    ),
    (
        "weather_actual_vapor_pressure_kpa_mean_7d",
        "mean",
        _weather_columns("actual_vapor_pressure_kpa"),
    ),
    (
        "weather_wind_speed_m_s_mean_7d",
        "mean",
        _weather_columns("wind_speed_m_s"),
    ),
    (
        "weather_solar_kj_m2_day_mean_7d",
        "mean",
        _weather_columns("solar_kj_m2_day"),
    ),
)
GATE_FEATURE_NAMES = tuple(item[0] for item in GATE_FEATURE_DEFINITIONS)

ZERO_ACCESS_FIELDS = (
    "P3_2022_ERA5_rows_read",
    "P3_2022_GEFS_rows_read",
    "P3_2022_model_inference_rows",
    "P3_2022_recommendation_rows",
    "P3_2022_SWAP_candidate_labels_read",
    "P3_2022_network_requests",
    "P3_2023_rows_or_labels_read",
    "P3_2024_rows_or_labels_read",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected!r}, received {actual!r}")


def require_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} changed: expected {expected.lower()}, received {actual.lower()}"
        )
    return actual.lower()


def model_payload_sha256(model: dict[str, Any]) -> str:
    hashable = {key: value for key, value in model.items() if key != "model_sha256"}
    payload = json.dumps(
        hashable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_gate_feature_contract() -> dict[str, Any]:
    return {
        "contract_id": "gefs-p3-strict-loso-gate-features-v1",
        "target_site_id": "P3",
        "candidate_representative": "zero_irrigation_row_after_invariance_check",
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "features": [
            {
                "gate_feature_name": name,
                "aggregation": aggregation,
                "source_columns": list(columns),
            }
            for name, aggregation, columns in GATE_FEATURE_DEFINITIONS
        ],
        "candidate_dependent_columns_forbidden": ["irrigation_mm"],
        "surrogate_predictions_used_as_gate_features": False,
        "swap_targets_used_as_gate_features": False,
        "future_observations_used_as_gate_features": False,
        "standardization_scope": "same_outer_fold_historical_P3_actionable_cycles_only",
    }


def schedule_contract() -> dict[str, Any]:
    return {
        "contract_id": "gefs-p3-2022-era5-dvs-schedule-rule-v1",
        "target_site": "P3",
        "target_year": 2022,
        "trunk_weather_source": "ERA5_2022_observed_daily_weather",
        "trunk_irrigation_policy": "zero_irrigation",
        "sowing_month_day": "04-26",
        "dvs_threshold": 0.1,
        "mature_dvs_excluded": True,
        "interval_days": 7,
        "horizon_days": 7,
        "state_checkpoint_offset_days": -1,
        "first_decision_rule": "first_day_after_checkpoint_DVS_reaches_0.1",
        "last_decision_rule": "retain_complete_D_through_D_plus_6_pre_maturity_cycles",
        "all_eligible_cycles_retained": True,
        "minimum_decision_cycles": 8,
        "manual_date_selection_allowed": False,
        "2021_dates_copied": False,
        "GEFS_values_used_to_select_dates": False,
        "SWAP_candidate_labels_used_to_select_dates": False,
        "decision_dates_resolved": False,
        "checkpoint_hashes_resolved": False,
    }


def weather_contract() -> dict[str, Any]:
    return {
        "contract_id": "gefs-p3-2022-frozen-operational-weather-v1",
        "archive": "NOAA_GEFS_operational_public_archive",
        "model_version": "GEFSv12",
        "cycle_hour_utc": 0,
        "product_grid": "atmos/pgrb2sp25_0p25_degree",
        "members": list(OPERATIONAL_MEMBERS),
        "member_count": len(OPERATIONAL_MEMBERS),
        "forecast_horizon": "local_decision_date_D_through_D_plus_6",
        "derived_daily_variables": [
            "precipitation_mm",
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
        ],
        "timezone_conversion": "America/Chicago_local_day",
        "precipitation_correction": {
            "method": "weekly_two_stage_linear_site_factor",
            "fit_period": "2000-2019",
            "factor_shrinkage_alpha": 0.75,
            "2022_reference_weather_used_for_fit": False,
        },
        "nonprecipitation_correction": {
            "method": "2015_2019_fitted_affine_and_solar_correction",
            "alphas": {
                "temperature_center": 1.0,
                "temperature_range": 0.0,
                "actual_vapor_pressure": 0.75,
                "wind_speed": 1.0,
                "solar_radiation_affine": 0.25,
            },
            "2022_reference_weather_used_for_fit": False,
        },
        "server_external_weather_download_allowed": False,
        "network_access_performed": False,
    }


def independent_promotion_conditions() -> list[dict[str, Any]]:
    conditions = [
        (
            "mean_regret_better_than_always_P1",
            "B2_mean_regret <= always_P1_mean_regret - 1e-6",
        ),
        (
            "mean_regret_better_than_always_P15",
            "B2_mean_regret <= always_P15_mean_regret - 1e-6",
        ),
        (
            "maximum_regret_no_worse_than_always_P1",
            "B2_maximum_regret <= always_P1_maximum_regret + 1e-6",
        ),
        (
            "maximum_regret_no_worse_than_always_P15",
            "B2_maximum_regret <= always_P15_maximum_regret + 1e-6",
        ),
        (
            "false_positive_count_no_worse_than_both_fixed_experts",
            "B2_false_positive_count <= min(always_P1,always_P15)",
        ),
        (
            "selected_60mm_count_no_worse_than_both_fixed_experts",
            "B2_selected_60mm_count <= min(always_P1,always_P15)",
        ),
        (
            "at_least_one_effective_route",
            "at_least_one_changed_irrigation_when_P1_and_P15_recommendations_differ",
        ),
        (
            "all_scheduled_cycles_have_eight_labels",
            "every_cycle_has_all_eight_successful_fixed_candidate_SWAP_labels",
        ),
        (
            "no_runtime_or_gate_fallback",
            "zero_endpoint_fallback_mismatch_nonfinite_or_gate_fallback",
        ),
        (
            "all_protocol_audits_pass",
            "zero_access_hash_recommendation_before_label_and_shared_label_audits_pass",
        ),
    ]
    return [
        {
            "condition_order": order,
            "condition_id": condition_id,
            "frozen_rule": rule,
            "required": True,
        }
        for order, (condition_id, rule) in enumerate(conditions, start=1)
    ]


def evaluation_contract() -> dict[str, Any]:
    return {
        "contract_id": "gefs-p3-2022-b2-single-independent-evaluation-v1",
        "target_site": "P3",
        "target_year": 2022,
        "candidate_policy": "B2_regret_sensitive_router",
        "fixed_baselines": ["always_P1", "always_P15"],
        "diagnostic_only_policy": "two_expert_label_oracle",
        "irrigation_candidates_mm": list(IRRIGATION_GRID),
        "formal_metrics_use_actual_fixed_candidate_SWAP_runs": True,
        "formal_metrics_use_interpolation": False,
        "continuous_optimization_allowed": False,
        "reported_metrics": [
            "cycle_count",
            "mean_regret_7d",
            "maximum_regret_7d",
            "false_positive_count",
            "missed_beneficial_count",
            "selected_60mm_count",
            "P1_route_count",
            "P15_route_count",
            "effective_route_count",
            "per_cycle_decisions",
        ],
        "independent_promotion_conditions": independent_promotion_conditions(),
        "failure_policy": "freeze_negative_without_criterion_changes",
        "passing_scope": "P3_2022_only_pending_separate_P3_2023_replication",
    }


def build_protocol_contract() -> dict[str, Any]:
    return {
        "protocol_id": "gefs-p3-2022-b2-independent-protocol-v1",
        "status": SUCCESS_STATUS,
        "frozen_on": "2026-07-31",
        "target": {
            "site_id": "P3",
            "site_code": "N3",
            "target_year": 2022,
            "latitude": 46.321,
            "longitude": -96.877,
            "timezone": "America/Chicago",
        },
        "sealed_reserves": {
            "P3_2023_replication": True,
            "P3_2024_TTA_workflow": True,
        },
        "candidate_policy": "B2_regret_sensitive_router_only",
        "excluded_policies": [
            "A_P3_historical_supervised_router",
            "B1_source_only_router",
        ],
        "routing_contract": {
            "source_experts": ["P1", "P15"],
            "source_fit_years": [2015, 2016, 2017, 2018, 2019],
            "source_checkpoint_track": "composite",
            "lambda_balance": 1.0,
            "feature_names": list(GATE_FEATURE_NAMES),
            "feature_contract": frozen_gate_feature_contract(),
            "threshold": 0.5,
            "tie_route": "P15",
            "audited_failure_fallback": "always_P15",
            "post_freeze_refit_reselection_or_search_allowed": False,
        },
        "schedule_rule": schedule_contract(),
        "operational_weather": weather_contract(),
        "evaluation": evaluation_contract(),
        "protocol_freeze_performs_target_data_access": False,
        "next_stage_requires_returned_artifact_verification": True,
    }


def build_stage_registry() -> pd.DataFrame:
    stages = [
        (1, "protocol_freeze", "historical contracts and frozen hashes only"),
        (2, "local_ERA5_2022_acquisition", "P3 2022 ERA5 daily weather"),
        (
            3,
            "server_ERA5_zero_irrigation_trunk_and_schedule",
            "zero-irrigation trunk and DVS-rule dates",
        ),
        (
            4,
            "local_operational_GEFSv12_five_member_acquisition",
            "frozen dates and five operational members",
        ),
        (5, "frozen_weather_correction", "pre-2022 fitted transformations"),
        (
            6,
            "B2_and_fixed_expert_recommendation_freeze",
            "recommendations without candidate labels",
        ),
        (
            7,
            "shared_fixed_eight_SWAP_generation",
            "one eight-candidate label table for all policies",
        ),
        (
            8,
            "single_independent_evaluation",
            "apply the ten frozen promotion conditions once",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "stage_order": order,
                "stage_id": stage_id,
                "frozen_input_scope": scope,
                "eligible_now": order == 1,
                "predecessor_hash_verification_required": order > 1,
                "model_or_threshold_selection_allowed": False,
                "P3_2023_access_allowed": False,
                "P3_2024_access_allowed": False,
            }
            for order, stage_id, scope in stages
        ]
    )


def _manifest_row(frame: pd.DataFrame, role: str) -> pd.Series:
    rows = frame.loc[frame["role"].astype(str).eq(role)]
    if len(rows) != 1:
        raise ValueError(f"B2 manifest requires exactly one {role!r} row")
    return rows.iloc[0]


def _validate_manifest_binding(
    frame: pd.DataFrame, role: str, expected_sha256: str
) -> None:
    row = _manifest_row(frame, role)
    require_equal(str(row["sha256"]).lower(), expected_sha256.lower(), f"{role} hash")


def validate_b2(b2_dir: Path) -> dict[str, Any]:
    gate_path = require_file(b2_dir / B2_FINAL_GATE_NAME, "B2 final gate")
    audit_path = require_file(b2_dir / B2_AUDIT_NAME, "B2 audit")
    manifest_path = require_file(b2_dir / B2_MANIFEST_NAME, "B2 manifest")
    gate_file_hash = require_sha256(
        gate_path, EXPECTED_B2_FINAL_GATE_SHA256, "B2 final gate SHA256"
    )
    audit_hash = require_sha256(audit_path, EXPECTED_B2_AUDIT_SHA256, "B2 audit SHA256")
    manifest_hash = require_sha256(
        manifest_path, EXPECTED_B2_MANIFEST_SHA256, "B2 manifest SHA256"
    )

    audit = read_json(audit_path)
    required_audit_values = {
        "status": EXPECTED_B2_STATUS,
        "mandatory_gate_passed": True,
        "development_gate_passed": True,
        "final_gate_written": True,
        "formal_promotion": False,
        "P3_rows_retained": 0,
        "P3_target_values_used": 0,
        "year_2020_rows_retained": 0,
        "year_2021_rows_retained": 0,
        "year_2024_rows_retained": 0,
        "source_checkpoint_retraining_performed": False,
        "source_checkpoint_reselection_performed": False,
        "gate_feature_search_performed": False,
        "gate_weight_search_performed": False,
        "gate_threshold_search_performed": False,
        "network_access_performed": False,
        "gate_threshold": 0.5,
        "gate_tie_route": "P15",
    }
    for key, expected in required_audit_values.items():
        require_equal(audit.get(key), expected, f"B2 audit {key}")
    require_equal(
        audit.get("scope_filter", {}).get("disallowed_rows_retained"),
        0,
        "B2 audit scope_filter.disallowed_rows_retained",
    )

    gate = read_json(gate_path)
    require_equal(gate.get("fallback"), False, "B2 final gate fallback")
    require_equal(gate.get("feature_names"), list(GATE_FEATURE_NAMES), "B2 feature_names")
    require_equal(gate.get("threshold"), 0.5, "B2 final gate threshold")
    require_equal(gate.get("tie_route"), "P15", "B2 final gate tie route")
    for key, expected in {
        "P3_rows_used": 0,
        "year_2021_rows_used": 0,
        "year_2024_rows_used": 0,
        "training_sites": ["P2", "P4"],
        "training_years": [2015, 2016, 2017, 2018, 2019],
        "policy_state": "pending_new_independent_freeze",
    }.items():
        require_equal(gate.get(key), expected, f"B2 final gate {key}")
    recorded_model_hash = str(gate.get("model_sha256", "")).lower()
    require_equal(recorded_model_hash, EXPECTED_B2_MODEL_SHA256.lower(), "B2 model SHA256")
    recomputed_model_hash = model_payload_sha256(gate)
    if recomputed_model_hash != recorded_model_hash:
        raise ValueError(
            "B2 model payload SHA256 mismatch: "
            f"recorded {recorded_model_hash}, recomputed {recomputed_model_hash}"
        )

    manifest = pd.read_csv(manifest_path, dtype={"sha256": str})
    required_columns = {"role", "path", "bytes", "sha256"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError("B2 manifest columns changed")
    _validate_manifest_binding(manifest, "output_final_gate", gate_file_hash)
    _validate_manifest_binding(manifest, "output_audit", audit_hash)
    _validate_manifest_binding(
        manifest, "input_final_full_history_policy", EXPECTED_FINAL_POLICY_SHA256
    )
    for source, expected_hash in EXPECTED_SOURCE_CHECKPOINT_SHA256.items():
        _validate_manifest_binding(
            manifest, f"input_final_source_checkpoint_{source}", expected_hash
        )
    return {
        "gate": gate,
        "audit": audit,
        "manifest": manifest,
        "paths": {
            "gate": gate_path,
            "audit": audit_path,
            "manifest": manifest_path,
        },
        "hashes": {
            "gate_file": gate_file_hash,
            "gate_model": recomputed_model_hash,
            "audit": audit_hash,
            "manifest": manifest_hash,
        },
    }


def _resolve_checkpoint(final_model_dir: Path, relative: str) -> Path:
    candidate = (final_model_dir / relative).resolve()
    root = final_model_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source checkpoint escapes final model directory: {relative}") from exc
    return candidate


def validate_final_models(final_model_dir: Path) -> dict[str, Any]:
    policy_path = require_file(final_model_dir / FINAL_POLICY_NAME, "final source policy")
    policy_hash = require_sha256(
        policy_path, EXPECTED_FINAL_POLICY_SHA256, "final source policy SHA256"
    )
    policy = read_json(policy_path)
    require_equal(
        policy.get("policy_id"),
        "gefs-p3-2021-final-full-history-two-expert-gate-v1r2",
        "final source policy id",
    )
    require_equal(policy.get("fit_years"), [2015, 2016, 2017, 2018, 2019], "fit years")
    require_equal(policy.get("2021_rows_used"), 0, "final policy 2021_rows_used")
    require_equal(
        policy.get("post_freeze_reselection_allowed"),
        False,
        "final policy post-freeze reselection",
    )
    experts = policy.get("source_experts", {})
    require_equal(set(experts), {"P1", "P15"}, "final policy source expert set")
    checkpoint_paths: dict[str, Path] = {}
    for source in ("P1", "P15"):
        record = experts[source]
        expected_hash = EXPECTED_SOURCE_CHECKPOINT_SHA256[source].lower()
        require_equal(
            str(record.get("checkpoint_sha256", "")).lower(),
            expected_hash,
            f"final policy {source} checkpoint hash",
        )
        require_equal(record.get("full_history_refit"), True, f"{source} full history refit")
        require_equal(record.get("checkpoint_track"), "composite", f"{source} track")
        require_equal(record.get("lambda_balance"), 1.0, f"{source} lambda_balance")
        checkpoint = _resolve_checkpoint(final_model_dir, str(record["checkpoint_path"]))
        require_file(checkpoint, f"{source} source checkpoint")
        require_sha256(checkpoint, expected_hash, f"{source} checkpoint SHA256")
        checkpoint_paths[source] = checkpoint
    return {
        "policy": policy,
        "policy_path": policy_path,
        "policy_sha256": policy_hash,
        "checkpoint_paths": checkpoint_paths,
    }


def validate_dual_protocol(dual_protocol_dir: Path, b2_manifest: pd.DataFrame) -> dict[str, Any]:
    protocol_path = require_file(
        dual_protocol_dir / DUAL_PROTOCOL_NAME, "dual-gate protocol"
    )
    audit_path = require_file(dual_protocol_dir / DUAL_AUDIT_NAME, "dual-gate audit")
    manifest_path = require_file(
        dual_protocol_dir / DUAL_MANIFEST_NAME, "dual-gate manifest"
    )
    dual_manifest_hash = sha256_file(manifest_path)
    _validate_manifest_binding(
        b2_manifest, "input_dual_protocol_manifest", dual_manifest_hash
    )

    protocol = read_json(protocol_path)
    require_equal(
        protocol.get("formal_primary_experiment"),
        "B_source_only_router",
        "dual protocol formal primary experiment",
    )
    candidate = protocol.get("B_source_only_router", {})
    require_equal(candidate.get("gate_fit_sites"), ["P2", "P4"], "B gate fit sites")
    require_equal(
        candidate.get("gate_development_years"),
        [2015, 2016, 2017, 2018, 2019],
        "B gate development years",
    )
    for key in (
        "P3_rows_allowed_in_gate_fit",
        "P3_rows_allowed_in_gate_preprocessing",
        "P3_rows_allowed_in_gate_label_construction",
    ):
        require_equal(candidate.get(key), 0, f"dual protocol {key}")
    require_equal(
        candidate.get("gate_hyperparameter_search_allowed"),
        False,
        "dual protocol gate hyperparameter search",
    )
    require_equal(candidate.get("gate_threshold"), 0.5, "dual protocol threshold")
    require_equal(candidate.get("gate_tie_route"), "P15", "dual protocol tie route")
    require_equal(
        candidate.get("irrigation_candidates_mm"),
        list(IRRIGATION_GRID),
        "dual protocol irrigation grid",
    )
    require_equal(
        candidate.get("gate_feature_contract"),
        frozen_gate_feature_contract(),
        "dual protocol B feature contract",
    )

    audit = read_json(audit_path)
    for key, expected in {
        "mandatory_gate_passed": True,
        "experiment_B_P3_rows_allowed": 0,
        "2021_GEFS_rows_read": 0,
        "2021_SWAP_candidate_labels_read": 0,
        "2021_model_inference_performed": False,
        "2021_results_read": False,
        "network_access_performed": False,
    }.items():
        require_equal(audit.get(key), expected, f"dual protocol audit {key}")

    manifest = pd.read_csv(manifest_path, dtype={"sha256": str})
    row = _manifest_row(manifest, "output_dual_gate")
    require_equal(
        str(row["sha256"]).lower(),
        sha256_file(protocol_path),
        "dual protocol manifest feature contract binding",
    )
    return {
        "protocol": protocol,
        "paths": {
            "protocol": protocol_path,
            "audit": audit_path,
            "manifest": manifest_path,
        },
        "hashes": {
            "protocol": sha256_file(protocol_path),
            "audit": sha256_file(audit_path),
            "manifest": dual_manifest_hash,
        },
    }


def build_component_registry(
    b2: dict[str, Any], final_models: dict[str, Any], dual: dict[str, Any]
) -> pd.DataFrame:
    rows = [
        {
            "component_order": 1,
            "component_role": "B2_router_gate_file",
            "component_id": "B2_regret_sensitive_router",
            "canonical_path": B2_FINAL_GATE_NAME,
            "sha256": b2["hashes"]["gate_file"],
            "fit_sites": "P2,P4",
            "fit_years": "2015,2016,2017,2018,2019",
            "target_rows_used": 0,
        },
        {
            "component_order": 2,
            "component_role": "B2_router_model_payload",
            "component_id": "B2_regret_sensitive_router_payload",
            "canonical_path": "embedded_in_B2_router_gate_file",
            "sha256": b2["hashes"]["gate_model"],
            "fit_sites": "P2,P4",
            "fit_years": "2015,2016,2017,2018,2019",
            "target_rows_used": 0,
        },
        {
            "component_order": 3,
            "component_role": "final_source_policy",
            "component_id": "P1_P15_full_history_policy_v1r2",
            "canonical_path": FINAL_POLICY_NAME,
            "sha256": final_models["policy_sha256"],
            "fit_sites": "P1,P15",
            "fit_years": "2015,2016,2017,2018,2019",
            "target_rows_used": 0,
        },
    ]
    for order, source in enumerate(("P1", "P15"), start=4):
        rows.append(
            {
                "component_order": order,
                "component_role": "source_expert_checkpoint",
                "component_id": f"{source}_full_history_composite",
                "canonical_path": (
                    "source_experts/final_2015_2019_full_history_refit/"
                    f"{source}/full_history_composite.pt"
                ),
                "sha256": EXPECTED_SOURCE_CHECKPOINT_SHA256[source],
                "fit_sites": source,
                "fit_years": "2015,2016,2017,2018,2019",
                "target_rows_used": 0,
            }
        )
    rows.append(
        {
            "component_order": 6,
            "component_role": "frozen_feature_contract",
            "component_id": "B_source_only_router_dual_protocol_contract",
            "canonical_path": DUAL_PROTOCOL_NAME,
            "sha256": dual["hashes"]["protocol"],
            "fit_sites": "P2,P4",
            "fit_years": "2015,2016,2017,2018,2019",
            "target_rows_used": 0,
        }
    )
    return pd.DataFrame(rows)


def _manifest_record(role: str, path: Path, display_path: str | None = None) -> dict[str, Any]:
    return {
        "role": role,
        "path": display_path if display_path is not None else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    b2 = validate_b2(Path(args.b2_dir))
    final_models = validate_final_models(Path(args.final_model_dir))
    dual = validate_dual_protocol(Path(args.dual_protocol_dir), b2["manifest"])

    output_dir.mkdir(parents=True)
    outputs = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    protocol = build_protocol_contract()
    components = build_component_registry(b2, final_models, dual)
    stages = build_stage_registry()
    write_json(outputs["protocol"], protocol)
    components.to_csv(outputs["components"], index=False)
    stages.to_csv(outputs["stages"], index=False)

    audit = {
        "status": SUCCESS_STATUS,
        "mandatory_gate_passed": True,
        "target_site": "P3",
        "target_year": 2022,
        "B2_development_status_verified": True,
        "B2_final_gate_file_sha256_verified": True,
        "B2_model_payload_sha256_recomputed_and_verified": True,
        "B2_manifest_bindings_verified": True,
        "final_source_policy_and_checkpoint_bytes_verified": True,
        "dual_protocol_feature_names_and_definitions_verified": True,
        "source_checkpoint_retraining_performed": False,
        "source_checkpoint_reselection_performed": False,
        "gate_feature_search_performed": False,
        "gate_weight_search_performed": False,
        "gate_threshold_search_performed": False,
        "decision_dates_resolved": False,
        "stage_1_only_eligible": True,
        "next_gate": "verify_returned_protocol_artifacts_before_local_ERA5_2022_acquisition",
        **{field: 0 for field in ZERO_ACCESS_FIELDS},
    }
    write_json(outputs["audit"], audit)

    input_paths = [
        ("input_B2_final_gate", b2["paths"]["gate"]),
        ("input_B2_audit", b2["paths"]["audit"]),
        ("input_B2_manifest", b2["paths"]["manifest"]),
        ("input_final_source_policy", final_models["policy_path"]),
        *[
            (f"input_final_source_checkpoint_{source}", path)
            for source, path in final_models["checkpoint_paths"].items()
        ],
        ("input_dual_protocol", dual["paths"]["protocol"]),
        ("input_dual_protocol_audit", dual["paths"]["audit"]),
        ("input_dual_protocol_manifest", dual["paths"]["manifest"]),
    ]
    manifest_rows = [_manifest_record(role, path) for role, path in input_paths]
    code_path = Path(__file__).resolve()
    try:
        code_display = code_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        code_display = code_path.name
    manifest_rows.append(_manifest_record("code", code_path, code_display))
    for name in ("protocol", "components", "stages", "audit"):
        path = outputs[name]
        manifest_rows.append(_manifest_record(f"output_{name}", path, path.name))
    pd.DataFrame(manifest_rows).to_csv(outputs["manifest"], index=False)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b2-dir", type=Path, default=DEFAULT_B2_DIR)
    parser.add_argument(
        "--dual-protocol-dir", type=Path, default=DEFAULT_DUAL_PROTOCOL_DIR
    )
    parser.add_argument("--final-model-dir", type=Path, default=DEFAULT_FINAL_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)

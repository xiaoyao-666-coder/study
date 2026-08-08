#!/usr/bin/env python3
"""Freeze target-blind 2019 recommendations for the passed GAM v6 rule."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import (
    CONTRACT_NAME,
    CYCLE_KEYS,
    DATASET_NAME,
    SOURCE_AUDIT_NAME,
    derive_state_frame,
)
from scripts.diagnostics.build_gefs_gam_expert_gate_feature_contract_v1 import (
    MODEL_FEATURE_COLUMNS,
    build_static_features,
)
from scripts.evaluation.freeze_gefs_gam_expert_cross_fitted_gate_protocol_v1 import (
    sha256_file,
)
from scripts.training.run_gefs_gam_action_aware_support_gate_development_v3 import (
    ACTION_FEATURE_COLUMNS,
    build_action_pairwise_frame,
    select_supported_expert,
    support_scores,
)
from scripts.training.run_gefs_gam_expert_cross_fitted_gate_development_v1 import (
    CANONICAL_EXPERTS,
    attach_peak_errors,
    expert_recommendations,
    select_gate_decisions,
)
from scripts.training.run_gefs_gam_expert_stability_gate_development_v2 import (
    _anchor_expert,
)
from scripts.training.run_gefs_gam_positive_peak_protected_support_gate_development_v4 import (
    cross_fit_positive_models,
)
from scripts.training.run_gefs_hierarchical_gam_bspline_development_v1 import (
    _dataset_index,
    _read_selected_rows,
    add_states,
    validate_complete_cycles,
)
from scripts.training.run_gefs_robust_envelope_gam_development_v1 import (
    _fit_model,
    build_jackknife_training_sets,
    read_frozen_penalties,
)
from s2s_rtist.models.gefs_hierarchical_gam_bspline_v1 import FORMAL_TARGETS, GAM_VC


PROTOCOL_ID = "teacher-guided-gam-v6-2019-independent-confirmation-v1"
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
FIT_YEARS = (2015, 2016, 2017, 2018)
CONFIRMATION_YEAR = 2019
EXPECTED_CYCLE_COUNT = 69
EXPECTED_CANDIDATE_ROW_COUNT = 552
EXPECTED_SITE_CYCLES = {"P1": 13, "P2": 13, "P3": 15, "P4": 14, "P15": 14}
TRUST_REGION_RADIUS_MM = 2.5
LIMITED_LCB_THRESHOLD_MM = 5.0
TOLERANCE = 1.0e-6
V6_GATE_NAME = "gefs_gam_v6_confidence_protected_anchor_replay_gate_v1.json"
V6_AUDIT_NAME = "gefs_gam_v6_confidence_protected_anchor_replay_audit_v1.json"
V6_MANIFEST_NAME = "gefs_gam_v6_confidence_protected_anchor_replay_manifest_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected 2019 confirmation protocol id")
    data = protocol.get("data_boundary", {})
    expert = protocol.get("expert_refit", {})
    gate = protocol.get("gate_model", {})
    rule = protocol.get("selection_rule", {})
    stage = protocol.get("stage_1_recommendation_freeze", {})
    outcomes = protocol.get("outcomes", {})
    if data.get("fit_years") != list(FIT_YEARS) or int(data.get("confirmation_year", -1)) != CONFIRMATION_YEAR:
        raise ValueError("fit or confirmation years changed")
    if tuple(data.get("sites", ())) != EXPECTED_SITES:
        raise ValueError("formal site list changed")
    if int(data.get("expected_confirmation_candidate_rows", -1)) != EXPECTED_CANDIDATE_ROW_COUNT or int(data.get("expected_confirmation_cycles", -1)) != EXPECTED_CYCLE_COUNT:
        raise ValueError("2019 row or cycle count changed")
    if data.get("expected_confirmation_site_cycles") != EXPECTED_SITE_CYCLES:
        raise ValueError("2019 site-cycle contract changed")
    if bool(data.get("2019_target_read_before_recommendation_freeze", True)):
        raise ValueError("2019 target access was enabled before freeze")
    if bool(data.get("2024_access", True)) or bool(outcomes.get("2024_access", True)):
        raise ValueError("2024 access was enabled")
    if expert.get("family") != "unchanged_hierarchical_gam_vc_bspline" or expert.get("penalty_source_fold_suffix") != "rolling_to_2018":
        raise ValueError("expert family or penalty inheritance changed")
    if not bool(expert.get("target_site_excluded", False)) or bool(expert.get("penalty_search", True)):
        raise ValueError("expert isolation or search boundary changed")
    expected_gate = {
        "environment_feature_count": 19,
        "action_feature_count": 13,
        "l2_penalty": 1.0,
        "positive_oracle_source_row_weight": 2.0,
        "absolute_residual_quantile": 0.9,
        "support_score_quantile": 0.95,
        "minimum_action_difference_mm": 1.0,
    }
    for name, expected in expected_gate.items():
        if float(gate.get(name, np.nan)) != float(expected):
            raise ValueError(f"frozen gate constant changed: {name}")
    if bool(gate.get("site_id_as_model_feature", True)) or any(bool(gate.get(name, True)) for name in ("hyperparameter_search", "feature_search", "threshold_search")):
        raise ValueError("gate feature or search boundary changed")
    if float(rule.get("v5_trust_region_radius_mm", np.nan)) != TRUST_REGION_RADIUS_MM:
        raise ValueError("v5 trust region changed")
    if float(rule.get("limited_candidate_required_lcb_mm", np.nan)) != LIMITED_LCB_THRESHOLD_MM:
        raise ValueError("v6 LCB threshold changed")
    if any(bool(rule.get(name, True)) for name in ("threshold_search", "site_id_as_selection_feature", "year_as_selection_feature", "target_or_oracle_features", "swap_gain_or_regret_features")):
        raise ValueError("v6 selection inputs or search boundary changed")
    if int(stage.get("expected_fold_count", -1)) != 5 or int(stage.get("expected_cycle_count", -1)) != EXPECTED_CYCLE_COUNT or stage.get("target_columns_loaded") != []:
        raise ValueError("Stage 1 contract changed")
    if bool(outcomes.get("automatic_model_promotion", True)) or bool(outcomes.get("final_refit_performed", True)) or bool(outcomes.get("tta_performed", True)):
        raise ValueError("a post-confirmation action was enabled")


def load_feature_rows_without_targets(
    dataset_path: Path,
    index: pd.DataFrame,
    selected: pd.Series,
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    selected_values = selected.to_numpy(dtype=bool)
    if selected_values.shape != (len(index),) or not selected_values.any():
        raise ValueError("physical feature selector is empty or misaligned")
    forbidden = sorted(
        column
        for column in feature_columns
        if column.startswith("target_") or column in set(FORMAL_TARGETS)
    )
    if forbidden:
        raise ValueError(f"target columns requested by target-blind loader: {forbidden}")
    base = ["sample_id", *CYCLE_KEYS, "irrigation_mm"]
    usecols = list(dict.fromkeys([*base, *feature_columns]))
    skipped_lines = set((index.index[~selected_values] + 1).tolist())
    frame = pd.read_csv(
        dataset_path,
        usecols=usecols,
        skiprows=lambda line_number: line_number in skipped_lines,
    )
    if len(frame) != int(selected_values.sum()):
        raise ValueError("physical feature loader returned the wrong number of rows")
    return frame


def inherit_rolling_2018_penalties(summary_path: Path, target_site: str) -> dict[str, float]:
    if target_site not in EXPECTED_SITES:
        raise ValueError(f"unexpected target site: {target_site}")
    return read_frozen_penalties(
        summary_path,
        expected_fold_id=f"holdout_{target_site}_rolling_to_2018",
    )


def apply_frozen_v5_v6_rule(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"anchor_recommendation_mm", "gate_recommendation_mm", "selected_lcb_mm"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"frozen v5/v6 decision fields are missing: {missing}")
    result = frame.copy()
    anchor = pd.to_numeric(result["anchor_recommendation_mm"], errors="raise").to_numpy(float)
    gate = pd.to_numeric(result["gate_recommendation_mm"], errors="raise").to_numpy(float)
    lcb = pd.to_numeric(result["selected_lcb_mm"], errors="raise").to_numpy(float)
    if not np.isfinite(np.column_stack([anchor, gate, lcb])).all():
        raise ValueError("frozen v5/v6 decision inputs are non-finite")
    delta = gate - anchor
    movement = np.clip(delta, -TRUST_REGION_RADIUS_MM, TRUST_REGION_RADIUS_MM)
    v5 = np.clip(anchor + movement, 0.0, 60.0)
    gate_changed = ~np.isclose(gate, anchor, atol=TOLERANCE, rtol=0.0)
    limited = np.abs(delta) > TRUST_REGION_RADIUS_MM + TOLERANCE
    use_v6 = gate_changed & (~limited | (lcb >= LIMITED_LCB_THRESHOLD_MM))
    selected = np.where(use_v6, v5, anchor)
    reasons = np.select(
        [
            ~gate_changed,
            gate_changed & ~limited,
            gate_changed & limited & (lcb >= LIMITED_LCB_THRESHOLD_MM),
        ],
        [
            "gate_unchanged_anchor_endpoint",
            "v5_unlimited_endpoint",
            "v5_limited_lcb_passed",
        ],
        default="v5_limited_lcb_failed_anchor_fallback",
    )
    result["gate_candidate_minus_anchor_mm"] = delta
    result["v5_movement_from_anchor_mm"] = movement
    result["v5_recommendation_mm"] = v5
    result["trust_region_limited"] = limited
    result["use_v6"] = use_v6
    result["selected_irrigation_mm"] = selected
    result["selection_reason"] = reasons
    if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("frozen v5/v6 outputs are non-finite")
    return result


def _cycle_features(frame: pd.DataFrame, static_features: pd.DataFrame) -> pd.DataFrame:
    states = derive_state_frame(frame)
    augmented = pd.concat(
        [frame.loc[:, list(CYCLE_KEYS)].reset_index(drop=True), states.reset_index(drop=True)],
        axis=1,
    )
    if not augmented.groupby(list(CYCLE_KEYS), sort=False)[list(states.columns)].nunique(dropna=False).le(1).all().all():
        raise ValueError("decision-time state changes within an irrigation cycle")
    cycles = augmented.groupby(list(CYCLE_KEYS), sort=False, as_index=False).first()
    static = static_features.rename(columns={"paper_site_id": "site_id"})
    cycles = cycles.merge(static, on="site_id", how="left", validate="many_to_one")
    if not np.isfinite(cycles.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(float)).all():
        raise ValueError("gate cycle features are non-finite")
    return cycles


def _verify_v6(v6_dir: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "v6_gate": v6_dir / V6_GATE_NAME,
        "v6_audit": v6_dir / V6_AUDIT_NAME,
        "v6_manifest": v6_dir / V6_MANIFEST_NAME,
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"required v6 artifacts are missing: {missing}")
    gate = read_json(paths["v6_gate"])
    audit = read_json(paths["v6_audit"])
    manifest = read_json(paths["v6_manifest"])
    upstream = protocol["upstream_v6"]
    if gate.get("passed") is not True or audit.get("status") != upstream["required_status"]:
        raise ValueError("required passed v6 result is absent")
    if audit.get("mandatory_execution_gate_passed") is not True or audit.get("predeclared_performance_gate_passed") is not True:
        raise ValueError("v6 execution or performance gate changed")
    if audit.get("inputs", {}).get("protocol_sha256") != upstream["required_protocol_sha256"]:
        raise ValueError("v6 protocol hash lineage changed")
    for name in ("gate", "audit"):
        path = paths[f"v6_{name}"]
        entry = manifest.get("outputs", {}).get(name, {})
        if entry.get("sha256") != sha256_file(path):
            raise ValueError(f"v6 manifest hash changed: {name}")
    return paths


def _fit_deployment_experts(
    *,
    target_site: str,
    source: pd.DataFrame,
    parameters: dict[str, float],
    source_protocol: dict[str, Any],
    fold_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_sites = tuple(site for site in EXPECTED_SITES if site != target_site)
    splits = build_jackknife_training_sets(
        source,
        target_site=target_site,
        source_sites=source_sites,
    )
    models: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []
    model_dir = fold_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=False)
    definitions = [("four_source_baseline", None, source)] + [
        (f"delete_{split.deleted_source_site}", split.deleted_source_site, split.frame)
        for split in splits
    ]
    for role, deleted_site, training in definitions:
        started = time.perf_counter()
        model = _fit_model(training, GAM_VC, source_protocol, parameters)
        model_path = model_dir / f"{role}.npz"
        model.save_coefficients(model_path)
        models[role] = model
        audit.append({
            "model_role": role,
            "deleted_source_site": deleted_site,
            "training_row_count": int(len(training)),
            "training_cycle_count": int(len(training[list(CYCLE_KEYS)].drop_duplicates())),
            "training_sites": sorted(training["site_id"].astype(str).unique()),
            "training_years": sorted(pd.to_numeric(training["target_year"]).astype(int).unique().tolist()),
            "target_site_training_rows": int(training["site_id"].astype(str).eq(target_site).sum()),
            "fit_seconds": float(time.perf_counter() - started),
            "coefficient_sha256": sha256_file(model_path),
            **parameters,
        })
    return models, audit


def _freeze_site_recommendations(
    *,
    target_site: str,
    source: pd.DataFrame,
    target_features: pd.DataFrame,
    source_features: pd.DataFrame,
    models: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate_protocol = protocol["gate_model"]
    deployment_resolution = 0.000001
    dense_step = 0.25
    source_decisions = expert_recommendations(
        source_features,
        models,
        deployment_resolution_mm=deployment_resolution,
        dense_step_mm=dense_step,
    )
    source_errors = attach_peak_errors(source_decisions, source)
    anchor = _anchor_expert(source_errors)
    pairwise_columns = [*MODEL_FEATURE_COLUMNS, *ACTION_FEATURE_COLUMNS]
    pairwise = build_action_pairwise_frame(
        source_decisions,
        source_features,
        anchor=anchor,
        environment_feature_columns=MODEL_FEATURE_COLUMNS,
    )
    anchor_error = source_errors.loc[
        source_errors["expert_id"].eq(anchor),
        [*CYCLE_KEYS, "absolute_peak_distance_mm", "true_peak_mm"],
    ].rename(columns={"absolute_peak_distance_mm": "anchor_peak_distance_mm"})
    advantage = source_errors.loc[
        ~source_errors["expert_id"].eq(anchor),
        [*CYCLE_KEYS, "expert_id", "absolute_peak_distance_mm"],
    ].merge(anchor_error, on=list(CYCLE_KEYS), validate="many_to_one")
    advantage["advantage_mm"] = advantage["anchor_peak_distance_mm"] - advantage["absolute_peak_distance_mm"]
    advantage["true_oracle_is_positive"] = advantage["true_peak_mm"] > TOLERANCE
    advantage = advantage[
        [*CYCLE_KEYS, "expert_id", "advantage_mm", "true_oracle_is_positive"]
    ].merge(pairwise, on=[*CYCLE_KEYS, "expert_id"], validate="one_to_one")
    candidates = [name for name in CANONICAL_EXPERTS if name in models and name != anchor]
    fitted = cross_fit_positive_models(
        advantage,
        feature_columns=pairwise_columns,
        experts=candidates,
        l2=float(gate_protocol["l2_penalty"]),
        residual_quantile=float(gate_protocol["absolute_residual_quantile"]),
        support_quantile=float(gate_protocol["support_score_quantile"]),
    )

    target_decisions = expert_recommendations(
        target_features,
        models,
        deployment_resolution_mm=deployment_resolution,
        dense_step_mm=dense_step,
    )
    target_pairwise = build_action_pairwise_frame(
        target_decisions,
        target_features,
        anchor=anchor,
        environment_feature_columns=MODEL_FEATURE_COLUMNS,
    )
    target_index = pd.MultiIndex.from_frame(target_features[list(CYCLE_KEYS)])
    predictions: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    actions: dict[str, np.ndarray] = {}
    for expert_id in candidates:
        rows = target_pairwise.loc[target_pairwise["expert_id"].eq(expert_id)].set_index(list(CYCLE_KEYS)).reindex(target_index)
        values = rows[pairwise_columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"2019 pairwise features are incomplete for {expert_id}")
        predictions[expert_id] = fitted["models"][expert_id].predict(values)
        supports[expert_id] = support_scores(fitted["models"][expert_id], values)
        actions[expert_id] = rows["absolute_candidate_minus_anchor_mm"].to_numpy(float)
    choices = []
    for row_index, item in enumerate(target_features.itertuples(index=False)):
        choice = select_supported_expert(
            anchor=anchor,
            candidates=candidates,
            predicted_advantage={name: float(values[row_index]) for name, values in predictions.items()},
            residual_bounds=fitted["residual_bounds_mm"],
            candidate_support_scores={name: float(values[row_index]) for name, values in supports.items()},
            support_bounds=fitted["support_bounds"],
            action_differences_mm={name: float(values[row_index]) for name, values in actions.items()},
            minimum_action_difference_mm=float(gate_protocol["minimum_action_difference_mm"]),
        )
        choices.append({
            **{key: getattr(item, key) for key in CYCLE_KEYS},
            "gate_selected_expert_id": choice.pop("selected_expert_id"),
            **choice,
        })
    choice_frame = pd.DataFrame(choices)
    selected = select_gate_decisions(
        target_decisions,
        target_features,
        choice_frame["gate_selected_expert_id"].to_numpy(object),
    ).merge(
        choice_frame.drop(columns=["gate_selected_expert_id"]),
        on=list(CYCLE_KEYS),
        validate="one_to_one",
    )
    anchor_rows = target_decisions.loc[
        target_decisions["expert_id"].eq(anchor),
        [*CYCLE_KEYS, "recommendation_mm"],
    ].rename(columns={"recommendation_mm": "anchor_recommendation_mm"})
    selected = selected.merge(anchor_rows, on=list(CYCLE_KEYS), validate="one_to_one")
    selected = selected.rename(columns={"recommendation_mm": "gate_recommendation_mm"})
    selected["anchor_expert_id"] = anchor
    selected["gate_selected_expert_id"] = selected["expert_id"]
    selected = apply_frozen_v5_v6_rule(selected)
    selected.insert(0, "fold_id", f"holdout_{target_site}_rolling_to_2019")
    selected.insert(1, "target_site", target_site)
    model_audit = {
        "target_site": target_site,
        "anchor_expert_id": anchor,
        "source_sites": list(fitted["source_sites"]),
        "source_cycle_count": int(len(source_features)),
        "target_cycle_count": int(len(target_features)),
        "residual_bounds_mm": fitted["residual_bounds_mm"],
        "support_bounds": fitted["support_bounds"],
        "advantage_models": {name: model.to_dict() for name, model in fitted["models"].items()},
    }
    return selected, model_audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    v6_paths = _verify_v6(args.v6_dir.resolve(), protocol)

    dataset_dir = args.dataset_dir.resolve()
    dataset_path = dataset_dir / DATASET_NAME
    contract_path = dataset_dir / CONTRACT_NAME
    dataset_audit_path = dataset_dir / SOURCE_AUDIT_NAME
    data_protocol = protocol["data_boundary"]
    if sha256_file(dataset_path) != data_protocol["dataset_sha256"] or sha256_file(contract_path) != data_protocol["dataset_contract_sha256"]:
        raise ValueError("formal surrogate dataset or contract hash changed")
    dataset_audit = read_json(dataset_audit_path)
    if dataset_audit.get("mandatory_gate_passed") is not True or dataset_audit.get("dataset_construction_gate_passed") is not True:
        raise ValueError("formal surrogate dataset audit did not pass")
    contract = read_json(contract_path)
    feature_columns = list(contract.get("continuous_feature_columns", ()))
    if len(feature_columns) != 51:
        raise ValueError("formal continuous feature contract changed")

    source_protocol_path = protocol_path.parent / protocol["expert_refit"]["source_protocol_filename"]
    source_metrics_path = args.source_development_dir.resolve() / protocol["expert_refit"]["source_fold_metrics_filename"]
    if sha256_file(source_protocol_path) != protocol["expert_refit"]["source_protocol_sha256"] or sha256_file(source_metrics_path) != protocol["expert_refit"]["source_fold_metrics_sha256"]:
        raise ValueError("frozen source protocol or penalty summary hash changed")
    source_protocol = read_json(source_protocol_path)
    static_path = args.static_features.resolve()
    if sha256_file(static_path) != protocol["expert_refit"]["static_features_sha256"]:
        raise ValueError("frozen static feature file hash changed")
    static = build_static_features(static_path)
    index = _dataset_index(dataset_path)
    years = pd.to_numeric(index["target_year"], errors="raise").astype(int)
    sites = index["site_id"].astype(str)
    target_selector = years.eq(CONFIRMATION_YEAR)
    if int(target_selector.sum()) != EXPECTED_CANDIDATE_ROW_COUNT:
        raise ValueError("2019 candidate row count changed")
    target_feature_rows = load_feature_rows_without_targets(
        dataset_path,
        index,
        target_selector,
        feature_columns=feature_columns,
    )
    if len(target_feature_rows[list(CYCLE_KEYS)].drop_duplicates()) != EXPECTED_CYCLE_COUNT:
        raise ValueError("2019 target-blind cycle count changed")

    output_dir.mkdir(parents=True)
    decisions, fit_audits = [], []
    for target_site in EXPECTED_SITES:
        print(f"site_start target_site={target_site}", flush=True)
        source_selector = years.isin(FIT_YEARS) & sites.ne(target_site)
        source = add_states(_read_selected_rows(dataset_path, index, source_selector))
        validate_complete_cycles(source)
        if target_site in set(source["site_id"].astype(str)):
            raise ValueError(f"target site leaked into source fit: {target_site}")
        site_target_rows = target_feature_rows.loc[target_feature_rows["site_id"].astype(str).eq(target_site)].copy()
        validate_complete_cycles(site_target_rows)
        source_features = _cycle_features(source, static)
        target_features = _cycle_features(site_target_rows, static)
        penalties = inherit_rolling_2018_penalties(source_metrics_path, target_site)
        fold_dir = output_dir / "folds" / f"holdout_{target_site}_rolling_to_2019"
        models, expert_audit = _fit_deployment_experts(
            target_site=target_site,
            source=source,
            parameters=penalties,
            source_protocol=source_protocol,
            fold_dir=fold_dir,
        )
        selected, gate_audit = _freeze_site_recommendations(
            target_site=target_site,
            source=source,
            target_features=target_features,
            source_features=source_features,
            models=models,
            protocol=protocol,
        )
        decisions.append(selected)
        fit_audits.append({
            "target_site": target_site,
            "fold_id": f"holdout_{target_site}_rolling_to_2019",
            "penalty_source_fold_id": f"holdout_{target_site}_rolling_to_2018",
            "penalties": penalties,
            "expert_models": expert_audit,
            "gate": gate_audit,
        })
        print(f"site_complete target_site={target_site} cycles={len(selected)} retained={int(selected['use_v6'].sum())}", flush=True)

    inventory = pd.concat(decisions, ignore_index=True).sort_values(["site_id", "decision_date"]).reset_index(drop=True)
    decision_numeric = inventory.select_dtypes(include=[np.number]).to_numpy(float)
    gate_conditions = {
        "all_69_confirmation_cycles_complete": len(inventory) == EXPECTED_CYCLE_COUNT and not inventory.duplicated(["site_id", "decision_date"]).any(),
        "all_5_target_sites_complete": inventory["target_site"].nunique() == 5,
        "all_site_cycle_counts_match_frozen_2019_schedule": inventory.groupby("target_site").size().astype(int).to_dict() == EXPECTED_SITE_CYCLES,
        "all_numeric_decision_outputs_finite": bool(np.isfinite(decision_numeric).all()),
        "all_source_site_holds_complete": len(fit_audits) == 5 and all(len(item["gate"]["source_sites"]) == 4 for item in fit_audits),
        "all_target_site_training_rows_zero": all(all(model["target_site_training_rows"] == 0 for model in item["expert_models"]) for item in fit_audits),
        "2019_target_rows_read_before_freeze_zero": True,
        "2024_rows_read_zero": True,
    }
    freeze_gate = {
        "protocol_id": PROTOCOL_ID,
        "gate_type": "mandatory_target_blind_2019_recommendation_freeze_gate",
        "conditions": gate_conditions,
        "observed": {
            "cycle_count": int(len(inventory)),
            "site_count": int(inventory["target_site"].nunique()),
            "retained_v6_decision_count": int(inventory["use_v6"].sum()),
            "retained_v6_site_count": int(inventory.loc[inventory["use_v6"], "target_site"].nunique()),
        },
        "passed": bool(all(gate_conditions.values())),
        "passing_action": protocol["stage_1_recommendation_freeze"]["passing_action"],
        "failing_action": protocol["stage_1_recommendation_freeze"]["failing_action"],
    }
    outputs = {
        "recommendations": output_dir / "gefs_gam_v6_2019_frozen_recommendations_v1.csv",
        "models": output_dir / "gefs_gam_v6_2019_recommendation_model_audit_v1.json",
        "gate": output_dir / "gefs_gam_v6_2019_recommendation_freeze_gate_v1.json",
        "audit": output_dir / "gefs_gam_v6_2019_recommendation_freeze_audit_v1.json",
        "manifest": output_dir / "gefs_gam_v6_2019_recommendation_freeze_manifest_v1.json",
    }
    inventory.to_csv(outputs["recommendations"], index=False)
    write_json(outputs["models"], fit_audits)
    write_json(outputs["gate"], freeze_gate)
    audit = {
        "status": "gam_v6_2019_recommendations_frozen_pending_exact_paired_swap_confirmation" if freeze_gate["passed"] else "gam_v6_2019_recommendation_freeze_failed_before_target_read",
        "protocol_id": PROTOCOL_ID,
        "mandatory_execution_gate_passed": bool(freeze_gate["passed"]),
        "recommendations_frozen_before_2019_target_read": True,
        "2019_feature_rows_read": EXPECTED_CANDIDATE_ROW_COUNT,
        "2019_target_rows_read": 0,
        "2019_swap_results_read": 0,
        "2024_rows_read": 0,
        "gam_bspline_family_changed": False,
        "penalty_search_performed": False,
        "gate_hyperparameter_search_performed": False,
        "v6_threshold_search_performed": False,
        "automatic_model_promotion": False,
        "next_gate": freeze_gate["passing_action"] if freeze_gate["passed"] else freeze_gate["failing_action"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_contract_sha256": sha256_file(contract_path),
            "dataset_audit_sha256": sha256_file(dataset_audit_path),
            "static_features_sha256": sha256_file(static_path),
            "source_protocol_sha256": sha256_file(source_protocol_path),
            "source_fold_metrics_sha256": sha256_file(source_metrics_path),
            **{name: sha256_file(path) for name, path in v6_paths.items()},
        },
    }
    write_json(outputs["audit"], audit)
    write_json(outputs["manifest"], {
        "status": audit["status"],
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in {
            "protocol": protocol_path,
            "dataset": dataset_path,
            "dataset_contract": contract_path,
            "dataset_audit": dataset_audit_path,
            "static_features": static_path,
            "source_protocol": source_protocol_path,
            "source_fold_metrics": source_metrics_path,
            **v6_paths,
        }.items()},
        "outputs": {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in outputs.items() if name != "manifest"},
    })
    if not freeze_gate["passed"]:
        raise RuntimeError(f"2019 recommendation freeze gate failed; see {outputs['gate']}")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v6-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-development-dir", type=Path, required=True)
    parser.add_argument("--static-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    for output_name, output_path in run(parse_args()).items():
        print(f"{output_name}: {output_path}", flush=True)

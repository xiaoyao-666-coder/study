#!/usr/bin/env python3
"""Qualify only v8 changed development cycles with exact paired SWAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluation.freeze_gefs_gam_expert_cross_fitted_gate_protocol_v1 import sha256_file
from scripts.simulation.run_gefs_controlled_continuous_swap_evaluation_v1 import (
    evaluate_cycle,
    run_swap_stage,
    unit_resources,
)
from scripts.simulation.run_gefs_gam_v5_changed_cycle_paired_swap_qualification_v1 import (
    prior_cycle_candidates,
    validate_prior_cycle_grid,
    zero_irrigation_cwdm_reference,
)


PROTOCOL_ID = "teacher-guided-gam-v8-changed-cycle-paired-swap-qualification-v1"
EXPECTED_CHANGED_CYCLES = 12
EXPECTED_PLAN_ROWS = 24
EXPECTED_FULL_CYCLES = 196
EXPECTED_CHANGED_SITE_CYCLES = {"P1": 3, "P2": 2, "P3": 1, "P4": 6}
EXPECTED_FULL_SITE_CYCLES = {"P1": 38, "P2": 38, "P3": 41, "P4": 39, "P15": 40}
EXPECTED_FOLDS = 15
EXPECTED_SITES = 5
TOLERANCE = 1.0e-6
ANCHOR_ROLE = "frozen_same_fold_best_single_anchor"
V8_ROLE = "frozen_v8_unlimited_preserving_confidence_shrinkage"
FIXED_IRRIGATION_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)

STAGE_A_DECISIONS_NAME = "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_decisions_v1.csv"
STAGE_A_GATE_NAME = "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_gate_v1.json"
STAGE_A_AUDIT_NAME = "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_audit_v1.json"
STAGE_A_MANIFEST_NAME = "gefs_gam_v8_unlimited_preserving_confidence_shrinkage_replay_manifest_v1.json"
PAIR_AUDIT_NAME = "gefs_gam_v5_changed_cycle_paired_swap_qualification_audit_v1.json"
PAIR_MANIFEST_NAME = "gefs_gam_v5_changed_cycle_paired_swap_qualification_manifest_v1.json"
FORMAL_WEATHER_NAME = "gefs_exact_schedule_2015_2019_formal_weather_v1.csv"
PAIR_DECISIONS_NAME = "gefs_gam_v5_changed_cycle_paired_swap_decisions_v1.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected v8 changed-cycle SWAP protocol id")
    contract = protocol.get("changed_cycle_contract", {})
    expected = {
        "expected_changed_cycle_count": EXPECTED_CHANGED_CYCLES,
        "expected_full_cycle_count": EXPECTED_FULL_CYCLES,
        "expected_unchanged_cycle_count": EXPECTED_FULL_CYCLES - EXPECTED_CHANGED_CYCLES,
        "expected_plan_row_count": EXPECTED_PLAN_ROWS,
        "expected_changed_site_count": len(EXPECTED_CHANGED_SITE_CYCLES),
        "expected_changed_site_cycle_counts": EXPECTED_CHANGED_SITE_CYCLES,
        "expected_full_site_cycle_counts": EXPECTED_FULL_SITE_CYCLES,
    }
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("changed-cycle contract changed")
    stage_a = protocol.get("source_stage_a", {})
    if stage_a.get("required_status") != "gam_v8_unlimited_preserving_confidence_shrinkage_replay_passed_pending_changed_cycle_exact_swap_design" or stage_a.get("required_gate_passed") is not True:
        raise ValueError("source Stage A boundary changed")
    source_v5 = protocol.get("source_v5_paired_swap", {})
    if source_v5.get("required_status") != "gam_v5_changed_cycle_paired_swap_qualification_failed_stop_before_2019" or source_v5.get("required_execution_gate_passed") is not True or source_v5.get("required_performance_gate_passed") is not False:
        raise ValueError("source v5 paired SWAP boundary changed")
    reuse = protocol.get("candidate_reuse", {})
    if float(reuse.get("exact_irrigation_tolerance_mm", -1.0)) != TOLERANCE:
        raise ValueError("candidate reuse tolerance changed")
    if int(reuse.get("expected_reused_recommendation_count", -1)) != 16:
        raise ValueError("candidate reuse count changed")
    if int(reuse.get("maximum_new_exact_swap_runs", -1)) != 8:
        raise ValueError("candidate reuse run boundary changed")
    if reuse.get("interpolation_allowed") is not False or reuse.get("nearest_neighbor_allowed") is not False:
        raise ValueError("candidate reuse matching policy changed")
    boundary = protocol.get("evidence_boundary", {})
    if boundary.get("development_years") != [2015, 2016, 2017, 2018]:
        raise ValueError("development year boundary changed")
    if any(boundary.get(k) is not False for k in (
        "2019_access", "2024_access", "automatic_model_promotion",
        "recommendation_changes_after_plan_freeze", "target_oracle_or_swap_outcome_selection",
    )):
        raise ValueError("sealed evidence boundary changed")
    gate = protocol.get("performance_gate", {})
    if int(gate.get("minimum_nonworse_outer_folds", -1)) != 10 or int(gate.get("minimum_nonworse_target_sites", -1)) != 4:
        raise ValueError("paired performance gate changed")
    if float(gate.get("tolerance", -1.0)) != TOLERANCE:
        raise ValueError("paired performance tolerance changed")
    mandatory_gate_fields = (
        "all_changed_pairs_finite", "all_full_cycle_differences_finite",
        "all_reused_and_new_execution_audits_passed",
        "changed_cycle_maximum_adaptive_regret_nonworse",
        "overall_fixed_eight_regret_nonworse", "pooled_mean_swap_gain_nonworse",
        "positive_oracle_mean_swap_gain_nonworse", "zero_oracle_mean_swap_gain_nonworse",
    )
    if any(gate.get(name) is not True for name in mandatory_gate_fields):
        raise ValueError("a mandatory paired performance gate was disabled")
    swap = protocol.get("swap", {})
    if (
        int(swap.get("restart_nprintday", -1)) != 48
        or float(swap.get("swap_dtmax_days", -1)) != 0.01
        or swap.get("fallback_offsets_mm") != [0.1, 0.2, 0.3]
        or float(swap.get("max_fallback_adjustment_mm", -1)) != 0.3
        or swap.get("fallback_exhaustion_fails_stage") is not True
    ):
        raise ValueError("SWAP fallback policy changed")


def build_changed_plan(inventory: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold_id", "target_site", "validation_year", "target_year", "site_id", "decision_date",
        "anchor_recommendation_mm", "v5_recommendation_mm", "v8_recommendation_mm", "recommendation_mm",
        "true_peak_mm", "true_oracle_is_positive", "irrigation_changed",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"v8 inventory fields are missing: {missing}")
    if len(inventory) != EXPECTED_FULL_CYCLES or inventory.duplicated(["site_id", "decision_date"]).any():
        raise ValueError("v8 full-cycle inventory changed")
    if inventory["fold_id"].nunique() != EXPECTED_FOLDS or inventory["site_id"].nunique() != EXPECTED_SITES:
        raise ValueError("v8 outer validation structure changed")
    if inventory.groupby("site_id").size().astype(int).to_dict() != EXPECTED_FULL_SITE_CYCLES:
        raise ValueError("v8 full site-cycle counts changed")
    changed = inventory.loc[inventory["irrigation_changed"].astype(bool)].copy()
    if len(changed) != EXPECTED_CHANGED_CYCLES or changed.groupby("site_id").size().astype(int).to_dict() != EXPECTED_CHANGED_SITE_CYCLES:
        raise ValueError("v8 changed-cycle contract changed")
    numeric = changed[["anchor_recommendation_mm", "v5_recommendation_mm", "v8_recommendation_mm", "recommendation_mm"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(numeric).all() or not ((numeric >= 0.0) & (numeric <= 60.0)).all():
        raise ValueError("v8 changed recommendations are invalid")
    if not np.allclose(numeric[:, 2], numeric[:, 3], rtol=0.0, atol=TOLERANCE):
        raise ValueError("v8 recommendation differs from frozen selected endpoint")
    anchor, v5, v8 = numeric[:, 0], numeric[:, 1], numeric[:, 2]
    movement = v5 - anchor
    if np.isclose(movement, 0.0, rtol=0.0, atol=TOLERANCE).any():
        raise ValueError("changed v8 cycle has no frozen v5 movement")
    if ((v8 - anchor) * movement < -TOLERANCE).any() or (np.abs(v8 - anchor) > np.abs(movement) + TOLERANCE).any():
        raise ValueError("v8 endpoint is outside the frozen anchor-v5 segment")
    rows: list[dict[str, Any]] = []
    for row in changed.itertuples(index=False):
        common = {
            "fold_id": str(row.fold_id), "target_site": str(row.target_site),
            "validation_year": int(row.validation_year), "target_year": int(row.target_year),
            "site_id": str(row.site_id), "decision_date": pd.Timestamp(row.decision_date).strftime("%Y-%m-%d"),
            "true_peak_mm": float(row.true_peak_mm), "true_oracle_is_positive": bool(row.true_oracle_is_positive),
        }
        rows.append({**common, "model_id": f"gam_v8_anchor_{row.site_id}", "model_role": ANCHOR_ROLE, "continuous_irrigation_mm": float(row.anchor_recommendation_mm)})
        rows.append({**common, "model_id": f"gam_v8_v8_{row.site_id}", "model_role": V8_ROLE, "continuous_irrigation_mm": float(row.v8_recommendation_mm)})
    plan = pd.DataFrame(rows).sort_values(["site_id", "decision_date", "model_role"]).reset_index(drop=True)
    if len(plan) != EXPECTED_PLAN_ROWS or plan.duplicated(["model_role", "site_id", "decision_date"]).any() or not plan.groupby(["site_id", "decision_date"])["model_role"].nunique().eq(2).all():
        raise ValueError("v8 paired changed-cycle plan is incomplete")
    return plan


def split_exact_reuse(prior_candidates: pd.DataFrame, requested_values_mm: list[float]) -> tuple[list[float], list[float]]:
    if "ir" not in prior_candidates:
        raise ValueError("prior candidates are missing irrigation values")
    observed = np.unique(pd.to_numeric(prior_candidates["ir"], errors="raise").to_numpy(float))
    if not np.isfinite(observed).all():
        raise ValueError("prior candidate irrigation values are non-finite")
    reusable, missing = [], []
    for requested in requested_values_mm:
        value = float(requested)
        if int(np.isclose(observed, value, rtol=0.0, atol=TOLERANCE).sum()) == 1:
            reusable.append(value)
        else:
            missing.append(value)
    return reusable, missing


def build_changed_comparison(decisions: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    required = {"model_role", "site_id", "decision_date", "continuous_irrigation_mm", "continuous_recommendation_swap_gain_7d", "regret_vs_fixed_eight_swap_7d", "regret_vs_adaptive_swap_oracle_7d"}
    missing = sorted(required - set(decisions.columns))
    if missing:
        raise ValueError(f"changed SWAP decisions are missing fields: {missing}")
    if len(decisions) != EXPECTED_PLAN_ROWS or set(decisions["model_role"].astype(str)) != {ANCHOR_ROLE, V8_ROLE}:
        raise ValueError("changed SWAP decisions do not contain both frozen roles")
    keys = ["site_id", "decision_date"]
    value_columns = ["continuous_irrigation_mm", "continuous_recommendation_swap_gain_7d", "regret_vs_fixed_eight_swap_7d", "regret_vs_adaptive_swap_oracle_7d"]
    def role_frame(role: str, prefix: str) -> pd.DataFrame:
        frame = decisions.loc[decisions["model_role"].eq(role), [*keys, *value_columns]].copy()
        if len(frame) != EXPECTED_CHANGED_CYCLES or frame.duplicated(keys).any():
            raise ValueError(f"changed SWAP role is incomplete: {role}")
        return frame.rename(columns={
            "continuous_irrigation_mm": f"{prefix}_irrigation_mm",
            "continuous_recommendation_swap_gain_7d": f"{prefix}_gain_7d",
            "regret_vs_fixed_eight_swap_7d": f"{prefix}_regret_fixed_7d",
            "regret_vs_adaptive_swap_oracle_7d": f"{prefix}_regret_adaptive_7d",
        })
    comparison = role_frame(ANCHOR_ROLE, "anchor").merge(role_frame(V8_ROLE, "v8"), on=keys, validate="one_to_one")
    info = plan.drop_duplicates(keys)[["fold_id", "target_site", "target_year", "site_id", "decision_date", "true_peak_mm", "true_oracle_is_positive"]]
    comparison = comparison.merge(info, on=keys, validate="one_to_one")
    frozen = plan.pivot(index=keys, columns="model_role", values="continuous_irrigation_mm").reset_index().rename(columns={ANCHOR_ROLE: "frozen_anchor_irrigation_mm", V8_ROLE: "frozen_v8_irrigation_mm"})
    comparison = comparison.merge(frozen, on=keys, validate="one_to_one")
    if not np.allclose(comparison["anchor_irrigation_mm"], comparison["frozen_anchor_irrigation_mm"], rtol=0.0, atol=TOLERANCE) or not np.allclose(comparison["v8_irrigation_mm"], comparison["frozen_v8_irrigation_mm"], rtol=0.0, atol=TOLERANCE):
        raise ValueError("executed recommendations differ from the frozen v8 plan")
    comparison["v8_minus_anchor_gain_7d"] = comparison["v8_gain_7d"] - comparison["anchor_gain_7d"]
    comparison["v8_minus_anchor_regret_fixed_7d"] = comparison["v8_regret_fixed_7d"] - comparison["anchor_regret_fixed_7d"]
    comparison["v8_minus_anchor_regret_adaptive_7d"] = comparison["v8_regret_adaptive_7d"] - comparison["anchor_regret_adaptive_7d"]
    if not np.isfinite(comparison.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("changed-cycle comparison is incomplete")
    return comparison.sort_values(keys).reset_index(drop=True)


def build_full_ledger(inventory: pd.DataFrame, changed_comparison: pd.DataFrame) -> pd.DataFrame:
    required = {"fold_id", "target_site", "site_id", "decision_date", "true_oracle_is_positive", "irrigation_changed"}
    if missing := sorted(required - set(inventory.columns)):
        raise ValueError(f"v8 inventory fields are missing: {missing}")
    keys = ["fold_id", "target_site", "site_id", "decision_date"]
    comparison = changed_comparison.copy()
    comparison["decision_date"] = pd.to_datetime(comparison["decision_date"]).dt.strftime("%Y-%m-%d")
    if len(comparison) != EXPECTED_CHANGED_CYCLES or comparison.duplicated(keys).any():
        raise ValueError("changed-cycle SWAP comparison is incomplete")
    deltas = ["v8_minus_anchor_gain_7d", "v8_minus_anchor_regret_fixed_7d", "v8_minus_anchor_regret_adaptive_7d"]
    ledger = inventory[[*keys, "validation_year", "target_year", "true_oracle_is_positive", "irrigation_changed"]].copy()
    ledger["decision_date"] = pd.to_datetime(ledger["decision_date"]).dt.strftime("%Y-%m-%d")
    ledger = ledger.merge(comparison[[*keys, *deltas]], on=keys, how="left", validate="one_to_one", indicator=True)
    changed = ledger["irrigation_changed"].astype(bool)
    if not ledger.loc[changed, "_merge"].eq("both").all() or not ledger.loc[~changed, "_merge"].eq("left_only").all():
        raise ValueError("changed-cycle comparison keys differ from the v8 inventory")
    ledger.loc[~changed, deltas] = 0.0
    ledger["pair_source"] = np.where(changed, "swap_changed_pair", "identical_recommendation_zero_difference")
    ledger = ledger.drop(columns="_merge")
    if len(ledger) != EXPECTED_FULL_CYCLES or not np.isfinite(ledger[deltas].to_numpy(float)).all():
        raise ValueError("full paired-difference ledger is incomplete")
    return ledger.sort_values(["site_id", "decision_date"]).reset_index(drop=True)


def evaluate_gate(ledger: pd.DataFrame, changed_comparison: pd.DataFrame, *, execution_complete: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    deltas = ["v8_minus_anchor_gain_7d", "v8_minus_anchor_regret_fixed_7d", "v8_minus_anchor_regret_adaptive_7d"]
    required = {"fold_id", "target_site", "true_oracle_is_positive", "irrigation_changed", *deltas}
    if missing := sorted(required - set(ledger.columns)):
        raise ValueError(f"full ledger gate fields are missing: {missing}")
    if len(ledger) != EXPECTED_FULL_CYCLES or ledger["fold_id"].nunique() != EXPECTED_FOLDS or ledger["target_site"].nunique() != EXPECTED_SITES:
        raise ValueError("full paired ledger structure changed")
    fold = ledger.groupby("fold_id", as_index=False).agg(cycle_count=(deltas[0], "size"), mean_v8_minus_anchor_gain_7d=(deltas[0], "mean"), mean_v8_minus_anchor_regret_fixed_7d=(deltas[1], "mean"), mean_v8_minus_anchor_regret_adaptive_7d=(deltas[2], "mean"))
    site = ledger.groupby("target_site", as_index=False).agg(cycle_count=(deltas[0], "size"), mean_v8_minus_anchor_gain_7d=(deltas[0], "mean"), mean_v8_minus_anchor_regret_fixed_7d=(deltas[1], "mean"), mean_v8_minus_anchor_regret_adaptive_7d=(deltas[2], "mean"))
    positive = ledger["true_oracle_is_positive"].astype(bool)
    zero = ~positive
    folds_nonworse = int(fold["mean_v8_minus_anchor_gain_7d"].ge(-TOLERANCE).sum())
    sites_nonworse = int(site["mean_v8_minus_anchor_gain_7d"].ge(-TOLERANCE).sum())
    conditions = {
        "pooled_full_196_mean_swap_gain_nonworse": float(ledger[deltas[0]].mean()) >= -TOLERANCE,
        "at_least_10_of_15_fold_mean_swap_gain_nonworse": folds_nonworse >= 10,
        "at_least_4_of_5_site_mean_swap_gain_nonworse": sites_nonworse >= 4,
        "overall_fixed_eight_regret_nonworse": float(ledger[deltas[1]].mean()) <= TOLERANCE,
        "changed_cycle_maximum_adaptive_regret_nonworse_sufficient_for_full_inventory": float(changed_comparison["v8_regret_adaptive_7d"].max()) <= float(changed_comparison["anchor_regret_adaptive_7d"].max()) + TOLERANCE,
        "zero_oracle_mean_swap_gain_nonworse": bool(zero.any()) and float(ledger.loc[zero, deltas[0]].mean()) >= -TOLERANCE,
        "positive_oracle_mean_swap_gain_nonworse": bool(positive.any()) and float(ledger.loc[positive, deltas[0]].mean()) >= -TOLERANCE,
        "all_12_changed_pairs_finite": len(changed_comparison) == EXPECTED_CHANGED_CYCLES and bool(np.isfinite(changed_comparison.select_dtypes(include=[np.number]).to_numpy(float)).all()),
        "all_196_pair_differences_finite": bool(np.isfinite(ledger[deltas].to_numpy(float)).all()),
        "all_reused_and_new_swap_execution_audits_passed": bool(execution_complete),
    }
    observed = {
        "full_cycle_count": int(len(ledger)), "changed_cycle_count": int(ledger["irrigation_changed"].astype(bool).sum()),
        "unchanged_cycle_count": int((~ledger["irrigation_changed"].astype(bool)).sum()),
        "pooled_mean_v8_minus_anchor_swap_gain_7d": float(ledger[deltas[0]].mean()),
        "positive_mean_v8_minus_anchor_swap_gain_7d": float(ledger.loc[positive, deltas[0]].mean()),
        "zero_mean_v8_minus_anchor_swap_gain_7d": float(ledger.loc[zero, deltas[0]].mean()),
        "folds_with_nonworse_mean_swap_gain": folds_nonworse, "sites_with_nonworse_mean_swap_gain": sites_nonworse,
        "anchor_changed_cycle_max_adaptive_regret_7d": float(changed_comparison["anchor_regret_adaptive_7d"].max()),
        "v8_changed_cycle_max_adaptive_regret_7d": float(changed_comparison["v8_regret_adaptive_7d"].max()),
    }
    gate = {"passed": bool(all(conditions.values())), "protocol_id": PROTOCOL_ID, "gate_type": "predeclared_changed_cycle_paired_continuous_engineering_gate_not_significance_test", "conditions": conditions, "observed": observed, "automatic_model_promotion": False, "passing_action": "authorize_separate_2019_frozen_after_development_time_out_confirmation_protocol_design_only", "failing_action": "freeze_v8_changed_cycle_paired_swap_as_negative_stop_before_2019"}
    return fold, site, gate


def _require_manifest_hash(manifest: dict[str, Any], section: str, name: str, path: Path) -> None:
    expected = manifest.get(section, {}).get(name, {}).get("sha256")
    if not isinstance(expected, str) or sha256_file(path).lower() != expected.lower():
        raise ValueError(f"manifest hash changed: {section}.{name}")


def load_and_validate_upstream(*, protocol: dict[str, Any], stage_a_dir: Path, paired_swap_dir: Path, formal_label_dir: Path) -> dict[str, Path]:
    paths = {
        "stage_a_decisions": stage_a_dir / STAGE_A_DECISIONS_NAME,
        "stage_a_gate": stage_a_dir / STAGE_A_GATE_NAME,
        "stage_a_audit": stage_a_dir / STAGE_A_AUDIT_NAME,
        "stage_a_manifest": stage_a_dir / STAGE_A_MANIFEST_NAME,
        "pair_audit": paired_swap_dir / PAIR_AUDIT_NAME,
        "pair_manifest": paired_swap_dir / PAIR_MANIFEST_NAME,
        "pair_plan": paired_swap_dir / "gefs_gam_v5_changed_cycle_paired_swap_plan_v1.csv",
        "pair_decisions": paired_swap_dir / PAIR_DECISIONS_NAME,
        "prior_candidates": paired_swap_dir / "gefs_gam_v5_changed_cycle_paired_swap_candidates_v1.csv",
        "formal_weather": formal_label_dir / FORMAL_WEATHER_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required v8 changed-cycle inputs are missing: {missing}")
    stage_gate = read_json(paths["stage_a_gate"])
    stage_audit = read_json(paths["stage_a_audit"])
    if stage_gate.get("passed") is not True or stage_audit.get("status") != protocol["source_stage_a"]["required_status"]:
        raise ValueError("frozen Stage A v8 result is not the required passed boundary")
    stage_manifest = read_json(paths["stage_a_manifest"])
    for name in ("decisions", "gate", "audit"):
        _require_manifest_hash(stage_manifest, "outputs", name, paths[f"stage_a_{name}"])
    pair_audit = read_json(paths["pair_audit"])
    if pair_audit.get("status") != protocol["source_v5_paired_swap"]["required_status"]:
        raise ValueError("frozen v5 paired SWAP status changed")
    if pair_audit.get("mandatory_execution_gate_passed") is not protocol["source_v5_paired_swap"]["required_execution_gate_passed"]:
        raise ValueError("frozen v5 paired SWAP execution boundary changed")
    if pair_audit.get("predeclared_performance_gate_passed") is not protocol["source_v5_paired_swap"]["required_performance_gate_passed"]:
        raise ValueError("frozen v5 paired SWAP performance boundary changed")
    if int(pair_audit.get("2019_rows_read", -1)) != 0 or int(pair_audit.get("2024_rows_read", -1)) != 0:
        raise ValueError("prior paired SWAP evidence boundary changed")
    pair_manifest = read_json(paths["pair_manifest"])
    for name in ("plan", "decisions", "candidates", "audit"):
        path_name = {"plan": "pair_plan", "decisions": "pair_decisions", "candidates": "prior_candidates", "audit": "pair_audit"}[name]
        _require_manifest_hash(pair_manifest, "outputs", name, paths[path_name])
    return paths


def load_prior_candidate_bundle(paired_swap_dir: Path, aggregate_path: Path, official_decisions_path: Path) -> tuple[pd.DataFrame, list[Path]]:
    frames = [pd.read_csv(aggregate_path)]
    branch_paths = sorted(paired_swap_dir.rglob("swap_candidates.csv"))
    for path in branch_paths:
        frame = pd.read_csv(path)
        required = {"site", "decision_date", "ir", "target_value"}
        if missing := sorted(required - set(frame.columns)):
            raise ValueError(f"prior exact branch fields are missing in {path}: {missing}")
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["evaluation_cycle_id"] = frame["site"].astype(str) + "_" + frame["decision_date"]
        frame["evaluation_stage"] = "v5_exact_recommendation"
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    official = pd.read_csv(official_decisions_path)
    required_official = {"site_id", "decision_date", "continuous_irrigation_mm", "continuous_recommendation_swap_gain_7d"}
    if missing := sorted(required_official - set(official.columns)):
        raise ValueError(f"prior official decision fields are missing: {missing}")
    for row in official.itertuples(index=False):
        cycle_id = f"{row.site_id}_{pd.Timestamp(row.decision_date).strftime('%Y-%m-%d')}"
        cycle = combined.loc[combined["evaluation_cycle_id"].astype(str).eq(cycle_id)]
        irrigation = pd.to_numeric(cycle["ir"], errors="raise").to_numpy(float)
        matches = np.isclose(irrigation, float(row.continuous_irrigation_mm), rtol=0.0, atol=TOLERANCE)
        gains = pd.to_numeric(cycle.loc[matches, "target_value"], errors="raise").to_numpy(float)
        if not len(gains) or not np.isfinite(gains).all() or not np.allclose(gains, float(row.continuous_recommendation_swap_gain_7d), rtol=0.0, atol=TOLERANCE):
            raise ValueError(f"prior exact endpoint does not reproduce official v5 decision: {cycle_id}")
    return combined, branch_paths


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = args.protocol.resolve()
    stage_a_dir = args.stage_a_dir.resolve()
    paired_swap_dir = args.paired_swap_dir.resolve()
    formal_label_dir = args.formal_label_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    swap = protocol["swap"]
    if int(args.restart_nprintday) != int(swap["restart_nprintday"]) or float(args.swap_dtmax_days) != float(swap["swap_dtmax_days"]):
        raise ValueError("execution arguments differ from the frozen SWAP protocol")
    input_paths = load_and_validate_upstream(protocol=protocol, stage_a_dir=stage_a_dir, paired_swap_dir=paired_swap_dir, formal_label_dir=formal_label_dir)
    inventory = pd.read_csv(input_paths["stage_a_decisions"])
    plan = build_changed_plan(inventory)
    prior_plan = pd.read_csv(input_paths["pair_plan"])
    prior_candidates, prior_branch_paths = load_prior_candidate_bundle(
        paired_swap_dir, input_paths["prior_candidates"], input_paths["pair_decisions"]
    )
    weather = pd.read_csv(input_paths["formal_weather"])
    output_dir.mkdir(parents=True, exist_ok=args.resume)

    checkpoint_by_cycle: dict[tuple[str, str], str] = {}
    if "checkpoint_sha256" in prior_plan.columns:
        for row in prior_plan.drop_duplicates(["site_id", "decision_date"]).itertuples(index=False):
            checkpoint_by_cycle[(str(row.site_id), str(row.decision_date))] = str(row.checkpoint_sha256)
    if checkpoint_by_cycle:
        plan["checkpoint_sha256"] = [checkpoint_by_cycle.get((str(r.site_id), str(r.decision_date)), "") for r in plan.itertuples(index=False)]
        if plan["checkpoint_sha256"].eq("").any():
            raise ValueError("v8 changed-cycle plan is missing frozen checkpoint hashes")
    preflight_reused = 0
    preflight_missing = 0
    for cycle in plan.drop_duplicates(["site_id", "decision_date"]).itertuples(index=False):
        cycle_prior = prior_cycle_candidates(
            prior_candidates,
            site_id=str(cycle.site_id),
            decision_date=str(cycle.decision_date),
        )
        requested = plan.loc[
            plan["site_id"].astype(str).eq(str(cycle.site_id))
            & plan["decision_date"].astype(str).eq(str(cycle.decision_date)),
            "continuous_irrigation_mm",
        ].astype(float).tolist()
        reusable, missing_values = split_exact_reuse(cycle_prior, requested)
        preflight_reused += len(reusable)
        preflight_missing += len(missing_values)
    if (
        preflight_reused
        != int(protocol["candidate_reuse"]["expected_reused_recommendation_count"])
        or preflight_missing
        != int(protocol["candidate_reuse"]["maximum_new_exact_swap_runs"])
    ):
        raise ValueError(
            "candidate reuse preflight differs from the frozen 16-reused/8-new contract"
        )
    plan_path = output_dir / "gefs_gam_v8_changed_cycle_paired_swap_plan_v1.csv"
    plan.to_csv(plan_path, index=False)

    decision_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    combined_frames: list[pd.DataFrame] = []
    stage_roots: list[str] = []
    unique_cycles = plan.drop_duplicates(["site_id", "decision_date"])
    for cycle_index, cycle in enumerate(unique_cycles.itertuples(index=False), start=1):
        site_id, decision_date, target_year = str(cycle.site_id), str(cycle.decision_date), int(cycle.target_year)
        print(f"cycle={cycle_index}/{EXPECTED_CHANGED_CYCLES} site={site_id} date={decision_date}", flush=True)
        recommendations = plan.loc[plan["site_id"].astype(str).eq(site_id) & plan["decision_date"].astype(str).eq(decision_date)].copy()
        requested = recommendations["continuous_irrigation_mm"].astype(float).tolist()
        prior = prior_cycle_candidates(prior_candidates, site_id=site_id, decision_date=decision_date)
        zero_reference = zero_irrigation_cwdm_reference(prior)
        reusable, missing_values = split_exact_reuse(prior, requested)
        new_candidates = pd.DataFrame()
        stage_root = ""
        if missing_values:
            source_workspace, checkpoint_dir, checkpoint_audit = unit_resources(formal_label_dir, site_id, decision_date, target_year)
            new_candidates, generated_root = run_swap_stage(
                base_root=output_dir / "exact_pair_branches" / site_id / pd.Timestamp(decision_date).strftime("%Y%m%d") / "missing_exact_recommendations",
                requested_values=missing_values, source_workspace=source_workspace, checkpoint_dir=checkpoint_dir,
                checkpoint_audit=checkpoint_audit, weather=weather, site_id=site_id, decision_date=decision_date,
                resume=bool(args.resume), restart_nprintday=int(args.restart_nprintday), swap_dtmax_days=float(args.swap_dtmax_days),
                fallback_offsets_mm=tuple(float(v) for v in swap["fallback_offsets_mm"]), target_year=target_year,
                zero_irrigation_cwdm_reference=zero_reference,
            )
            new_candidates = new_candidates.assign(evaluation_stage="v8_missing_exact_recommendations")
            stage_root = str(generated_root)
            stage_roots.append(stage_root)
        combined = pd.concat([prior, new_candidates], ignore_index=True)
        validate_prior_cycle_grid(combined)
        rows, summary = evaluate_cycle(combined, recommendations)
        decision_rows.extend(rows)
        cycle_rows.append(summary)
        combined.insert(0, "v8_changed_cycle_id", f"{site_id}_{decision_date}")
        combined_frames.append(combined)
        reuse_rows.append({
            "site_id": site_id, "decision_date": decision_date, "requested_recommendation_count": len(requested),
            "exact_reused_recommendation_count": len(reusable), "new_exact_swap_run_count": len(missing_values),
            "exact_reused_values_mm": json.dumps(reusable), "new_exact_values_mm": json.dumps(missing_values),
            "external_zero_irrigation_cwdm_reference": zero_reference, "new_stage_root": stage_root,
            "prior_grid_candidate_count": int(len(prior)), "combined_candidate_count": int(len(combined)),
            "mandatory_execution_gate_passed": True,
        })

    decisions = pd.DataFrame(decision_rows)
    changed_comparison = build_changed_comparison(decisions, plan)
    ledger = build_full_ledger(inventory, changed_comparison)
    reuse_audit = pd.DataFrame(reuse_rows)
    new_run_count = int(reuse_audit["new_exact_swap_run_count"].sum())
    reused_count = int(reuse_audit["exact_reused_recommendation_count"].sum())
    execution_complete = bool(
        len(decisions) == EXPECTED_PLAN_ROWS
        and len(reuse_audit) == EXPECTED_CHANGED_CYCLES
        and new_run_count == int(protocol["candidate_reuse"]["maximum_new_exact_swap_runs"])
        and reused_count == int(protocol["candidate_reuse"]["expected_reused_recommendation_count"])
        and new_run_count + reused_count == EXPECTED_PLAN_ROWS
        and reuse_audit["mandatory_execution_gate_passed"].all()
    )
    fold_summary, site_summary, gate = evaluate_gate(ledger, changed_comparison, execution_complete=execution_complete)
    gate["conditions"]["retained_stage_a_peak_gate_passed"] = True
    gate["passed"] = bool(all(gate["conditions"].values()))
    gate.update({"protocol_id": PROTOCOL_ID, "2019_access_authorized": False, "new_exact_swap_run_count": new_run_count, "exact_reused_recommendation_count": reused_count})
    outputs = {
        "plan": plan_path,
        "decisions": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_decisions_v1.csv",
        "changed_comparison": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_comparison_v1.csv",
        "full_ledger": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_full_ledger_v1.csv",
        "fold_summary": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_fold_summary_v1.csv",
        "site_summary": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_site_summary_v1.csv",
        "reuse_audit": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_reuse_audit_v1.csv",
        "cycle_oracles": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_oracles_v1.csv",
        "candidates": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_candidates_v1.csv",
        "gate": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_performance_gate_v1.json",
        "audit": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_qualification_audit_v1.json",
        "manifest": output_dir / "gefs_gam_v8_changed_cycle_paired_swap_qualification_manifest_v1.json",
    }
    decisions.to_csv(outputs["decisions"], index=False)
    changed_comparison.to_csv(outputs["changed_comparison"], index=False)
    ledger.to_csv(outputs["full_ledger"], index=False)
    fold_summary.to_csv(outputs["fold_summary"], index=False)
    site_summary.to_csv(outputs["site_summary"], index=False)
    reuse_audit.to_csv(outputs["reuse_audit"], index=False)
    pd.DataFrame(cycle_rows).to_csv(outputs["cycle_oracles"], index=False)
    pd.concat(combined_frames, ignore_index=True).to_csv(outputs["candidates"], index=False)
    write_json(outputs["gate"], gate)
    audit = {
        "status": "gam_v8_changed_cycle_paired_swap_qualification_passed_pending_2019_time_out_confirmation_design" if gate["passed"] else "gam_v8_changed_cycle_paired_swap_qualification_failed_stop_before_2019",
        "protocol_id": PROTOCOL_ID, "mandatory_execution_gate_passed": execution_complete,
        "predeclared_performance_gate_passed": bool(gate["passed"]), "full_cycle_count": EXPECTED_FULL_CYCLES,
        "changed_cycle_count": EXPECTED_CHANGED_CYCLES, "unchanged_cycle_zero_difference_count": EXPECTED_FULL_CYCLES - EXPECTED_CHANGED_CYCLES,
        "recommendation_plan_rows": EXPECTED_PLAN_ROWS, "exact_reused_recommendation_count": reused_count,
        "new_exact_swap_run_count": new_run_count, "new_stage_roots": stage_roots,
        "prior_exact_branch_candidate_file_count": len(prior_branch_paths),
        "prior_exact_branch_candidate_sha256": {str(path): sha256_file(path) for path in prior_branch_paths},
        "prior_numerical_candidates_reused": True, "interpolation_used": False,
        "recommendation_changes_after_swap": False, "2019_rows_read": 0, "2024_rows_read": 0,
        "automatic_model_promotion": False, "next_gate": gate["passing_action"] if gate["passed"] else gate["failing_action"],
        "inputs": {"protocol_sha256": sha256_file(protocol_path), **{name: sha256_file(path) for name, path in input_paths.items()}},
    }
    write_json(outputs["audit"], audit)
    write_json(outputs["manifest"], {"status": audit["status"], "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()}, "outputs": {name: {"path": path.name, "sha256": sha256_file(path)} for name, path in outputs.items() if name != "manifest"}})
    if not execution_complete:
        raise RuntimeError(f"v8 changed-cycle SWAP execution gate failed; see {outputs['audit']}")
    if not gate["passed"]:
        raise RuntimeError(f"v8 changed-cycle paired SWAP performance gate failed; see {outputs['gate']}")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-a-dir", type=Path, required=True)
    parser.add_argument("--paired-swap-dir", type=Path, required=True)
    parser.add_argument("--formal-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restart-nprintday", type=int, default=48)
    parser.add_argument("--swap-dtmax-days", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    for name, path in run(parse_args()).items():
        print(f"{name}: {path}", flush=True)

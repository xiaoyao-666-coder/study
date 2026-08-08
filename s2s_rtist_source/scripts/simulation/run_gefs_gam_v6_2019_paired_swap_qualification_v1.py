#!/usr/bin/env python3
"""Confirm frozen GAM v6 recommendations on 2019 with exact paired SWAP."""

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

from scripts.diagnostics.audit_gefs_gam_state_interaction_features_v1 import (
    CONTRACT_NAME,
    CYCLE_KEYS,
    DATASET_NAME,
    SOURCE_AUDIT_NAME,
)
from scripts.evaluation.freeze_gefs_gam_expert_cross_fitted_gate_protocol_v1 import (
    sha256_file,
)
from scripts.simulation.run_gefs_controlled_continuous_swap_evaluation_v1 import (
    evaluate_cycle,
    run_swap_stage,
    unit_resources,
)
from scripts.training.run_gefs_hierarchical_gam_bspline_development_v1 import (
    _dataset_index,
    _read_selected_rows,
    validate_complete_cycles,
)
from scripts.training.run_gefs_robust_envelope_gam_development_v1 import (
    true_fixed_grid_peak,
)


PROTOCOL_ID = "teacher-guided-gam-v6-2019-independent-confirmation-v1"
CONFIRMATION_YEAR = 2019
SEALED_TEST_YEAR = 2024
EXPECTED_CYCLE_COUNT = 69
EXPECTED_CANDIDATE_ROW_COUNT = 552
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
EXPECTED_SITE_CYCLES = {"P1": 13, "P2": 13, "P3": 15, "P4": 14, "P15": 14}
FIXED_IRRIGATION_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
TOLERANCE = 1.0e-6
ANCHOR_ROLE = "frozen_source_selected_anchor"
SELECTED_ROLE = "frozen_v6_confidence_protected"
FORMAL_WEATHER_NAME = "gefs_exact_schedule_2015_2019_formal_weather_v1.csv"
RECOMMENDATIONS_NAME = "gefs_gam_v6_2019_frozen_recommendations_v1.csv"
FREEZE_MODELS_NAME = "gefs_gam_v6_2019_recommendation_model_audit_v1.json"
FREEZE_GATE_NAME = "gefs_gam_v6_2019_recommendation_freeze_gate_v1.json"
FREEZE_AUDIT_NAME = "gefs_gam_v6_2019_recommendation_freeze_audit_v1.json"
FREEZE_MANIFEST_NAME = "gefs_gam_v6_2019_recommendation_freeze_manifest_v1.json"


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
    swap = protocol.get("swap", {})
    gate = protocol.get("performance_gate", {})
    outcomes = protocol.get("outcomes", {})
    if int(data.get("confirmation_year", -1)) != CONFIRMATION_YEAR:
        raise ValueError("confirmation year changed")
    if int(data.get("sealed_test_year", -1)) != SEALED_TEST_YEAR:
        raise ValueError("sealed test year changed")
    if tuple(data.get("sites", ())) != EXPECTED_SITES:
        raise ValueError("confirmation sites changed")
    if (
        int(data.get("expected_confirmation_candidate_rows", -1))
        != EXPECTED_CANDIDATE_ROW_COUNT
        or int(data.get("expected_confirmation_cycles", -1))
        != EXPECTED_CYCLE_COUNT
    ):
        raise ValueError("2019 row or cycle contract changed")
    if data.get("expected_confirmation_site_cycles") != EXPECTED_SITE_CYCLES:
        raise ValueError("2019 site-cycle contract changed")
    expected_swap = {
        "fixed_irrigation_mm": list(FIXED_IRRIGATION_MM),
        "fallback_offsets_mm": [0.1, 0.2, 0.3],
        "restart_nprintday": 48,
        "swap_dtmax_days": 0.001,
        "interpolation": False,
        "unchanged_cycle_difference": 0.0,
    }
    if any(swap.get(name) != value for name, value in expected_swap.items()):
        raise ValueError("exact paired SWAP policy changed")
    if swap.get("formal_weather_sha256") != "2d2c79c79329f45eac13895630de98463ae9c53b376ed4d5c498b022e61fe335":
        raise ValueError("formal weather hash contract changed")
    required_gate = {
        "minimum_changed_cycles": 2,
        "minimum_changed_sites": 2,
        "minimum_nonworse_sites": 4,
        "pooled_mean_swap_gain_nonworse": True,
        "positive_oracle_mean_swap_gain_nonworse": True,
        "zero_oracle_mean_swap_gain_nonworse": True,
        "overall_fixed_eight_regret_nonworse": True,
        "changed_cycle_maximum_adaptive_regret_nonworse": True,
        "overall_mean_peak_distance_nonworse": True,
        "positive_oracle_mean_peak_distance_nonworse": True,
        "zero_oracle_mean_peak_distance_nonworse": True,
        "global_maximum_peak_distance_nonworse": True,
        "all_numeric_outputs_finite": True,
        "tolerance": TOLERANCE,
    }
    if any(gate.get(name) != value for name, value in required_gate.items()):
        raise ValueError("joint confirmation gate changed")
    if (
        bool(data.get("2024_access", True))
        or bool(outcomes.get("2024_access", True))
        or bool(outcomes.get("automatic_model_promotion", True))
        or bool(outcomes.get("final_refit_performed", True))
        or bool(outcomes.get("tta_performed", True))
    ):
        raise ValueError("a post-confirmation action was enabled")


def validate_execution_args(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    swap = protocol["swap"]
    if (
        int(args.restart_nprintday) != int(swap["restart_nprintday"])
        or float(args.swap_dtmax_days) != float(swap["swap_dtmax_days"])
    ):
        raise ValueError("execution arguments differ from the frozen protocol")


def _normalize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    required = {
        "fold_id",
        "target_site",
        "target_year",
        "site_id",
        "decision_date",
        "anchor_recommendation_mm",
        "selected_irrigation_mm",
        "use_v6",
    }
    if missing := sorted(required - set(inventory.columns)):
        raise ValueError(f"frozen recommendation fields are missing: {missing}")
    frame = inventory.copy()
    frame["target_year"] = pd.to_numeric(frame["target_year"], errors="raise").astype(int)
    frame["decision_date"] = pd.to_datetime(
        frame["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    frame["use_v6"] = frame["use_v6"].astype(bool)
    numeric = frame[["anchor_recommendation_mm", "selected_irrigation_mm"]].apply(
        pd.to_numeric, errors="raise"
    )
    frame[numeric.columns] = numeric
    site_counts = frame.groupby("site_id").size().astype(int).to_dict()
    if (
        len(frame) != EXPECTED_CYCLE_COUNT
        or frame.duplicated(["site_id", "decision_date"]).any()
        or set(frame["target_site"].astype(str)) != set(EXPECTED_SITES)
        or site_counts != EXPECTED_SITE_CYCLES
        or not frame["target_year"].eq(CONFIRMATION_YEAR).all()
        or not frame["site_id"].astype(str).eq(frame["target_site"].astype(str)).all()
        or not numeric.apply(lambda column: column.between(0.0, 60.0).all()).all()
        or not np.isfinite(numeric.to_numpy(float)).all()
    ):
        raise ValueError("frozen recommendation inventory structure changed")
    changed_by_value = ~np.isclose(
        frame["anchor_recommendation_mm"],
        frame["selected_irrigation_mm"],
        rtol=0.0,
        atol=TOLERANCE,
    )
    if not np.array_equal(frame["use_v6"].to_numpy(bool), changed_by_value):
        raise ValueError("use_v6 does not match frozen endpoint differences")
    return frame.sort_values(["site_id", "decision_date"]).reset_index(drop=True)


def score_peak_inventory(inventory: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_inventory(inventory)
    keys = ["target_year", "site_id", "decision_date"]
    required_truth = {
        *keys,
        "true_peak_mm",
        "true_fixed_list_net_gain_7d",
        "true_oracle_is_positive",
    }
    if missing := sorted(required_truth - set(truth.columns)):
        raise ValueError(f"2019 truth fields are missing: {missing}")
    labels = truth[list(required_truth)].copy()
    labels["decision_date"] = pd.to_datetime(
        labels["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if len(labels) != EXPECTED_CYCLE_COUNT or labels.duplicated(keys).any():
        raise ValueError("2019 truth inventory is incomplete")
    scored = frame.merge(labels, on=keys, how="left", validate="one_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("2019 truth keys differ from the frozen recommendations")
    scored = scored.drop(columns="_merge")
    scored["anchor_peak_distance_mm"] = np.abs(
        scored["anchor_recommendation_mm"] - scored["true_peak_mm"]
    )
    scored["selected_peak_distance_mm"] = np.abs(
        scored["selected_irrigation_mm"] - scored["true_peak_mm"]
    )
    numeric = scored.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("2019 peak scoring contains non-finite values")
    return scored.sort_values(["site_id", "decision_date"]).reset_index(drop=True)


def build_changed_plan(inventory: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_inventory(inventory)
    changed = frame.loc[frame["use_v6"]].copy()
    rows: list[dict[str, Any]] = []
    for row in changed.itertuples(index=False):
        common = {
            "fold_id": str(row.fold_id),
            "target_site": str(row.target_site),
            "target_year": int(row.target_year),
            "site_id": str(row.site_id),
            "decision_date": str(row.decision_date),
        }
        rows.extend(
            [
                {
                    **common,
                    "model_id": f"gam_v6_2019_anchor_{row.site_id}",
                    "model_role": ANCHOR_ROLE,
                    "continuous_irrigation_mm": float(row.anchor_recommendation_mm),
                },
                {
                    **common,
                    "model_id": f"gam_v6_2019_selected_{row.site_id}",
                    "model_role": SELECTED_ROLE,
                    "continuous_irrigation_mm": float(row.selected_irrigation_mm),
                },
            ]
        )
    columns = [
        "fold_id",
        "target_site",
        "target_year",
        "site_id",
        "decision_date",
        "model_id",
        "model_role",
        "continuous_irrigation_mm",
    ]
    plan = pd.DataFrame(rows, columns=columns)
    if plan.empty:
        return plan
    plan = plan.sort_values(["site_id", "decision_date", "model_role"]).reset_index(drop=True)
    if (
        len(plan) != 2 * int(frame["use_v6"].sum())
        or plan.duplicated(["site_id", "decision_date", "model_role"]).any()
        or not plan.groupby(["site_id", "decision_date"])["model_role"].nunique().eq(2).all()
    ):
        raise ValueError("changed-cycle exact endpoint plan is incomplete")
    return plan


def build_full_ledger(
    inventory: pd.DataFrame,
    changed_comparison: pd.DataFrame,
) -> pd.DataFrame:
    frame = _normalize_inventory(inventory)
    keys = ["fold_id", "target_site", "site_id", "decision_date"]
    required = {
        *keys,
        "anchor_gain_7d",
        "selected_gain_7d",
        "anchor_regret_fixed_7d",
        "selected_regret_fixed_7d",
        "anchor_regret_adaptive_7d",
        "selected_regret_adaptive_7d",
    }
    if missing := sorted(required - set(changed_comparison.columns)):
        raise ValueError(f"changed comparison fields are missing: {missing}")
    comparison = changed_comparison.copy()
    comparison["decision_date"] = pd.to_datetime(
        comparison["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    expected_changed = int(frame["use_v6"].sum())
    if len(comparison) != expected_changed or comparison.duplicated(keys).any():
        raise ValueError("changed-cycle SWAP comparison is incomplete")
    comparison["selected_minus_anchor_gain_7d"] = (
        comparison["selected_gain_7d"] - comparison["anchor_gain_7d"]
    )
    comparison["selected_minus_anchor_regret_fixed_7d"] = (
        comparison["selected_regret_fixed_7d"]
        - comparison["anchor_regret_fixed_7d"]
    )
    comparison["selected_minus_anchor_regret_adaptive_7d"] = (
        comparison["selected_regret_adaptive_7d"]
        - comparison["anchor_regret_adaptive_7d"]
    )
    deltas = [
        "selected_minus_anchor_gain_7d",
        "selected_minus_anchor_regret_fixed_7d",
        "selected_minus_anchor_regret_adaptive_7d",
    ]
    keep = [
        *keys,
        "target_year",
        "true_peak_mm",
        "true_fixed_list_net_gain_7d",
        "true_oracle_is_positive",
        "anchor_peak_distance_mm",
        "selected_peak_distance_mm",
        "use_v6",
    ]
    missing_inventory = sorted(set(keep) - set(inventory.columns))
    if missing_inventory:
        raise ValueError(f"scored inventory fields are missing: {missing_inventory}")
    ledger = inventory[keep].copy()
    ledger["decision_date"] = pd.to_datetime(
        ledger["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    ledger = ledger.merge(
        comparison[[*keys, *deltas]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    changed = ledger["use_v6"].astype(bool)
    if (
        not ledger.loc[changed, "_merge"].eq("both").all()
        or not ledger.loc[~changed, "_merge"].eq("left_only").all()
    ):
        raise ValueError("changed SWAP keys differ from the frozen inventory")
    ledger.loc[~changed, deltas] = 0.0
    ledger["pair_source"] = np.where(
        changed,
        "exact_changed_cycle_swap_pair",
        "identical_frozen_recommendation_zero_difference",
    )
    ledger = ledger.drop(columns="_merge")
    if len(ledger) != EXPECTED_CYCLE_COUNT or not np.isfinite(
        ledger[deltas].to_numpy(float)
    ).all():
        raise ValueError("full 2019 paired ledger is incomplete")
    return ledger.sort_values(["site_id", "decision_date"]).reset_index(drop=True)


def _peak_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    strata = {
        "overall": pd.Series(True, index=ledger.index),
        "positive_oracle": ledger["true_oracle_is_positive"].astype(bool),
        "zero_oracle": ~ledger["true_oracle_is_positive"].astype(bool),
    }
    rows = []
    for name, selected in strata.items():
        part = ledger.loc[selected]
        rows.append(
            {
                "stratum": name,
                "cycle_count": int(len(part)),
                "anchor_mean_peak_distance_mm": float(part["anchor_peak_distance_mm"].mean()),
                "selected_mean_peak_distance_mm": float(part["selected_peak_distance_mm"].mean()),
                "anchor_max_peak_distance_mm": float(part["anchor_peak_distance_mm"].max()),
                "selected_max_peak_distance_mm": float(part["selected_peak_distance_mm"].max()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_gate(
    ledger: pd.DataFrame,
    changed_comparison: pd.DataFrame,
    *,
    execution_complete: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(ledger) != EXPECTED_CYCLE_COUNT or ledger["target_site"].nunique() != 5:
        raise ValueError("full 2019 ledger structure changed")
    changed_count = int(ledger["use_v6"].astype(bool).sum())
    changed_site_count = int(
        ledger.loc[ledger["use_v6"].astype(bool), "target_site"].nunique()
    )
    if len(changed_comparison) != changed_count:
        raise ValueError("changed comparison count differs from full ledger")
    deltas = [
        "selected_minus_anchor_gain_7d",
        "selected_minus_anchor_regret_fixed_7d",
        "selected_minus_anchor_regret_adaptive_7d",
    ]
    if missing := sorted(set(deltas) - set(ledger.columns)):
        raise ValueError(f"paired gate fields are missing: {missing}")
    site_summary = ledger.groupby("target_site", as_index=False).agg(
        cycle_count=("selected_minus_anchor_gain_7d", "size"),
        mean_selected_minus_anchor_gain_7d=("selected_minus_anchor_gain_7d", "mean"),
        mean_selected_minus_anchor_regret_fixed_7d=(
            "selected_minus_anchor_regret_fixed_7d",
            "mean",
        ),
        mean_selected_minus_anchor_regret_adaptive_7d=(
            "selected_minus_anchor_regret_adaptive_7d",
            "mean",
        ),
    )
    peak_summary = _peak_summary(ledger)
    peak = peak_summary.set_index("stratum")
    positive = ledger["true_oracle_is_positive"].astype(bool)
    zero = ~positive
    numeric_complete = bool(
        np.isfinite(ledger.select_dtypes(include=[np.number]).to_numpy(float)).all()
        and np.isfinite(
            changed_comparison.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()
    )
    sites_nonworse = int(
        site_summary["mean_selected_minus_anchor_gain_7d"].ge(-TOLERANCE).sum()
    )
    changed_adaptive_ok = bool(
        changed_count > 0
        and float(changed_comparison["selected_regret_adaptive_7d"].max())
        <= float(changed_comparison["anchor_regret_adaptive_7d"].max()) + TOLERANCE
    )
    conditions = {
        "all_execution_and_hash_audits_passed": bool(execution_complete),
        "all_numeric_outputs_finite": numeric_complete,
        "at_least_2_irrigation_decisions_changed": changed_count >= 2,
        "changed_irrigation_at_least_2_target_sites": changed_site_count >= 2,
        "at_least_4_of_5_site_mean_swap_gain_nonworse": sites_nonworse >= 4,
        "pooled_mean_swap_gain_nonworse": float(
            ledger["selected_minus_anchor_gain_7d"].mean()
        )
        >= -TOLERANCE,
        "positive_oracle_mean_swap_gain_nonworse": bool(positive.any())
        and float(ledger.loc[positive, "selected_minus_anchor_gain_7d"].mean())
        >= -TOLERANCE,
        "zero_oracle_mean_swap_gain_nonworse": bool(zero.any())
        and float(ledger.loc[zero, "selected_minus_anchor_gain_7d"].mean())
        >= -TOLERANCE,
        "overall_fixed_eight_regret_nonworse": float(
            ledger["selected_minus_anchor_regret_fixed_7d"].mean()
        )
        <= TOLERANCE,
        "changed_cycle_maximum_adaptive_regret_nonworse": changed_adaptive_ok,
        "overall_mean_peak_distance_nonworse": float(
            peak.loc["overall", "selected_mean_peak_distance_mm"]
        )
        <= float(peak.loc["overall", "anchor_mean_peak_distance_mm"]) + TOLERANCE,
        "positive_oracle_mean_peak_distance_nonworse": float(
            peak.loc["positive_oracle", "selected_mean_peak_distance_mm"]
        )
        <= float(peak.loc["positive_oracle", "anchor_mean_peak_distance_mm"])
        + TOLERANCE,
        "zero_oracle_mean_peak_distance_nonworse": float(
            peak.loc["zero_oracle", "selected_mean_peak_distance_mm"]
        )
        <= float(peak.loc["zero_oracle", "anchor_mean_peak_distance_mm"])
        + TOLERANCE,
        "global_maximum_peak_distance_nonworse": float(
            peak.loc["overall", "selected_max_peak_distance_mm"]
        )
        <= float(peak.loc["overall", "anchor_max_peak_distance_mm"]) + TOLERANCE,
    }
    observed = {
        "cycle_count": int(len(ledger)),
        "changed_cycle_count": changed_count,
        "changed_site_count": changed_site_count,
        "sites_with_nonworse_mean_swap_gain": sites_nonworse,
        "pooled_mean_selected_minus_anchor_swap_gain_7d": float(
            ledger["selected_minus_anchor_gain_7d"].mean()
        ),
        "positive_mean_selected_minus_anchor_swap_gain_7d": float(
            ledger.loc[positive, "selected_minus_anchor_gain_7d"].mean()
        ),
        "zero_mean_selected_minus_anchor_swap_gain_7d": float(
            ledger.loc[zero, "selected_minus_anchor_gain_7d"].mean()
        ),
        "anchor_changed_cycle_max_adaptive_regret_7d": (
            float(changed_comparison["anchor_regret_adaptive_7d"].max())
            if changed_count
            else None
        ),
        "selected_changed_cycle_max_adaptive_regret_7d": (
            float(changed_comparison["selected_regret_adaptive_7d"].max())
            if changed_count
            else None
        ),
        "anchor_overall_mean_peak_distance_mm": float(
            peak.loc["overall", "anchor_mean_peak_distance_mm"]
        ),
        "selected_overall_mean_peak_distance_mm": float(
            peak.loc["overall", "selected_mean_peak_distance_mm"]
        ),
        "anchor_global_maximum_peak_distance_mm": float(
            peak.loc["overall", "anchor_max_peak_distance_mm"]
        ),
        "selected_global_maximum_peak_distance_mm": float(
            peak.loc["overall", "selected_max_peak_distance_mm"]
        ),
    }
    return site_summary, peak_summary, {
        "protocol_id": PROTOCOL_ID,
        "gate_type": "predeclared_hash_separated_2019_peak_and_exact_paired_swap_confirmation_gate",
        "conditions": conditions,
        "observed": observed,
        "passed": bool(all(conditions.values())),
        "automatic_model_promotion": False,
        "passing_action": "authorize_2015_2019_final_refit_and_frozen_2024_test_protocol_design_only",
        "failing_action": "freeze_v6_2019_confirmation_as_negative_or_inconclusive",
    }


def _require_manifest_hash(
    manifest: dict[str, Any], section: str, name: str, path: Path
) -> None:
    expected = manifest.get(section, {}).get(name, {}).get("sha256")
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"Stage 1 manifest hash changed: {section}.{name}")


def verify_recommendation_freeze(
    protocol_path: Path,
    recommendation_dir: Path,
) -> dict[str, Path]:
    paths = {
        "recommendations": recommendation_dir / RECOMMENDATIONS_NAME,
        "models": recommendation_dir / FREEZE_MODELS_NAME,
        "freeze_gate": recommendation_dir / FREEZE_GATE_NAME,
        "freeze_audit": recommendation_dir / FREEZE_AUDIT_NAME,
        "freeze_manifest": recommendation_dir / FREEZE_MANIFEST_NAME,
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(f"Stage 1 artifacts are missing: {missing}")
    gate = read_json(paths["freeze_gate"])
    audit = read_json(paths["freeze_audit"])
    manifest = read_json(paths["freeze_manifest"])
    expected_status = (
        "gam_v6_2019_recommendations_frozen_pending_exact_paired_swap_confirmation"
    )
    if (
        audit.get("status") != expected_status
        or manifest.get("status") != expected_status
        or audit.get("mandatory_execution_gate_passed") is not True
        or audit.get("recommendations_frozen_before_2019_target_read") is not True
        or int(audit.get("2019_target_rows_read", -1)) != 0
        or int(audit.get("2019_swap_results_read", -1)) != 0
        or int(audit.get("2024_rows_read", -1)) != 0
        or gate.get("passed") is not True
    ):
        raise ValueError("Stage 1 freeze boundary is not valid")
    _require_manifest_hash(manifest, "inputs", "protocol", protocol_path)
    for name in ("recommendations", "models", "gate", "audit"):
        manifest_name = "freeze_gate" if name == "gate" else "freeze_audit" if name == "audit" else name
        _require_manifest_hash(manifest, "outputs", name, paths[manifest_name])
    return paths


def _truth_inventory(target_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cycle in target_frame.groupby(list(CYCLE_KEYS), sort=False):
        peak, gain = true_fixed_grid_peak(cycle)
        rows.append(
            {
                **dict(zip(CYCLE_KEYS, keys)),
                "true_peak_mm": float(peak),
                "true_fixed_list_net_gain_7d": float(gain),
                "true_oracle_is_positive": bool(peak > TOLERANCE),
            }
        )
    return pd.DataFrame(rows)


def _fixed_cycle_candidates(
    target_frame: pd.DataFrame, *, site_id: str, decision_date: str
) -> pd.DataFrame:
    selected = (
        target_frame["site_id"].astype(str).eq(site_id)
        & pd.to_datetime(target_frame["decision_date"]).dt.strftime("%Y-%m-%d").eq(
            decision_date
        )
    )
    frame = target_frame.loc[selected, ["irrigation_mm", "target_net_gain_7d"]].copy()
    if len(frame) != len(FIXED_IRRIGATION_MM):
        raise ValueError(f"fixed 2019 candidate grid is incomplete: {site_id} {decision_date}")
    frame = frame.rename(
        columns={"irrigation_mm": "ir", "target_net_gain_7d": "target_value"}
    )
    frame["evaluation_stage"] = "frozen_fixed_eight_2019_labels"
    return frame


def _split_exact_fixed_reuse(requested: Sequence[float]) -> tuple[list[float], list[float]]:
    reusable, missing = [], []
    for value in requested:
        target = reusable if any(
            np.isclose(float(value), option, rtol=0.0, atol=TOLERANCE)
            for option in FIXED_IRRIGATION_MM
        ) else missing
        target.append(float(value))
    return reusable, missing


def _build_changed_comparison(decisions: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    keys = ["site_id", "decision_date"]
    value_columns = [
        "continuous_irrigation_mm",
        "continuous_recommendation_swap_gain_7d",
        "regret_vs_fixed_eight_swap_7d",
        "regret_vs_adaptive_swap_oracle_7d",
    ]

    def role_frame(role: str, prefix: str) -> pd.DataFrame:
        frame = decisions.loc[decisions["model_role"].eq(role), [*keys, *value_columns]].copy()
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate changed SWAP role rows: {role}")
        return frame.rename(
            columns={
                "continuous_irrigation_mm": f"{prefix}_irrigation_mm",
                "continuous_recommendation_swap_gain_7d": f"{prefix}_gain_7d",
                "regret_vs_fixed_eight_swap_7d": f"{prefix}_regret_fixed_7d",
                "regret_vs_adaptive_swap_oracle_7d": f"{prefix}_regret_adaptive_7d",
            }
        )

    comparison = role_frame(ANCHOR_ROLE, "anchor").merge(
        role_frame(SELECTED_ROLE, "selected"), on=keys, validate="one_to_one"
    )
    info = plan.drop_duplicates(keys)[
        ["fold_id", "target_site", "target_year", "site_id", "decision_date"]
    ]
    comparison = comparison.merge(info, on=keys, validate="one_to_one")
    if len(comparison) * 2 != len(plan):
        raise ValueError("changed comparison does not cover the exact endpoint plan")
    numeric = comparison.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("changed comparison contains non-finite values")
    return comparison.sort_values(keys).reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    protocol_path = args.protocol.resolve()
    recommendation_dir = args.recommendation_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    formal_label_dir = args.formal_label_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    validate_execution_args(args, protocol)

    # This verification must complete before the first physical 2019 target read.
    freeze_paths = verify_recommendation_freeze(protocol_path, recommendation_dir)
    inventory = _normalize_inventory(pd.read_csv(freeze_paths["recommendations"]))
    dataset_path = dataset_dir / DATASET_NAME
    contract_path = dataset_dir / CONTRACT_NAME
    dataset_audit_path = dataset_dir / SOURCE_AUDIT_NAME
    if not all(path.is_file() for path in (dataset_path, contract_path, dataset_audit_path)):
        raise FileNotFoundError("frozen surrogate dataset inputs are incomplete")
    data_protocol = protocol["data_boundary"]
    freeze_audit = read_json(freeze_paths["freeze_audit"])
    if (
        sha256_file(dataset_path) != data_protocol["dataset_sha256"]
        or sha256_file(contract_path) != data_protocol["dataset_contract_sha256"]
        or freeze_audit.get("inputs", {}).get("dataset_sha256") != sha256_file(dataset_path)
        or freeze_audit.get("inputs", {}).get("dataset_contract_sha256")
        != sha256_file(contract_path)
    ):
        raise ValueError("frozen dataset hashes changed between Stage 1 and Stage 2")

    index = _dataset_index(dataset_path)
    years = pd.to_numeric(index["target_year"], errors="raise").astype(int)
    target_selector = years.eq(CONFIRMATION_YEAR)
    target_frame = _read_selected_rows(dataset_path, index, target_selector)
    validate_complete_cycles(target_frame)
    if len(target_frame) != EXPECTED_CANDIDATE_ROW_COUNT:
        raise ValueError("2019 target row count changed")
    scored = score_peak_inventory(inventory, _truth_inventory(target_frame))
    plan = build_changed_plan(scored)

    swap = protocol["swap"]
    weather_path = formal_label_dir / FORMAL_WEATHER_NAME
    if not weather_path.is_file():
        raise FileNotFoundError(weather_path)
    if sha256_file(weather_path) != swap["formal_weather_sha256"]:
        raise ValueError("frozen formal weather hash changed")
    weather = pd.read_csv(weather_path)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan_path = output_dir / "gefs_gam_v6_2019_changed_cycle_swap_plan_v1.csv"
    plan.to_csv(plan_path, index=False)

    decision_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    reuse_rows: list[dict[str, Any]] = []
    unique_cycles = plan.drop_duplicates(["site_id", "decision_date"])
    for cycle_index, cycle in enumerate(unique_cycles.itertuples(index=False), start=1):
        site_id = str(cycle.site_id)
        decision_date = str(cycle.decision_date)
        recommendations = plan.loc[
            plan["site_id"].astype(str).eq(site_id)
            & plan["decision_date"].astype(str).eq(decision_date)
        ].copy()
        print(
            f"cycle={cycle_index}/{len(unique_cycles)} site={site_id} date={decision_date}",
            flush=True,
        )
        requested = recommendations["continuous_irrigation_mm"].astype(float).tolist()
        reusable, missing = _split_exact_fixed_reuse(requested)
        fixed = _fixed_cycle_candidates(
            target_frame, site_id=site_id, decision_date=decision_date
        )
        new_candidates = pd.DataFrame()
        stage_root = ""
        if missing:
            source_workspace, checkpoint_dir, checkpoint_audit = unit_resources(
                formal_label_dir, site_id, decision_date, CONFIRMATION_YEAR
            )
            requested_with_zero = sorted(set([0.0, *missing]))
            new_candidates, generated_root = run_swap_stage(
                base_root=(
                    output_dir
                    / "exact_pair_branches"
                    / site_id
                    / pd.Timestamp(decision_date).strftime("%Y%m%d")
                    / "missing_exact_recommendations_with_zero_reference"
                ),
                requested_values=requested_with_zero,
                source_workspace=source_workspace,
                checkpoint_dir=checkpoint_dir,
                checkpoint_audit=checkpoint_audit,
                weather=weather,
                site_id=site_id,
                decision_date=decision_date,
                resume=bool(args.resume),
                restart_nprintday=int(args.restart_nprintday),
                swap_dtmax_days=float(args.swap_dtmax_days),
                fallback_offsets_mm=tuple(
                    float(value) for value in swap["fallback_offsets_mm"]
                ),
                target_year=CONFIRMATION_YEAR,
                zero_irrigation_cwdm_reference=None,
            )
            new_candidates = new_candidates.assign(
                evaluation_stage="2019_missing_exact_recommendations"
            )
            stage_root = str(generated_root)
            new_zero = new_candidates.loc[
                np.isclose(new_candidates["ir"].astype(float), 0.0, atol=TOLERANCE)
            ]
            fixed_zero = fixed.loc[np.isclose(fixed["ir"].astype(float), 0.0, atol=TOLERANCE)]
            if (
                len(new_zero) != 1
                or len(fixed_zero) != 1
                or not np.isclose(
                    float(new_zero["target_value"].iloc[0]),
                    float(fixed_zero["target_value"].iloc[0]),
                    rtol=0.0,
                    atol=TOLERANCE,
                )
            ):
                raise ValueError(f"2019 zero-irrigation gain mismatch: {site_id} {decision_date}")
        combined = pd.concat([new_candidates, fixed], ignore_index=True, sort=False)
        rows, summary = evaluate_cycle(combined, recommendations)
        decision_rows.extend(rows)
        oracle_rows.append(summary)
        combined.insert(0, "confirmation_cycle_id", f"{site_id}_{decision_date}")
        candidate_frames.append(combined)
        reuse_rows.append(
            {
                "site_id": site_id,
                "decision_date": decision_date,
                "exact_fixed_endpoint_reuse_count": len(reusable),
                "new_exact_endpoint_count": len(missing),
                "new_zero_reference_run_count": 1 if missing else 0,
                "exact_fixed_endpoint_values_mm": json.dumps(reusable),
                "new_exact_endpoint_values_mm": json.dumps(missing),
                "stage_root": stage_root,
                "mandatory_execution_gate_passed": True,
            }
        )

    decision_columns = [
        "model_id",
        "model_role",
        "site_id",
        "decision_date",
        "continuous_irrigation_mm",
        "continuous_recommendation_swap_gain_7d",
        "fixed_eight_swap_oracle_irrigation_mm",
        "fixed_eight_swap_oracle_gain_7d",
        "regret_vs_fixed_eight_swap_7d",
        "adaptive_swap_oracle_irrigation_mm",
        "adaptive_swap_oracle_gain_7d",
        "regret_vs_adaptive_swap_oracle_7d",
        "continuous_beats_fixed_eight",
    ]
    decisions = pd.DataFrame(decision_rows, columns=decision_columns)
    comparison = _build_changed_comparison(decisions, plan)
    ledger = build_full_ledger(scored, comparison)
    reuse_audit = pd.DataFrame(
        reuse_rows,
        columns=[
            "site_id",
            "decision_date",
            "exact_fixed_endpoint_reuse_count",
            "new_exact_endpoint_count",
            "new_zero_reference_run_count",
            "exact_fixed_endpoint_values_mm",
            "new_exact_endpoint_values_mm",
            "stage_root",
            "mandatory_execution_gate_passed",
        ],
    )
    expected_decisions = 2 * int(scored["use_v6"].sum())
    execution_complete = bool(
        len(decisions) == expected_decisions
        and len(reuse_audit) == int(scored["use_v6"].sum())
        and (reuse_audit.empty or reuse_audit["mandatory_execution_gate_passed"].all())
    )
    site_summary, peak_summary, gate = evaluate_gate(
        ledger, comparison, execution_complete=execution_complete
    )

    outputs = {
        "plan": plan_path,
        "peak_inventory": output_dir / "gefs_gam_v6_2019_peak_inventory_v1.csv",
        "decisions": output_dir / "gefs_gam_v6_2019_changed_cycle_swap_decisions_v1.csv",
        "changed_comparison": output_dir / "gefs_gam_v6_2019_changed_cycle_swap_comparison_v1.csv",
        "full_ledger": output_dir / "gefs_gam_v6_2019_full_paired_ledger_v1.csv",
        "site_summary": output_dir / "gefs_gam_v6_2019_site_summary_v1.csv",
        "peak_summary": output_dir / "gefs_gam_v6_2019_peak_summary_v1.csv",
        "reuse_audit": output_dir / "gefs_gam_v6_2019_swap_reuse_audit_v1.csv",
        "cycle_oracles": output_dir / "gefs_gam_v6_2019_changed_cycle_oracles_v1.csv",
        "candidates": output_dir / "gefs_gam_v6_2019_changed_cycle_candidates_v1.csv",
        "gate": output_dir / "gefs_gam_v6_2019_confirmation_gate_v1.json",
        "audit": output_dir / "gefs_gam_v6_2019_confirmation_audit_v1.json",
        "manifest": output_dir / "gefs_gam_v6_2019_confirmation_manifest_v1.json",
    }
    scored.to_csv(outputs["peak_inventory"], index=False)
    decisions.to_csv(outputs["decisions"], index=False)
    comparison.to_csv(outputs["changed_comparison"], index=False)
    ledger.to_csv(outputs["full_ledger"], index=False)
    site_summary.to_csv(outputs["site_summary"], index=False)
    peak_summary.to_csv(outputs["peak_summary"], index=False)
    reuse_audit.to_csv(outputs["reuse_audit"], index=False)
    pd.DataFrame(oracle_rows).to_csv(outputs["cycle_oracles"], index=False)
    if candidate_frames:
        pd.concat(candidate_frames, ignore_index=True).to_csv(outputs["candidates"], index=False)
    else:
        pd.DataFrame(columns=["confirmation_cycle_id", "ir", "target_value", "evaluation_stage"]).to_csv(
            outputs["candidates"], index=False
        )
    write_json(outputs["gate"], gate)
    audit = {
        "status": (
            "gam_v6_2019_independent_confirmation_passed_pending_final_refit_2024_protocol_design"
            if gate["passed"]
            else "gam_v6_2019_independent_confirmation_failed_or_inconclusive"
        ),
        "protocol_id": PROTOCOL_ID,
        "mandatory_execution_gate_passed": execution_complete,
        "predeclared_performance_gate_passed": bool(gate["passed"]),
        "stage_1_manifest_verified_before_2019_target_read": True,
        "recommendation_selection_after_2019_target_read": False,
        "2019_target_rows_read": EXPECTED_CANDIDATE_ROW_COUNT,
        "2019_target_cycles_scored": EXPECTED_CYCLE_COUNT,
        "2024_rows_read": 0,
        "changed_cycle_count": int(scored["use_v6"].sum()),
        "unchanged_cycle_exact_zero_difference_count": int((~scored["use_v6"]).sum()),
        "interpolation_used": False,
        "gam_bspline_refit_performed_in_stage_2": False,
        "gate_training_performed_in_stage_2": False,
        "automatic_model_promotion": False,
        "independent_of_v6_rule_and_threshold_selection": True,
        "pristine_final_generalization_test": False,
        "earlier_2019_context_was_inspected": True,
        "final_untouched_test_year": SEALED_TEST_YEAR,
        "next_gate": gate["passing_action"] if gate["passed"] else gate["failing_action"],
        "inputs": {
            "protocol_sha256": sha256_file(protocol_path),
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_contract_sha256": sha256_file(contract_path),
            "dataset_audit_sha256": sha256_file(dataset_audit_path),
            "formal_weather_sha256": sha256_file(weather_path),
            **{name: sha256_file(path) for name, path in freeze_paths.items()},
        },
    }
    write_json(outputs["audit"], audit)
    write_json(
        outputs["manifest"],
        {
            "status": audit["status"],
            "inputs": {
                "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
                "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
                "dataset_contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
                "dataset_audit": {"path": str(dataset_audit_path), "sha256": sha256_file(dataset_audit_path)},
                "formal_weather": {"path": str(weather_path), "sha256": sha256_file(weather_path)},
                **{
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in freeze_paths.items()
                },
            },
            "outputs": {
                name: {"path": path.name, "sha256": sha256_file(path)}
                for name, path in outputs.items()
                if name != "manifest"
            },
        },
    )
    if not execution_complete:
        raise RuntimeError(f"2019 exact SWAP execution gate failed; see {outputs['audit']}")
    if not gate["passed"]:
        raise RuntimeError(f"2019 joint confirmation gate failed; see {outputs['gate']}")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--recommendation-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--formal-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--restart-nprintday", type=int, default=48)
    parser.add_argument("--swap-dtmax-days", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    for output_name, output_path in run(parse_args()).items():
        print(f"{output_name}: {output_path}", flush=True)

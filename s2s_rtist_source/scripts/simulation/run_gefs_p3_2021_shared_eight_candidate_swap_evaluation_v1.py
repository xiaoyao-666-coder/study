#!/usr/bin/env python3
"""Generate one shared P3 2021 SWAP label table and evaluate frozen routers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation.run_gefs_controlled_continuous_swap_evaluation_v1 import (
    run_swap_stage,
)
from scripts.simulation.run_gefs_checkpoint_one_date_eight_ir_smoke_v1 import (
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECOMMENDATION_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2021_A_B_frozen_recommendations_v1"
)
DEFAULT_TRUNK_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2021_era5_trunk_schedule_freeze_v1"
)
DEFAULT_WEATHER_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2021_five_member_frozen_weather_correction_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_p3_2021_shared_eight_candidate_swap_evaluation_v1"
)

TARGET_SITE = "P3"
TARGET_YEAR = 2021
SOWING_MONTH_DAY = "04-26"
FIXED_IRRIGATION_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
POLICY_ROLES = {
    "B_source_only_router": "formal_primary_P3_label_isolated",
    "A_P3_historical_supervised_router": "formal_secondary_P3_calibrated",
    "always_P15": "formal_baseline",
    "always_P1": "diagnostic_baseline",
}
FORMAL_PRIMARY = "B_source_only_router"
FORMAL_SECONDARY = "A_P3_historical_supervised_router"
FORMAL_BASELINE = "always_P15"

RECOMMENDATION_AUDIT_NAME = "gefs_p3_2021_A_B_recommendation_freeze_audit_v1.json"
RECOMMENDATION_MANIFEST_NAME = "gefs_p3_2021_A_B_recommendation_freeze_manifest_v1.csv"
RECOMMENDATION_NAME = "gefs_p3_2021_A_B_policy_recommendations_v1.csv"
TRUNK_AUDIT_NAME = "gefs_p3_2021_era5_trunk_schedule_freeze_audit_v1.json"
WEATHER_AUDIT_NAME = "gefs_p3_2021_frozen_weather_correction_audit_v1.json"
WEATHER_NAME = "gefs_p3_2021_five_member_frozen_corrected_weather_v1.csv"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def boolean_mask(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            bool(value)
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() in {"1", "true", "yes"}
        )
    )


def fixed_grid_matches(values: pd.Series) -> bool:
    observed = sorted(pd.to_numeric(values, errors="raise").astype(float).tolist())
    return len(observed) == len(FIXED_IRRIGATION_MM) and np.allclose(
        observed,
        FIXED_IRRIGATION_MM,
        rtol=0.0,
        atol=1.0e-9,
    )


def verify_recommendation_manifest(
    recommendation_dir: Path, recommendation_path: Path
) -> bool:
    manifest = pd.read_csv(recommendation_dir / RECOMMENDATION_MANIFEST_NAME)
    required = {"role", "path", "sha256"}
    if missing := sorted(required - set(manifest.columns)):
        raise ValueError(f"recommendation manifest is missing fields: {missing}")
    rows = manifest.loc[manifest["role"].eq("output_policy_recommendations")]
    if len(rows) != 1:
        raise ValueError("recommendation manifest must bind one policy table")
    row = rows.iloc[0]
    return bool(
        Path(str(row["path"])).name == recommendation_path.name
        and str(row["sha256"]) == sha256_file(recommendation_path)
    )


def validate_frozen_inputs(
    recommendation_dir: Path,
    trunk_dir: Path,
    weather_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recommendation_audit_path = recommendation_dir / RECOMMENDATION_AUDIT_NAME
    recommendation_manifest_path = recommendation_dir / RECOMMENDATION_MANIFEST_NAME
    recommendation_path = recommendation_dir / RECOMMENDATION_NAME
    trunk_audit_path = trunk_dir / TRUNK_AUDIT_NAME
    weather_audit_path = weather_dir / WEATHER_AUDIT_NAME
    weather_path = weather_dir / WEATHER_NAME
    unit = trunk_dir / "units" / "Y2021" / TARGET_SITE
    schedule_path = unit / "trunk" / "swap_season_decision_schedule_v1.csv"
    checkpoint_audit_path = (
        unit
        / "all_checkpoints_v1"
        / "swap_season_checkpoint_equivalence_v1.csv"
    )
    required_paths = [
        recommendation_audit_path,
        recommendation_manifest_path,
        recommendation_path,
        trunk_audit_path,
        weather_audit_path,
        weather_path,
        schedule_path,
        checkpoint_audit_path,
        unit / "workspace",
        unit / "all_checkpoints_v1" / "checkpoints",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"P3 2021 evaluation inputs are missing: {missing}")

    recommendation_audit = read_json(recommendation_audit_path)
    trunk_audit = read_json(trunk_audit_path)
    weather_audit = read_json(weather_audit_path)
    manifest_verified = verify_recommendation_manifest(
        recommendation_dir, recommendation_path
    )
    frozen_gate = bool(
        recommendation_audit.get("mandatory_gate_passed") is True
        and recommendation_audit.get("status")
        == "p3_2021_A_B_recommendations_frozen_pending_shared_independent_SWAP_labels"
        and int(recommendation_audit.get("2021_SWAP_candidate_labels_read", -1)) == 0
        and recommendation_audit.get("SWAP_simulation_performed") is False
        and recommendation_audit.get("model_training_or_reselection_performed")
        is False
        and recommendation_audit.get("formal_primary_experiment") == FORMAL_PRIMARY
        and recommendation_audit.get("formal_secondary_experiment")
        == FORMAL_SECONDARY
        and recommendation_audit.get("formal_baseline") == FORMAL_BASELINE
        and manifest_verified
    )
    if not frozen_gate:
        raise ValueError("frozen A/B recommendation gate is not valid")
    if not (
        trunk_audit.get("mandatory_gate_passed") is True
        and trunk_audit.get("target_site") == TARGET_SITE
        and int(trunk_audit.get("target_year", -1)) == TARGET_YEAR
        and int(trunk_audit.get("SWAP_candidate_label_rows_read", -1)) == 0
        and trunk_audit.get("SWAP_candidate_label_generation_performed") is False
    ):
        raise ValueError("P3 2021 trunk gate is not valid")
    if not (
        weather_audit.get("mandatory_gate_passed") is True
        and weather_audit.get("target_site") == TARGET_SITE
        and int(weather_audit.get("target_year", -1)) == TARGET_YEAR
        and int(weather_audit.get("2021_SWAP_candidate_labels_read", -1)) == 0
        and weather_audit.get("model_inference_performed") is False
    ):
        raise ValueError("P3 2021 corrected-weather gate is not valid")

    recommendations = pd.read_csv(recommendation_path)
    required_recommendation_fields = {
        "experiment_id",
        "site_id",
        "decision_date",
        "gate_probability_p1",
        "selected_source_site_id",
        "selected_irrigation_mm",
        "selected_predicted_net_gain_7d",
    }
    if missing := sorted(required_recommendation_fields - set(recommendations.columns)):
        raise ValueError(f"frozen recommendations are missing fields: {missing}")
    recommendations["decision_date"] = pd.to_datetime(
        recommendations["decision_date"]
    ).dt.strftime("%Y-%m-%d")
    if not (
        len(recommendations) == 48
        and set(recommendations["experiment_id"].astype(str)) == set(POLICY_ROLES)
        and recommendations.groupby("experiment_id").size().eq(12).all()
        and recommendations["site_id"].astype(str).eq(TARGET_SITE).all()
        and pd.to_datetime(recommendations["decision_date"]).dt.year.eq(TARGET_YEAR).all()
        and recommendations["selected_irrigation_mm"].map(
            lambda value: any(
                np.isclose(float(value), option, rtol=0.0, atol=1.0e-9)
                for option in FIXED_IRRIGATION_MM
            )
        ).all()
    ):
        raise ValueError("frozen policy table does not contain four valid 12-cycle policies")

    schedule = pd.read_csv(schedule_path)
    schedule["decision_date"] = pd.to_datetime(schedule["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    checkpoint_audit = pd.read_csv(checkpoint_audit_path)
    if not (
        len(schedule) == 12
        and schedule["site_id"].astype(str).eq(TARGET_SITE).all()
        and set(schedule["decision_date"])
        == set(recommendations["decision_date"])
        and len(checkpoint_audit) == 12
        and boolean_mask(checkpoint_audit["checkpoint_equivalence_passed"]).all()
    ):
        raise ValueError("frozen schedule/checkpoint scope differs from recommendations")

    weather = pd.read_csv(weather_path)
    weather["decision_date"] = pd.to_datetime(weather["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    if not (
        len(weather) == 420
        and weather["site_id"].astype(str).eq(TARGET_SITE).all()
        and set(weather["decision_date"]) == set(schedule["decision_date"])
    ):
        raise ValueError("corrected weather does not cover the frozen 12-cycle schedule")

    metadata = {
        "recommendation_audit": recommendation_audit_path,
        "recommendation_manifest": recommendation_manifest_path,
        "recommendations": recommendation_path,
        "trunk_audit": trunk_audit_path,
        "weather_audit": weather_audit_path,
        "weather": weather_path,
        "schedule": schedule_path,
        "checkpoint_audit": checkpoint_audit_path,
        "unit": unit,
        "recommendation_manifest_hash_verified": manifest_verified,
    }
    return recommendations, schedule, weather, metadata


def unit_resources(unit: Path, decision_date: str) -> tuple[Path, Path, Path]:
    checkpoint_root = unit / "all_checkpoints_v1"
    checkpoint_dir = (
        checkpoint_root
        / "checkpoints"
        / pd.Timestamp(decision_date).strftime("%Y%m%d")
    )
    return (
        unit / "workspace",
        checkpoint_dir,
        checkpoint_root / "swap_season_checkpoint_equivalence_v1.csv",
    )


def normalize_cycle_labels(candidates: pd.DataFrame, decision_date: str) -> pd.DataFrame:
    labels = candidates.copy()
    required = {
        "ir",
        "target_value",
        "water_balance_residual_0_100cm_7d_mm",
        "requested_ir_mm",
        "simulated_ir_mm",
        "numerical_endpoint_fallback",
        "numerical_endpoint_fallback_delta_mm",
    }
    if missing := sorted(required - set(labels.columns)):
        raise ValueError(f"SWAP candidate output is missing fields: {missing}")
    labels.insert(0, "evaluation_cycle_id", f"{TARGET_SITE}_{decision_date}")
    labels["site_id"] = TARGET_SITE
    labels["decision_date"] = decision_date
    labels["irrigation_mm"] = pd.to_numeric(labels["ir"], errors="raise").astype(float)
    labels["swap_gain_7d"] = pd.to_numeric(
        labels["target_value"], errors="raise"
    ).astype(float)
    return labels


def evaluate_frozen_policies(
    labels: pd.DataFrame, recommendations: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for decision_date, cycle_labels in labels.groupby("decision_date", sort=True):
        if not fixed_grid_matches(cycle_labels["irrigation_mm"]):
            raise ValueError(f"{decision_date}: shared SWAP labels do not contain the fixed grid")
        ordered = cycle_labels.sort_values(
            ["swap_gain_7d", "irrigation_mm"], ascending=[False, True]
        )
        oracle = ordered.iloc[0]
        cycle_recommendations = recommendations.loc[
            recommendations["decision_date"].eq(str(decision_date))
        ]
        for recommendation in cycle_recommendations.itertuples(index=False):
            selected_irrigation = float(recommendation.selected_irrigation_mm)
            matches = np.isclose(
                cycle_labels["irrigation_mm"].to_numpy(float),
                selected_irrigation,
                rtol=0.0,
                atol=1.0e-9,
            )
            if int(matches.sum()) != 1:
                raise ValueError(
                    f"{decision_date}: selected irrigation is not unique in shared labels"
                )
            selected = cycle_labels.loc[matches].iloc[0]
            rows.append(
                {
                    "experiment_id": recommendation.experiment_id,
                    "policy_role": POLICY_ROLES[recommendation.experiment_id],
                    "site_id": TARGET_SITE,
                    "decision_date": str(decision_date),
                    "gate_probability_p1": float(recommendation.gate_probability_p1),
                    "selected_source_site_id": recommendation.selected_source_site_id,
                    "selected_irrigation_mm": selected_irrigation,
                    "selected_predicted_net_gain_7d": float(
                        recommendation.selected_predicted_net_gain_7d
                    ),
                    "selected_swap_gain_7d": float(selected["swap_gain_7d"]),
                    "fixed_eight_swap_oracle_irrigation_mm": float(
                        oracle["irrigation_mm"]
                    ),
                    "fixed_eight_swap_oracle_gain_7d": float(oracle["swap_gain_7d"]),
                    "regret_vs_fixed_eight_swap_7d": float(
                        oracle["swap_gain_7d"] - selected["swap_gain_7d"]
                    ),
                }
            )
    decisions = pd.DataFrame(rows).sort_values(
        ["experiment_id", "decision_date"]
    ).reset_index(drop=True)
    baseline = decisions.loc[
        decisions["experiment_id"].eq(FORMAL_BASELINE),
        ["decision_date", "selected_swap_gain_7d", "regret_vs_fixed_eight_swap_7d"],
    ].rename(
        columns={
            "selected_swap_gain_7d": "always_P15_swap_gain_7d",
            "regret_vs_fixed_eight_swap_7d": "always_P15_regret_7d",
        }
    )
    decisions = decisions.merge(baseline, on="decision_date", how="left", validate="many_to_one")
    decisions["swap_gain_delta_vs_always_P15_7d"] = (
        decisions["selected_swap_gain_7d"] - decisions["always_P15_swap_gain_7d"]
    )
    decisions["regret_reduction_vs_always_P15_7d"] = (
        decisions["always_P15_regret_7d"]
        - decisions["regret_vs_fixed_eight_swap_7d"]
    )
    summary = decisions.groupby(
        ["experiment_id", "policy_role"], as_index=False
    ).agg(
        cycle_count=("decision_date", "size"),
        mean_selected_irrigation_mm=("selected_irrigation_mm", "mean"),
        diagnostic_selected_60mm_count=(
            "selected_irrigation_mm",
            lambda values: int(np.isclose(values.astype(float), 60.0).sum()),
        ),
        mean_selected_swap_gain_7d=("selected_swap_gain_7d", "mean"),
        mean_regret_vs_fixed_eight_swap_7d=(
            "regret_vs_fixed_eight_swap_7d",
            "mean",
        ),
        maximum_regret_vs_fixed_eight_swap_7d=(
            "regret_vs_fixed_eight_swap_7d",
            "max",
        ),
        mean_swap_gain_delta_vs_always_P15_7d=(
            "swap_gain_delta_vs_always_P15_7d",
            "mean",
        ),
        mean_regret_reduction_vs_always_P15_7d=(
            "regret_reduction_vs_always_P15_7d",
            "mean",
        ),
        cycles_better_than_always_P15=(
            "swap_gain_delta_vs_always_P15_7d",
            lambda values: int((values > 1.0e-9).sum()),
        ),
        cycles_equal_to_always_P15=(
            "swap_gain_delta_vs_always_P15_7d",
            lambda values: int(np.isclose(values, 0.0, rtol=0.0, atol=1.0e-9).sum()),
        ),
        cycles_worse_than_always_P15=(
            "swap_gain_delta_vs_always_P15_7d",
            lambda values: int((values < -1.0e-9).sum()),
        ),
    )
    return decisions, summary


def run(args: argparse.Namespace) -> dict[str, Path]:
    recommendation_dir = args.recommendation_dir.resolve()
    trunk_dir = args.trunk_dir.resolve()
    weather_dir = args.weather_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)

    recommendations, schedule, weather, metadata = validate_frozen_inputs(
        recommendation_dir, trunk_dir, weather_dir
    )
    cycle_plan = schedule[["decision_date"]].sort_values("decision_date")
    if args.max_cycles is not None:
        if args.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        cycle_plan = cycle_plan.head(args.max_cycles)
    selected_dates = set(cycle_plan["decision_date"].astype(str))
    evaluation_recommendations = recommendations.loc[
        recommendations["decision_date"].isin(selected_dates)
    ].copy()

    label_frames: list[pd.DataFrame] = []
    branch_roots: list[Path] = []
    for index, cycle in enumerate(cycle_plan.itertuples(index=False), start=1):
        decision_date = str(cycle.decision_date)
        print(
            f"cycle={index}/{len(cycle_plan)} site={TARGET_SITE} date={decision_date}",
            flush=True,
        )
        source_workspace, checkpoint_dir, checkpoint_audit = unit_resources(
            metadata["unit"], decision_date
        )
        branch_root = (
            output_dir
            / "branches"
            / TARGET_SITE
            / pd.Timestamp(decision_date).strftime("%Y%m%d")
            / "fixed_eight"
        )
        candidates, used_root = run_swap_stage(
            base_root=branch_root,
            requested_values=list(FIXED_IRRIGATION_MM),
            source_workspace=source_workspace,
            checkpoint_dir=checkpoint_dir,
            checkpoint_audit=checkpoint_audit,
            weather=weather,
            site_id=TARGET_SITE,
            decision_date=decision_date,
            resume=args.resume,
            restart_nprintday=args.restart_nprintday,
            swap_dtmax_days=args.swap_dtmax_days,
            target_year=TARGET_YEAR,
            sowing_month_day=SOWING_MONTH_DAY,
        )
        label_frames.append(normalize_cycle_labels(candidates, decision_date))
        branch_roots.append(used_root)

    labels = pd.concat(label_frames, ignore_index=True)
    decisions, summary = evaluate_frozen_policies(
        labels, evaluation_recommendations
    )
    full_run = len(cycle_plan) == 12
    expected_label_rows = 8 * len(cycle_plan)
    expected_decision_rows = len(POLICY_ROLES) * len(cycle_plan)
    fixed_grid_per_cycle = labels.groupby("decision_date")["irrigation_mm"].apply(
        fixed_grid_matches
    )
    finite_decisions = np.isfinite(
        decisions[
            [
                "selected_swap_gain_7d",
                "fixed_eight_swap_oracle_gain_7d",
                "regret_vs_fixed_eight_swap_7d",
                "swap_gain_delta_vs_always_P15_7d",
            ]
        ].to_numpy(float)
    ).all()
    fallback_mask = boolean_mask(labels["numerical_endpoint_fallback"])
    fallback_rows = labels.loc[fallback_mask]
    fallback_delta = pd.to_numeric(
        labels["numerical_endpoint_fallback_delta_mm"], errors="raise"
    ).astype(float)
    gates = {
        "frozen_recommendation_manifest_hash_verified": bool(
            metadata["recommendation_manifest_hash_verified"]
        ),
        "shared_label_rows_expected": len(labels) == expected_label_rows,
        "one_row_per_cycle_candidate": not labels[
            ["site_id", "decision_date", "irrigation_mm"]
        ].duplicated().any(),
        "fixed_eight_grid_each_cycle": bool(fixed_grid_per_cycle.all()),
        "four_policies_use_same_cycle_labels": len(decisions)
        == expected_decision_rows,
        "finite_policy_metrics": bool(finite_decisions),
        "nonnegative_fixed_eight_regret": bool(
            (decisions["regret_vs_fixed_eight_swap_7d"] >= -1.0e-6).all()
        ),
        "formal_primary_present": int(
            decisions["experiment_id"].eq(FORMAL_PRIMARY).sum()
        )
        == len(cycle_plan),
        "formal_secondary_present": int(
            decisions["experiment_id"].eq(FORMAL_SECONDARY).sum()
        )
        == len(cycle_plan),
        "formal_baseline_present": int(
            decisions["experiment_id"].eq(FORMAL_BASELINE).sum()
        )
        == len(cycle_plan),
    }
    mandatory = bool(all(gates.values()))
    outputs = {
        "labels": output_dir
        / "gefs_p3_2021_shared_eight_candidate_swap_labels_v1.csv",
        "decisions": output_dir
        / "gefs_p3_2021_frozen_policy_swap_decisions_v1.csv",
        "summary": output_dir
        / "gefs_p3_2021_frozen_policy_swap_summary_v1.csv",
        "audit": output_dir
        / "gefs_p3_2021_shared_eight_candidate_swap_evaluation_audit_v1.json",
        "manifest": output_dir
        / "gefs_p3_2021_shared_eight_candidate_swap_evaluation_manifest_v1.json",
    }
    labels.to_csv(outputs["labels"], index=False)
    decisions.to_csv(outputs["decisions"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    frozen_recommendation_audit = read_json(metadata["recommendation_audit"])
    audit = {
        "status": (
            "p3_2021_shared_eight_candidate_swap_evaluation_complete_passed"
            if mandatory and full_run
            else "p3_2021_shared_eight_candidate_swap_evaluation_partial_passed"
            if mandatory
            else "p3_2021_shared_eight_candidate_swap_evaluation_failed"
        ),
        "mandatory_gate_passed": mandatory,
        "full_run": full_run,
        "target_site": TARGET_SITE,
        "target_year": TARGET_YEAR,
        "evaluated_cycle_count": int(len(cycle_plan)),
        "shared_candidate_label_rows": int(len(labels)),
        "policy_decision_rows": int(len(decisions)),
        "formal_primary_experiment": FORMAL_PRIMARY,
        "formal_secondary_experiment": FORMAL_SECONDARY,
        "formal_baseline": FORMAL_BASELINE,
        "diagnostic_baseline": "always_P1",
        "irrigation_candidates_mm": list(FIXED_IRRIGATION_MM),
        "single_shared_label_table_used_for_all_policies": True,
        "recommendations_frozen_before_2021_SWAP_labels": True,
        "experiment_B_P3_rows_used_in_gate_fit": 0,
        "experiment_A_uses_P3_2015_2019_gate_labels": True,
        "source_checkpoint_sha256": frozen_recommendation_audit.get(
            "source_checkpoint_sha256"
        ),
        "gate_sha256": frozen_recommendation_audit.get("gate_sha256"),
        "formal_metric": "SWAP_fixed_eight_discrete_regret_7d",
        "exact_irrigation_match_metric_reported": False,
        "interpolation_used_for_formal_metrics": False,
        "continuous_oracle_evaluated_in_this_stage": False,
        "candidate_labels_training_eligible": False,
        "model_checkpoint_files_loaded_by_this_stage": 0,
        "model_training_or_reselection_performed": False,
        "gate_or_threshold_reselection_performed": False,
        "network_access_performed": False,
        "2024_rows_read": 0,
        "restart_nprintday": int(args.restart_nprintday),
        "swap_dtmax_days": float(args.swap_dtmax_days),
        "numerical_endpoint_fallback_count": int(fallback_mask.sum()),
        "numerical_endpoint_fallback_requested_values_mm": sorted(
            pd.to_numeric(
                fallback_rows.get("requested_ir_mm"), errors="coerce"
            )
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        if not fallback_rows.empty
        else [],
        "numerical_endpoint_fallback_simulated_values_mm": sorted(
            pd.to_numeric(
                fallback_rows.get("simulated_ir_mm"), errors="coerce"
            )
            .dropna()
            .astype(float)
            .unique()
            .tolist()
        )
        if not fallback_rows.empty
        else [],
        "maximum_absolute_numerical_endpoint_adjustment_mm": (
            float(fallback_delta.loc[fallback_mask].abs().max())
            if fallback_mask.any()
            else 0.0
        ),
        "maximum_absolute_water_balance_residual_mm": float(
            pd.to_numeric(
                labels["water_balance_residual_0_100cm_7d_mm"], errors="raise"
            )
            .abs()
            .max()
        ),
        "shared_label_table_sha256": sha256_file(outputs["labels"]),
        "branch_stage_roots": [str(path) for path in branch_roots],
        "gates": gates,
        "next_gate": (
            "review_primary_B_vs_always_P15_and_secondary_A_without_2021_reselection"
            if mandatory and full_run
            else "resume_shared_2021_SWAP_label_generation"
            if mandatory
            else "repair_shared_2021_SWAP_evaluation"
        ),
    }
    write_json(outputs["audit"], audit)
    input_paths = {
        "frozen_recommendation_audit": metadata["recommendation_audit"],
        "frozen_recommendation_manifest": metadata["recommendation_manifest"],
        "frozen_policy_recommendations": metadata["recommendations"],
        "frozen_trunk_audit": metadata["trunk_audit"],
        "frozen_weather_audit": metadata["weather_audit"],
        "frozen_corrected_weather": metadata["weather"],
        "frozen_decision_schedule": metadata["schedule"],
        "frozen_checkpoint_audit": metadata["checkpoint_audit"],
    }
    manifest = {
        "status": audit["status"],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    write_json(outputs["manifest"], manifest)
    if not mandatory:
        raise RuntimeError(f"shared P3 2021 SWAP evaluation failed; see {outputs['audit']}")
    print(
        "completed shared P3 2021 fixed-eight SWAP evaluation "
        f"cycles={len(cycle_plan)} labels={len(labels)} policies={len(POLICY_ROLES)}",
        flush=True,
    )
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recommendation-dir", type=Path, default=DEFAULT_RECOMMENDATION_DIR
    )
    parser.add_argument("--trunk-dir", type=Path, default=DEFAULT_TRUNK_DIR)
    parser.add_argument("--weather-dir", type=Path, default=DEFAULT_WEATHER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--restart-nprintday", type=int, default=48)
    parser.add_argument("--swap-dtmax-days", type=float, default=0.01)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({name: str(path) for name, path in generated.items()}, indent=2))

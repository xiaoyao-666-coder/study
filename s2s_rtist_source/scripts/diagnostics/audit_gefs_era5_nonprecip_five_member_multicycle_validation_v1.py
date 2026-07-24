#!/usr/bin/env python3
"""Run formal multicycle five-member validation for causal nonprecipitation correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_application_smoke_v1 import (
    BRANCH_RAW,
    BRANCH_SELECTED_AFFINE_SOLAR,
    BRANCH_SELECTED_RAW_SOLAR,
    build_validation,
)
from scripts.diagnostics.run_gefs_era5_nonprecip_causal_correction_v1 import VARIABLES


def aggregate_cycle_metrics(cycle_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (branch_id, variable), group in cycle_metrics.groupby(
        ["branch_id", "variable"], sort=False
    ):
        weights = group["sample_count"].to_numpy(dtype=float)
        total = float(weights.sum())
        rows.append(
            {
                "branch_id": branch_id,
                "variable": variable,
                "sample_count": int(total),
                "bias_ensemble_mean_minus_era5": float(
                    np.sum(weights * group["bias_ensemble_mean_minus_era5"]) / total
                ),
                "mae": float(np.sum(weights * group["mae"]) / total),
                "rmse": float(
                    np.sqrt(np.sum(weights * np.square(group["rmse"])) / total)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["variable", "branch_id"]).reset_index(
        drop=True
    )


def evaluate_performance_gates(
    cycle_metrics: pd.DataFrame, pooled_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    cycle_count = int(cycle_metrics["decision_date"].nunique())
    for variable in VARIABLES:
        baseline_branch = (
            BRANCH_SELECTED_RAW_SOLAR if variable == "solar_kj_m2_day" else BRANCH_RAW
        )
        candidate_branch = (
            BRANCH_SELECTED_AFFINE_SOLAR
            if variable == "solar_kj_m2_day"
            else BRANCH_SELECTED_RAW_SOLAR
        )
        baseline = pooled_metrics.loc[
            pooled_metrics["variable"].eq(variable)
            & pooled_metrics["branch_id"].eq(baseline_branch)
        ].iloc[0]
        candidate = pooled_metrics.loc[
            pooled_metrics["variable"].eq(variable)
            & pooled_metrics["branch_id"].eq(candidate_branch)
        ].iloc[0]
        cycle = cycle_metrics.loc[cycle_metrics["variable"].eq(variable)]
        baseline_cycles = cycle.loc[cycle["branch_id"].eq(baseline_branch)].set_index(
            "decision_date"
        )
        candidate_cycles = cycle.loc[cycle["branch_id"].eq(candidate_branch)].set_index(
            "decision_date"
        )
        joined = baseline_cycles[["mae", "rmse"]].join(
            candidate_cycles[["mae", "rmse"]],
            lsuffix="_raw",
            rsuffix="_candidate",
            how="inner",
            validate="one_to_one",
        )
        mae_improved_cycles = int((joined["mae_candidate"] <= joined["mae_raw"]).sum())
        rmse_improved_cycles = int(
            (joined["rmse_candidate"] <= joined["rmse_raw"]).sum()
        )
        pooled_mae_passed = bool(candidate["mae"] <= baseline["mae"] + 1e-12)
        pooled_rmse_passed = bool(candidate["rmse"] <= baseline["rmse"] + 1e-12)
        pooled_bias_passed = bool(
            abs(candidate["bias_ensemble_mean_minus_era5"])
            <= abs(baseline["bias_ensemble_mean_minus_era5"]) + 1e-12
        )
        majority_cycle_passed = (
            mae_improved_cycles >= (cycle_count // 2 + 1)
            and rmse_improved_cycles >= (cycle_count // 2 + 1)
        )
        rows.append(
            {
                "variable": variable,
                "baseline_branch": baseline_branch,
                "candidate_branch": candidate_branch,
                "pooled_raw_absolute_bias": float(
                    abs(baseline["bias_ensemble_mean_minus_era5"])
                ),
                "pooled_candidate_absolute_bias": float(
                    abs(candidate["bias_ensemble_mean_minus_era5"])
                ),
                "pooled_raw_mae": float(baseline["mae"]),
                "pooled_candidate_mae": float(candidate["mae"]),
                "pooled_raw_rmse": float(baseline["rmse"]),
                "pooled_candidate_rmse": float(candidate["rmse"]),
                "mae_improved_cycle_count": mae_improved_cycles,
                "rmse_improved_cycle_count": rmse_improved_cycles,
                "cycle_count": cycle_count,
                "pooled_mae_gate_passed": pooled_mae_passed,
                "pooled_rmse_gate_passed": pooled_rmse_passed,
                "pooled_absolute_bias_gate_passed": pooled_bias_passed,
                "majority_cycle_gate_passed": majority_cycle_passed,
                "all_performance_gates_passed": bool(
                    pooled_mae_passed
                    and pooled_rmse_passed
                    and pooled_bias_passed
                    and majority_cycle_passed
                ),
            }
        )
    gates = pd.DataFrame(rows)
    non_solar = gates.loc[~gates["variable"].eq("solar_kj_m2_day")]
    solar = gates.loc[gates["variable"].eq("solar_kj_m2_day")].iloc[0]
    policy = {
        "non_solar_all_performance_gates_passed": bool(
            non_solar["all_performance_gates_passed"].all()
        ),
        "solar_affine_all_performance_gates_passed": bool(
            solar["all_performance_gates_passed"]
        ),
        "recommended_solar_branch": (
            "affine_alpha_0.25"
            if bool(solar["all_performance_gates_passed"])
            else "raw_solar_fallback"
        ),
    }
    return gates, policy


def build_multicycle_validation(
    *,
    five_member_weather: pd.DataFrame,
    history_gefs_c00: pd.DataFrame,
    history_era5: pd.DataFrame,
    cycle_era5: pd.DataFrame,
    metrics: pd.DataFrame,
    robust_selection: pd.DataFrame,
    minimum_samples: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    weather = five_member_weather.copy()
    reference = cycle_era5.copy()
    for frame in (weather, reference):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime(
            "%Y-%m-%d"
        )
    cycles = sorted(weather["decision_date"].unique())
    if len(cycles) != 5 or sorted(pd.to_datetime(cycles).year) != list(
        range(2015, 2020)
    ):
        raise ValueError("formal validation requires one cycle in each year 2015-2019")

    output_parts = []
    metric_parts = []
    factor_parts = []
    cycle_audits = []
    frozen_policy = None
    for cycle in cycles:
        cycle_weather = weather.loc[weather["decision_date"].eq(cycle)].copy()
        cycle_reference = reference.loc[reference["decision_date"].eq(cycle)].copy()
        output, cycle_metrics, factors, policy, audit = build_validation(
            five_member_weather=cycle_weather,
            history_gefs_c00=history_gefs_c00,
            history_era5=history_era5,
            cycle_era5=cycle_reference,
            metrics=metrics,
            robust_selection=robust_selection,
            minimum_samples=minimum_samples,
        )
        cycle_metrics.insert(0, "decision_date", cycle)
        factors.insert(0, "validation_cycle", cycle)
        output_parts.append(output)
        metric_parts.append(cycle_metrics)
        factor_parts.append(factors)
        cycle_audits.append(audit)
        frozen_policy = policy if frozen_policy is None else frozen_policy
        if policy != frozen_policy:
            raise ValueError("correction policy changed between validation cycles")

    output = pd.concat(output_parts, ignore_index=True)
    cycle_metrics = pd.concat(metric_parts, ignore_index=True)
    factors = pd.concat(factor_parts, ignore_index=True)
    pooled_metrics = aggregate_cycle_metrics(cycle_metrics)
    gates, performance_policy = evaluate_performance_gates(
        cycle_metrics, pooled_metrics
    )
    structural_passed = all(audit["mandatory_gate_passed"] for audit in cycle_audits)
    ready = bool(
        structural_passed
        and performance_policy["non_solar_all_performance_gates_passed"]
    )
    final_policy = {
        **(frozen_policy or {}),
        **performance_policy,
        "policy_freeze_allowed": ready,
    }
    audit = {
        "status": (
            "formal_multicycle_five_member_validation_passed_ready_to_freeze_policy"
            if ready
            else "formal_multicycle_five_member_validation_completed_policy_revision_required"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "formal_performance_gate_passed": bool(
            gates["all_performance_gates_passed"].all()
        ),
        "non_solar_performance_gate_passed": performance_policy[
            "non_solar_all_performance_gates_passed"
        ],
        "solar_affine_performance_gate_passed": performance_policy[
            "solar_affine_all_performance_gates_passed"
        ],
        "recommended_solar_branch": performance_policy["recommended_solar_branch"],
        "policy_freeze_allowed": ready,
        "cycle_count": len(cycles),
        "cycles": cycles,
        "years": sorted(pd.to_datetime(cycles).year.tolist()),
        "input_member_rows": int(len(weather)),
        "output_branch_rows": int(len(output)),
        "cycle_metric_rows": int(len(cycle_metrics)),
        "pooled_metric_rows": int(len(pooled_metrics)),
        "performance_gate_rows": int(len(gates)),
        "all_cycle_structural_gates_passed": structural_passed,
        "formal_multicycle_five_member_validation_performed": True,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "freeze_nonprecip_policy_then_integrate_precipitation_correction"
            if ready
            else "revise_failed_nonprecipitation_variables"
        ),
    }
    return output, cycle_metrics, pooled_metrics, gates, factors, final_policy, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    generated = build_multicycle_validation(
        five_member_weather=pd.read_csv(args.five_member_weather),
        history_gefs_c00=pd.read_csv(args.history_gefs_c00),
        history_era5=pd.read_csv(args.history_era5),
        cycle_era5=pd.read_csv(args.cycle_era5),
        metrics=pd.read_csv(args.causal_metrics),
        robust_selection=pd.read_csv(args.robust_selection),
        minimum_samples=args.minimum_samples,
    )
    output, cycle_metrics, pooled_metrics, gates, factors, policy, audit = generated
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "weather": args.output_dir / "gefs_five_member_multicycle_branches_v1.csv",
        "cycle_metrics": args.output_dir / "gefs_five_member_multicycle_cycle_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "gefs_five_member_multicycle_pooled_metrics_v1.csv",
        "gates": args.output_dir / "gefs_five_member_multicycle_performance_gates_v1.csv",
        "factors": args.output_dir / "gefs_five_member_multicycle_factors_v1.csv",
        "policy": args.output_dir / "gefs_five_member_multicycle_policy_v1.json",
        "audit": args.output_dir / "gefs_five_member_multicycle_audit_v1.json",
    }
    output.to_csv(outputs["weather"], index=False)
    cycle_metrics.to_csv(outputs["cycle_metrics"], index=False)
    pooled_metrics.to_csv(outputs["pooled_metrics"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    factors.to_csv(outputs["factors"], index=False)
    outputs["policy"].write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"multicycle structural gate failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five-member-weather", type=Path, required=True)
    parser.add_argument("--history-gefs-c00", type=Path, required=True)
    parser.add_argument("--history-era5", type=Path, required=True)
    parser.add_argument("--cycle-era5", type=Path, required=True)
    parser.add_argument("--causal-metrics", type=Path, required=True)
    parser.add_argument("--robust-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

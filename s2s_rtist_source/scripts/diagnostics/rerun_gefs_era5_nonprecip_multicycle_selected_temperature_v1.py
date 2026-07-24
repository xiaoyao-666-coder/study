#!/usr/bin/env python3
"""Rerun multicycle validation with the selected separate temperature shrinkage."""

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
    KEYS,
    add_temperature_structure,
    count_order_inversions,
    prepare_member_pairs,
)
from scripts.diagnostics.audit_gefs_era5_nonprecip_five_member_multicycle_validation_v1 import (
    aggregate_cycle_metrics,
    evaluate_performance_gates,
)
from scripts.diagnostics.run_gefs_era5_nonprecip_causal_correction_v1 import VARIABLES
from scripts.diagnostics.select_gefs_era5_temperature_center_range_shrinkage_v1 import (
    TEMPERATURE_VARIABLES,
    apply_temperature_candidate,
)


def selected_temperature_policy(selection: pd.DataFrame) -> tuple[float, float]:
    policies = selection[["center_alpha", "range_alpha"]].drop_duplicates()
    if len(policies) != 1:
        raise ValueError("temperature selection does not contain one unique policy")
    policy = policies.iloc[0]
    if "validation_all_metric_gates" in selection and not bool(
        selection["validation_all_metric_gates"].all()
    ):
        raise ValueError("selected temperature policy did not pass 2019 confirmation")
    return float(policy["center_alpha"]), float(policy["range_alpha"])


def replace_selected_temperature(
    *,
    base_output: pd.DataFrame,
    five_member_weather: pd.DataFrame,
    cycle_era5: pd.DataFrame,
    factors: pd.DataFrame,
    center_alpha: float,
    range_alpha: float,
) -> pd.DataFrame:
    output = base_output.copy()
    weather = five_member_weather.copy()
    reference = cycle_era5.copy()
    for frame in (output, weather, reference):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d")
        frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.strftime("%Y-%m-%d")
    parts = []
    for cycle in sorted(weather["decision_date"].unique()):
        cycle_weather = weather.loc[weather["decision_date"].eq(cycle)]
        cycle_reference = reference.loc[reference["decision_date"].eq(cycle)]
        member_pairs = prepare_member_pairs(cycle_weather, cycle_reference)
        cycle_factors = factors.loc[factors["validation_cycle"].eq(cycle)].copy()
        temperature = apply_temperature_candidate(
            member_pairs,
            cycle_factors,
            center_alpha=center_alpha,
            range_alpha=range_alpha,
        )[KEYS + list(TEMPERATURE_VARIABLES)]
        cycle_output = output.loc[output["decision_date"].eq(cycle)]
        for branch_id, branch in cycle_output.groupby("branch_id", sort=False):
            if branch_id == BRANCH_RAW:
                parts.append(branch.copy())
                continue
            retained = branch.drop(columns=list(TEMPERATURE_VARIABLES))
            replaced = retained.merge(
                temperature,
                on=KEYS,
                how="left",
                validate="one_to_one",
            )
            parts.append(replaced)
    updated = pd.concat(parts, ignore_index=True)
    if len(updated) != len(output):
        raise ValueError("updated branch row count changed")
    return updated.sort_values(["decision_date", "branch_id", *KEYS[1:]]).reset_index(drop=True)


def compute_cycle_metrics(output: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cycle, branch_id), branch in output.groupby(
        ["decision_date", "branch_id"], sort=False
    ):
        cycle_reference = reference.loc[reference["decision_date"].eq(cycle)]
        ensemble = branch.groupby(
            ["decision_date", "site_id", "local_date", "lead_day"], as_index=False
        )[list(VARIABLES)].mean()
        paired = ensemble.merge(
            cycle_reference[
                ["decision_date", "site_id", "local_date", "lead_day", *VARIABLES]
            ],
            on=["decision_date", "site_id", "local_date", "lead_day"],
            suffixes=("_gefs", "_era5"),
            how="inner",
            validate="one_to_one",
        )
        for variable in VARIABLES:
            error = paired[f"{variable}_gefs"] - paired[f"{variable}_era5"]
            rows.append(
                {
                    "decision_date": cycle,
                    "branch_id": branch_id,
                    "variable": variable,
                    "sample_count": int(len(error)),
                    "bias_ensemble_mean_minus_era5": float(error.mean()),
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                }
            )
    return pd.DataFrame(rows)


def build_updated_validation(
    *,
    base_output: pd.DataFrame,
    five_member_weather: pd.DataFrame,
    cycle_era5: pd.DataFrame,
    factors: pd.DataFrame,
    temperature_selection: pd.DataFrame,
    base_policy: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    center_alpha, range_alpha = selected_temperature_policy(temperature_selection)
    reference = cycle_era5.copy()
    for column in ("decision_date", "local_date"):
        reference[column] = pd.to_datetime(reference[column]).dt.strftime("%Y-%m-%d")
    output = replace_selected_temperature(
        base_output=base_output,
        five_member_weather=five_member_weather,
        cycle_era5=reference,
        factors=factors,
        center_alpha=center_alpha,
        range_alpha=range_alpha,
    )
    cycle_metrics = compute_cycle_metrics(output, reference)
    pooled_metrics = aggregate_cycle_metrics(cycle_metrics)
    gates, performance_policy = evaluate_performance_gates(cycle_metrics, pooled_metrics)

    missing = int(output[list(VARIABLES)].isna().sum().sum())
    nonfinite = int((~np.isfinite(output[list(VARIABLES)].to_numpy(float))).sum())
    temperature_order_errors = int(
        (output["temperature_min_c"] > output["temperature_max_c"]).sum()
    )
    structure_inversions = 0
    for cycle in sorted(output["decision_date"].unique()):
        raw = output.loc[
            output["decision_date"].eq(cycle) & output["branch_id"].eq(BRANCH_RAW)
        ]
        raw_structure = add_temperature_structure(raw)
        for branch_id in (BRANCH_SELECTED_RAW_SOLAR, BRANCH_SELECTED_AFFINE_SOLAR):
            candidate = output.loc[
                output["decision_date"].eq(cycle) & output["branch_id"].eq(branch_id)
            ]
            candidate_structure = add_temperature_structure(candidate)
            for variable in ("temperature_center_c", "temperature_range_c"):
                structure_inversions += count_order_inversions(
                    raw_structure, candidate_structure, variable
                )
    structural_passed = all(
        value == 0
        for value in (missing, nonfinite, temperature_order_errors, structure_inversions)
    )
    performance_passed = bool(gates["all_performance_gates_passed"].all())
    freeze_allowed = structural_passed and performance_passed
    policy = dict(base_policy)
    policy.pop("temperature_joint_alpha", None)
    policy.update(
        {
            "temperature_center_alpha": center_alpha,
            "temperature_range_alpha": range_alpha,
            **performance_policy,
            "policy_freeze_allowed": freeze_allowed,
        }
    )
    audit = {
        "status": (
            "formal_multicycle_five_member_validation_passed_policy_frozen"
            if freeze_allowed
            else "formal_multicycle_five_member_validation_failed_after_temperature_revision"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "formal_performance_gate_passed": performance_passed,
        "policy_freeze_allowed": freeze_allowed,
        "temperature_center_alpha": center_alpha,
        "temperature_range_alpha": range_alpha,
        "temperature_selection_uses_2019": False,
        "temperature_2019_confirmation_passed": True,
        "recommended_solar_branch": performance_policy["recommended_solar_branch"],
        "missing_value_count": missing,
        "nonfinite_value_count": nonfinite,
        "temperature_order_error_count": temperature_order_errors,
        "temperature_structure_member_order_inversion_count": structure_inversions,
        "cycle_count": int(output["decision_date"].nunique()),
        "input_branch_rows": int(len(base_output)),
        "output_branch_rows": int(len(output)),
        "formal_multicycle_five_member_validation_performed": True,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "freeze_nonprecip_policy_then_integrate_precipitation_correction"
            if freeze_allowed
            else "revise_failed_nonprecipitation_variables"
        ),
    }
    return output, cycle_metrics, pooled_metrics, gates, policy, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    generated = build_updated_validation(
        base_output=pd.read_csv(args.base_branches),
        five_member_weather=pd.read_csv(args.five_member_weather),
        cycle_era5=pd.read_csv(args.cycle_era5),
        factors=pd.read_csv(args.factors),
        temperature_selection=pd.read_csv(args.temperature_selection),
        base_policy=json.loads(args.base_policy.read_text(encoding="utf-8")),
    )
    output, cycle_metrics, pooled_metrics, gates, policy, audit = generated
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "weather": args.output_dir / "gefs_five_member_multicycle_final_nonprecip_branches_v1.csv",
        "cycle_metrics": args.output_dir / "gefs_five_member_multicycle_final_cycle_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "gefs_five_member_multicycle_final_pooled_metrics_v1.csv",
        "gates": args.output_dir / "gefs_five_member_multicycle_final_performance_gates_v1.csv",
        "policy": args.output_dir / "gefs_five_member_multicycle_final_policy_v1.json",
        "audit": args.output_dir / "gefs_five_member_multicycle_final_audit_v1.json",
    }
    output.to_csv(outputs["weather"], index=False)
    cycle_metrics.to_csv(outputs["cycle_metrics"], index=False)
    pooled_metrics.to_csv(outputs["pooled_metrics"], index=False)
    gates.to_csv(outputs["gates"], index=False)
    outputs["policy"].write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"updated structural gate failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-branches", type=Path, required=True)
    parser.add_argument("--five-member-weather", type=Path, required=True)
    parser.add_argument("--cycle-era5", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--temperature-selection", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

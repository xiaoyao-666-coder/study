#!/usr/bin/env python3
"""Apply causal nonprecipitation corrections to one five-member GEFS cycle."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_era5_nonprecip_causal_correction_v1 import (
    POSITIVE_VARIABLES,
    VARIABLES,
    apply_candidate,
    fit_target_factors,
    prepare_pairs,
)
from scripts.diagnostics.select_gefs_era5_nonprecip_causal_candidate_v1 import (
    SELECTION_YEARS,
    VALIDATION_YEAR,
    aggregate_metrics,
    compare_to_raw,
)


KEYS = ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
TEMPERATURE_VARIABLES = ("temperature_min_c", "temperature_max_c")
BRANCH_RAW = "raw_all_nonprecip"
BRANCH_SELECTED_RAW_SOLAR = "selected_nonprecip_raw_solar"
BRANCH_SELECTED_AFFINE_SOLAR = "selected_nonprecip_affine_solar"


def resolve_policy(
    metrics: pd.DataFrame, robust_selection: pd.DataFrame
) -> dict[str, Any]:
    pooled = compare_to_raw(aggregate_metrics(metrics, SELECTION_YEARS))
    validation = compare_to_raw(aggregate_metrics(metrics, (VALIDATION_YEAR,)))
    common_alphas: set[float] | None = None
    for variable in TEMPERATURE_VARIABLES:
        passed = set(
            pooled.loc[
                pooled["variable"].eq(variable)
                & pooled["candidate_id"].ne("raw_gefs")
                & pooled["all_metric_gates_passed"],
                "shrinkage_alpha",
            ].astype(float)
        )
        common_alphas = passed if common_alphas is None else common_alphas & passed
    if not common_alphas:
        raise ValueError("no common temperature alpha passes 2015-2018 gates")
    temperature_alpha = min(common_alphas)
    temperature_confirmation = validation.loc[
        validation["variable"].isin(TEMPERATURE_VARIABLES)
        & np.isclose(validation["shrinkage_alpha"], temperature_alpha)
    ]
    if len(temperature_confirmation) != 2 or not bool(
        temperature_confirmation["all_metric_gates_passed"].all()
    ):
        raise ValueError("common temperature alpha failed 2019 confirmation")

    selected = robust_selection.set_index("variable")
    required = {
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "solar_kj_m2_day",
    }
    if not required.issubset(selected.index):
        raise ValueError("robust selection is missing required variables")
    if not bool(selected.loc["solar_kj_m2_day", "raw_solar_sensitivity_required"]):
        raise ValueError("solar raw sensitivity branch is not required by selection")
    return {
        "temperature_joint_alpha": float(temperature_alpha),
        "actual_vapor_pressure_kpa_alpha": float(
            selected.loc["actual_vapor_pressure_kpa", "shrinkage_alpha"]
        ),
        "wind_speed_m_s_alpha": float(
            selected.loc["wind_speed_m_s", "shrinkage_alpha"]
        ),
        "solar_kj_m2_day_alpha": float(
            selected.loc["solar_kj_m2_day", "shrinkage_alpha"]
        ),
        "solar_branches": ["raw", "affine"],
        "temperature_selection_uses_2019": False,
        "temperature_2019_confirmation_passed": True,
    }


def prepare_member_pairs(weather: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for member, group in weather.groupby("gefs_member", sort=True):
        paired = prepare_pairs(group, reference)
        paired["gefs_member"] = member
        parts.append(paired)
    return pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def count_order_inversions(
    raw: pd.DataFrame, corrected: pd.DataFrame, variable: str
) -> int:
    joined = raw[KEYS + [variable]].merge(
        corrected[KEYS + [variable]],
        on=KEYS,
        how="inner",
        suffixes=("_raw", "_corrected"),
        validate="one_to_one",
    )
    inversions = 0
    for _, group in joined.groupby(
        ["decision_date", "site_id", "local_date", "lead_day"], sort=False
    ):
        group = group.sort_values("gefs_member")
        raw_values = group[f"{variable}_raw"].to_numpy(dtype=float)
        corrected_values = group[f"{variable}_corrected"].to_numpy(dtype=float)
        for left, right in itertools.combinations(range(len(group)), 2):
            if (
                (raw_values[left] - raw_values[right])
                * (corrected_values[left] - corrected_values[right])
                < -1e-12
            ):
                inversions += 1
    return inversions


def add_temperature_structure(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["temperature_center_c"] = (
        output["temperature_min_c"] + output["temperature_max_c"]
    ) / 2.0
    output["temperature_range_c"] = (
        output["temperature_max_c"] - output["temperature_min_c"]
    )
    return output


def build_validation(
    *,
    five_member_weather: pd.DataFrame,
    history_gefs_c00: pd.DataFrame,
    history_era5: pd.DataFrame,
    cycle_era5: pd.DataFrame,
    metrics: pd.DataFrame,
    robust_selection: pd.DataFrame,
    minimum_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    weather = five_member_weather.copy()
    for column in ("decision_date", "local_date"):
        weather[column] = pd.to_datetime(weather[column]).dt.strftime("%Y-%m-%d")
    decision_dates = sorted(weather["decision_date"].unique())
    if len(decision_dates) != 1:
        raise ValueError("five-member smoke requires exactly one decision date")
    target_date = decision_dates[0]
    if weather[KEYS].duplicated().any():
        raise ValueError("five-member weather contains duplicate sample keys")

    policy = resolve_policy(metrics, robust_selection)
    c00 = weather.loc[weather["gefs_member"].eq("c00")].copy()
    target_pairs = prepare_pairs(c00, cycle_era5)
    history_pairs = prepare_pairs(history_gefs_c00, history_era5)
    fitting_pairs = pd.concat([history_pairs, target_pairs], ignore_index=True)
    factors = fit_target_factors(
        fitting_pairs,
        target_date=target_date,
        minimum_samples=minimum_samples,
    )
    if factors is None:
        raise ValueError("insufficient causal history for five-member target cycle")

    member_pairs = prepare_member_pairs(weather, cycle_era5)
    alphas = {
        policy["temperature_joint_alpha"],
        policy["actual_vapor_pressure_kpa_alpha"],
        policy["wind_speed_m_s_alpha"],
        policy["solar_kj_m2_day_alpha"],
    }
    corrected_by_alpha = {
        alpha: apply_candidate(member_pairs, factors, alpha=alpha)
        for alpha in sorted(alphas)
    }
    precipitation = weather[KEYS + ["precipitation_mm_raw"]].copy()
    raw = member_pairs[KEYS].copy()
    for variable in VARIABLES:
        raw[variable] = member_pairs[f"{variable}_gefs"].to_numpy()
    raw = raw.merge(precipitation, on=KEYS, how="left", validate="one_to_one")

    branches = []
    for branch_id in (
        BRANCH_RAW,
        BRANCH_SELECTED_RAW_SOLAR,
        BRANCH_SELECTED_AFFINE_SOLAR,
    ):
        branch = raw.copy()
        if branch_id != BRANCH_RAW:
            temperature = corrected_by_alpha[policy["temperature_joint_alpha"]]
            vapor = corrected_by_alpha[policy["actual_vapor_pressure_kpa_alpha"]]
            wind = corrected_by_alpha[policy["wind_speed_m_s_alpha"]]
            branch["temperature_min_c"] = temperature[
                "temperature_min_c_corrected"
            ].to_numpy()
            branch["temperature_max_c"] = temperature[
                "temperature_max_c_corrected"
            ].to_numpy()
            branch["actual_vapor_pressure_kpa"] = vapor[
                "actual_vapor_pressure_kpa_corrected"
            ].to_numpy()
            branch["wind_speed_m_s"] = wind["wind_speed_m_s_corrected"].to_numpy()
            if branch_id == BRANCH_SELECTED_AFFINE_SOLAR:
                solar = corrected_by_alpha[policy["solar_kj_m2_day_alpha"]]
                branch["solar_kj_m2_day"] = solar[
                    "solar_kj_m2_day_corrected"
                ].to_numpy()
        branch.insert(0, "branch_id", branch_id)
        branches.append(branch)
    output = pd.concat(branches, ignore_index=True)

    reference = cycle_era5.copy()
    for column in ("decision_date", "local_date"):
        reference[column] = pd.to_datetime(reference[column]).dt.strftime("%Y-%m-%d")
    metric_rows = []
    for branch_id, branch in output.groupby("branch_id", sort=False):
        ensemble = (
            branch.groupby(
                ["decision_date", "site_id", "local_date", "lead_day"],
                as_index=False,
            )[list(VARIABLES)]
            .mean()
        )
        paired = ensemble.merge(
            reference[
                ["decision_date", "site_id", "local_date", "lead_day", *VARIABLES]
            ],
            on=["decision_date", "site_id", "local_date", "lead_day"],
            how="inner",
            suffixes=("_gefs", "_era5"),
            validate="one_to_one",
        )
        for variable in VARIABLES:
            error = paired[f"{variable}_gefs"] - paired[f"{variable}_era5"]
            metric_rows.append(
                {
                    "branch_id": branch_id,
                    "variable": variable,
                    "sample_count": int(len(error)),
                    "bias_ensemble_mean_minus_era5": float(error.mean()),
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.square(error).mean())),
                }
            )
    metric_frame = pd.DataFrame(metric_rows)

    missing = int(output[[*KEYS, *VARIABLES]].isna().sum().sum())
    nonfinite = int(
        (~np.isfinite(output[list(VARIABLES)].to_numpy(dtype=float))).sum()
    )
    negative = int((output[list(POSITIVE_VARIABLES)] < 0.0).sum().sum())
    temperature_order_errors = int(
        (output["temperature_min_c"] > output["temperature_max_c"]).sum()
    )
    positive_variable_order_inversions = 0
    temperature_structure_order_inversions = 0
    temperature_output_order_inversions = 0
    raw_temperature_structure = add_temperature_structure(raw)
    for branch_id, branch in output.groupby("branch_id", sort=False):
        if branch_id == BRANCH_RAW:
            continue
        for variable in POSITIVE_VARIABLES:
            positive_variable_order_inversions += count_order_inversions(
                raw, branch, variable
            )
        branch_temperature_structure = add_temperature_structure(branch)
        for variable in ("temperature_center_c", "temperature_range_c"):
            temperature_structure_order_inversions += count_order_inversions(
                raw_temperature_structure,
                branch_temperature_structure,
                variable,
            )
        for variable in TEMPERATURE_VARIABLES:
            temperature_output_order_inversions += count_order_inversions(
                raw, branch, variable
            )
    target = pd.Timestamp(target_date)
    fit_last = pd.to_datetime(factors["fit_last_decision_date"])
    leakage = int(((fit_last + pd.Timedelta(days=6)) >= target).sum())
    member_counts = weather.groupby(
        ["site_id", "local_date", "lead_day"]
    )["gefs_member"].nunique()
    mandatory_passed = all(
        [
            missing == 0,
            nonfinite == 0,
            negative == 0,
            temperature_order_errors == 0,
            positive_variable_order_inversions == 0,
            temperature_structure_order_inversions == 0,
            leakage == 0,
            int(member_counts.min()) == 5,
            int(member_counts.max()) == 5,
            int(weather["site_id"].nunique()) == 5,
            int(weather["lead_day"].nunique()) == 7,
        ]
    )
    audit = {
        "status": (
            "five_member_nonprecip_application_smoke_passed"
            if mandatory_passed
            else "five_member_nonprecip_application_smoke_failed"
        ),
        "mandatory_gate_passed": mandatory_passed,
        "decision_date": target_date,
        "input_member_rows": int(len(weather)),
        "output_branch_rows": int(len(output)),
        "branch_count": int(output["branch_id"].nunique()),
        "member_count_minimum": int(member_counts.min()),
        "member_count_maximum": int(member_counts.max()),
        "site_count": int(weather["site_id"].nunique()),
        "lead_day_count": int(weather["lead_day"].nunique()),
        "factor_rows": int(len(factors)),
        "minimum_fit_samples_per_site_lead": int(factors["fit_sample_count"].min()),
        "fit_leakage_rows": leakage,
        "missing_value_count": missing,
        "nonfinite_value_count": nonfinite,
        "negative_positive_only_value_count": negative,
        "temperature_order_error_count": temperature_order_errors,
        "positive_variable_member_order_inversion_count": (
            positive_variable_order_inversions
        ),
        "temperature_structure_member_order_inversion_count": (
            temperature_structure_order_inversions
        ),
        "temperature_output_member_order_inversion_count_diagnostic": (
            temperature_output_order_inversions
        ),
        "member_order_gate_definition": (
            "temperature_center_and_diurnal_range_for_temperature;_"
            "native_variable_for_vapor_pressure_wind_and_solar"
        ),
        "precipitation_correction_applied": False,
        "solar_raw_sensitivity_branch_included": True,
        "solar_affine_sensitivity_branch_included": True,
        "one_cycle_five_member_application_smoke_performed": True,
        "formal_multicycle_five_member_validation_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "design_multicycle_five_member_validation_sample"
            if mandatory_passed
            else "repair_five_member_application_smoke"
        ),
    }
    return output, metric_frame, factors, policy, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    output, metrics, factors, policy, audit = build_validation(
        five_member_weather=pd.read_csv(args.five_member_weather),
        history_gefs_c00=pd.read_csv(args.history_gefs_c00),
        history_era5=pd.read_csv(args.history_era5),
        cycle_era5=pd.read_csv(args.cycle_era5),
        metrics=pd.read_csv(args.causal_metrics),
        robust_selection=pd.read_csv(args.robust_selection),
        minimum_samples=args.minimum_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "weather": args.output_dir / "gefs_five_member_nonprecip_branches_v1.csv",
        "metrics": args.output_dir / "gefs_five_member_nonprecip_metrics_v1.csv",
        "factors": args.output_dir / "gefs_five_member_nonprecip_factors_v1.csv",
        "policy": args.output_dir / "gefs_five_member_nonprecip_policy_v1.json",
        "audit": args.output_dir / "gefs_five_member_nonprecip_audit_v1.json",
    }
    output.to_csv(outputs["weather"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    factors.to_csv(outputs["factors"], index=False)
    outputs["policy"].write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"five-member application smoke failed; see {outputs['audit']}")
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

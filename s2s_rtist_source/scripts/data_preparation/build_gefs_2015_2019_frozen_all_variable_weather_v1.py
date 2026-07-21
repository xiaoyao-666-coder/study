#!/usr/bin/env python3
"""Combine frozen causal precipitation and nonprecipitation GEFS corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMBER_KEYS = ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
NONPRECIP_VARIABLES = [
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]
POSITIVE_VARIABLES = [
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
    "precipitation_mm",
]
PRECIPITATION_CANDIDATE = "weekly_two_stage_linear_site_only"
NONPRECIP_BRANCH = "selected_nonprecip_affine_solar"
PRECIPITATION_ALPHA = 0.75
EXPECTED_POLICY = {
    "actual_vapor_pressure_kpa_alpha": 0.75,
    "solar_kj_m2_day_alpha": 0.25,
    "temperature_center_alpha": 1.0,
    "temperature_range_alpha": 0.0,
    "wind_speed_m_s_alpha": 1.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ("decision_date", "local_date"):
        output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    return output


def validate_nonprecip_policy(policy: dict[str, Any]) -> None:
    if not bool(policy.get("policy_freeze_allowed")):
        raise ValueError("nonprecipitation policy is not frozen")
    if policy.get("recommended_solar_branch") != "affine_alpha_0.25":
        raise ValueError("nonprecipitation solar branch is not the frozen affine policy")
    for key, expected in EXPECTED_POLICY.items():
        actual = policy.get(key)
        if actual is None or not np.isclose(float(actual), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"nonprecipitation policy mismatch for {key}: {actual!r}")
    if bool(policy.get("temperature_selection_uses_2019")):
        raise ValueError("temperature selection unexpectedly uses 2019")
    if not bool(policy.get("temperature_2019_confirmation_passed")):
        raise ValueError("temperature policy did not pass 2019 confirmation")


def load_causal_precipitation_factors(
    cv_factors: pd.DataFrame, validation_factors: pd.DataFrame
) -> pd.DataFrame:
    cv = cv_factors.loc[
        cv_factors["candidate_id"].eq(PRECIPITATION_CANDIDATE)
        & cv_factors["validation_year"].isin([2015, 2016, 2017, 2018])
    ].copy()
    validation = validation_factors.loc[
        validation_factors["candidate_id"].eq(PRECIPITATION_CANDIDATE)
        & validation_factors["validation_year"].eq(2019)
    ].copy()
    factors = pd.concat([cv, validation], ignore_index=True).rename(
        columns={"validation_year": "target_year"}
    )
    required = [
        "target_year",
        "site_id",
        "fit_first_year",
        "fit_last_year",
        "validation_rows_used_for_fit",
        "raw_ensemble_mean_7d_q90_mm",
        "overall_factor",
        "final_extreme_factor",
    ]
    missing = sorted(set(required) - set(factors.columns))
    if missing:
        raise ValueError(f"precipitation factors are missing columns: {missing}")
    factors = factors[required].copy()
    factors["target_year"] = factors["target_year"].astype(int)
    if len(factors) != 25 or factors[["target_year", "site_id"]].duplicated().any():
        raise ValueError("expected 25 unique causal year-site precipitation factor rows")
    if set(factors["target_year"]) != set(range(2015, 2020)):
        raise ValueError("precipitation factors do not cover exactly 2015-2019")
    if factors["site_id"].nunique() != 5:
        raise ValueError("precipitation factors do not cover five sites")
    if (factors["fit_last_year"] >= factors["target_year"]).any():
        raise ValueError("precipitation factor table leaks target or future years")
    if (factors["validation_rows_used_for_fit"] != 0).any():
        raise ValueError("precipitation factor table used validation rows for fitting")
    numeric = factors[
        ["raw_ensemble_mean_7d_q90_mm", "overall_factor", "final_extreme_factor"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise ValueError("precipitation factor table contains invalid values")
    return factors.sort_values(["target_year", "site_id"]).reset_index(drop=True)


def count_member_order_inversions(
    raw: pd.DataFrame, corrected: pd.DataFrame, raw_column: str, corrected_column: str
) -> int:
    joined = raw[MEMBER_KEYS + [raw_column]].merge(
        corrected[MEMBER_KEYS + [corrected_column]],
        on=MEMBER_KEYS,
        how="inner",
        validate="one_to_one",
    )
    inversions = 0
    groups = ["decision_date", "site_id", "local_date", "lead_day"]
    for _, group in joined.groupby(groups, sort=False):
        values_raw = group[raw_column].to_numpy(dtype=float)
        values_corrected = group[corrected_column].to_numpy(dtype=float)
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                if (
                    (values_raw[left] - values_raw[right])
                    * (values_corrected[left] - values_corrected[right])
                    < -1e-12
                ):
                    inversions += 1
    return inversions


def build_frozen_weather(
    *,
    raw_weather: pd.DataFrame,
    nonprecip_branches: pd.DataFrame,
    nonprecip_policy: dict[str, Any],
    precipitation_factors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_nonprecip_policy(nonprecip_policy)
    raw = normalize_dates(raw_weather)
    branches = normalize_dates(nonprecip_branches)
    raw["target_year"] = pd.to_datetime(raw["decision_date"]).dt.year.astype(int)

    if len(raw) != 875 or raw[MEMBER_KEYS].duplicated().any():
        raise ValueError("raw weather must contain 875 unique five-cycle member rows")
    selected = branches.loc[branches["branch_id"].eq(NONPRECIP_BRANCH)].copy()
    if len(selected) != len(raw) or selected[MEMBER_KEYS].duplicated().any():
        raise ValueError("frozen nonprecipitation branch does not cover all raw member rows")

    branch_fields = MEMBER_KEYS + NONPRECIP_VARIABLES + ["precipitation_mm_raw"]
    selected = selected[branch_fields]
    raw_precipitation = raw[MEMBER_KEYS + ["precipitation_mm_raw"]].merge(
        selected[MEMBER_KEYS + ["precipitation_mm_raw"]],
        on=MEMBER_KEYS,
        how="inner",
        suffixes=("_weather", "_branch"),
        validate="one_to_one",
    )
    raw_precipitation_error = float(
        np.max(
            np.abs(
                raw_precipitation["precipitation_mm_raw_weather"].to_numpy(float)
                - raw_precipitation["precipitation_mm_raw_branch"].to_numpy(float)
            )
        )
    )
    if raw_precipitation_error > 1e-12:
        raise ValueError("raw precipitation differs between weather and nonprecip branches")

    replacement = selected.drop(columns="precipitation_mm_raw").rename(
        columns={field: f"{field}_corrected" for field in NONPRECIP_VARIABLES}
    )
    output = raw.merge(replacement, on=MEMBER_KEYS, how="left", validate="one_to_one")
    for field in NONPRECIP_VARIABLES:
        output[f"{field}_raw"] = output[field]
        output[field] = output.pop(f"{field}_corrected")

    weekly = (
        output.groupby(["target_year", "decision_date", "site_id"], as_index=False)[
            "precipitation_mm_raw"
        ]
        .sum()
        .rename(columns={"precipitation_mm_raw": "five_member_sum_7d_mm"})
    )
    weekly["raw_ensemble_mean_7d_mm"] = weekly["five_member_sum_7d_mm"] / 5.0
    weekly = weekly.drop(columns="five_member_sum_7d_mm").merge(
        precipitation_factors,
        on=["target_year", "site_id"],
        how="left",
        validate="one_to_one",
    )
    if weekly[["overall_factor", "final_extreme_factor"]].isna().any().any():
        raise ValueError("causal precipitation factor coverage is incomplete")
    weekly["weekly_extreme_regime"] = weekly["raw_ensemble_mean_7d_mm"].gt(
        weekly["raw_ensemble_mean_7d_q90_mm"]
    )
    weekly["base_factor"] = np.where(
        weekly["weekly_extreme_regime"],
        weekly["final_extreme_factor"],
        weekly["overall_factor"],
    )
    weekly["factor_shrinkage_alpha"] = PRECIPITATION_ALPHA
    weekly["effective_factor"] = 1.0 + PRECIPITATION_ALPHA * (
        weekly["base_factor"] - 1.0
    )
    output = output.merge(
        weekly[
            [
                "target_year",
                "decision_date",
                "site_id",
                "fit_first_year",
                "fit_last_year",
                "raw_ensemble_mean_7d_mm",
                "raw_ensemble_mean_7d_q90_mm",
                "weekly_extreme_regime",
                "factor_shrinkage_alpha",
                "effective_factor",
            ]
        ],
        on=["target_year", "decision_date", "site_id"],
        how="left",
        validate="many_to_one",
    )
    output["precipitation_mm"] = (
        output["precipitation_mm_raw"] * output["effective_factor"]
    )
    output["weather_source"] = (
        "GEFSv12_5member_frozen_causal_all_variable_correction_v1"
    )

    numeric_fields = NONPRECIP_VARIABLES + ["precipitation_mm"]
    missing_count = int(output[numeric_fields].isna().sum().sum())
    nonfinite_count = int(
        (~np.isfinite(output[numeric_fields].to_numpy(dtype=float))).sum()
    )
    negative_count = int((output[POSITIVE_VARIABLES] < 0.0).sum().sum())
    temperature_order_count = int(
        (output["temperature_min_c"] > output["temperature_max_c"]).sum()
    )
    leakage_count = int((output["fit_last_year"] >= output["target_year"]).sum())
    zero_pattern_mismatch = int(
        (
            output["precipitation_mm_raw"].eq(0.0)
            != output["precipitation_mm"].eq(0.0)
        ).sum()
    )
    order_inversions = count_member_order_inversions(
        raw, output, "precipitation_mm_raw", "precipitation_mm"
    )

    weekly_output = output.groupby(
        ["target_year", "decision_date", "site_id"], as_index=False
    ).agg(
        corrected_five_member_sum_7d_mm=("precipitation_mm", "sum"),
        effective_factor=("effective_factor", "first"),
        raw_ensemble_mean_7d_mm=("raw_ensemble_mean_7d_mm", "first"),
    )
    weekly_output["corrected_ensemble_mean_7d_mm"] = (
        weekly_output["corrected_five_member_sum_7d_mm"] / 5.0
    )
    weekly_output["expected_corrected_ensemble_mean_7d_mm"] = (
        weekly_output["raw_ensemble_mean_7d_mm"]
        * weekly_output["effective_factor"]
    )
    weekly_total_error = float(
        np.max(
            np.abs(
                weekly_output["corrected_ensemble_mean_7d_mm"]
                - weekly_output["expected_corrected_ensemble_mean_7d_mm"]
            )
        )
    )

    corrected_comparison = output[MEMBER_KEYS + NONPRECIP_VARIABLES].merge(
        selected[MEMBER_KEYS + NONPRECIP_VARIABLES],
        on=MEMBER_KEYS,
        suffixes=("_output", "_selected"),
        validate="one_to_one",
    )
    nonprecip_errors = [
        np.max(
            np.abs(
                corrected_comparison[f"{field}_output"].to_numpy(float)
                - corrected_comparison[f"{field}_selected"].to_numpy(float)
            )
        )
        for field in NONPRECIP_VARIABLES
    ]
    maximum_nonprecip_error = float(max(nonprecip_errors))

    member_counts = output.groupby(
        ["decision_date", "site_id", "local_date", "lead_day"]
    )["gefs_member"].nunique()
    structural_passed = all(
        [
            missing_count == 0,
            nonfinite_count == 0,
            negative_count == 0,
            temperature_order_count == 0,
            leakage_count == 0,
            zero_pattern_mismatch == 0,
            order_inversions == 0,
            weekly_total_error <= 1e-10,
            maximum_nonprecip_error <= 1e-12,
            int(member_counts.min()) == 5,
            int(member_counts.max()) == 5,
            int(output["decision_date"].nunique()) == 5,
            int(output["site_id"].nunique()) == 5,
            int(output["lead_day"].nunique()) == 7,
        ]
    )
    audit = {
        "status": (
            "frozen_all_variable_weather_integration_passed"
            if structural_passed
            else "frozen_all_variable_weather_integration_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "member_rows": int(len(output)),
        "site_cycle_count": int(len(weekly)),
        "cycle_count": int(output["decision_date"].nunique()),
        "site_count": int(output["site_id"].nunique()),
        "member_count_minimum": int(member_counts.min()),
        "member_count_maximum": int(member_counts.max()),
        "lead_day_count": int(output["lead_day"].nunique()),
        "precipitation_candidate_id": (
            "weekly_two_stage_linear_site_factor_shrink_a075"
        ),
        "precipitation_factor_shrinkage_alpha": PRECIPITATION_ALPHA,
        "precipitation_factor_fit_leakage_rows": leakage_count,
        "precipitation_zero_pattern_mismatch_count": zero_pattern_mismatch,
        "precipitation_member_order_inversion_count": order_inversions,
        "maximum_absolute_weekly_precipitation_total_error_mm": weekly_total_error,
        "maximum_absolute_nonprecipitation_branch_copy_error": maximum_nonprecip_error,
        "maximum_absolute_raw_precipitation_join_error_mm": raw_precipitation_error,
        "missing_value_count": missing_count,
        "nonfinite_value_count": nonfinite_count,
        "negative_positive_variable_count": negative_count,
        "temperature_order_error_count": temperature_order_count,
        "nonprecipitation_policy_frozen": True,
        "selected_nonprecipitation_branch": NONPRECIP_BRANCH,
        "contains_2024": bool(output["target_year"].eq(2024).any()),
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "one_date_eight_irrigation_swap_branch_smoke_from_verified_checkpoint"
            if structural_passed
            else "repair_all_variable_weather_integration"
        ),
    }
    if not structural_passed:
        raise RuntimeError("frozen all-variable weather integration gate failed")
    return output.sort_values(MEMBER_KEYS).reset_index(drop=True), weekly, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    policy = json.loads(args.nonprecip_policy.read_text(encoding="utf-8"))
    factors = load_causal_precipitation_factors(
        pd.read_csv(args.precipitation_cv_factors),
        pd.read_csv(args.precipitation_2019_factors),
    )
    weather, site_cycles, audit = build_frozen_weather(
        raw_weather=pd.read_csv(args.raw_weather),
        nonprecip_branches=pd.read_csv(args.nonprecip_branches),
        nonprecip_policy=policy,
        precipitation_factors=factors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "weather": args.output_dir / "gefs_2015_2019_frozen_all_variable_member_weather_v1.csv",
        "site_cycles": args.output_dir / "gefs_2015_2019_frozen_precipitation_site_cycles_v1.csv",
        "audit": args.output_dir / "gefs_2015_2019_frozen_all_variable_weather_audit_v1.json",
        "policy": args.output_dir / "gefs_2015_2019_frozen_all_variable_weather_policy_v1.json",
        "manifest": args.output_dir / "gefs_2015_2019_frozen_all_variable_weather_manifest_v1.json",
    }
    weather.to_csv(outputs["weather"], index=False)
    site_cycles.to_csv(outputs["site_cycles"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    combined_policy = {
        "precipitation": {
            "candidate_id": "weekly_two_stage_linear_site_factor_shrink_a075",
            "factor_shrinkage_alpha": PRECIPITATION_ALPHA,
            "historical_application": "target_year_causal_expanding_fit",
        },
        "nonprecipitation": policy,
        "selected_nonprecipitation_branch": NONPRECIP_BRANCH,
        "policy_frozen": True,
    }
    outputs["policy"].write_text(
        json.dumps(combined_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "raw_weather_sha256": sha256_file(args.raw_weather),
            "nonprecip_branches_sha256": sha256_file(args.nonprecip_branches),
            "nonprecip_policy_sha256": sha256_file(args.nonprecip_policy),
            "precipitation_cv_factors_sha256": sha256_file(
                args.precipitation_cv_factors
            ),
            "precipitation_2019_factors_sha256": sha256_file(
                args.precipitation_2019_factors
            ),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-weather", type=Path, required=True)
    parser.add_argument("--nonprecip-branches", type=Path, required=True)
    parser.add_argument("--nonprecip-policy", type=Path, required=True)
    parser.add_argument("--precipitation-cv-factors", type=Path, required=True)
    parser.add_argument("--precipitation-2019-factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

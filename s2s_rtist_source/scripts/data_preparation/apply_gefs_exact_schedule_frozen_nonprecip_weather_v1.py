#!/usr/bin/env python3
"""Apply the frozen causal nonprecipitation policy to an exact GEFS schedule."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_frozen_all_variable_weather_v1 import (
    MEMBER_KEYS,
    NONPRECIP_BRANCH,
    validate_nonprecip_policy,
)
from scripts.diagnostics.run_gefs_era5_nonprecip_causal_correction_v1 import (
    POSITIVE_VARIABLES,
    VARIABLES,
    fit_target_factors,
    prepare_pairs,
)


FROZEN_MEMBERS = ("c00", "p01", "p02", "p03", "p04")


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


def prepare_target_members(weather: pd.DataFrame) -> pd.DataFrame:
    target = normalize_dates(weather)
    target["target_year"] = pd.to_datetime(target["decision_date"]).dt.year.astype(int)
    for variable in VARIABLES:
        target[f"{variable}_gefs"] = target[variable].astype(float)
    target["temperature_center_c_gefs"] = (
        target["temperature_min_c_gefs"] + target["temperature_max_c_gefs"]
    ) / 2.0
    target["temperature_range_c_gefs"] = (
        target["temperature_max_c_gefs"] - target["temperature_min_c_gefs"]
    )
    if (target["temperature_range_c_gefs"] < 0.0).any():
        raise ValueError("raw target weather has Tmin above Tmax")
    return target


def apply_frozen_policy(
    target: pd.DataFrame,
    factors: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    data = target.merge(
        factors,
        on=["target_year", "site_id", "lead_day"],
        how="left",
        validate="many_to_one",
    )
    if data["fit_sample_count"].isna().any():
        raise ValueError("causal nonprecipitation factor coverage is incomplete")

    center_alpha = float(policy["temperature_center_alpha"])
    range_alpha = float(policy["temperature_range_alpha"])
    center = data["temperature_center_c_gefs"] + center_alpha * data[
        "temperature_center_additive_delta_c"
    ]
    range_factor = 1.0 + range_alpha * (data["temperature_range_ratio"] - 1.0)
    temperature_range = data["temperature_range_c_gefs"] * range_factor
    data["temperature_min_c"] = center - temperature_range / 2.0
    data["temperature_max_c"] = center + temperature_range / 2.0

    for variable in ("actual_vapor_pressure_kpa", "wind_speed_m_s"):
        alpha = float(policy[f"{variable}_alpha"])
        effective_factor = 1.0 + alpha * (data[f"{variable}_ratio"] - 1.0)
        data[variable] = data[f"{variable}_gefs"] * effective_factor

    solar_alpha = float(policy["solar_kj_m2_day_alpha"])
    solar_intercept = solar_alpha * data["solar_kj_m2_day_affine_intercept"]
    solar_slope = 1.0 + solar_alpha * (
        data["solar_kj_m2_day_affine_slope"] - 1.0
    )
    data["solar_kj_m2_day"] = (
        solar_intercept + solar_slope * data["solar_kj_m2_day_gefs"]
    ).clip(lower=0.0)

    output = data[
        [*MEMBER_KEYS, *VARIABLES, "precipitation_mm_raw"]
    ].copy()
    output.insert(0, "branch_id", NONPRECIP_BRANCH)
    return output


def count_order_inversions(
    raw: pd.DataFrame,
    corrected: pd.DataFrame,
    raw_column: str,
    corrected_column: str,
) -> int:
    joined = raw[MEMBER_KEYS + [raw_column]].merge(
        corrected[MEMBER_KEYS + [corrected_column]],
        on=MEMBER_KEYS,
        how="inner",
        suffixes=("_raw", "_corrected"),
        validate="one_to_one",
    )
    raw_value_column = f"{raw_column}_raw"
    corrected_value_column = f"{corrected_column}_corrected"
    inversions = 0
    group_keys = ["decision_date", "site_id", "local_date", "lead_day"]
    for _, group in joined.groupby(group_keys, sort=False):
        group = group.sort_values("gefs_member")
        raw_values = group[raw_value_column].to_numpy(dtype=float)
        corrected_values = group[corrected_value_column].to_numpy(dtype=float)
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


def build_frozen_nonprecipitation(
    *,
    target_weather: pd.DataFrame,
    history_gefs_c00: pd.DataFrame,
    history_era5: pd.DataFrame,
    policy: dict[str, Any],
    minimum_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_nonprecip_policy(policy)
    target = prepare_target_members(target_weather)
    required = set(MEMBER_KEYS) | set(VARIABLES) | {"precipitation_mm_raw"}
    missing_columns = sorted(required - set(target.columns))
    if missing_columns:
        raise ValueError(f"target weather is missing columns: {missing_columns}")
    if target.empty or target[MEMBER_KEYS].duplicated().any():
        raise ValueError("target weather must contain nonempty unique member rows")
    if tuple(sorted(target["gefs_member"].astype(str).unique())) != tuple(
        sorted(FROZEN_MEMBERS)
    ):
        raise ValueError("target weather does not contain the frozen five members")
    if target["target_year"].nunique() != 1:
        raise ValueError("formal application requires exactly one target year")

    history_gefs = normalize_dates(history_gefs_c00)
    if "gefs_member" in history_gefs.columns:
        if set(history_gefs["gefs_member"].astype(str)) != {"c00"}:
            raise ValueError("causal fit history must contain c00 only")
    history_pairs = prepare_pairs(history_gefs, history_era5)

    branch_parts: list[pd.DataFrame] = []
    factor_parts: list[pd.DataFrame] = []
    for target_date in sorted(target["decision_date"].unique()):
        cycle = target.loc[target["decision_date"].eq(target_date)].copy()
        expected_groups = cycle[["site_id", "lead_day"]].drop_duplicates()
        factors = fit_target_factors(
            history_pairs,
            target_date=target_date,
            minimum_samples=minimum_samples,
            expected_groups=expected_groups,
        )
        if factors is None:
            raise ValueError(
                f"insufficient strictly causal nonprecipitation history for {target_date}"
            )
        factor_parts.append(factors)
        branch_parts.append(apply_frozen_policy(cycle, factors, policy))

    output = pd.concat(branch_parts, ignore_index=True).sort_values(MEMBER_KEYS)
    factors = pd.concat(factor_parts, ignore_index=True).sort_values(
        ["target_date", "site_id", "lead_day"]
    )
    output = output.reset_index(drop=True)
    factors = factors.reset_index(drop=True)

    raw_keys = target[MEMBER_KEYS].sort_values(MEMBER_KEYS).reset_index(drop=True)
    output_keys = output[MEMBER_KEYS].sort_values(MEMBER_KEYS).reset_index(drop=True)
    exact_key_coverage = raw_keys.equals(output_keys)
    numeric = output[list(VARIABLES)].to_numpy(dtype=float)
    missing_count = int(output[list(VARIABLES)].isna().sum().sum())
    nonfinite_count = int((~np.isfinite(numeric)).sum())
    negative_count = int((output[list(POSITIVE_VARIABLES)] < 0.0).sum().sum())
    temperature_order_count = int(
        (output["temperature_min_c"] > output["temperature_max_c"]).sum()
    )

    raw_canonical = target[MEMBER_KEYS + list(VARIABLES)].copy()
    raw_structure = add_temperature_structure(raw_canonical)
    corrected_structure = add_temperature_structure(output)
    order_inversions = sum(
        count_order_inversions(raw_canonical, output, variable, variable)
        for variable in POSITIVE_VARIABLES
    )
    structure_inversions = sum(
        count_order_inversions(
            raw_structure, corrected_structure, variable, variable
        )
        for variable in ("temperature_center_c", "temperature_range_c")
    )

    target_dates = pd.to_datetime(factors["target_date"])
    fit_last_dates = pd.to_datetime(factors["fit_last_decision_date"])
    historical_2015 = target_dates.dt.year.eq(2015)
    leakage = int(
        (
            historical_2015
            & ((fit_last_dates + pd.Timedelta(days=6)) >= target_dates)
        ).sum()
        + (
            ~historical_2015
            & (factors["fit_last_year"].astype(int) >= target_dates.dt.year)
        ).sum()
    )
    member_counts = output.groupby(
        ["decision_date", "site_id", "local_date", "lead_day"]
    )["gefs_member"].nunique()
    lead_sequences = output.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=False
    )["lead_day"].apply(lambda values: sorted(values.astype(int).tolist()))
    full_seven_day_horizons = bool(
        lead_sequences.apply(lambda values: values == list(range(1, 8))).all()
    )
    mandatory_passed = all(
        [
            exact_key_coverage,
            missing_count == 0,
            nonfinite_count == 0,
            negative_count == 0,
            temperature_order_count == 0,
            order_inversions == 0,
            structure_inversions == 0,
            leakage == 0,
            int(member_counts.min()) == 5,
            int(member_counts.max()) == 5,
            full_seven_day_horizons,
        ]
    )
    audit = {
        "status": (
            "exact_schedule_frozen_causal_nonprecipitation_passed"
            if mandatory_passed
            else "exact_schedule_frozen_causal_nonprecipitation_failed"
        ),
        "mandatory_gate_passed": mandatory_passed,
        "target_year": int(target["target_year"].iloc[0]),
        "input_member_rows": int(len(target)),
        "output_branch_rows": int(len(output)),
        "decision_date_count": int(target["decision_date"].nunique()),
        "site_cycle_rows": int(
            target[["decision_date", "site_id"]].drop_duplicates().shape[0]
        ),
        "factor_rows": int(len(factors)),
        "minimum_fit_samples_per_site_lead": int(factors["fit_sample_count"].min()),
        "maximum_fit_samples_per_site_lead": int(factors["fit_sample_count"].max()),
        "fit_leakage_rows": leakage,
        "exact_sample_key_coverage": exact_key_coverage,
        "member_count_minimum": int(member_counts.min()),
        "member_count_maximum": int(member_counts.max()),
        "full_seven_day_horizons": full_seven_day_horizons,
        "missing_value_count": missing_count,
        "nonfinite_value_count": nonfinite_count,
        "negative_positive_only_value_count": negative_count,
        "temperature_order_error_count": temperature_order_count,
        "positive_variable_member_order_inversion_count": order_inversions,
        "temperature_structure_member_order_inversion_count": structure_inversions,
        "selected_nonprecipitation_branch": NONPRECIP_BRANCH,
        "target_era5_input_required": False,
        "target_or_future_era5_used_for_fit": False,
        "precipitation_correction_applied": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": "integrate_frozen_causal_precipitation_correction",
    }
    return output, factors, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    policy = json.loads(args.nonprecip_policy.read_text(encoding="utf-8"))
    output, factors, audit = build_frozen_nonprecipitation(
        target_weather=pd.read_csv(args.target_weather),
        history_gefs_c00=pd.read_csv(args.history_gefs_c00),
        history_era5=pd.read_csv(args.history_era5),
        policy=policy,
        minimum_samples=args.minimum_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    year = int(audit["target_year"])
    outputs = {
        "branches": args.output_dir
        / f"gefs_exact_schedule_{year}_frozen_nonprecip_branches_v1.csv",
        "factors": args.output_dir
        / f"gefs_exact_schedule_{year}_frozen_nonprecip_factors_v1.csv",
        "audit": args.output_dir
        / f"gefs_exact_schedule_{year}_frozen_nonprecip_audit_v1.json",
        "manifest": args.output_dir
        / f"gefs_exact_schedule_{year}_frozen_nonprecip_manifest_v1.json",
    }
    output.to_csv(outputs["branches"], index=False)
    factors.to_csv(outputs["factors"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "target_weather_sha256": sha256_file(args.target_weather),
            "history_gefs_c00_sha256": sha256_file(args.history_gefs_c00),
            "history_era5_sha256": sha256_file(args.history_era5),
            "nonprecip_policy_sha256": sha256_file(args.nonprecip_policy),
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
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"formal nonprecipitation gate failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-weather", type=Path, required=True)
    parser.add_argument("--history-gefs-c00", type=Path, required=True)
    parser.add_argument("--history-era5", type=Path, required=True)
    parser.add_argument("--nonprecip-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-samples", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

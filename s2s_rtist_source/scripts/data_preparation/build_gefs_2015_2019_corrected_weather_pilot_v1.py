#!/usr/bin/env python3
"""Apply frozen year-specific precipitation factors to the historical GEFS pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_full_weather_pilot_local_v1"
) / "gefs_2015_2019_full_weather_member_daily_v1.csv"
DEFAULT_OUTPUT_DIR = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_full_weather_pilot_local_v1"
) / "corrected_weather_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_scenario_consistent_pilot_contract_v1.json"
)
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_scenario_consistent_pilot_plan_v1.csv"
)
DEFAULT_OLD_EVIDENCE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_pre_crop_gate_cycle_evidence_v1.csv"
)
DEFAULT_CV_FACTORS = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "_local_gefs_weekly_two_stage_linear_scaling_cv_v1"
    / "weekly_two_stage_linear_fold_factors_v1.csv"
)
DEFAULT_2019_FACTORS = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "_local_gefs_weekly_two_stage_linear_scaling_2019_validation_v1"
    / "weekly_two_stage_linear_site_factors_2000_2018_v1.csv"
)

MEMBER_KEYS = ["target_year", "decision_date", "site_id", "gefs_member", "lead_day"]
PASSTHROUGH_FIELDS = [
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_factor_table(
    cv_path: Path, validation_path: Path, plan: pd.DataFrame
) -> pd.DataFrame:
    cv = pd.read_csv(cv_path)
    cv = cv.loc[cv["candidate_id"] == "weekly_two_stage_linear_site_only"].copy()
    cv = cv.rename(columns={"validation_year": "target_year"})
    validation = pd.read_csv(validation_path).copy()
    if "validation_year" not in validation:
        validation["validation_year"] = 2019
    validation = validation.rename(columns={"validation_year": "target_year"})
    columns = [
        "target_year",
        "site_id",
        "fit_first_year",
        "fit_last_year",
        "raw_ensemble_mean_7d_q90_mm",
        "overall_factor",
        "final_extreme_factor",
    ]
    factors = pd.concat([cv[columns], validation[columns]], ignore_index=True)
    factors["target_year"] = factors["target_year"].astype(int)
    factors = factors.merge(
        plan[["target_year", "decision_date", "fit_first_year", "fit_last_year"]],
        on=["target_year", "fit_first_year", "fit_last_year"],
        how="inner",
        validate="many_to_one",
    )
    if len(factors) != 25 or factors[["target_year", "site_id"]].duplicated().any():
        raise ValueError("expected exactly 25 unique year-site factor rows")
    if not (factors["fit_last_year"] < factors["target_year"]).all():
        raise ValueError("factor table leaks target or future years")
    if (factors["target_year"] == 2024).any():
        raise ValueError("2024 is forbidden in the historical pilot")
    return factors


def apply_frozen_factors(
    member: pd.DataFrame, factors: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = member.copy()
    data["target_year"] = pd.to_datetime(data["decision_date"]).dt.year
    if len(data) != 875 or data[MEMBER_KEYS].duplicated().any():
        raise ValueError("member input must contain 875 unique member-site-day rows")
    weekly = (
        data.groupby(["target_year", "decision_date", "site_id"], as_index=False)[
            "precipitation_mm_raw"
        ]
        .sum()
        .rename(columns={"precipitation_mm_raw": "member_sum_7d_mm"})
    )
    weekly["raw_ensemble_mean_7d_mm"] = weekly["member_sum_7d_mm"] / 5.0
    weekly = weekly.drop(columns="member_sum_7d_mm").merge(
        factors,
        on=["target_year", "decision_date", "site_id"],
        how="left",
        validate="one_to_one",
    )
    if weekly["overall_factor"].isna().any() or len(weekly) != 25:
        raise ValueError("factor coverage is incomplete")
    weekly["weekly_extreme_regime"] = (
        weekly["raw_ensemble_mean_7d_mm"]
        > weekly["raw_ensemble_mean_7d_q90_mm"]
    )
    weekly["base_factor"] = np.where(
        weekly["weekly_extreme_regime"],
        weekly["final_extreme_factor"],
        weekly["overall_factor"],
    )
    weekly["factor_shrinkage_alpha"] = float(alpha)
    weekly["effective_factor"] = 1.0 + alpha * (weekly["base_factor"] - 1.0)
    weekly["corrected_ensemble_mean_7d_mm"] = (
        weekly["raw_ensemble_mean_7d_mm"] * weekly["effective_factor"]
    )

    merge_columns = [
        "target_year",
        "decision_date",
        "site_id",
        "fit_first_year",
        "fit_last_year",
        "raw_ensemble_mean_7d_mm",
        "raw_ensemble_mean_7d_q90_mm",
        "weekly_extreme_regime",
        "base_factor",
        "factor_shrinkage_alpha",
        "effective_factor",
    ]
    corrected = data.merge(
        weekly[merge_columns],
        on=["target_year", "decision_date", "site_id"],
        how="left",
        validate="many_to_one",
    )
    corrected["precipitation_mm_corrected"] = (
        corrected["precipitation_mm_raw"] * corrected["effective_factor"]
    )
    if (corrected["precipitation_mm_corrected"] < 0).any():
        raise ValueError("corrected precipitation contains negative values")
    return corrected.sort_values(MEMBER_KEYS).reset_index(drop=True), weekly.sort_values(
        ["target_year", "site_id"]
    ).reset_index(drop=True)


def build_ensemble_daily(corrected: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "target_year",
        "decision_date",
        "site_id",
        "site_timezone",
        "local_date",
        "lead_day",
    ]
    aggregations: dict[str, Any] = {
        field: "mean" for field in PASSTHROUGH_FIELDS
    }
    aggregations.update(
        {
            "precipitation_mm_raw": "mean",
            "precipitation_mm_corrected": "mean",
            "effective_factor": "first",
            "weekly_extreme_regime": "first",
            "gefs_member": "nunique",
        }
    )
    daily = corrected.groupby(keys, as_index=False).agg(aggregations)
    daily = daily.rename(
        columns={
            "gefs_member": "member_count",
            **{field: f"{field}_mean" for field in PASSTHROUGH_FIELDS},
            "precipitation_mm_raw": "precipitation_mm_raw_mean",
            "precipitation_mm_corrected": "precipitation_mm_corrected_mean",
        }
    )
    if len(daily) != 175 or not (daily["member_count"] == 5).all():
        raise ValueError("ensemble daily output must contain 175 five-member rows")
    return daily.sort_values(["target_year", "site_id", "lead_day"]).reset_index(
        drop=True
    )


def build_audit(
    corrected: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    old_evidence: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    old = old_evidence.rename(
        columns={
            "effective_factor": "old_utc_day_effective_factor",
            "raw_ensemble_mean_7d_mm": "old_utc_day_raw_ensemble_mean_7d_mm",
        }
    )
    old["decision_date"] = pd.to_datetime(old["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    comparison = weekly.merge(
        old[
            [
                "target_year",
                "decision_date",
                "site_id",
                "old_utc_day_raw_ensemble_mean_7d_mm",
                "old_utc_day_effective_factor",
            ]
        ],
        on=["target_year", "decision_date", "site_id"],
        how="left",
        validate="one_to_one",
    )
    comparison["old_utc_evidence_comparison_available"] = comparison[
        "old_utc_day_effective_factor"
    ].notna()
    comparison["effective_factor_changed_from_old_utc_evidence"] = (
        comparison["old_utc_evidence_comparison_available"]
        & ~np.isclose(
            comparison["effective_factor"],
            comparison["old_utc_day_effective_factor"],
            rtol=0.0,
            atol=1e-6,
        )
    )
    weekly_from_daily = daily.groupby(["target_year", "site_id"])[
        "precipitation_mm_corrected_mean"
    ].sum()
    expected = comparison.set_index(["target_year", "site_id"])[
        "corrected_ensemble_mean_7d_mm"
    ]
    max_total_error = float((weekly_from_daily - expected).abs().max())
    audit = {
        "status": "corrected_weather_pilot_passed",
        "member_site_day_rows": int(len(corrected)),
        "ensemble_site_day_rows": int(len(daily)),
        "site_cycle_rows": int(len(weekly)),
        "factor_fit_leakage_rows": int(
            (weekly["fit_last_year"] >= weekly["target_year"]).sum()
        ),
        "target_year_2024_rows": int((corrected["target_year"] == 2024).sum()),
        "negative_corrected_precipitation_rows": int(
            (corrected["precipitation_mm_corrected"] < 0).sum()
        ),
        "non_precipitation_passthrough": True,
        "old_utc_evidence_comparable_rows": int(
            comparison["old_utc_evidence_comparison_available"].sum()
        ),
        "old_utc_evidence_factor_change_count": int(
            comparison["effective_factor_changed_from_old_utc_evidence"].sum()
        ),
        "maximum_absolute_weekly_total_error_mm": max_total_error,
        "correction_candidate_id": "weekly_two_stage_linear_site_factor_shrink_a075",
        "factor_shrinkage_alpha": 0.75,
        "swap_labels_generated": 0,
        "surrogate_training_started": False,
        "tta_started": False,
    }
    return audit, comparison


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["precipitation_correction"]["candidate_id"] != (
        "weekly_two_stage_linear_site_factor_shrink_a075"
    ):
        raise ValueError("pilot correction candidate mismatch")
    alpha = float(contract["precipitation_correction"]["factor_shrinkage_alpha"])
    plan = pd.read_csv(args.plan)
    factors = load_factor_table(args.cv_factors, args.validation_factors, plan)
    source = pd.read_csv(args.member_weather)
    corrected, weekly = apply_frozen_factors(source, factors, alpha)
    daily = build_ensemble_daily(corrected)
    audit, comparison = build_audit(
        corrected, daily, weekly, pd.read_csv(args.old_evidence)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "member": args.output_dir / "gefs_2015_2019_corrected_member_daily_v1.csv",
        "ensemble": args.output_dir / "gefs_2015_2019_corrected_ensemble_daily_v1.csv",
        "site_cycle": args.output_dir / "gefs_2015_2019_corrected_site_cycle_v1.csv",
        "comparison": args.output_dir
        / "gefs_2015_2019_local_vs_old_utc_factor_comparison_v1.csv",
        "audit": args.output_dir / "gefs_2015_2019_corrected_weather_audit_v1.json",
        "manifest": args.output_dir
        / "gefs_2015_2019_corrected_weather_manifest_v1.json",
    }
    corrected.to_csv(outputs["member"], index=False)
    daily.to_csv(outputs["ensemble"], index=False)
    weekly.to_csv(outputs["site_cycle"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "source_member_weather": str(args.member_weather),
        "source_member_weather_sha256": sha256_file(args.member_weather),
        "contract_sha256": sha256_file(args.contract),
        "cv_factor_table_sha256": sha256_file(args.cv_factors),
        "validation_factor_table_sha256": sha256_file(args.validation_factors),
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member-weather", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--old-evidence", type=Path, default=DEFAULT_OLD_EVIDENCE)
    parser.add_argument("--cv-factors", type=Path, default=DEFAULT_CV_FACTORS)
    parser.add_argument(
        "--validation-factors", type=Path, default=DEFAULT_2019_FACTORS
    )
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

#!/usr/bin/env python3
"""Aggregate stage-17 season schedules and audit the formal date-density budget."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_YEARS = (2015, 2016, 2017, 2018, 2019)
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
IRRIGATION_CANDIDATE_COUNT = 8
GEFS_MEMBER_COUNT = 5
HORIZON_DAYS = 7
REQUIRED_SCHEDULE_COLUMNS = {
    "site_id",
    "target_year",
    "schedule_index",
    "state_checkpoint_date",
    "state_dvs",
    "decision_date",
    "decision_doy",
    "horizon_start_date",
    "horizon_end_date",
    "horizon_days",
    "harvest_date",
    "dvs_threshold",
    "sampling_interval_days",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_split(year: int) -> str:
    if int(year) in {2015, 2016, 2017, 2018}:
        return "training"
    if int(year) == 2019:
        return "validation"
    raise ValueError(f"unsupported formal year: {year}")


def precipitation_fit_last_year(year: int) -> int:
    return int(year) - 1


def validate_schedule(
    schedule: pd.DataFrame,
    *,
    site_id: str,
    target_year: int,
) -> pd.DataFrame:
    missing = REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing fields: {sorted(missing)}")
    if schedule.empty:
        raise ValueError(f"empty season schedule for {target_year}/{site_id}")
    data = schedule.copy()
    if set(data["site_id"].astype(str)) != {str(site_id)}:
        raise ValueError(f"schedule site mismatch for {target_year}/{site_id}")
    years = set(pd.to_numeric(data["target_year"], errors="raise").astype(int))
    if years != {int(target_year)}:
        raise ValueError(f"schedule year mismatch for {target_year}/{site_id}")

    date_columns = (
        "state_checkpoint_date",
        "decision_date",
        "horizon_start_date",
        "horizon_end_date",
        "harvest_date",
    )
    for column in date_columns:
        data[column] = pd.to_datetime(data[column], errors="raise")
    data["state_dvs"] = pd.to_numeric(data["state_dvs"], errors="raise")
    data = data.sort_values("decision_date").reset_index(drop=True)

    expected_indices = list(range(len(data)))
    if data["schedule_index"].astype(int).tolist() != expected_indices:
        raise ValueError(f"schedule indices are not contiguous for {target_year}/{site_id}")
    if data["decision_date"].duplicated().any():
        raise ValueError(f"duplicate decision dates for {target_year}/{site_id}")
    gaps = data["decision_date"].diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError(f"decision dates are not spaced by seven days for {target_year}/{site_id}")
    if not data["state_checkpoint_date"].eq(
        data["decision_date"] - pd.Timedelta(days=1)
    ).all():
        raise ValueError(f"checkpoint dates are not decision minus one for {target_year}/{site_id}")
    if not data["horizon_start_date"].eq(data["decision_date"]).all():
        raise ValueError(f"horizon start mismatch for {target_year}/{site_id}")
    if not data["horizon_end_date"].eq(
        data["decision_date"] + pd.Timedelta(days=HORIZON_DAYS - 1)
    ).all():
        raise ValueError(f"horizon end mismatch for {target_year}/{site_id}")
    if not pd.to_numeric(data["horizon_days"], errors="raise").eq(HORIZON_DAYS).all():
        raise ValueError(f"horizon length mismatch for {target_year}/{site_id}")
    if not pd.to_numeric(data["sampling_interval_days"], errors="raise").eq(7).all():
        raise ValueError(f"sampling interval mismatch for {target_year}/{site_id}")
    if (data["state_dvs"] < 0.1 - 1e-12).any():
        raise ValueError(f"schedule contains pre-emergence decisions for {target_year}/{site_id}")
    if (data["horizon_end_date"] > data["harvest_date"]).any():
        raise ValueError(f"schedule contains incomplete harvest horizons for {target_year}/{site_id}")
    if not data["decision_date"].dt.year.eq(int(target_year)).all():
        raise ValueError(f"decision date year mismatch for {target_year}/{site_id}")
    if not pd.to_numeric(data["decision_doy"], errors="raise").astype(int).eq(
        data["decision_date"].dt.dayofyear
    ).all():
        raise ValueError(f"decision DOY mismatch for {target_year}/{site_id}")

    data["site_id"] = str(site_id)
    data["target_year"] = int(target_year)
    data["formal_split"] = formal_split(target_year)
    data["precipitation_fit_last_year"] = precipitation_fit_last_year(target_year)
    data["is_mature_checkpoint_dvs_ge_2"] = data["state_dvs"].ge(2.0)
    data["expected_irrigation_candidate_rows"] = IRRIGATION_CANDIDATE_COUNT
    data["expected_gefs_member_day_rows"] = GEFS_MEMBER_COUNT * HORIZON_DAYS
    data["weather_preparation_status"] = "pending_exact_cycle_all_variable_correction"
    for column in date_columns:
        data[column] = data[column].dt.strftime("%Y-%m-%d")
    return data


def load_trunk_schedule(
    input_root: Path,
    *,
    target_year: int,
    site_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trunk_root = input_root / str(target_year) / site_id / "trunk"
    schedule_path = trunk_root / "swap_season_decision_schedule_v1.csv"
    audit_path = trunk_root / "swap_season_trunk_smoke_audit_v1.json"
    if not schedule_path.is_file():
        raise FileNotFoundError(f"missing trunk schedule: {schedule_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"missing trunk audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not str(audit.get("status", "")).endswith("full_season_trunk_smoke_passed"):
        raise ValueError(f"trunk audit did not pass: {audit_path}")
    if str(audit.get("site_id")) != str(site_id):
        raise ValueError(f"trunk audit site mismatch: {audit_path}")
    if int(audit.get("target_year", -1)) != int(target_year):
        raise ValueError(f"trunk audit year mismatch: {audit_path}")
    if not bool(audit.get("swap_normal_completion", False)):
        raise ValueError(f"trunk SWAP completion missing: {audit_path}")
    schedule = validate_schedule(
        pd.read_csv(schedule_path),
        site_id=site_id,
        target_year=target_year,
    )
    source = {
        "target_year": int(target_year),
        "site_id": str(site_id),
        "schedule_path": str(schedule_path),
        "schedule_sha256": sha256_file(schedule_path),
        "trunk_audit_path": str(audit_path),
        "trunk_audit_sha256": sha256_file(audit_path),
        "decision_rows": int(len(schedule)),
        "first_decision_date": str(schedule["decision_date"].iloc[0]),
        "last_decision_date": str(schedule["decision_date"].iloc[-1]),
        "effective_crop_end_date": str(audit.get("effective_crop_end_date", "")),
        "weather_years": audit.get("weather_years", []),
        "swap_returncode": int(audit.get("swap_returncode", 0)),
        "swap_normal_completion": True,
    }
    return schedule, source


def build_summary(schedule: pd.DataFrame) -> pd.DataFrame:
    return (
        schedule.groupby(["target_year", "site_id", "formal_split"], as_index=False)
        .agg(
            decision_rows=("decision_date", "size"),
            first_decision_date=("decision_date", "min"),
            last_decision_date=("decision_date", "max"),
            minimum_state_dvs=("state_dvs", "min"),
            maximum_state_dvs=("state_dvs", "max"),
            mature_checkpoint_rows=("is_mature_checkpoint_dvs_ge_2", "sum"),
            expected_candidate_rows=("expected_irrigation_candidate_rows", "sum"),
            expected_gefs_member_day_rows=("expected_gefs_member_day_rows", "sum"),
        )
        .sort_values(["target_year", "site_id"])
        .reset_index(drop=True)
    )


def build_audit(
    schedule: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
    expected_sites: tuple[str, ...] = EXPECTED_SITES,
) -> dict[str, Any]:
    expected_pairs = {(year, site) for year in expected_years for site in expected_sites}
    actual_pairs = set(
        sources[["target_year", "site_id"]].itertuples(index=False, name=None)
    )
    duplicate_keys = int(
        schedule[["target_year", "site_id", "decision_date"]].duplicated().sum()
    )
    mature_rows = int(schedule["is_mature_checkpoint_dvs_ge_2"].sum())
    decision_rows = int(len(schedule))
    structural_passed = all(
        [
            actual_pairs == expected_pairs,
            len(sources) == len(expected_pairs),
            decision_rows > 0,
            duplicate_keys == 0,
            np.isfinite(schedule["state_dvs"].to_numpy(dtype=float)).all(),
            int((pd.to_datetime(schedule["horizon_end_date"]) > pd.to_datetime(schedule["harvest_date"])).sum()) == 0,
            int((schedule["state_dvs"] < 0.1 - 1e-12).sum()) == 0,
        ]
    )
    weather_expansion_allowed = structural_passed and mature_rows == 0
    if not structural_passed:
        status = "seasonal_date_density_design_failed"
        next_gate = "repair_season_schedule_inputs"
    elif mature_rows > 0:
        status = "seasonal_date_density_design_completed_maturity_rule_confirmation_required"
        next_gate = "confirm_whether_dvs_ge_2_checkpoints_remain_in_formal_schedule"
    else:
        status = "seasonal_date_density_design_passed"
        next_gate = "prepare_exact_schedule_frozen_corrected_gefs_without_swap_generation"
    return {
        "status": status,
        "structural_gate_passed": structural_passed,
        "teacher_rule_start_dvs_threshold": 0.1,
        "sampling_interval_days": 7,
        "forecast_horizon_days": 7,
        "full_horizon_before_harvest_required": True,
        "site_year_count": int(len(actual_pairs)),
        "expected_site_year_count": int(len(expected_pairs)),
        "decision_rows": decision_rows,
        "decision_rows_by_year": {
            str(year): int(schedule["target_year"].eq(year).sum()) for year in expected_years
        },
        "decision_rows_by_site": {
            site: int(schedule["site_id"].astype(str).eq(site).sum()) for site in expected_sites
        },
        "unique_calendar_decision_date_count": int(schedule["decision_date"].nunique()),
        "duplicate_site_decision_key_count": duplicate_keys,
        "minimum_state_dvs": float(schedule["state_dvs"].min()),
        "maximum_state_dvs": float(schedule["state_dvs"].max()),
        "mature_checkpoint_dvs_ge_2_row_count": mature_rows,
        "pre_maturity_checkpoint_dvs_lt_2_row_count": int(decision_rows - mature_rows),
        "maturity_upper_bound_teacher_confirmed": False,
        "expected_candidate_label_rows_all_teacher_rule_dates": int(decision_rows * IRRIGATION_CANDIDATE_COUNT),
        "expected_candidate_label_rows_if_dvs_ge_2_excluded": int((decision_rows - mature_rows) * IRRIGATION_CANDIDATE_COUNT),
        "expected_gefs_member_day_rows_all_teacher_rule_dates": int(decision_rows * GEFS_MEMBER_COUNT * HORIZON_DAYS),
        "estimated_checkpoint_prefix_swap_invocations": decision_rows,
        "estimated_candidate_branch_swap_invocations": int(decision_rows * IRRIGATION_CANDIDATE_COUNT),
        "estimated_additional_swap_invocations_current_verified_method": int(decision_rows * (IRRIGATION_CANDIDATE_COUNT + 1)),
        "weather_expansion_allowed": weather_expansion_allowed,
        "swap_simulation_performed": False,
        "weather_download_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": next_gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded-pilot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not args.bounded_pilot_root.is_dir():
        raise FileNotFoundError(f"bounded pilot root is missing: {args.bounded_pilot_root}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    schedules: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for year in EXPECTED_YEARS:
        for site in EXPECTED_SITES:
            schedule, source = load_trunk_schedule(
                args.bounded_pilot_root,
                target_year=year,
                site_id=site,
            )
            schedules.append(schedule)
            sources.append(source)
    combined = pd.concat(schedules, ignore_index=True).sort_values(
        ["target_year", "site_id", "decision_date"]
    ).reset_index(drop=True)
    source_frame = pd.DataFrame(sources).sort_values(
        ["target_year", "site_id"]
    ).reset_index(drop=True)
    summary = build_summary(combined)
    weather_plan = combined[
        [
            "target_year",
            "formal_split",
            "site_id",
            "decision_date",
            "state_checkpoint_date",
            "state_dvs",
            "horizon_end_date",
            "harvest_date",
            "precipitation_fit_last_year",
            "is_mature_checkpoint_dvs_ge_2",
            "expected_gefs_member_day_rows",
            "weather_preparation_status",
        ]
    ].copy()
    audit = build_audit(combined, source_frame)
    args.output_dir.mkdir(parents=True)
    outputs = {
        "schedule": args.output_dir / "gefs_2015_2019_seasonal_decision_schedule_v1.csv",
        "summary": args.output_dir / "gefs_2015_2019_seasonal_date_density_summary_v1.csv",
        "weather_plan": args.output_dir / "gefs_2015_2019_exact_cycle_weather_plan_v1.csv",
        "sources": args.output_dir / "gefs_2015_2019_trunk_schedule_sources_v1.csv",
        "audit": args.output_dir / "gefs_2015_2019_seasonal_date_density_audit_v1.json",
        "manifest": args.output_dir / "gefs_2015_2019_seasonal_date_density_manifest_v1.json",
    }
    combined.to_csv(outputs["schedule"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    weather_plan.to_csv(outputs["weather_plan"], index=False)
    source_frame.to_csv(outputs["sources"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "bounded_pilot_root": str(args.bounded_pilot_root),
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "swap_simulation_performed": False,
        "weather_download_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["structural_gate_passed"]:
        raise RuntimeError(f"seasonal date-density design failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

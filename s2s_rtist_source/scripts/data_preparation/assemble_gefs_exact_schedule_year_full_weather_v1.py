#!/usr/bin/env python3
"""Close one formal GEFS year by merging and re-auditing its completed batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    sha256_file,
    write_json,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS


BATCH_AUDIT_NAME = "gefs_exact_schedule_batch_full_weather_audit_v1.json"
CYCLE_AUDIT_NAME = "gefs_2015_2019_full_weather_audit_v1.json"
CYCLE_WEATHER_NAME = "gefs_2015_2019_full_weather_member_daily_v1.csv"
KEY_COLUMNS = ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
WEATHER_COLUMNS = [
    "precipitation_mm_raw",
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]
NONNEGATIVE_COLUMNS = [
    "precipitation_mm_raw",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required audit is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _planned_year(
    batch_budget_path: Path, cycle_plan_path: Path, target_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batches = pd.read_csv(batch_budget_path)
    cycles = pd.read_csv(cycle_plan_path)
    required_batch = {"batch_id", "target_year", "cycle_count", "expected_output_rows"}
    required_cycle = {
        "batch_id",
        "target_year",
        "decision_date",
        "required_site_count",
        "required_sites",
        "expected_member_count",
        "expected_lead_day_count",
        "expected_output_rows",
    }
    missing_batch = required_batch.difference(batches.columns)
    missing_cycle = required_cycle.difference(cycles.columns)
    if missing_batch:
        raise ValueError(f"batch budget missing fields: {sorted(missing_batch)}")
    if missing_cycle:
        raise ValueError(f"cycle plan missing fields: {sorted(missing_cycle)}")

    batches = batches.loc[batches["target_year"].eq(target_year)].copy()
    cycles = cycles.loc[cycles["target_year"].eq(target_year)].copy()
    if batches.empty or cycles.empty:
        raise ValueError(f"target year is absent from the formal plan: {target_year}")
    if batches["batch_id"].duplicated().any():
        raise ValueError(f"duplicate batch id in target year {target_year}")
    if cycles["decision_date"].duplicated().any():
        raise ValueError(f"duplicate decision date in target year {target_year}")
    if set(cycles["batch_id"]) != set(batches["batch_id"]):
        raise ValueError(f"batch/cycle plan membership mismatch for {target_year}")

    expected_cycles = batches.set_index("batch_id")["cycle_count"].astype(int)
    actual_cycles = cycles.groupby("batch_id").size()
    if not actual_cycles.equals(expected_cycles.reindex(actual_cycles.index)):
        raise ValueError(f"batch cycle counts mismatch for {target_year}")
    if int(cycles["expected_output_rows"].sum()) != int(
        batches["expected_output_rows"].sum()
    ):
        raise ValueError(f"year output row budget mismatch for {target_year}")

    cycles["decision_date"] = pd.to_datetime(
        cycles["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    batches = batches.sort_values("batch_id").reset_index(drop=True)
    cycles = cycles.sort_values(["decision_date", "batch_id"]).reset_index(drop=True)
    return batches, cycles


def _expected_cycle_keys(cycle: Any) -> pd.DataFrame:
    sites = [item for item in str(cycle.required_sites).split(",") if item]
    if len(sites) != int(cycle.required_site_count):
        raise ValueError(f"site count mismatch in plan for {cycle.decision_date}")
    if int(cycle.expected_member_count) != len(GEFS_REFORECAST_MEMBERS):
        raise ValueError(f"member count mismatch in plan for {cycle.decision_date}")
    if int(cycle.expected_lead_day_count) != 7:
        raise ValueError(f"lead-day count mismatch in plan for {cycle.decision_date}")
    decision = pd.Timestamp(cycle.decision_date)
    rows = [
        {
            "decision_date": decision.strftime("%Y-%m-%d"),
            "site_id": site,
            "gefs_member": member,
            "local_date": (decision + pd.Timedelta(days=lead - 1)).strftime("%Y-%m-%d"),
            "lead_day": lead,
        }
        for site in sites
        for member in GEFS_REFORECAST_MEMBERS
        for lead in range(1, 8)
    ]
    expected = pd.DataFrame(rows, columns=KEY_COLUMNS)
    if len(expected) != int(cycle.expected_output_rows):
        raise ValueError(f"planned row count mismatch for {cycle.decision_date}")
    return expected


def _validate_cycle(
    output_root: Path, cycle: Any
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cycle_dir = output_root / str(cycle.batch_id) / str(cycle.decision_date).replace("-", "")
    batch_audit = _read_json(output_root / str(cycle.batch_id) / BATCH_AUDIT_NAME)
    if not (
        batch_audit.get("status") == "exact_schedule_full_weather_local_batch_passed"
        and batch_audit.get("mandatory_structural_gate_passed") is True
        and batch_audit.get("full_three_hourly_records") is True
        and batch_audit.get("all_required_weather_variables_retained") is True
        and int(batch_audit.get("member_count", -1)) == len(GEFS_REFORECAST_MEMBERS)
        and batch_audit.get("temporary_grib_retained") is False
    ):
        raise ValueError(f"batch gate did not pass: {cycle.batch_id}")

    cycle_audit = _read_json(cycle_dir / CYCLE_AUDIT_NAME)
    if not (
        cycle_audit.get("status") == "full_weather_local_extraction_passed"
        and int(cycle_audit.get("row_count", -1)) == int(cycle.expected_output_rows)
        and int(cycle_audit.get("expected_row_count", -1))
        == int(cycle.expected_output_rows)
        and int(cycle_audit.get("member_count", -1)) == len(GEFS_REFORECAST_MEMBERS)
        and int(cycle_audit.get("canonical_missing_value_count", -1)) == 0
        and int(cycle_audit.get("canonical_nonfinite_value_count", -1)) == 0
        and int(cycle_audit.get("duplicate_sample_key_count", -1)) == 0
        and int(cycle_audit.get("retained_grib_file_count", -1)) == 0
    ):
        raise ValueError(f"cycle gate did not pass: {cycle.decision_date}")
    retained_gribs = list(cycle_dir.rglob("*.grib2"))
    if retained_gribs:
        raise ValueError(f"cycle retained temporary GRIB files: {cycle.decision_date}")

    weather_path = cycle_dir / CYCLE_WEATHER_NAME
    if not weather_path.is_file():
        raise FileNotFoundError(f"cycle weather is missing: {weather_path}")
    weather = pd.read_csv(weather_path)
    missing = set(KEY_COLUMNS + WEATHER_COLUMNS).difference(weather.columns)
    if missing:
        raise ValueError(f"cycle weather missing fields: {sorted(missing)}")
    for column in ("decision_date", "local_date"):
        weather[column] = pd.to_datetime(weather[column], errors="raise").dt.strftime(
            "%Y-%m-%d"
        )
    weather["lead_day"] = pd.to_numeric(weather["lead_day"], errors="raise").astype(int)

    expected = _expected_cycle_keys(cycle).sort_values(KEY_COLUMNS).reset_index(drop=True)
    actual = weather[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError(f"exact sample-key coverage mismatch: {cycle.decision_date}")
    numeric = weather[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"missing or non-finite weather values: {cycle.decision_date}")
    if (numeric[NONNEGATIVE_COLUMNS] < 0.0).any().any():
        raise ValueError(f"negative physical weather values: {cycle.decision_date}")
    if (numeric["temperature_min_c"] > numeric["temperature_max_c"]).any():
        raise ValueError(f"minimum temperature exceeds maximum: {cycle.decision_date}")
    return weather, cycle_audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    for path in (args.batch_budget, args.cycle_plan):
        if not path.is_file():
            raise FileNotFoundError(f"required plan input is missing: {path}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite year output: {args.output_dir}")

    batches, cycles = _planned_year(
        args.batch_budget, args.cycle_plan, int(args.target_year)
    )
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    weather_parts: list[pd.DataFrame] = []
    cycle_status_rows: list[dict[str, Any]] = []
    checked_batches: set[str] = set()
    for cycle in cycles.itertuples(index=False):
        weather, cycle_audit = _validate_cycle(args.output_root, cycle)
        weather_parts.append(weather)
        checked_batches.add(str(cycle.batch_id))
        cycle_status_rows.append(
            {
                "target_year": int(args.target_year),
                "batch_id": str(cycle.batch_id),
                "decision_date": str(cycle.decision_date),
                "site_count": int(cycle.required_site_count),
                "member_count": int(cycle_audit["member_count"]),
                "expected_rows": int(cycle.expected_output_rows),
                "actual_rows": len(weather),
                "status": "verified_completed_cycle",
            }
        )

    merged = pd.concat(weather_parts, ignore_index=True)
    merged = merged.sort_values(KEY_COLUMNS).reset_index(drop=True)
    expected_rows = int(cycles["expected_output_rows"].sum())
    duplicate_count = int(merged.duplicated(KEY_COLUMNS).sum())
    if len(merged) != expected_rows or duplicate_count:
        raise ValueError("year row count or duplicate-key gate failed")
    if checked_batches != set(batches["batch_id"].astype(str)):
        raise ValueError("not all planned year batches were checked")
    retained_year_gribs = [
        path
        for batch_id in batches["batch_id"].astype(str)
        for path in (args.output_root / batch_id).rglob("*.grib2")
    ]
    if retained_year_gribs:
        raise ValueError("temporary GRIB files remain under the target-year batch roots")

    weather_path = args.output_dir / f"gefs_exact_schedule_{args.target_year}_raw_full_weather_v1.csv"
    status_path = args.output_dir / f"gefs_exact_schedule_{args.target_year}_cycle_status_v1.csv"
    audit_path = args.output_dir / f"gefs_exact_schedule_{args.target_year}_raw_full_weather_audit_v1.json"
    manifest_path = args.output_dir / f"gefs_exact_schedule_{args.target_year}_raw_full_weather_manifest_v1.json"
    merged.to_csv(weather_path, index=False)
    pd.DataFrame(cycle_status_rows).to_csv(status_path, index=False)
    audit = {
        "status": "exact_schedule_year_raw_full_weather_passed",
        "mandatory_year_gate_passed": True,
        "target_year": int(args.target_year),
        "batch_count": len(batches),
        "cycle_count": len(cycles),
        "site_cycle_rows": int(cycles["required_site_count"].sum()),
        "row_count": len(merged),
        "expected_row_count": expected_rows,
        "member_count": len(GEFS_REFORECAST_MEMBERS),
        "members": list(GEFS_REFORECAST_MEMBERS),
        "lead_day_count": 7,
        "weather_variable_count": len(WEATHER_COLUMNS),
        "weather_variables": WEATHER_COLUMNS,
        "duplicate_sample_key_count": duplicate_count,
        "canonical_missing_value_count": 0,
        "canonical_nonfinite_value_count": 0,
        "full_three_hourly_records": True,
        "temporary_grib_retained": False,
        "weather_correction_applied": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "training_eligible": False,
        "tta_performed": False,
        "next_gate": "apply_frozen_causal_all_variable_weather_correction_for_target_year",
    }
    write_json(audit_path, audit)
    manifest = {
        "status": audit["status"],
        "inputs": {
            "batch_budget_sha256": sha256_file(args.batch_budget),
            "cycle_plan_sha256": sha256_file(args.cycle_plan),
        },
        "outputs": {
            "weather": {"path": weather_path.name, "sha256": sha256_file(weather_path)},
            "cycle_status": {"path": status_path.name, "sha256": sha256_file(status_path)},
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
        },
    }
    write_json(manifest_path, manifest)
    return {
        "weather": weather_path,
        "cycle_status": status_path,
        "audit": audit_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-budget", type=Path, required=True)
    parser.add_argument("--cycle-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

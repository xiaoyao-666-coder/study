#!/usr/bin/env python3
"""Preflight exact-schedule GEFS full-weather bytes without payload download."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data_preparation.extract_gefs_2015_2019_full_weather_pilot_v1 import (
    preflight_extraction,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS
from s2s_rtist.weather.gefs_reforecast_full_weather import REQUIRED_PRODUCT_SPECS


EXPECTED_YEARS = (2015, 2016, 2017, 2018, 2019)
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
EXPECTED_SITE_CYCLE_ROWS = 338
EXPECTED_UNIQUE_CYCLES = 239
LOCAL_ONE_PACKAGE_LIMIT_BYTES = 6_500_000_000
REQUIRED_PLAN_COLUMNS = {
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
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def strict_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("boolean plan field contains values other than true/false")
    return normalized.eq("true")


def validate_weather_plan(
    plan: pd.DataFrame,
    *,
    expected_site_cycle_rows: int = EXPECTED_SITE_CYCLE_ROWS,
    expected_unique_cycles: int = EXPECTED_UNIQUE_CYCLES,
    expected_years: tuple[int, ...] = EXPECTED_YEARS,
    expected_sites: tuple[str, ...] = EXPECTED_SITES,
) -> pd.DataFrame:
    missing = REQUIRED_PLAN_COLUMNS.difference(plan.columns)
    if missing:
        raise ValueError(f"exact-cycle weather plan missing fields: {sorted(missing)}")
    data = plan.copy()
    for column in (
        "decision_date",
        "state_checkpoint_date",
        "horizon_end_date",
        "harvest_date",
    ):
        data[column] = pd.to_datetime(data[column], errors="raise")
    data["target_year"] = pd.to_numeric(data["target_year"], errors="raise").astype(int)
    data["state_dvs"] = pd.to_numeric(data["state_dvs"], errors="raise")
    data["precipitation_fit_last_year"] = pd.to_numeric(
        data["precipitation_fit_last_year"], errors="raise"
    ).astype(int)
    data["expected_gefs_member_day_rows"] = pd.to_numeric(
        data["expected_gefs_member_day_rows"], errors="raise"
    ).astype(int)
    data["is_mature_checkpoint_dvs_ge_2"] = strict_bool(
        data["is_mature_checkpoint_dvs_ge_2"]
    )

    if len(data) != int(expected_site_cycle_rows):
        raise ValueError(
            f"site-cycle row count mismatch: {len(data)} != {expected_site_cycle_rows}"
        )
    if data[["target_year", "site_id", "decision_date"]].duplicated().any():
        raise ValueError("exact-cycle weather plan contains duplicate site-date keys")
    if tuple(sorted(data["target_year"].unique().tolist())) != expected_years:
        raise ValueError("exact-cycle weather plan year set mismatch")
    if set(data["site_id"].astype(str)) != set(expected_sites):
        raise ValueError("exact-cycle weather plan site set mismatch")
    if data["decision_date"].nunique() != int(expected_unique_cycles):
        raise ValueError(
            f"unique cycle count mismatch: {data['decision_date'].nunique()} != {expected_unique_cycles}"
        )
    if not data["decision_date"].dt.year.eq(data["target_year"]).all():
        raise ValueError("decision date year differs from target year")
    if not data["state_checkpoint_date"].eq(
        data["decision_date"] - pd.Timedelta(days=1)
    ).all():
        raise ValueError("checkpoint date differs from decision minus one")
    if not data["horizon_end_date"].eq(
        data["decision_date"] + pd.Timedelta(days=6)
    ).all():
        raise ValueError("weather horizon is not seven inclusive days")
    if (data["horizon_end_date"] > data["harvest_date"]).any():
        raise ValueError("weather plan contains a horizon after harvest")
    if (data["state_dvs"] < 0.1 - 1e-12).any():
        raise ValueError("weather plan contains a pre-emergence checkpoint")
    if data["is_mature_checkpoint_dvs_ge_2"].any() or data["state_dvs"].ge(2.0).any():
        raise ValueError("weather plan contains a mature DVS>=2 checkpoint")
    if not data["precipitation_fit_last_year"].eq(data["target_year"] - 1).all():
        raise ValueError("historical precipitation correction fit boundary is not causal")
    expected_splits = data["target_year"].map(
        {2015: "training", 2016: "training", 2017: "training", 2018: "training", 2019: "validation"}
    )
    if not data["formal_split"].astype(str).eq(expected_splits).all():
        raise ValueError("formal training/validation split mismatch")
    if not data["expected_gefs_member_day_rows"].eq(35).all():
        raise ValueError("each site-cycle must require five members by seven days")
    numeric = data[["state_dvs"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("weather plan contains nonfinite numeric values")
    for column in (
        "decision_date",
        "state_checkpoint_date",
        "horizon_end_date",
        "harvest_date",
    ):
        data[column] = data[column].dt.strftime("%Y-%m-%d")
    return data.sort_values(["decision_date", "site_id"]).reset_index(drop=True)


def build_cycle_plan(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decision_date, group in plan.groupby("decision_date", sort=True):
        sites = sorted(group["site_id"].astype(str).unique().tolist())
        years = sorted(group["target_year"].astype(int).unique().tolist())
        if len(years) != 1:
            raise ValueError(f"one cycle spans multiple target years: {decision_date}")
        rows.append(
            {
                "target_year": years[0],
                "decision_date": str(decision_date),
                "required_site_count": len(sites),
                "required_sites": ",".join(sites),
                "expected_member_count": len(GEFS_REFORECAST_MEMBERS),
                "expected_lead_day_count": 7,
                "expected_output_rows": len(sites) * len(GEFS_REFORECAST_MEMBERS) * 7,
            }
        )
    return pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)


def build_budget_tables(
    preflight: pd.DataFrame,
    cycle_plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "cycle_date",
        "gefs_member",
        "product_id",
        "selected_range_bytes",
        "index_network_bytes_this_run",
    }
    missing = required.difference(preflight.columns)
    if missing:
        raise ValueError(f"preflight table missing fields: {sorted(missing)}")
    if preflight[["cycle_date", "gefs_member", "product_id"]].duplicated().any():
        raise ValueError("preflight contains duplicate cycle-member-product rows")
    cycle = (
        preflight.groupby("cycle_date", as_index=False)
        .agg(
            task_member_count=("gefs_member", "nunique"),
            product_row_count=("product_id", "size"),
            selected_range_bytes=("selected_range_bytes", "sum"),
            index_network_bytes_this_run=("index_network_bytes_this_run", "sum"),
        )
        .rename(columns={"cycle_date": "decision_date"})
    )
    cycle = cycle.merge(cycle_plan, on="decision_date", how="left", validate="one_to_one")
    if cycle["required_site_count"].isna().any():
        raise ValueError("preflight contains a cycle outside the exact schedule")
    cycle["selected_range_gib"] = cycle["selected_range_bytes"] / (1024**3)
    cycle = cycle.sort_values("decision_date").reset_index(drop=True)
    year = (
        cycle.groupby("target_year", as_index=False)
        .agg(
            cycle_count=("decision_date", "size"),
            site_cycle_rows=("required_site_count", "sum"),
            expected_output_rows=("expected_output_rows", "sum"),
            selected_range_bytes=("selected_range_bytes", "sum"),
            index_network_bytes_this_run=("index_network_bytes_this_run", "sum"),
        )
        .sort_values("target_year")
        .reset_index(drop=True)
    )
    year["selected_range_gib"] = year["selected_range_bytes"] / (1024**3)
    return cycle, year


def build_audit(
    *,
    plan: pd.DataFrame,
    cycle_plan: pd.DataFrame,
    preflight: pd.DataFrame,
    inventory: pd.DataFrame,
    cycle_budget: pd.DataFrame,
) -> dict[str, Any]:
    expected_tasks = len(cycle_plan) * len(GEFS_REFORECAST_MEMBERS)
    expected_product_rows = expected_tasks * len(REQUIRED_PRODUCT_SPECS)
    selected_bytes = int(preflight["selected_range_bytes"].sum())
    preflight_network_bytes = int(
        preflight["index_network_bytes_this_run"].sum()
        + inventory["network_bytes_this_run"].sum()
    )
    structural_passed = all(
        [
            len(plan) == EXPECTED_SITE_CYCLE_ROWS,
            len(cycle_plan) == EXPECTED_UNIQUE_CYCLES,
            len(inventory) == expected_tasks,
            len(preflight) == expected_product_rows,
            preflight[["cycle_date", "gefs_member", "product_id"]].duplicated().sum() == 0,
            int(cycle_budget["selected_range_bytes"].sum()) == selected_bytes,
        ]
    )
    within_local_limit = selected_bytes <= LOCAL_ONE_PACKAGE_LIMIT_BYTES
    return {
        "status": (
            "exact_schedule_full_weather_preflight_passed"
            if structural_passed
            else "exact_schedule_full_weather_preflight_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "site_cycle_rows": int(len(plan)),
        "unique_cycle_count": int(len(cycle_plan)),
        "member_count": int(len(GEFS_REFORECAST_MEMBERS)),
        "cycle_member_task_count": int(len(inventory)),
        "required_product_count_per_task": int(len(REQUIRED_PRODUCT_SPECS)),
        "cycle_member_product_row_count": int(len(preflight)),
        "expected_corrected_member_day_rows": int(plan["expected_gefs_member_day_rows"].sum()),
        "selected_range_bytes": selected_bytes,
        "selected_range_gib": float(selected_bytes / (1024**3)),
        "preflight_inventory_and_index_network_bytes_this_run": preflight_network_bytes,
        "local_one_package_limit_bytes": LOCAL_ONE_PACKAGE_LIMIT_BYTES,
        "within_local_one_package_limit": within_local_limit,
        "largest_cycle_selected_range_bytes": int(cycle_budget["selected_range_bytes"].max()),
        "largest_cycle_selected_range_gib": float(cycle_budget["selected_range_gib"].max()),
        "smallest_cycle_selected_range_bytes": int(cycle_budget["selected_range_bytes"].min()),
        "product_payload_download_started": False,
        "temporary_grib_retained": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "prepare_local_exact_schedule_full_weather_extraction"
            if structural_passed and within_local_limit
            else "review_budget_and_design_resumable_batched_extraction"
            if structural_passed
            else "repair_exact_schedule_preflight"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not args.weather_plan.is_file():
        raise FileNotFoundError(f"exact-cycle weather plan is missing: {args.weather_plan}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    plan = validate_weather_plan(pd.read_csv(args.weather_plan))
    cycle_plan = build_cycle_plan(plan)
    tasks = [
        (decision_date, member)
        for decision_date in cycle_plan["decision_date"].tolist()
        for member in GEFS_REFORECAST_MEMBERS
    ]
    _, preflight, inventory = preflight_extraction(
        tasks=tasks,
        cache_dir=args.output_dir / "cache",
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
    )
    cycle_budget, year_budget = build_budget_tables(preflight, cycle_plan)
    audit = build_audit(
        plan=plan,
        cycle_plan=cycle_plan,
        preflight=preflight,
        inventory=inventory,
        cycle_budget=cycle_budget,
    )
    outputs = {
        "cycle_plan": args.output_dir / "gefs_exact_schedule_cycle_plan_v1.csv",
        "preflight": args.output_dir / "gefs_exact_schedule_product_preflight_v1.csv",
        "inventory": args.output_dir / "gefs_exact_schedule_inventory_manifest_v1.csv",
        "cycle_budget": args.output_dir / "gefs_exact_schedule_cycle_budget_v1.csv",
        "year_budget": args.output_dir / "gefs_exact_schedule_year_budget_v1.csv",
        "audit": args.output_dir / "gefs_exact_schedule_full_weather_preflight_audit_v1.json",
        "manifest": args.output_dir / "gefs_exact_schedule_full_weather_preflight_manifest_v1.json",
    }
    cycle_plan.to_csv(outputs["cycle_plan"], index=False)
    preflight.to_csv(outputs["preflight"], index=False)
    inventory.to_csv(outputs["inventory"], index=False)
    cycle_budget.to_csv(outputs["cycle_budget"], index=False)
    year_budget.to_csv(outputs["year_budget"], index=False)
    write_json(outputs["audit"], audit)
    manifest = {
        "status": audit["status"],
        "weather_plan": {
            "path": str(args.weather_plan),
            "sha256": sha256_file(args.weather_plan),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "cache_policy": {
            "inventory_xml_retained": True,
            "index_files_retained": True,
            "product_payload_downloaded": False,
            "temporary_grib_retained": False,
        },
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    write_json(outputs["manifest"], manifest)
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"exact-schedule preflight failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

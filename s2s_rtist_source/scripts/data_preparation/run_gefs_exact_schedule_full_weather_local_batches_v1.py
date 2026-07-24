#!/usr/bin/env python3
"""Run the formal exact-schedule GEFS full-weather extraction in resumable local batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.data_preparation.extract_gefs_2015_2019_full_weather_pilot_v1 import (
    run_extraction,
)
from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    EXPECTED_SITES,
    sha256_file,
    write_json,
)
from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS


REQUIRED_BATCH_COLUMNS = {
    "batch_id",
    "target_year",
    "cycle_count",
    "first_decision_date",
    "last_decision_date",
}
REQUIRED_CYCLE_COLUMNS = {
    "batch_id",
    "target_year",
    "decision_date",
    "required_site_count",
    "required_sites",
    "expected_output_rows",
}
EXPECTED_CYCLES = 239
EXPECTED_BATCHES = 61


def read_batch_and_cycle_plan(
    batch_budget_path: Path, cycle_plan_path: Path, batch_id: str
) -> tuple[pd.Series, pd.DataFrame]:
    batch = pd.read_csv(batch_budget_path)
    cycles = pd.read_csv(cycle_plan_path)
    missing_batch = REQUIRED_BATCH_COLUMNS.difference(batch.columns)
    missing_cycles = REQUIRED_CYCLE_COLUMNS.difference(cycles.columns)
    if missing_batch:
        raise ValueError(f"batch budget missing fields: {sorted(missing_batch)}")
    if missing_cycles:
        raise ValueError(f"cycle plan missing fields: {sorted(missing_cycles)}")
    if len(batch) != EXPECTED_BATCHES:
        raise ValueError("batch budget count mismatch")
    if len(cycles) != EXPECTED_CYCLES:
        raise ValueError("cycle plan count mismatch")
    selected_batch = batch.loc[batch["batch_id"].eq(batch_id)]
    if len(selected_batch) != 1:
        raise ValueError(f"expected one batch row for {batch_id!r}")
    selected_cycles = cycles.loc[cycles["batch_id"].eq(batch_id)].copy()
    selected_cycles["decision_date"] = pd.to_datetime(
        selected_cycles["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    selected_cycles = selected_cycles.sort_values("decision_date").reset_index(drop=True)
    expected_count = int(selected_batch.iloc[0]["cycle_count"])
    if len(selected_cycles) != expected_count:
        raise ValueError(f"cycle count mismatch inside {batch_id}")
    if selected_cycles["decision_date"].duplicated().any():
        raise ValueError(f"duplicate decision date inside {batch_id}")
    for row in selected_cycles.itertuples(index=False):
        sites = [item for item in str(row.required_sites).split(",") if item]
        if len(sites) != int(row.required_site_count):
            raise ValueError(f"site count mismatch for {row.decision_date}")
        if not set(sites).issubset(EXPECTED_SITES):
            raise ValueError(f"unexpected site in {row.decision_date}: {sites}")
        if int(row.expected_output_rows) != len(sites) * len(GEFS_REFORECAST_MEMBERS) * 7:
            raise ValueError(f"expected row count mismatch for {row.decision_date}")
    return selected_batch.iloc[0], selected_cycles


def cycle_is_complete(cycle_dir: Path, expected_rows: int) -> bool:
    audit_path = cycle_dir / "gefs_2015_2019_full_weather_audit_v1.json"
    weather_path = cycle_dir / "gefs_2015_2019_full_weather_member_daily_v1.csv"
    if not audit_path.is_file() or not weather_path.is_file():
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return (
        audit.get("status") == "full_weather_local_extraction_passed"
        and int(audit.get("row_count", -1)) == expected_rows
        and int(audit.get("retained_grib_file_count", -1)) == 0
        and not list(cycle_dir.rglob("*.grib2"))
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    for path in (args.batch_budget, args.cycle_plan):
        if not path.is_file():
            raise FileNotFoundError(f"required plan input is missing: {path}")
    batch, cycles = read_batch_and_cycle_plan(
        args.batch_budget, args.cycle_plan, args.batch_id
    )
    batch_dir = args.output_root / args.batch_id
    if batch_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite batch directory: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=args.resume)
    cycle_rows: list[dict[str, Any]] = []
    for row in cycles.itertuples(index=False):
        cycle_dir = batch_dir / str(row.decision_date).replace("-", "")
        sites = tuple(item for item in str(row.required_sites).split(",") if item)
        if cycle_is_complete(cycle_dir, int(row.expected_output_rows)):
            status = "reused_completed_cycle"
        else:
            if cycle_dir.exists() and not args.resume:
                raise FileExistsError(f"incomplete cycle directory exists: {cycle_dir}")
            cycle_dir.mkdir(parents=True, exist_ok=True)
            outputs = run_extraction(
                cycles=(str(row.decision_date),),
                site_ids=sites,
                members=GEFS_REFORECAST_MEMBERS,
                output_dir=cycle_dir,
                timeout=args.timeout,
                retries=args.retries,
                workers=min(args.download_workers, len(GEFS_REFORECAST_MEMBERS)),
                product_workers=args.download_workers,
                product_range_workers=args.range_workers,
            )
            status = "generated_cycle"
            if not cycle_is_complete(cycle_dir, int(row.expected_output_rows)):
                raise RuntimeError(f"cycle did not pass full-weather audit: {cycle_dir}")
            cycle_rows.append(
                {
                    "decision_date": str(row.decision_date),
                    "status": status,
                    "site_count": len(sites),
                    "expected_output_rows": int(row.expected_output_rows),
                    "cycle_audit": str(outputs["audit"]),
                }
            )
            continue
        cycle_rows.append(
            {
                "decision_date": str(row.decision_date),
                "status": status,
                "site_count": len(sites),
                "expected_output_rows": int(row.expected_output_rows),
                "cycle_audit": str(
                    cycle_dir / "gefs_2015_2019_full_weather_audit_v1.json"
                ),
            }
        )
    cycle_status = pd.DataFrame(cycle_rows).sort_values("decision_date")
    status_path = batch_dir / "gefs_exact_schedule_batch_cycle_status_v1.csv"
    cycle_status.to_csv(status_path, index=False)
    audit = {
        "status": "exact_schedule_full_weather_local_batch_passed",
        "mandatory_structural_gate_passed": True,
        "batch_id": args.batch_id,
        "target_year": int(batch["target_year"]),
        "cycle_count": len(cycles),
        "expected_rows": int(cycles["expected_output_rows"].sum()),
        "member_count": len(GEFS_REFORECAST_MEMBERS),
        "required_product_count_per_task": 7,
        "download_workers": int(args.download_workers),
        "download_parallelism_unit": "cycle_member_product",
        "range_workers_per_product": int(args.range_workers),
        "full_three_hourly_records": True,
        "all_required_weather_variables_retained": True,
        "payload_download_started": True,
        "temporary_grib_retained": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "training_eligible": False,
        "tta_performed": False,
        "completed_cycle_count": int(
            cycle_status["status"].isin(
                ["generated_cycle", "reused_completed_cycle"]
            ).sum()
        ),
        "next_gate": "review_batch_full_weather_outputs_before_next_batch",
    }
    audit_path = batch_dir / "gefs_exact_schedule_batch_full_weather_audit_v1.json"
    write_json(audit_path, audit)
    manifest = {
        "status": audit["status"],
        "inputs": {
            "batch_budget_sha256": sha256_file(args.batch_budget),
            "cycle_plan_sha256": sha256_file(args.cycle_plan),
        },
        "outputs": {
            "cycle_status": {
                "path": status_path.name,
                "sha256": sha256_file(status_path),
            },
            "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path)},
        },
        "resume_policy": "completed cycle audits are reused; incomplete cycles rerun with --resume",
        "full_three_hourly_records": True,
        "product_payload_download_started": True,
        "temporary_grib_retained": False,
    }
    manifest_path = batch_dir / "gefs_exact_schedule_batch_full_weather_manifest_v1.json"
    write_json(manifest_path, manifest)
    return {"cycle_status": status_path, "audit": audit_path, "manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-budget", type=Path, required=True)
    parser.add_argument("--cycle-plan", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--range-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

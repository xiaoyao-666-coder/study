#!/usr/bin/env python3
"""Plan resumable, year-isolated GEFS payload extraction batches without downloading data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    EXPECTED_SITE_CYCLE_ROWS,
    EXPECTED_UNIQUE_CYCLES,
    EXPECTED_YEARS,
    build_cycle_plan,
    sha256_file,
    validate_weather_plan,
    write_json,
)


DEFAULT_MAX_CYCLES_PER_BATCH = 4
DEFAULT_MAX_BATCH_BYTES = 5 * 1024**3
REQUIRED_CYCLE_BUDGET_COLUMNS = {
    "target_year",
    "decision_date",
    "required_site_count",
    "required_sites",
    "expected_output_rows",
    "selected_range_bytes",
    "index_network_bytes_this_run",
}


def validate_preflight_inputs(
    *,
    weather_plan: pd.DataFrame,
    cycle_budget: pd.DataFrame,
    preflight_audit: dict[str, Any],
) -> pd.DataFrame:
    """Validate that the schedule and budget describe the formal exact run."""
    plan = validate_weather_plan(weather_plan)
    expected_cycles = build_cycle_plan(plan)
    missing = REQUIRED_CYCLE_BUDGET_COLUMNS.difference(cycle_budget.columns)
    if missing:
        raise ValueError(f"cycle budget missing fields: {sorted(missing)}")
    budget = cycle_budget.copy()
    budget["decision_date"] = pd.to_datetime(
        budget["decision_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    for column in (
        "target_year",
        "required_site_count",
        "expected_output_rows",
        "selected_range_bytes",
        "index_network_bytes_this_run",
    ):
        budget[column] = pd.to_numeric(budget[column], errors="raise").astype(int)
    if budget["decision_date"].duplicated().any():
        raise ValueError("cycle budget contains duplicate decision dates")
    if len(budget) != EXPECTED_UNIQUE_CYCLES:
        raise ValueError("cycle budget unique-cycle count mismatch")
    if tuple(sorted(budget["target_year"].unique())) != EXPECTED_YEARS:
        raise ValueError("cycle budget year set mismatch")
    if (budget["selected_range_bytes"] <= 0).any():
        raise ValueError("cycle budget contains a nonpositive payload estimate")

    comparison_columns = [
        "target_year",
        "decision_date",
        "required_site_count",
        "required_sites",
        "expected_output_rows",
    ]
    expected = expected_cycles[comparison_columns].sort_values("decision_date")
    actual = budget[comparison_columns].sort_values("decision_date")
    if not expected.reset_index(drop=True).equals(actual.reset_index(drop=True)):
        raise ValueError("cycle budget does not match the exact weather schedule")

    required_audit = {
        "status": "exact_schedule_full_weather_preflight_passed",
        "mandatory_structural_gate_passed": True,
        "product_payload_download_started": False,
        "site_cycle_rows": EXPECTED_SITE_CYCLE_ROWS,
        "unique_cycle_count": EXPECTED_UNIQUE_CYCLES,
        "cycle_member_task_count": 1195,
        "cycle_member_product_row_count": 8365,
    }
    for key, expected_value in required_audit.items():
        if preflight_audit.get(key) != expected_value:
            raise ValueError(
                f"preflight audit contract mismatch for {key}: "
                f"{preflight_audit.get(key)!r} != {expected_value!r}"
            )
    if int(budget["selected_range_bytes"].sum()) != int(
        preflight_audit.get("selected_range_bytes", -1)
    ):
        raise ValueError("cycle budget bytes do not reconcile to preflight audit")
    return budget.sort_values(["target_year", "decision_date"]).reset_index(drop=True)


def assign_year_isolated_batches(
    cycle_budget: pd.DataFrame,
    *,
    max_cycles_per_batch: int = DEFAULT_MAX_CYCLES_PER_BATCH,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign chronological cycles to stable batches, never crossing a year boundary."""
    if max_cycles_per_batch < 1:
        raise ValueError("max_cycles_per_batch must be at least one")
    if max_batch_bytes < 1:
        raise ValueError("max_batch_bytes must be positive")
    data = cycle_budget.sort_values(["target_year", "decision_date"]).copy()
    if (data["selected_range_bytes"] > max_batch_bytes).any():
        raise ValueError("a single cycle exceeds the configured batch byte limit")

    assigned: list[dict[str, Any]] = []
    for target_year, year_cycles in data.groupby("target_year", sort=True):
        batch_number = 0
        current: list[dict[str, Any]] = []
        current_bytes = 0

        def flush() -> None:
            nonlocal batch_number, current, current_bytes
            if not current:
                return
            batch_number += 1
            batch_id = f"Y{int(target_year)}_B{batch_number:02d}"
            for sequence, record in enumerate(current, start=1):
                record["batch_id"] = batch_id
                record["batch_number_in_year"] = batch_number
                record["cycle_number_in_batch"] = sequence
                assigned.append(record)
            current = []
            current_bytes = 0

        for record in year_cycles.to_dict("records"):
            payload_bytes = int(record["selected_range_bytes"])
            would_exceed_cycles = len(current) >= max_cycles_per_batch
            would_exceed_bytes = current and current_bytes + payload_bytes > max_batch_bytes
            if would_exceed_cycles or would_exceed_bytes:
                flush()
            current.append(record)
            current_bytes += payload_bytes
        flush()

    cycle_plan = pd.DataFrame(assigned).sort_values(
        ["target_year", "batch_number_in_year", "cycle_number_in_batch"]
    ).reset_index(drop=True)
    batch_budget = (
        cycle_plan.groupby(
            ["batch_id", "target_year", "batch_number_in_year"], as_index=False
        )
        .agg(
            cycle_count=("decision_date", "size"),
            first_decision_date=("decision_date", "min"),
            last_decision_date=("decision_date", "max"),
            site_cycle_rows=("required_site_count", "sum"),
            expected_output_rows=("expected_output_rows", "sum"),
            selected_range_bytes=("selected_range_bytes", "sum"),
            index_network_bytes_this_run=("index_network_bytes_this_run", "sum"),
        )
        .sort_values(["target_year", "batch_number_in_year"])
        .reset_index(drop=True)
    )
    batch_budget["selected_range_gib"] = (
        batch_budget["selected_range_bytes"] / 1024**3
    )
    if batch_budget["target_year"].isna().any():
        raise ValueError("batch assignment lost target-year ownership")
    if (batch_budget["cycle_count"] > max_cycles_per_batch).any():
        raise ValueError("batch exceeds the cycle-count limit")
    if (batch_budget["selected_range_bytes"] > max_batch_bytes).any():
        raise ValueError("batch exceeds the payload byte limit")
    if len(cycle_plan) != len(cycle_budget):
        raise ValueError("batch assignment did not preserve every cycle")
    if int(batch_budget["selected_range_bytes"].sum()) != int(
        cycle_budget["selected_range_bytes"].sum()
    ):
        raise ValueError("batch bytes do not reconcile to cycle bytes")
    return cycle_plan, batch_budget


def build_contract(
    *,
    weather_plan_path: Path,
    cycle_budget_path: Path,
    preflight_audit_path: Path,
    max_cycles_per_batch: int,
    max_batch_bytes: int,
) -> dict[str, Any]:
    return {
        "contract_id": "gefs-exact-schedule-year-isolated-batches-v1",
        "input_sha256": {
            "weather_plan": sha256_file(weather_plan_path),
            "cycle_budget": sha256_file(cycle_budget_path),
            "preflight_audit": sha256_file(preflight_audit_path),
        },
        "batch_id_pattern": "Y<target_year>_B<two-digit-sequence>",
        "year_isolated": True,
        "max_cycles_per_batch": int(max_cycles_per_batch),
        "max_payload_bytes_per_batch": int(max_batch_bytes),
        "max_payload_gib_per_batch": float(max_batch_bytes / 1024**3),
        "payload_download_workers_required": 1,
        "resume_rules": [
            "A batch may resume only when these three input SHA-256 values match.",
            "A batch is complete only after its completion receipt is atomically written.",
            "A completed batch must not be downloaded again unless an explicit replacement run is authorized.",
            "Temporary GRIB files must be removed after every product decode.",
        ],
        "completion_receipt_relative_path": "batches/<batch_id>/batch_completion_receipt_v1.json",
        "payload_download_started": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }


def build_audit(
    *,
    cycle_plan: pd.DataFrame,
    batch_budget: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    cycle_bytes = int(cycle_plan["selected_range_bytes"].sum())
    structural_passed = all(
        [
            len(cycle_plan) == EXPECTED_UNIQUE_CYCLES,
            cycle_plan["decision_date"].nunique() == EXPECTED_UNIQUE_CYCLES,
            tuple(sorted(cycle_plan["target_year"].unique())) == EXPECTED_YEARS,
            batch_budget["target_year"].nunique() == len(EXPECTED_YEARS),
            batch_budget["cycle_count"].le(contract["max_cycles_per_batch"]).all(),
            batch_budget["selected_range_bytes"]
            .le(contract["max_payload_bytes_per_batch"])
            .all(),
            int(batch_budget["selected_range_bytes"].sum()) == cycle_bytes,
        ]
    )
    return {
        "status": (
            "exact_schedule_batched_extraction_plan_passed"
            if structural_passed
            else "exact_schedule_batched_extraction_plan_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "contract_id": contract["contract_id"],
        "year_isolated": True,
        "batch_count": int(len(batch_budget)),
        "cycle_count": int(len(cycle_plan)),
        "site_cycle_rows": int(cycle_plan["required_site_count"].sum()),
        "expected_corrected_member_day_rows": int(
            cycle_plan["expected_output_rows"].sum()
        ),
        "selected_range_bytes": cycle_bytes,
        "selected_range_gib": float(cycle_bytes / 1024**3),
        "max_cycles_per_batch": contract["max_cycles_per_batch"],
        "max_payload_bytes_per_batch": contract["max_payload_bytes_per_batch"],
        "max_payload_gib_per_batch": contract["max_payload_gib_per_batch"],
        "largest_batch_selected_range_bytes": int(
            batch_budget["selected_range_bytes"].max()
        ),
        "largest_batch_selected_range_gib": float(
            batch_budget["selected_range_gib"].max()
        ),
        "single_worker_required": True,
        "payload_download_started": False,
        "temporary_grib_retained": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": "explicit_approval_required_before_batched_gefs_payload_download"
        if structural_passed
        else "repair_batched_extraction_plan",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-plan", type=Path, required=True)
    parser.add_argument("--cycle-budget", type=Path, required=True)
    parser.add_argument("--preflight-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-cycles-per-batch", type=int, default=DEFAULT_MAX_CYCLES_PER_BATCH
    )
    parser.add_argument("--max-batch-bytes", type=int, default=DEFAULT_MAX_BATCH_BYTES)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    for path in (args.weather_plan, args.cycle_budget, args.preflight_audit):
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    weather_plan = pd.read_csv(args.weather_plan)
    cycle_budget = pd.read_csv(args.cycle_budget)
    preflight_audit = json.loads(args.preflight_audit.read_text(encoding="utf-8"))
    validated_budget = validate_preflight_inputs(
        weather_plan=weather_plan,
        cycle_budget=cycle_budget,
        preflight_audit=preflight_audit,
    )
    cycle_plan, batch_budget = assign_year_isolated_batches(
        validated_budget,
        max_cycles_per_batch=args.max_cycles_per_batch,
        max_batch_bytes=args.max_batch_bytes,
    )
    contract = build_contract(
        weather_plan_path=args.weather_plan,
        cycle_budget_path=args.cycle_budget,
        preflight_audit_path=args.preflight_audit,
        max_cycles_per_batch=args.max_cycles_per_batch,
        max_batch_bytes=args.max_batch_bytes,
    )
    audit = build_audit(
        cycle_plan=cycle_plan, batch_budget=batch_budget, contract=contract
    )
    args.output_dir.mkdir(parents=True)
    outputs = {
        "cycle_plan": args.output_dir / "gefs_exact_schedule_batched_cycle_plan_v1.csv",
        "batch_budget": args.output_dir / "gefs_exact_schedule_batch_budget_v1.csv",
        "contract": args.output_dir / "gefs_exact_schedule_batched_extraction_contract_v1.json",
        "audit": args.output_dir / "gefs_exact_schedule_batched_extraction_audit_v1.json",
        "manifest": args.output_dir / "gefs_exact_schedule_batched_extraction_manifest_v1.json",
    }
    cycle_plan.to_csv(outputs["cycle_plan"], index=False)
    batch_budget.to_csv(outputs["batch_budget"], index=False)
    write_json(outputs["contract"], contract)
    write_json(outputs["audit"], audit)
    manifest = {
        "status": audit["status"],
        "inputs": contract["input_sha256"],
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "payload_download_started": False,
    }
    write_json(outputs["manifest"], manifest)
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"batch plan failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

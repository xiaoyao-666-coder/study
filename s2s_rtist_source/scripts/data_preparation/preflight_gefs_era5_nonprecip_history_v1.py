#!/usr/bin/env python3
"""Preflight a sparse historical GEFS sample for ERA5 nonprecip correction."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from s2s_rtist.weather.gefs_reforecast_full_weather import (
    ReforecastProductSpec,
    load_or_download_inventory,
    preflight_product,
    select_product_objects,
)


NONPRECIP_PRODUCT_SPECS = (
    ReforecastProductSpec("tmp_2m", "TMP", "2 m above ground", "instant"),
    ReforecastProductSpec("spfh_2m", "SPFH", "2 m above ground", "instant"),
    ReforecastProductSpec("pres_sfc", "PRES", "surface", "instant"),
    ReforecastProductSpec("ugrd_hgt", "UGRD", "10 m above ground", "instant"),
    ReforecastProductSpec("vgrd_hgt", "VGRD", "10 m above ground", "instant"),
    ReforecastProductSpec("dswrf_sfc", "DSWRF", "surface", "ave"),
)
DEFAULT_MONTH_DAYS = ("05-15", "06-15", "07-15", "08-15")
DEFAULT_MAXIMUM_SELECTED_BYTES = 25_000_000_000


def build_cycles(years: Sequence[int], month_days: Sequence[str]) -> list[str]:
    cycles = []
    for year in years:
        for month_day in month_days:
            timestamp = pd.Timestamp(f"{int(year):04d}-{month_day}")
            if timestamp.year != int(year):
                raise ValueError(f"invalid month-day {month_day!r} for {year}")
            cycles.append(timestamp.strftime("%Y-%m-%d"))
    if len(cycles) != len(set(cycles)):
        raise ValueError("historical preflight cycles are not unique")
    return cycles


def preflight_task(
    *,
    cycle_date: str,
    member: str,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    objects, inventory = load_or_download_inventory(
        cycle_date=cycle_date,
        member=member,
        cache_dir=cache_dir,
        timeout=timeout,
        retries=retries,
    )
    pairs = select_product_objects(objects, specs=NONPRECIP_PRODUCT_SPECS)
    rows = [
        preflight_product(
            cycle_date=cycle_date,
            member=member,
            pair=pair,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        for pair in pairs
    ]
    return rows, inventory


def build_audit(
    manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    cycles: Sequence[str],
    members: Sequence[str],
    maximum_selected_bytes: int,
) -> dict[str, object]:
    selected_bytes = int(manifest["selected_range_bytes"].sum())
    task_count = len(cycles) * len(members)
    expected_rows = task_count * len(NONPRECIP_PRODUCT_SPECS)
    failures = []
    if len(manifest) != expected_rows:
        failures.append("product_manifest_row_count")
    if len(inventory) != task_count:
        failures.append("inventory_manifest_row_count")
    if manifest[["cycle_date", "gefs_member", "product_id"]].duplicated().any():
        failures.append("duplicate_product_key")
    if selected_bytes > maximum_selected_bytes:
        failures.append("selected_bytes_exceed_contract")
    product_totals = {
        str(key): int(value)
        for key, value in manifest.groupby("product_id")[
            "selected_range_bytes"
        ].sum().items()
    }
    return {
        "status": (
            "gefs_era5_nonprecip_history_preflight_passed"
            if not failures
            else "gefs_era5_nonprecip_history_preflight_failed"
        ),
        "mandatory_gate_passed": not failures,
        "gate_failures": failures,
        "cycle_count": len(cycles),
        "first_cycle": min(cycles),
        "last_cycle": max(cycles),
        "member_count": len(members),
        "members": list(members),
        "task_count": task_count,
        "product_count_per_task": len(NONPRECIP_PRODUCT_SPECS),
        "product_manifest_rows": int(len(manifest)),
        "selected_range_bytes": selected_bytes,
        "selected_range_gb_decimal": selected_bytes / 1e9,
        "maximum_selected_bytes": int(maximum_selected_bytes),
        "projected_five_member_bytes": int(selected_bytes * 5 / len(members)),
        "projected_five_member_gb_decimal": selected_bytes * 5 / len(members) / 1e9,
        "selected_bytes_by_product": product_totals,
        "download_started": False,
        "reference_dataset": "ERA5_corresponding_year",
        "correction_fit_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    cycles = build_cycles(args.years, args.month_days)
    tasks = [(cycle, member) for cycle in cycles for member in args.members]
    rows: list[dict[str, object]] = []
    inventories: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                preflight_task,
                cycle_date=cycle,
                member=member,
                cache_dir=args.output_dir / "cache",
                timeout=args.timeout,
                retries=args.retries,
            ): (cycle, member)
            for cycle, member in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_map), start=1
        ):
            cycle, member = future_map[future]
            product_rows, inventory = future.result()
            rows.extend(product_rows)
            inventories.append(inventory)
            print(
                f"[preflight] {cycle} {member} ready ({completed}/{len(tasks)})",
                flush=True,
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["cycle_date", "gefs_member", "product_id"]
    )
    inventory = pd.DataFrame(inventories).sort_values(
        ["cycle_date", "gefs_member"]
    )
    audit = build_audit(
        manifest,
        inventory,
        cycles=cycles,
        members=args.members,
        maximum_selected_bytes=args.maximum_selected_bytes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": args.output_dir / "gefs_era5_nonprecip_history_preflight_v1.csv",
        "inventory": args.output_dir
        / "gefs_era5_nonprecip_history_inventory_v1.csv",
        "audit": args.output_dir
        / "gefs_era5_nonprecip_history_preflight_audit_v1.json",
    }
    manifest.to_csv(outputs["manifest"], index=False)
    inventory.to_csv(outputs["inventory"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"historical preflight failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2000, 2020)))
    parser.add_argument("--month-days", nargs="+", default=list(DEFAULT_MONTH_DAYS))
    parser.add_argument("--members", nargs="+", default=["c00"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--maximum-selected-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_SELECTED_BYTES,
    )
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

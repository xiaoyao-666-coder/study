#!/usr/bin/env python3
"""Extract the frozen 2015-2019 full-variable GEFS reforecast pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    reforecast_site_frame,
)
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    REQUIRED_PRODUCT_SPECS,
    ProductObjectPair,
    aggregate_member_weather,
    download_product_points,
    load_or_download_inventory,
    preflight_product,
    select_product_objects,
    validate_full_weather,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_full_weather_pilot_local_v1"
)
DEFAULT_CYCLES = (
    "2015-07-15",
    "2016-07-15",
    "2017-07-15",
    "2018-07-15",
    "2019-07-15",
)
DEFAULT_SITES = ("P1", "P2", "P3", "P4", "P15")
MAXIMUM_LOCAL_DOWNLOAD_BYTES = 6_500_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_member(
    *,
    cycle_date: str,
    member: str,
    pairs: Sequence[ProductObjectPair],
    sites: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    point_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for pair in pairs:
        points, metadata = download_product_points(
            cycle_date=cycle_date,
            member=member,
            pair=pair,
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        point_parts.append(points)
        manifest_rows.append(metadata)
    points = pd.concat(point_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)
    daily = aggregate_member_weather(
        points,
        member=member,
        product_manifest=manifest,
    )
    return daily, manifest_rows


def preflight_extraction(
    *,
    tasks: Sequence[tuple[str, str]],
    cache_dir: Path,
    timeout: int,
    retries: int,
    workers: int,
) -> tuple[
    dict[tuple[str, str], list[ProductObjectPair]],
    pd.DataFrame,
    pd.DataFrame,
]:
    pairs_by_task: dict[tuple[str, str], list[ProductObjectPair]] = {}
    preflight_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    def inspect(task: tuple[str, str]):
        cycle_date, member = task
        objects, inventory = load_or_download_inventory(
            cycle_date=cycle_date,
            member=member,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )
        pairs = select_product_objects(objects)
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
        inventory.update(
            {
                "selected_product_ids": [pair.spec.product_id for pair in pairs],
                "required_product_count": len(REQUIRED_PRODUCT_SPECS),
            }
        )
        return task, pairs, rows, inventory

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = [executor.submit(inspect, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            task, pairs, rows, inventory = future.result()
            pairs_by_task[task] = pairs
            preflight_rows.extend(rows)
            inventory_rows.append(inventory)
            print(
                f"[preflight] {task[0]} {task[1]} ready "
                f"({completed}/{len(tasks)})",
                flush=True,
            )
    preflight = pd.DataFrame(preflight_rows).sort_values(
        ["cycle_date", "gefs_member", "product_id"]
    ).reset_index(drop=True)
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["cycle_date", "gefs_member"]
    ).reset_index(drop=True)
    return pairs_by_task, preflight, inventory


def run_extraction(
    *,
    cycles: Sequence[str],
    site_ids: Sequence[str],
    members: Sequence[str],
    output_dir: Path,
    timeout: int,
    retries: int,
    workers: int,
    preflight_only: bool = False,
    product_workers: int | None = None,
    product_range_workers: int = 1,
) -> dict[str, Path]:
    normalized_cycles = tuple(
        pd.Timestamp(cycle).strftime("%Y-%m-%d") for cycle in cycles
    )
    if len(normalized_cycles) != len(set(normalized_cycles)):
        raise ValueError("cycles must be unique")
    if any(pd.Timestamp(cycle).year == 2024 for cycle in normalized_cycles):
        raise ValueError("2024 is forbidden in the pilot extraction")
    members = tuple(members)
    unsupported = sorted(set(members).difference(GEFS_REFORECAST_MEMBERS))
    if unsupported:
        raise ValueError(f"unsupported reforecast members: {unsupported}")
    if len(members) != len(set(members)):
        raise ValueError("members must be unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    sites = reforecast_site_frame(tuple(site_ids))
    tasks = [
        (cycle_date, member)
        for cycle_date in normalized_cycles
        for member in members
    ]
    pairs_by_task, preflight_manifest, inventory_manifest = preflight_extraction(
        tasks=tasks,
        cache_dir=cache_dir,
        timeout=timeout,
        retries=retries,
        workers=workers,
    )
    preflight_path = output_dir / "gefs_2015_2019_full_weather_preflight_v1.csv"
    preflight_manifest.to_csv(preflight_path, index=False, encoding="utf-8-sig")
    selected_range_bytes = int(preflight_manifest["selected_range_bytes"].sum())
    preflight_network_bytes = int(
        preflight_manifest["index_network_bytes_this_run"].sum()
        + inventory_manifest["network_bytes_this_run"].sum()
    )
    preflight_audit_path = (
        output_dir / "gefs_2015_2019_full_weather_preflight_audit_v1.json"
    )
    preflight_audit = {
        "contract_id": "gefs-2015-2019-scenario-consistent-three-output-pilot-v1",
        "task_count": len(tasks),
        "required_product_count_per_task": len(REQUIRED_PRODUCT_SPECS),
        "selected_range_bytes": selected_range_bytes,
        "maximum_local_download_bytes": MAXIMUM_LOCAL_DOWNLOAD_BYTES,
        "download_limit_passed": selected_range_bytes
        <= MAXIMUM_LOCAL_DOWNLOAD_BYTES,
        "preflight_network_bytes_this_run": preflight_network_bytes,
        "download_started": False,
        "status": "preflight_passed"
        if selected_range_bytes <= MAXIMUM_LOCAL_DOWNLOAD_BYTES
        else "preflight_blocked_download_limit",
    }
    _write_json(preflight_audit_path, preflight_audit)
    if selected_range_bytes > MAXIMUM_LOCAL_DOWNLOAD_BYTES:
        raise ValueError(
            f"selected range bytes {selected_range_bytes} exceed the 6.5 GB contract limit"
        )
    if preflight_only:
        return {
            "preflight": preflight_path,
            "preflight_audit": preflight_audit_path,
        }

    weather_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    def extract(task: tuple[str, str]):
        cycle_date, member = task
        return task, _extract_member(
            cycle_date=cycle_date,
            member=member,
            pairs=pairs_by_task[task],
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )

    if product_workers is None:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(extract, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                (cycle_date, member), (daily, product_rows) = future.result()
                weather_parts.append(daily)
                manifest_rows.extend(product_rows)
                print(
                    f"[full-weather] {cycle_date} {member} ready "
                    f"({completed}/{len(tasks)})",
                    flush=True,
                )
    else:
        point_parts_by_task: dict[tuple[str, str], list[pd.DataFrame]] = {
            task: [] for task in tasks
        }
        product_rows_by_task: dict[
            tuple[str, str], list[dict[str, object]]
        ] = {task: [] for task in tasks}
        product_tasks = [
            (task, pair)
            for task in tasks
            for pair in pairs_by_task[task]
        ]

        def extract_product(task: tuple[str, str], pair: ProductObjectPair):
            cycle_date, member = task
            points, metadata = download_product_points(
                cycle_date=cycle_date,
                member=member,
                pair=pair,
                sites=sites,
                cache_dir=cache_dir,
                timeout=timeout,
                retries=retries,
                range_workers=product_range_workers,
            )
            return task, pair.spec.product_id, points, metadata

        with ThreadPoolExecutor(max_workers=max(1, int(product_workers))) as executor:
            futures = [
                executor.submit(extract_product, task, pair)
                for task, pair in product_tasks
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                task, product_id, points, metadata = future.result()
                point_parts_by_task[task].append(points)
                product_rows_by_task[task].append(metadata)
                print(
                    f"[full-weather-product] {task[0]} {task[1]} {product_id} ready "
                    f"({completed}/{len(product_tasks)})",
                    flush=True,
                )

        for completed, task in enumerate(tasks, start=1):
            product_rows = product_rows_by_task[task]
            daily = aggregate_member_weather(
                pd.concat(point_parts_by_task[task], ignore_index=True),
                member=task[1],
                product_manifest=pd.DataFrame(product_rows),
            )
            weather_parts.append(daily)
            manifest_rows.extend(product_rows)
            print(
                f"[full-weather] {task[0]} {task[1]} ready "
                f"({completed}/{len(tasks)})",
                flush=True,
            )

    weather = pd.concat(weather_parts, ignore_index=True).sort_values(
        ["decision_date", "site_id", "gefs_member", "local_date"]
    ).reset_index(drop=True)
    download_manifest = pd.DataFrame(manifest_rows).sort_values(
        ["cycle_date", "gefs_member", "product_id"]
    ).reset_index(drop=True)
    audit = validate_full_weather(
        weather,
        expected_cycles=normalized_cycles,
        expected_sites=tuple(site_ids),
        expected_members=members,
    )
    product_network_bytes = int(download_manifest["network_bytes_this_run"].sum())
    network_bytes_this_run = product_network_bytes + preflight_network_bytes
    retained_gribs = list(cache_dir.rglob("*.grib2"))
    if retained_gribs:
        raise ValueError(f"temporary GRIB files were retained: {retained_gribs[:5]}")
    audit.update(
        {
            "contract_id": "gefs-2015-2019-scenario-consistent-three-output-pilot-v1",
            "scope": "local_full_variable_extraction_only",
            "selected_range_bytes": selected_range_bytes,
            "network_bytes_this_run": network_bytes_this_run,
            "preflight_network_bytes_this_run": preflight_network_bytes,
            "maximum_local_download_bytes": MAXIMUM_LOCAL_DOWNLOAD_BYTES,
            "download_limit_passed": True,
            "retained_grib_file_count": 0,
            "server_network_download_performed": False,
            "precipitation_correction_applied": False,
            "swap_simulation_performed": False,
            "surrogate_training_performed": False,
            "tta_performed": False,
            "status": "full_weather_local_extraction_passed",
        }
    )

    weather_path = output_dir / "gefs_2015_2019_full_weather_member_daily_v1.csv"
    download_path = output_dir / "gefs_2015_2019_full_weather_download_manifest_v1.csv"
    inventory_path = output_dir / "gefs_2015_2019_full_weather_inventory_manifest_v1.csv"
    sites_path = output_dir / "gefs_2015_2019_full_weather_sites_v1.csv"
    audit_path = output_dir / "gefs_2015_2019_full_weather_audit_v1.json"
    manifest_path = output_dir / "gefs_2015_2019_full_weather_manifest_v1.json"
    weather.to_csv(weather_path, index=False, encoding="utf-8-sig")
    download_manifest.to_csv(download_path, index=False, encoding="utf-8-sig")
    inventory_manifest.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    sites.to_csv(sites_path, index=False, encoding="utf-8-sig")
    _write_json(audit_path, audit)
    output_manifest = {
        "contract_id": audit["contract_id"],
        "status": audit["status"],
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in (
                preflight_path,
                preflight_audit_path,
                weather_path,
                download_path,
                inventory_path,
                sites_path,
                audit_path,
            )
        },
        "cache_policy": {
            "inventory_xml_retained": True,
            "index_files_retained": True,
            "point_records_retained": True,
            "temporary_grib_retained": False,
        },
    }
    _write_json(manifest_path, output_manifest)
    return {
        "weather": weather_path,
        "preflight": preflight_path,
        "preflight_audit": preflight_audit_path,
        "download_manifest": download_path,
        "inventory_manifest": inventory_path,
        "sites": sites_path,
        "audit": audit_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", nargs="+", default=list(DEFAULT_CYCLES))
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument(
        "--members", nargs="+", default=list(GEFS_REFORECAST_MEMBERS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_extraction(
        cycles=args.cycles,
        site_ids=args.sites,
        members=args.members,
        output_dir=args.output_dir,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
        preflight_only=args.preflight_only,
    )
    print(
        json.dumps(
            {key: str(value) for key, value in outputs.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

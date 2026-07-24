#!/usr/bin/env python3
"""Extract sparse causal-history GEFS nonprecipitation weather for ERA5 correction."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_gridmet_bias import aggregate_gefs_point_records
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    reforecast_site_frame,
)
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    ProductObjectPair,
    download_product_points,
    load_or_download_inventory,
    preflight_product,
    select_product_objects,
    specific_humidity_to_vapor_pressure_kpa,
)
from scripts.data_preparation.preflight_gefs_era5_nonprecip_history_v1 import (
    NONPRECIP_PRODUCT_SPECS,
)


DEFAULT_SITES = ("P1", "P2", "P3", "P4", "P15")
DEFAULT_YEARS = tuple(range(2015, 2020))
MAXIMUM_LOCAL_DOWNLOAD_BYTES = 25_000_000_000
CANONICAL_COLUMNS = (
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_causal_cycles(years: Sequence[int]) -> list[str]:
    cycles = []
    for year in years:
        dates = pd.date_range(
            f"{int(year):04d}-01-15", f"{int(year):04d}-08-13", freq="14D"
        )
        if len(dates) != 16:
            raise ValueError(f"expected 16 causal calibration cycles for {year}")
        cycles.extend(dates.strftime("%Y-%m-%d").tolist())
    if len(cycles) != len(set(cycles)):
        raise ValueError("causal calibration cycles are not unique")
    return cycles


def resolve_causal_cycles(
    *,
    years: Sequence[int] | None,
    explicit_cycles: Sequence[str] | None,
) -> list[str]:
    if explicit_cycles:
        cycles = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in explicit_cycles]
        if len(cycles) != len(set(cycles)):
            raise ValueError("explicit causal calibration cycles are not unique")
        return sorted(cycles)
    return build_causal_cycles(years or DEFAULT_YEARS)


def aggregate_humidity_pressure(points: pd.DataFrame) -> pd.DataFrame:
    selected = points.loc[points["short_name"].isin(["SPFH", "PRES"])].copy()
    keys = ["site", "timezone", "cycle_init_utc", "lead_hour"]
    pivot = selected.pivot_table(
        index=keys, columns="short_name", values="value", aggfunc="first"
    ).reset_index()
    if not {"SPFH", "PRES"}.issubset(pivot.columns):
        raise ValueError("SPFH and PRES are required for vapor pressure")
    pivot["valid_time_utc"] = pd.to_datetime(
        pivot["cycle_init_utc"], utc=True
    ) + pd.to_timedelta(pivot["lead_hour"], unit="h")
    pivot["local_date"] = [
        timestamp.tz_convert(ZoneInfo(timezone_name)).tz_localize(None).normalize()
        for timestamp, timezone_name in zip(
            pivot["valid_time_utc"], pivot["timezone"]
        )
    ]
    pivot["actual_vapor_pressure_kpa"] = specific_humidity_to_vapor_pressure_kpa(
        pivot["SPFH"], pivot["PRES"]
    )
    return pivot.groupby(
        ["site", "timezone", "cycle_init_utc", "local_date"], as_index=False
    ).agg(
        actual_vapor_pressure_kpa=("actual_vapor_pressure_kpa", "mean"),
        specific_humidity_kg_kg=("SPFH", "mean"),
        surface_pressure_kpa=("PRES", lambda values: float(values.mean()) / 1000.0),
    )


def aggregate_nonprecip_member(
    points: pd.DataFrame,
    *,
    member: str,
    product_manifest: pd.DataFrame,
) -> pd.DataFrame:
    work = points.copy()
    work["lead_hour"] = work["end_hour"].astype(int)
    daily = aggregate_gefs_point_records(work)
    humidity = aggregate_humidity_pressure(work)
    daily = daily.merge(
        humidity,
        on=["site", "timezone", "cycle_init_utc", "local_date"],
        how="left",
        validate="one_to_one",
    )
    daily["gefs_member"] = member
    daily["solar_kj_m2_day"] = daily["shortwave_w_m2"].astype(float) * 86.4
    daily = daily.rename(
        columns={
            "site": "site_id",
            "timezone": "site_timezone",
            "cycle_init_utc": "forecast_init_utc",
        }
    )
    daily["source_product_keys"] = ";".join(
        sorted(product_manifest["source_key"].astype(str))
    )
    daily["source_product_etags"] = ";".join(
        sorted(product_manifest["source_etag"].astype(str))
    )
    columns = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        "local_date",
        "lead_day",
        *CANONICAL_COLUMNS,
        "specific_humidity_kg_kg",
        "surface_pressure_kpa",
        "shortwave_w_m2",
        "source_product_keys",
        "source_product_etags",
    ]
    return daily[columns].sort_values(["site_id", "lead_day"]).reset_index(drop=True)


def validate_history(
    frame: pd.DataFrame,
    *,
    cycles: Sequence[str],
    site_ids: Sequence[str],
    members: Sequence[str],
) -> dict[str, object]:
    expected_rows = len(cycles) * len(site_ids) * len(members) * 7
    key = ["decision_date", "site_id", "gefs_member", "lead_day"]
    failures = []
    if len(frame) != expected_rows:
        failures.append("row_count")
    if frame[key].duplicated().any():
        failures.append("duplicate_sample_key")
    if set(frame["site_id"].astype(str)) != set(site_ids):
        failures.append("site_set")
    if set(frame["gefs_member"].astype(str)) != set(members):
        failures.append("member_set")
    actual_cycles = set(pd.to_datetime(frame["decision_date"]).dt.strftime("%Y-%m-%d"))
    if actual_cycles != set(cycles):
        failures.append("cycle_set")
    if frame[list(CANONICAL_COLUMNS)].isna().any().any():
        failures.append("missing_canonical_values")
    if not np.isfinite(frame[list(CANONICAL_COLUMNS)].to_numpy(dtype=float)).all():
        failures.append("nonfinite_canonical_values")
    if (frame["temperature_min_c"] > frame["temperature_max_c"]).any():
        failures.append("temperature_order")
    if (
        frame[
            [
                "actual_vapor_pressure_kpa",
                "wind_speed_m_s",
                "solar_kj_m2_day",
            ]
        ]
        < 0.0
    ).any().any():
        failures.append("negative_positive_only_values")
    bad_leads = [
        group_key
        for group_key, group in frame.groupby(
            ["decision_date", "site_id", "gefs_member"], sort=False
        )
        if group.sort_values("lead_day")["lead_day"].astype(int).tolist()
        != list(range(1, 8))
    ]
    if bad_leads:
        failures.append("incomplete_lead_days")
    return {
        "mandatory_gate_passed": not failures,
        "gate_failures": failures,
        "row_count": int(len(frame)),
        "expected_row_count": int(expected_rows),
        "cycle_count": int(frame["decision_date"].nunique()),
        "site_count": int(frame["site_id"].nunique()),
        "member_count": int(frame["gefs_member"].nunique()),
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    cycles = resolve_causal_cycles(
        years=args.years,
        explicit_cycles=args.cycles,
    )
    members = tuple(args.members)
    unsupported = sorted(set(members) - set(GEFS_REFORECAST_MEMBERS))
    if unsupported:
        raise ValueError(f"unsupported GEFS members: {unsupported}")
    if members != ("c00",):
        raise ValueError("v1 causal-history extraction is restricted to c00")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    sites = reforecast_site_frame(tuple(args.sites))
    tasks = [(cycle, member) for cycle in cycles for member in members]

    pairs_by_task: dict[tuple[str, str], list[ProductObjectPair]] = {}
    preflight_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    def inspect(task: tuple[str, str]):
        cycle, member = task
        objects, inventory = load_or_download_inventory(
            cycle_date=cycle,
            member=member,
            cache_dir=cache_dir,
            timeout=args.timeout,
            retries=args.retries,
        )
        pairs = select_product_objects(objects, specs=NONPRECIP_PRODUCT_SPECS)
        rows = [
            preflight_product(
                cycle_date=cycle,
                member=member,
                pair=pair,
                cache_dir=cache_dir,
                timeout=args.timeout,
                retries=args.retries,
            )
            for pair in pairs
        ]
        return task, pairs, rows, inventory

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(inspect, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            task, pairs, rows, inventory = future.result()
            pairs_by_task[task] = pairs
            preflight_rows.extend(rows)
            inventory_rows.append(inventory)
            print(f"[preflight] {task[0]} {task[1]} ({completed}/{len(tasks)})", flush=True)

    preflight = pd.DataFrame(preflight_rows).sort_values(
        ["cycle_date", "gefs_member", "product_id"]
    )
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["cycle_date", "gefs_member"]
    )
    selected_bytes = int(preflight["selected_range_bytes"].sum())
    preflight_network_bytes = int(
        preflight["index_network_bytes_this_run"].sum()
        + inventory["network_bytes_this_run"].sum()
    )
    preflight_path = args.output_dir / "gefs_era5_nonprecip_causal_preflight_v1.csv"
    inventory_path = args.output_dir / "gefs_era5_nonprecip_causal_inventory_v1.csv"
    preflight_audit_path = (
        args.output_dir / "gefs_era5_nonprecip_causal_preflight_audit_v1.json"
    )
    preflight.to_csv(preflight_path, index=False)
    inventory.to_csv(inventory_path, index=False)
    preflight_audit = {
        "status": "preflight_passed"
        if selected_bytes <= MAXIMUM_LOCAL_DOWNLOAD_BYTES
        else "preflight_blocked_download_limit",
        "cycle_count": len(cycles),
        "task_count": len(tasks),
        "selected_range_bytes": selected_bytes,
        "selected_range_gb_decimal": selected_bytes / 1e9,
        "maximum_local_download_bytes": MAXIMUM_LOCAL_DOWNLOAD_BYTES,
        "download_limit_passed": selected_bytes <= MAXIMUM_LOCAL_DOWNLOAD_BYTES,
        "download_started": False,
        "preflight_network_bytes_this_run": preflight_network_bytes,
    }
    write_json(preflight_audit_path, preflight_audit)
    if selected_bytes > MAXIMUM_LOCAL_DOWNLOAD_BYTES:
        raise RuntimeError(f"selected bytes exceed contract; see {preflight_audit_path}")
    if args.preflight_only:
        return {
            "preflight": preflight_path,
            "audit": preflight_audit_path,
            "inventory": inventory_path,
        }

    weather_parts: list[pd.DataFrame] = []
    download_rows: list[dict[str, object]] = []

    def extract(task: tuple[str, str]):
        cycle, member = task
        point_parts = []
        product_rows = []
        for pair in pairs_by_task[task]:
            points, metadata = download_product_points(
                cycle_date=cycle,
                member=member,
                pair=pair,
                sites=sites,
                cache_dir=cache_dir,
                timeout=args.timeout,
                retries=args.retries,
            )
            point_parts.append(points)
            product_rows.append(metadata)
        manifest = pd.DataFrame(product_rows)
        daily = aggregate_nonprecip_member(
            pd.concat(point_parts, ignore_index=True),
            member=member,
            product_manifest=manifest,
        )
        return task, daily, product_rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract, task) for task in tasks]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            task, daily, product_rows = future.result()
            weather_parts.append(daily)
            download_rows.extend(product_rows)
            print(f"[download] {task[0]} {task[1]} ({completed}/{len(tasks)})", flush=True)

    weather = pd.concat(weather_parts, ignore_index=True).sort_values(
        ["decision_date", "site_id", "gefs_member", "lead_day"]
    ).reset_index(drop=True)
    download = pd.DataFrame(download_rows).sort_values(
        ["cycle_date", "gefs_member", "product_id"]
    )
    audit = validate_history(
        weather, cycles=cycles, site_ids=args.sites, members=members
    )
    retained_gribs = list(cache_dir.rglob("*.grib2"))
    if retained_gribs:
        audit["gate_failures"].append("temporary_grib_retained")
        audit["mandatory_gate_passed"] = False
    audit.update(
        {
            "status": "gefs_era5_nonprecip_causal_history_extraction_passed"
            if audit["mandatory_gate_passed"]
            else "gefs_era5_nonprecip_causal_history_extraction_failed",
            "selected_range_bytes": selected_bytes,
            "network_bytes_this_run": int(
                download["network_bytes_this_run"].sum()
                + preflight_network_bytes
            ),
            "retained_grib_file_count": len(retained_gribs),
            "reference_dataset": "ERA5_corresponding_year",
            "correction_fit_performed": False,
            "swap_simulation_performed": False,
            "surrogate_training_performed": False,
            "tta_performed": False,
        }
    )
    weather_path = args.output_dir / "gefs_era5_nonprecip_causal_c00_daily_v1.csv"
    download_path = args.output_dir / "gefs_era5_nonprecip_causal_download_v1.csv"
    sites_path = args.output_dir / "gefs_era5_nonprecip_causal_sites_v1.csv"
    audit_path = args.output_dir / "gefs_era5_nonprecip_causal_audit_v1.json"
    manifest_path = args.output_dir / "gefs_era5_nonprecip_causal_manifest_v1.json"
    weather.to_csv(weather_path, index=False)
    download.to_csv(download_path, index=False)
    sites.to_csv(sites_path, index=False)
    write_json(audit_path, audit)
    manifest = {
        "status": audit["status"],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (
                preflight_path,
                preflight_audit_path,
                inventory_path,
                weather_path,
                download_path,
                sites_path,
                audit_path,
            )
        },
        "cache_policy": {
            "inventories_retained": True,
            "indices_retained": True,
            "point_records_retained": True,
            "temporary_grib_retained": False,
        },
    }
    write_json(manifest_path, manifest)
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"extraction gate failed; see {audit_path}")
    return {
        "weather": weather_path,
        "audit": audit_path,
        "manifest": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--cycles", nargs="+")
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--members", nargs="+", default=["c00"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

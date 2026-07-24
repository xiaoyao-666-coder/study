#!/usr/bin/env python3
"""Build a lean GEFS weather smoke locally from an audited full-daily baseline."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from scripts.data_preparation.preflight_gefs_2015_2019_exact_schedule_full_weather_v1 import (
    sha256_file,
    write_json,
)
from scripts.diagnostics.run_gefs_temporal_sampling_one_cycle_weather_smoke_v1 import (
    EXACT_MATCH_VARIABLES,
    compare_weather,
)
from s2s_rtist.weather.gefs_gridmet_bias import (
    decode_gefs_minigrib_points,
    fetch_selected_byte_ranges,
    merge_contiguous_ranges,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    _request,
    reforecast_site_frame,
)
from s2s_rtist.weather.gefs_reforecast_full_weather import (
    CANONICAL_WEATHER_COLUMNS,
    REQUIRED_PRODUCT_SPECS,
    object_url,
    select_product_records,
    specific_humidity_to_vapor_pressure_kpa,
)


EXPECTED_BASELINE_ZIP_SHA256 = (
    "81ac419d3325978d5a446027d6bb678a6118e8bb1521b4db9a8d53cf5c644dff"
)
EXPECTED_CYCLE = "2015-07-06"
EXPECTED_SITES = ("P1", "P2", "P3", "P4", "P15")
STATE_PRODUCT_IDS = ("spfh_2m", "pres_sfc", "ugrd_hgt", "vgrd_hgt")
EXPECTED_BASELINE_ROWS = len(EXPECTED_SITES) * len(GEFS_REFORECAST_MEMBERS) * 7
BASELINE_WEATHER_NAME = "gefs_2015_2019_full_weather_member_daily_v1.csv"
BASELINE_DOWNLOAD_NAME = "gefs_2015_2019_full_weather_download_manifest_v1.csv"
BASELINE_AUDIT_NAME = "gefs_2015_2019_full_weather_audit_v1.json"
BASELINE_MANIFEST_NAME = "gefs_2015_2019_full_weather_manifest_v1.json"


def extract_and_validate_baseline(
    baseline_zip: Path, baseline_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if sha256_file(baseline_zip) != EXPECTED_BASELINE_ZIP_SHA256:
        raise ValueError("full-weather baseline ZIP SHA-256 mismatch")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(baseline_zip) as archive:
        names = set(archive.namelist())
        required = {
            BASELINE_WEATHER_NAME,
            BASELINE_DOWNLOAD_NAME,
            BASELINE_AUDIT_NAME,
            BASELINE_MANIFEST_NAME,
        }
        if not required.issubset(names):
            raise ValueError(f"baseline ZIP is missing files: {sorted(required - names)}")
        archive.extractall(baseline_dir)
    manifest = json.loads(
        (baseline_dir / BASELINE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    for name, metadata in manifest.get("files", {}).items():
        path = baseline_dir / name
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"baseline manifest verification failed for {name}")
    audit = json.loads((baseline_dir / BASELINE_AUDIT_NAME).read_text(encoding="utf-8"))
    required_audit = {
        "status": "full_weather_local_extraction_passed",
        "cycle_count": 1,
        "site_count": 5,
        "member_count": 5,
        "row_count": EXPECTED_BASELINE_ROWS,
        "retained_grib_file_count": 0,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise ValueError(f"baseline audit mismatch for {key}")
    weather = pd.read_csv(baseline_dir / BASELINE_WEATHER_NAME)
    downloads = pd.read_csv(baseline_dir / BASELINE_DOWNLOAD_NAME)
    if len(weather) != EXPECTED_BASELINE_ROWS:
        raise ValueError("baseline weather row count mismatch")
    if set(weather["site_id"]) != set(EXPECTED_SITES):
        raise ValueError("baseline weather site set mismatch")
    if set(weather["gefs_member"]) != set(GEFS_REFORECAST_MEMBERS):
        raise ValueError("baseline weather member set mismatch")
    if set(pd.to_datetime(weather["decision_date"]).dt.strftime("%Y-%m-%d")) != {
        EXPECTED_CYCLE
    }:
        raise ValueError("baseline weather cycle mismatch")
    state_downloads = downloads.loc[
        downloads["product_id"].isin(STATE_PRODUCT_IDS)
    ].copy()
    if len(state_downloads) != len(STATE_PRODUCT_IDS) * len(GEFS_REFORECAST_MEMBERS):
        raise ValueError("baseline state-product manifest row count mismatch")
    return weather, state_downloads, audit


def _lean_cache_paths(
    cache_dir: Path, *, cycle_date: str, member: str, product_id: str
) -> dict[str, Path]:
    cycle = pd.Timestamp(cycle_date).strftime("%Y%m%d")
    stem = f"{cycle}_{member}_{product_id}_f006-f174_step006"
    return {
        "index": cache_dir / "indices" / f"{stem}.idx",
        "grib": cache_dir / "minigrib" / f"{stem}.grib2",
        "points": cache_dir / "point_records" / f"{stem}.csv",
        "metadata": cache_dir / "metadata" / f"{stem}.json",
    }


def download_lean_state_product(
    *,
    manifest_row: pd.Series,
    sites: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cycle_date = str(manifest_row["cycle_date"])
    member = str(manifest_row["gefs_member"])
    product_id = str(manifest_row["product_id"])
    paths = _lean_cache_paths(
        cache_dir, cycle_date=cycle_date, member=member, product_id=product_id
    )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if paths["points"].is_file() and paths["metadata"].is_file():
        points = pd.read_csv(paths["points"], parse_dates=["cycle_init_utc"])
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        metadata["status"] = "cached_lean_point_records"
        metadata["network_bytes_this_run"] = 0
        return points, metadata

    if paths["index"].is_file():
        index_payload = paths["index"].read_bytes()
        index_network_bytes = 0
    else:
        index_payload, _ = _request(
            object_url(str(manifest_row["index_key"])),
            timeout=timeout,
            retries=retries,
        )
        paths["index"].write_bytes(index_payload)
        index_network_bytes = len(index_payload)
    spec = next(item for item in REQUIRED_PRODUCT_SPECS if item.product_id == product_id)
    full_records = select_product_records(
        index_payload.decode("utf-8"), spec=spec, maximum_end_hour=174
    )
    records = [item for item in full_records if item.step.end_hour % 6 == 0]
    if len(records) != 29 or records[-1].step.end_hour != 174:
        raise ValueError(f"incomplete six-hour record selection for {member}/{product_id}")
    ranges = merge_contiguous_ranges(records)
    product_network_bytes = 0

    def fetch(url: str, start: int, end: int) -> bytes:
        nonlocal product_network_bytes
        payload, _ = _request(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=timeout,
            retries=retries,
        )
        expected = end - start + 1
        if len(payload) != expected:
            raise ValueError(
                f"range length mismatch for {member}/{product_id}: {len(payload)} != {expected}"
            )
        product_network_bytes += len(payload)
        return payload

    if not paths["grib"].is_file():
        payload = fetch_selected_byte_ranges(
            object_url(str(manifest_row["source_key"])), ranges, fetcher=fetch
        )
        temporary = paths["grib"].with_suffix(".grib2.tmp")
        temporary.write_bytes(payload)
        temporary.replace(paths["grib"])
    try:
        points = decode_gefs_minigrib_points(
            paths["grib"],
            selected_records=records,
            sites=sites,
            cycle_init_utc=f"{cycle_date}T00:00:00Z",
            lead_hour=0,
        )
        points["gefs_member"] = member
        points["source_product_id"] = product_id
        points.to_csv(paths["points"], index=False)
        metadata: dict[str, Any] = {
            "status": "downloaded_lean_state_product",
            "cycle_date": cycle_date,
            "gefs_member": member,
            "product_id": product_id,
            "source_key": str(manifest_row["source_key"]),
            "source_etag": str(manifest_row["source_etag"]),
            "selected_message_count": len(records),
            "range_count": len(ranges),
            "selected_range_bytes": int(paths["grib"].stat().st_size),
            "selected_range_sha256": sha256_file(paths["grib"]),
            "index_bytes": len(index_payload),
            "network_bytes_this_run": index_network_bytes + product_network_bytes,
            "point_rows": len(points),
        }
        write_json(paths["metadata"], metadata)
    finally:
        paths["grib"].unlink(missing_ok=True)
    return points, metadata


def aggregate_lean_state(points: pd.DataFrame) -> pd.DataFrame:
    work = points.copy()
    work["cycle_init_utc"] = pd.to_datetime(work["cycle_init_utc"], utc=True)
    work["valid_time_utc"] = work["cycle_init_utc"] + pd.to_timedelta(
        pd.to_numeric(work["end_hour"], errors="raise"), unit="h"
    )
    work["local_date"] = [
        timestamp.tz_convert(ZoneInfo(timezone_name)).tz_localize(None).normalize()
        for timestamp, timezone_name in zip(work["valid_time_utc"], work["timezone"])
    ]
    pivot = work.pivot_table(
        index=[
            "site",
            "timezone",
            "cycle_init_utc",
            "gefs_member",
            "end_hour",
            "local_date",
        ],
        columns="short_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    if not {"SPFH", "PRES", "UGRD", "VGRD"}.issubset(pivot.columns):
        raise ValueError("lean state points are missing a required state variable")
    pivot["actual_vapor_pressure_kpa"] = specific_humidity_to_vapor_pressure_kpa(
        pivot["SPFH"], pivot["PRES"]
    )
    pivot["wind_speed_m_s"] = np.sqrt(pivot["UGRD"] ** 2 + pivot["VGRD"] ** 2)
    return (
        pivot.groupby(
            ["site", "timezone", "gefs_member", "local_date"], as_index=False
        )
        .agg(
            actual_vapor_pressure_kpa=("actual_vapor_pressure_kpa", "mean"),
            wind_speed_m_s=("wind_speed_m_s", "mean"),
            six_hour_sample_count=("end_hour", "size"),
        )
        .rename(columns={"site": "site_id", "timezone": "site_timezone"})
    )


def build_lean_weather(
    baseline_weather: pd.DataFrame, lean_state: pd.DataFrame
) -> pd.DataFrame:
    baseline = baseline_weather.copy()
    baseline["local_date"] = pd.to_datetime(baseline["local_date"]).dt.normalize()
    state = lean_state.copy()
    state["local_date"] = pd.to_datetime(state["local_date"]).dt.normalize()
    replacement = state[
        [
            "site_id",
            "gefs_member",
            "local_date",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "six_hour_sample_count",
        ]
    ]
    lean = baseline.drop(
        columns=["actual_vapor_pressure_kpa", "wind_speed_m_s"]
    ).merge(
        replacement,
        on=["site_id", "gefs_member", "local_date"],
        how="left",
        validate="one_to_one",
    )
    if lean[["actual_vapor_pressure_kpa", "wind_speed_m_s"]].isna().any().any():
        raise ValueError("lean state aggregation does not cover every baseline weather row")
    return lean.sort_values(
        ["decision_date", "site_id", "gefs_member", "local_date"]
    ).reset_index(drop=True)


def build_audit(
    *,
    baseline_audit: dict[str, Any],
    baseline_weather: pd.DataFrame,
    lean_weather: pd.DataFrame,
    metrics: pd.DataFrame,
    download_metadata: pd.DataFrame,
    retained_grib_count: int,
) -> dict[str, Any]:
    indexed = metrics.set_index("variable")
    exact_unchanged = all(
        bool(indexed.loc[variable, "exact_match"]) for variable in EXACT_MATCH_VARIABLES
    )
    structural_passed = all(
        [
            baseline_audit.get("status") == "full_weather_local_extraction_passed",
            len(baseline_weather) == EXPECTED_BASELINE_ROWS,
            len(lean_weather) == EXPECTED_BASELINE_ROWS,
            len(download_metadata) == len(STATE_PRODUCT_IDS) * len(GEFS_REFORECAST_MEMBERS),
            int(download_metadata["selected_message_count"].sum()) == 580,
            exact_unchanged,
            retained_grib_count == 0,
        ]
    )
    return {
        "status": (
            "local_baseline_full_vs_lean_weather_smoke_completed"
            if structural_passed
            else "local_baseline_full_vs_lean_weather_smoke_failed"
        ),
        "mandatory_structural_gate_passed": structural_passed,
        "baseline_zip_sha256": EXPECTED_BASELINE_ZIP_SHA256,
        "decision_date": EXPECTED_CYCLE,
        "site_count": len(EXPECTED_SITES),
        "member_count": len(GEFS_REFORECAST_MEMBERS),
        "lead_day_count": 7,
        "weather_row_count": len(lean_weather),
        "lean_state_product_count": len(STATE_PRODUCT_IDS),
        "lean_selected_message_count": int(
            download_metadata["selected_message_count"].sum()
        ),
        "lean_selected_range_bytes": int(
            download_metadata["selected_range_bytes"].sum()
        ),
        "lean_selected_range_gib": float(
            download_metadata["selected_range_bytes"].sum() / 1024**3
        ),
        "network_bytes_this_run": int(
            download_metadata["network_bytes_this_run"].sum()
        ),
        "full_baseline_payload_redownloaded": False,
        "unchanged_variables_reused_from_audited_baseline": list(EXACT_MATCH_VARIABLES),
        "all_required_weather_variables_retained": True,
        "unchanged_variables_exact_match": exact_unchanged,
        "weather_difference_metrics_computed": True,
        "weather_equivalence_approved": False,
        "teacher_review_required": True,
        "correction_applied": False,
        "retained_grib_file_count": retained_grib_count,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "training_eligible": False,
        "tta_performed": False,
        "next_gate": (
            "teacher_review_full_vs_lean_raw_daily_weather_before_policy_change"
            if structural_passed
            else "repair_local_baseline_weather_smoke"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Path]:
    if not args.baseline_zip.is_file():
        raise FileNotFoundError(f"baseline ZIP is missing: {args.baseline_zip}")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    baseline_weather, state_manifest, baseline_audit = extract_and_validate_baseline(
        args.baseline_zip, args.output_dir / "baseline"
    )
    sites = reforecast_site_frame(EXPECTED_SITES)
    point_parts: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    def extract(row: pd.Series):
        return download_lean_state_product(
            manifest_row=row,
            sites=sites,
            cache_dir=args.output_dir / "cache",
            timeout=args.timeout,
            retries=args.retries,
        )

    rows = [row for _, row in state_manifest.iterrows()]
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = [executor.submit(extract, row) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            points, metadata = future.result()
            point_parts.append(points)
            metadata_rows.append(metadata)
            print(
                f"[lean-state] {metadata['gefs_member']} {metadata['product_id']} "
                f"ready ({completed}/{len(rows)})",
                flush=True,
            )
    all_points = pd.concat(point_parts, ignore_index=True)
    lean_state = aggregate_lean_state(all_points)
    lean_weather = build_lean_weather(baseline_weather, lean_state)
    comparison, metrics = compare_weather(baseline_weather, lean_weather)
    download_metadata = pd.DataFrame(metadata_rows).sort_values(
        ["gefs_member", "product_id"]
    ).reset_index(drop=True)
    retained_gribs = list((args.output_dir / "cache").rglob("*.grib2"))
    audit = build_audit(
        baseline_audit=baseline_audit,
        baseline_weather=baseline_weather,
        lean_weather=lean_weather,
        metrics=metrics,
        download_metadata=download_metadata,
        retained_grib_count=len(retained_gribs),
    )
    outputs = {
        "baseline_weather": args.output_dir / "gefs_temporal_sampling_full_baseline_weather_v1.csv",
        "lean_weather": args.output_dir / "gefs_temporal_sampling_lean_weather_v1.csv",
        "comparison": args.output_dir / "gefs_temporal_sampling_row_comparison_v1.csv",
        "metrics": args.output_dir / "gefs_temporal_sampling_variable_metrics_v1.csv",
        "download_manifest": args.output_dir / "gefs_temporal_sampling_lean_download_manifest_v1.csv",
        "audit": args.output_dir / "gefs_temporal_sampling_local_smoke_audit_v1.json",
        "manifest": args.output_dir / "gefs_temporal_sampling_local_smoke_manifest_v1.json",
    }
    baseline_weather.to_csv(outputs["baseline_weather"], index=False)
    lean_weather.to_csv(outputs["lean_weather"], index=False)
    comparison.to_csv(outputs["comparison"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    download_metadata.to_csv(outputs["download_manifest"], index=False)
    write_json(outputs["audit"], audit)
    manifest = {
        "status": audit["status"],
        "baseline_zip": {
            "path": args.baseline_zip.name,
            "sha256": sha256_file(args.baseline_zip),
        },
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "temporary_grib_retained": False,
        "full_baseline_payload_redownloaded": False,
    }
    write_json(outputs["manifest"], manifest)
    if not audit["mandatory_structural_gate_passed"]:
        raise RuntimeError(f"local lean weather smoke failed; see {outputs['audit']}")
    return outputs


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

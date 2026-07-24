"""GEFSv12 reforecast precipitation extraction and empirical quantile mapping."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .gefs_gridmet_bias import (
    GribIndexRecord,
    aggregate_gefs_point_records,
    allocate_interval_to_local_days,
    decumulate_reset_intervals,
    decode_gefs_minigrib_points,
    fetch_selected_byte_ranges,
    merge_contiguous_ranges,
    parse_gefs_index,
)


CONTRACT_ID = "gefs-precipitation-quantile-mapping-v1"
CONTRACT_VERSION = 1
CONTRACT_ID_V2 = "gefs-precipitation-quantile-mapping-v2"
CONTRACT_VERSION_V2 = 2
CONTRACT_ID_TRAINING_CV = (
    "gefs-precipitation-qm-training-period-cross-validation-v1"
)
CONTRACT_VERSION_TRAINING_CV = 1
SUPPORTED_CONTRACTS = {
    CONTRACT_ID: CONTRACT_VERSION,
    CONTRACT_ID_V2: CONTRACT_VERSION_V2,
    CONTRACT_ID_TRAINING_CV: CONTRACT_VERSION_TRAINING_CV,
}
SITE_LOCAL_DAY_BOUNDARY = "SITE_LOCAL_00_to_24"
UTC_DAY_BOUNDARY = "UTC_00_to_24"
UPPER_TAIL_MULTIPLICATIVE = "multiplicative_exceedance_above_training_maximum"
UPPER_TAIL_CONSTANT_ADDITIVE = (
    "constant_additive_exceedance_above_training_maximum"
)
GEFS_REFORECAST_BASE = "https://noaa-gefs-retrospective.s3.amazonaws.com"
GEFS_REFORECAST_MEMBERS = ("c00", "p01", "p02", "p03", "p04")
ERA5_PRECIPITATION_SOURCE_UNIT = "m"
ERA5_PRECIPITATION_SCALE_TO_MM = 1000.0
ERA5_PRECIPITATION_METADATA_URL = (
    "https://developers.google.com/earth-engine/datasets/catalog/"
    "ECMWF_ERA5_LAND_DAILY_AGGR#bands"
)
SITE_METADATA = {
    "P1": (42.015928, -98.224144, "America/Chicago"),
    "P2": (40.595000, -88.415000, "America/Chicago"),
    "P3": (46.321000, -96.877000, "America/Chicago"),
    "P4": (42.681600, -94.668600, "America/Chicago"),
    "P15": (41.735000, -112.265000, "America/Denver"),
}


def reforecast_site_frame(site_ids: Sequence[str]) -> pd.DataFrame:
    unknown = sorted(set(site_ids).difference(SITE_METADATA))
    if unknown:
        raise ValueError(f"unknown reforecast sites: {unknown}")
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("site_ids must be unique")
    rows = []
    for site_id in site_ids:
        latitude, longitude, timezone_name = SITE_METADATA[site_id]
        rows.append(
            {
                "site": site_id,
                "site_id": site_id,
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone_name,
                "site_timezone": timezone_name,
            }
        )
    return pd.DataFrame(rows)


def build_reforecast_precipitation_url(
    cycle_date: str | date,
    member: str,
    *,
    index: bool = False,
) -> str:
    if member not in GEFS_REFORECAST_MEMBERS:
        raise ValueError(f"unsupported GEFS reforecast member: {member!r}")
    timestamp = pd.Timestamp(cycle_date)
    cycle = timestamp.strftime("%Y%m%d00")
    suffix = ".idx" if index else ""
    key = (
        f"GEFSv12/reforecast/{timestamp.year}/{cycle}/{member}/Days:1-10/"
        f"apcp_sfc_{cycle}_{member}.grib2{suffix}"
    )
    return f"{GEFS_REFORECAST_BASE}/{key}"


def reforecast_source_key(cycle_date: str | date, member: str) -> str:
    return build_reforecast_precipitation_url(cycle_date, member).split(
        ".com/", 1
    )[1]


def select_reforecast_precipitation_records(
    index_text: str,
    *,
    maximum_end_hour: int = 174,
) -> list[GribIndexRecord]:
    records = parse_gefs_index(index_text)
    selected = [
        record
        for record in records
        if record.short_name == "APCP"
        and record.level == "surface"
        and record.step.kind == "acc"
        and 0 < record.step.end_hour <= int(maximum_end_hour)
    ]
    if not selected:
        raise ValueError("no GEFS reforecast precipitation records were selected")
    if selected[0].step.start_hour != 0 or selected[0].step.end_hour != 3:
        raise ValueError("reforecast precipitation must begin with the 0-3 hour interval")
    if selected[-1].step.end_hour != int(maximum_end_hour):
        raise ValueError(
            "reforecast precipitation does not cover the required end hour "
            f"{maximum_end_hour}"
        )
    expected_end_hours = list(range(3, int(maximum_end_hour) + 1, 3))
    actual_end_hours = [record.step.end_hour for record in selected]
    if actual_end_hours != expected_end_hours:
        raise ValueError("reforecast precipitation has missing or duplicate end hours")
    if any(record.range_end is None for record in selected):
        raise ValueError("selected reforecast record has no byte-range end")
    return selected


def _request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    retries: int = 4,
) -> tuple[bytes, dict[str, str]]:
    request_headers = {"User-Agent": "s2s-rtist-gefs-qm/1.0"}
    request_headers.update(headers or {})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers=request_headers, method=method
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_headers = {
                    str(key).lower(): str(value)
                    for key, value in response.headers.items()
                }
                return response.read(), response_headers
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to request {url} after {retries} attempts") from last_error


def _range_fetcher(
    *,
    timeout: int,
    retries: int,
    total_ranges: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Callable[[str, int, int], bytes]:
    completed_ranges = 0

    def fetch(url: str, start: int, end: int) -> bytes:
        nonlocal completed_ranges
        payload, _ = _request(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=timeout,
            retries=retries,
        )
        completed_ranges += 1
        if progress_callback is not None:
            progress_callback(completed_ranges, int(total_ranges or 0))
        return payload

    return fetch


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_reforecast_member_points(
    *,
    cycle_date: str,
    member: str,
    sites: pd.DataFrame,
    cache_dir: Path,
    timeout: int = 120,
    retries: int = 4,
    keep_grib: bool = False,
    maximum_end_hour: int = 174,
    range_progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    product_url = build_reforecast_precipitation_url(cycle_date, member)
    index_url = build_reforecast_precipitation_url(cycle_date, member, index=True)
    stem = f"{pd.Timestamp(cycle_date).strftime('%Y%m%d')}_{member}_f000-f{maximum_end_hour:03d}"
    index_path = cache_dir / "indices" / f"{stem}.idx"
    grib_path = cache_dir / "minigrib" / f"{stem}.grib2"
    points_path = cache_dir / "point_records" / f"{stem}.csv"
    metadata_path = cache_dir / "metadata" / f"{stem}.json"
    for directory in (
        index_path.parent,
        grib_path.parent,
        points_path.parent,
        metadata_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if points_path.exists() and metadata_path.exists():
        points = pd.read_csv(points_path, parse_dates=["cycle_init_utc"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "cached_point_records"
        return points, metadata

    if index_path.exists():
        index_text = index_path.read_text(encoding="utf-8")
        _, index_headers = _request(
            index_url, method="HEAD", timeout=timeout, retries=retries
        )
    else:
        index_payload, index_headers = _request(
            index_url, timeout=timeout, retries=retries
        )
        index_text = index_payload.decode("utf-8")
        index_path.write_text(index_text, encoding="utf-8")

    records = select_reforecast_precipitation_records(
        index_text, maximum_end_hour=maximum_end_hour
    )
    _, product_headers = _request(
        product_url, method="HEAD", timeout=timeout, retries=retries
    )
    ranges = merge_contiguous_ranges(records)
    if not grib_path.exists():
        payload = fetch_selected_byte_ranges(
            product_url,
            ranges,
            fetcher=_range_fetcher(
                timeout=timeout,
                retries=retries,
                total_ranges=len(ranges),
                progress_callback=range_progress,
            ),
        )
        grib_path.write_bytes(payload)

    cycle_init = pd.Timestamp(f"{pd.Timestamp(cycle_date).strftime('%Y-%m-%d')}T00:00:00Z")
    points = decode_gefs_minigrib_points(
        grib_path,
        selected_records=records,
        sites=sites,
        cycle_init_utc=cycle_init,
        lead_hour=0,
    )
    points["gefs_member"] = member
    points.to_csv(points_path, index=False, encoding="utf-8-sig")
    metadata = {
        "status": "downloaded",
        "cycle_date": pd.Timestamp(cycle_date).strftime("%Y-%m-%d"),
        "forecast_init_utc": cycle_init.isoformat(),
        "gefs_member": member,
        "source_key": reforecast_source_key(cycle_date, member),
        "product_url": product_url,
        "index_url": index_url,
        "source_etag": product_headers.get("etag", "").strip('"'),
        "source_content_length": int(product_headers.get("content-length", 0)),
        "index_etag": index_headers.get("etag", "").strip('"'),
        "selected_message_count": len(records),
        "selected_start_step": min(record.step.start_hour for record in records),
        "selected_end_step": max(record.step.end_hour for record in records),
        "range_count": len(ranges),
        "downloaded_bytes": grib_path.stat().st_size,
        "downloaded_sha256": _sha256_file(grib_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not keep_grib:
        grib_path.unlink(missing_ok=True)
    return points, metadata


def _source_step_coverage(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["site", "timezone", "cycle_init_utc", "gefs_member"]
    for keys, group in points.groupby(group_columns, sort=False):
        site, timezone_name, cycle_init, member = keys
        tolerance = 1.0e-6
        if "packing_resolution" in group:
            tolerance = 2.0 * float(group["packing_resolution"].max())
        intervals = decumulate_reset_intervals(
            group[["start_hour", "end_hour", "value"]],
            kind="acc",
            negative_tolerance=tolerance,
        )
        cycle_timestamp = pd.Timestamp(cycle_init)
        cycle_timestamp = (
            cycle_timestamp.tz_localize("UTC")
            if cycle_timestamp.tzinfo is None
            else cycle_timestamp.tz_convert("UTC")
        )
        for interval in intervals.itertuples(index=False):
            allocated = allocate_interval_to_local_days(
                start_utc=(
                    cycle_timestamp
                    + pd.Timedelta(hours=int(interval.interval_start_hour))
                ).to_pydatetime(),
                end_utc=(
                    cycle_timestamp
                    + pd.Timedelta(hours=int(interval.interval_end_hour))
                ).to_pydatetime(),
                value=float(interval.interval_value),
                kind="acc",
                timezone_name=str(timezone_name),
            )
            for part in allocated.itertuples(index=False):
                rows.append(
                    {
                        "site": site,
                        "timezone": timezone_name,
                        "cycle_init_utc": cycle_timestamp,
                        "gefs_member": member,
                        "local_date": part.local_date,
                        "source_start_step": int(interval.interval_start_hour),
                        "source_end_step": int(interval.interval_end_hour),
                    }
                )
    coverage = pd.DataFrame(rows)
    keys = ["site", "timezone", "cycle_init_utc", "gefs_member", "local_date"]
    return coverage.groupby(keys, as_index=False).agg(
        source_start_step=("source_start_step", "min"),
        source_end_step=("source_end_step", "max"),
    )


def aggregate_reforecast_member_daily(
    points: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for member, member_points in points.groupby("gefs_member", sort=False):
        daily = aggregate_gefs_point_records(member_points)
        daily["gefs_member"] = member
        parts.append(daily)
    daily = pd.concat(parts, ignore_index=True)
    coverage = _source_step_coverage(points)
    keys = ["site", "timezone", "cycle_init_utc", "gefs_member", "local_date"]
    daily = daily.merge(coverage, on=keys, how="left", validate="one_to_one")
    manifest_columns = [
        "forecast_init_utc",
        "gefs_member",
        "source_key",
        "source_etag",
    ]
    source = manifest[manifest_columns].copy()
    source["forecast_init_utc"] = pd.to_datetime(
        source["forecast_init_utc"], utc=True
    )
    daily = daily.rename(
        columns={
            "site": "site_id",
            "timezone": "site_timezone",
            "cycle_init_utc": "forecast_init_utc",
            "precipitation_mm": "precipitation_mm_raw",
        }
    )
    daily["forecast_init_utc"] = pd.to_datetime(
        daily["forecast_init_utc"], utc=True
    )
    daily = daily.merge(
        source,
        on=["forecast_init_utc", "gefs_member"],
        how="left",
        validate="many_to_one",
    )
    columns = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        "local_date",
        "lead_day",
        "precipitation_mm_raw",
        "source_key",
        "source_etag",
        "source_start_step",
        "source_end_step",
    ]
    return daily[columns].sort_values(
        ["site_id", "forecast_init_utc", "gefs_member", "local_date"]
    ).reset_index(drop=True)


def aggregate_reforecast_member_daily_utc(
    points: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    utc_points = points.copy()
    utc_points["timezone"] = "UTC"
    daily = aggregate_reforecast_member_daily(utc_points, manifest=manifest)
    daily = daily.rename(columns={"local_date": "valid_date_utc"})
    daily["site_timezone"] = daily["site_id"].map(
        lambda site_id: SITE_METADATA[str(site_id)][2]
    )
    daily["aggregation_day_boundary"] = UTC_DAY_BOUNDARY
    columns = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        "valid_date_utc",
        "lead_day",
        "aggregation_day_boundary",
        "precipitation_mm_raw",
        "source_key",
        "source_etag",
        "source_start_step",
        "source_end_step",
    ]
    return daily[columns].sort_values(
        ["site_id", "forecast_init_utc", "gefs_member", "valid_date_utc"]
    ).reset_index(drop=True)


def _era5_tif_path(era5_root: Path, valid_date: pd.Timestamp) -> Path:
    day_index = int((valid_date.normalize() - pd.Timestamp(valid_date.year, 1, 1)).days)
    return (
        era5_root
        / f"era5_{valid_date.year}"
        / "total_precipitation_sum"
        / f"total_precipitation_sum_{day_index}.tif"
    )


def normalize_era5_precipitation_m(
    raw_m: float, *, negative_tolerance_m: float = 1.0e-7
) -> tuple[float, bool]:
    value = float(raw_m)
    if not math.isfinite(value):
        raise ValueError("ERA5 precipitation must be finite")
    if value < -float(negative_tolerance_m):
        raise ValueError(f"negative ERA5 precipitation exceeds tolerance: {value} m")
    clipped = value < 0.0
    return max(0.0, value) * ERA5_PRECIPITATION_SCALE_TO_MM, clipped


def extract_era5_reference_precipitation(
    *,
    era5_root: Path,
    sites: pd.DataFrame,
    valid_dates: Sequence[str | pd.Timestamp],
    negative_tolerance_m: float = 1.0e-7,
) -> pd.DataFrame:
    try:
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        raise RuntimeError("rasterio is required to extract ERA5 GeoTIFF values") from exc

    rows: list[dict[str, object]] = []
    for value in sorted({pd.Timestamp(item).normalize() for item in valid_dates}):
        path = _era5_tif_path(era5_root, value)
        if not path.is_file():
            raise FileNotFoundError(f"missing ERA5 precipitation GeoTIFF: {path}")
        with rasterio.open(path) as dataset:
            coordinates = []
            for site in sites.itertuples(index=False):
                xs, ys = transform(
                    "EPSG:4326",
                    dataset.crs,
                    [float(site.longitude)],
                    [float(site.latitude)],
                )
                coordinates.append((xs[0], ys[0]))
            samples = list(dataset.sample(coordinates))
            for site, sample in zip(sites.itertuples(index=False), samples):
                raw_m = float(sample[0])
                try:
                    normalized_mm, clipped = normalize_era5_precipitation_m(
                        raw_m, negative_tolerance_m=negative_tolerance_m
                    )
                except ValueError as exc:
                    raise ValueError(f"{exc} in {path} for {site.site}") from exc
                rows.append(
                    {
                        "site_id": str(site.site),
                        "local_date": value,
                        "precipitation_mm_reference": normalized_mm,
                        "reference_dataset": "ERA5_Land_daily_aggregated_project_GeoTIFF",
                        "reference_source_path": str(path),
                        "reference_source_unit": ERA5_PRECIPITATION_SOURCE_UNIT,
                        "reference_unit_conversion": "m * 1000 = mm",
                        "reference_unit_metadata_url": ERA5_PRECIPITATION_METADATA_URL,
                        "reference_negative_roundoff_clipped": clipped,
                    }
                )
    return pd.DataFrame(rows).sort_values(["site_id", "local_date"]).reset_index(
        drop=True
    )


def extract_era5_reference_precipitation_utc(
    *,
    era5_root: Path,
    sites: pd.DataFrame,
    valid_dates: Sequence[str | pd.Timestamp],
    negative_tolerance_m: float = 1.0e-7,
) -> pd.DataFrame:
    reference = extract_era5_reference_precipitation(
        era5_root=era5_root,
        sites=sites,
        valid_dates=valid_dates,
        negative_tolerance_m=negative_tolerance_m,
    ).rename(columns={"local_date": "valid_date_utc"})
    reference["aggregation_day_boundary"] = UTC_DAY_BOUNDARY
    return reference.sort_values(["site_id", "valid_date_utc"]).reset_index(
        drop=True
    )


def validate_member_daily_precipitation(
    frame: pd.DataFrame,
    *,
    expected_sites: Sequence[str],
    expected_members: Sequence[str],
    expected_cycles: Sequence[str],
    horizon_days: int = 7,
    date_column: str = "local_date",
) -> None:
    required = {
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        date_column,
        "lead_day",
        "precipitation_mm_raw",
        "source_key",
        "source_etag",
        "source_start_step",
        "source_end_step",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing member daily columns: {sorted(missing)}")
    expected_rows = (
        len(expected_sites)
        * len(expected_members)
        * len(expected_cycles)
        * int(horizon_days)
    )
    if len(frame) != expected_rows:
        raise ValueError(f"member daily rows={len(frame)}, expected={expected_rows}")
    keys = ["site_id", "forecast_init_utc", "gefs_member", date_column]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate member daily precipitation rows")
    numeric = frame[
        [
            "lead_day",
            "precipitation_mm_raw",
            "source_start_step",
            "source_end_step",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("member daily precipitation contains missing or nonfinite values")
    if (numeric["precipitation_mm_raw"] < 0.0).any():
        raise ValueError("member daily precipitation contains negative values")
    if set(frame["site_id"]) != set(expected_sites):
        raise ValueError("member daily precipitation site set is incomplete")
    if set(frame["gefs_member"]) != set(expected_members):
        raise ValueError("member daily precipitation member set is incomplete")
    grouped = frame.groupby(
        ["site_id", "forecast_init_utc", date_column],
        dropna=False,
        sort=False,
    )
    expected_member_set = set(expected_members)
    for key, group in grouped:
        if set(group["gefs_member"]) != expected_member_set:
            raise ValueError(f"incomplete member set for {key}")
    if frame["source_etag"].astype(str).str.strip().eq("").any():
        raise ValueError("source ETag is missing")


def validate_reference_daily_precipitation(
    frame: pd.DataFrame,
    *,
    expected_sites: Sequence[str],
    expected_dates: Sequence[str | pd.Timestamp],
    date_column: str = "local_date",
) -> None:
    required = {
        "site_id",
        date_column,
        "precipitation_mm_reference",
        "reference_dataset",
        "reference_source_path",
        "reference_unit_conversion",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing reference daily columns: {sorted(missing)}")
    expected_rows = len(expected_sites) * len(
        {pd.Timestamp(value).normalize() for value in expected_dates}
    )
    if len(frame) != expected_rows:
        raise ValueError(f"reference daily rows={len(frame)}, expected={expected_rows}")
    if frame.duplicated(["site_id", date_column]).any():
        raise ValueError("duplicate reference daily precipitation rows")
    values = pd.to_numeric(frame["precipitation_mm_reference"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("reference precipitation contains missing or nonfinite values")
    if (values < 0.0).any():
        raise ValueError("reference precipitation contains negative values")


def _stable_frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame[list(columns)].copy()
    for column in ordered.columns:
        if pd.api.types.is_datetime64_any_dtype(ordered[column]):
            ordered[column] = ordered[column].astype(str)
    ordered = ordered.sort_values(list(columns), kind="stable").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _sha256_bytes(payload)


def _canonical_artifact_hash(artifact: dict[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _normalise_group_value(column: str, value: Any) -> str | int | float:
    if pd.isna(value):
        raise ValueError(f"quantile mapping group key {column} contains missing values")
    if column == "lead_day":
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def _group_artifact_key(group_keys: Sequence[str], values: Sequence[Any]) -> str:
    normalised = [
        _normalise_group_value(column, value)
        for column, value in zip(group_keys, values, strict=True)
    ]
    if list(group_keys) == ["site_id", "lead_day"]:
        return f"{normalised[0]}|{normalised[1]}"
    if not group_keys:
        return "global"
    return "|".join(
        f"{column}={value}" for column, value in zip(group_keys, normalised, strict=True)
    )


def verify_quantile_mapping_artifact(
    artifact: dict[str, Any], *, expected_contract_id: str | None = None
) -> None:
    expected = str(artifact.get("artifact_sha256", ""))
    if not expected or expected != _canonical_artifact_hash(artifact):
        raise ValueError("quantile mapping artifact hash mismatch")
    contract_id = str(artifact.get("contract_id", ""))
    contract_version = artifact.get("contract_version")
    if contract_id not in SUPPORTED_CONTRACTS:
        raise ValueError("quantile mapping artifact contract mismatch")
    if contract_version != SUPPORTED_CONTRACTS[contract_id]:
        raise ValueError("quantile mapping artifact contract version mismatch")
    if expected_contract_id is not None and contract_id != expected_contract_id:
        raise ValueError(
            "quantile mapping artifact does not match the expected contract"
        )
    if contract_id in {CONTRACT_ID_V2, CONTRACT_ID_TRAINING_CV}:
        if artifact.get("aggregation_day_boundary") != UTC_DAY_BOUNDARY:
            raise ValueError("quantile mapping artifact requires UTC day boundary")
        if artifact.get("canonical_valid_date_column") != "valid_date_utc":
            raise ValueError("quantile mapping artifact date column mismatch")
        if artifact.get("upper_tail_policy") != UPPER_TAIL_CONSTANT_ADDITIVE:
            raise ValueError("quantile mapping artifact upper-tail policy mismatch")
    if contract_id == CONTRACT_ID_TRAINING_CV:
        context = artifact.get("artifact_context")
        if not isinstance(context, dict):
            raise ValueError("training CV artifact requires artifact_context")
        validation_year = int(context.get("validation_year", -1))
        if validation_year in set(int(year) for year in artifact.get("fit_years", [])):
            raise ValueError("training CV validation year leaked into fit_years")
        if validation_year not in {2015, 2016, 2017, 2018}:
            raise ValueError("training CV validation year is outside 2015-2018")
    group_keys = artifact.get("group_keys")
    if not isinstance(group_keys, list) or not all(
        isinstance(column, str) for column in group_keys
    ):
        raise ValueError("quantile mapping artifact group_keys are invalid")
    forbidden = {2019, 2024}.intersection(set(artifact.get("fit_years", [])))
    if forbidden:
        raise ValueError(f"validation/test years found in fit_years: {sorted(forbidden)}")


def _merge_duplicate_nodes(
    forecast_nodes: np.ndarray, reference_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"forecast": forecast_nodes, "reference": reference_nodes})
    merged = frame.groupby("forecast", as_index=False, sort=True)["reference"].mean()
    merged["reference"] = np.maximum.accumulate(merged["reference"].to_numpy())
    return (
        merged["forecast"].to_numpy(dtype=float),
        merged["reference"].to_numpy(dtype=float),
    )


def _fit_group_mapping(
    forecast: np.ndarray,
    reference: np.ndarray,
    *,
    wet_day_reference_threshold_mm: float,
    include_additive_offset: bool = False,
    occurrence_correction: bool = True,
) -> dict[str, Any]:
    if len(forecast) != len(reference):
        raise ValueError("forecast and repeated reference sample counts must match")
    if len(forecast) == 0:
        raise ValueError("cannot fit an empty quantile mapping group")
    if np.any(~np.isfinite(forecast)) or np.any(~np.isfinite(reference)):
        raise ValueError("quantile mapping samples must be finite")
    if np.any(forecast < 0.0) or np.any(reference < 0.0):
        raise ValueError("quantile mapping precipitation samples must be nonnegative")

    reference_wet = reference > float(wet_day_reference_threshold_mm)
    target_wet_count = int(reference_wet.sum())
    if target_wet_count == 0:
        raise ValueError("quantile mapping group has no positive reference samples")
    if occurrence_correction:
        sorted_forecast = np.sort(forecast)
        target_dry_count = len(sorted_forecast) - target_wet_count
        if target_dry_count <= 0:
            forecast_wet_threshold = -1.0e-12
        else:
            forecast_wet_threshold = float(sorted_forecast[target_dry_count - 1])
    else:
        forecast_wet_threshold = 0.0

    positive_forecast = forecast[forecast > forecast_wet_threshold]
    positive_reference = reference[reference_wet]
    if len(positive_forecast) == 0:
        raise ValueError("occurrence correction leaves no positive forecast samples")
    node_count = max(1, min(len(positive_forecast), len(positive_reference)))
    probabilities = np.linspace(0.0, 1.0, node_count)
    forecast_nodes = np.quantile(positive_forecast, probabilities)
    reference_nodes = np.quantile(positive_reference, probabilities)
    forecast_nodes, reference_nodes = _merge_duplicate_nodes(
        np.asarray(forecast_nodes, dtype=float),
        np.asarray(reference_nodes, dtype=float),
    )
    if len(forecast_nodes) == 0 or float(forecast_nodes[-1]) <= 0.0:
        raise ValueError("positive quantile mapping nodes require a positive forecast maximum")
    upper_ratio = float(reference_nodes[-1]) / float(forecast_nodes[-1])
    result = {
        "sample_count": int(len(forecast)),
        "reference_wet_sample_count": target_wet_count,
        "forecast_positive_sample_count": int(len(positive_forecast)),
        "forecast_wet_threshold_mm": forecast_wet_threshold,
        "forecast_quantile_nodes": forecast_nodes.tolist(),
        "reference_quantile_nodes": reference_nodes.tolist(),
        "effective_quantile_node_count": int(len(forecast_nodes)),
        "training_forecast_maximum_mm": float(forecast_nodes[-1]),
        "training_reference_maximum_mm": float(reference_nodes[-1]),
        "upper_tail_multiplicative_ratio": upper_ratio,
    }
    if include_additive_offset:
        result["upper_tail_additive_offset_mm"] = float(
            reference_nodes[-1] - forecast_nodes[-1]
        )
    if not occurrence_correction:
        result["occurrence_correction"] = False
    return result


def fit_empirical_precipitation_qm(
    frame: pd.DataFrame,
    *,
    fit_years: Sequence[int] = (2015, 2016, 2017, 2018),
    wet_day_reference_threshold_mm: float = 0.1,
    expected_members: Sequence[str] = GEFS_REFORECAST_MEMBERS,
    contract_id: str = CONTRACT_ID,
    contract_version: int = CONTRACT_VERSION,
    aggregation_day_boundary: str = SITE_LOCAL_DAY_BOUNDARY,
    canonical_valid_date_column: str = "local_date",
    upper_tail_policy: str = UPPER_TAIL_MULTIPLICATIVE,
    group_keys: Sequence[str] = ("site_id", "lead_day"),
    artifact_context: dict[str, Any] | None = None,
    occurrence_correction: bool = True,
) -> dict[str, Any]:
    if SUPPORTED_CONTRACTS.get(contract_id) != int(contract_version):
        raise ValueError("unsupported quantile mapping contract id/version")
    if contract_id in {CONTRACT_ID_V2, CONTRACT_ID_TRAINING_CV}:
        if aggregation_day_boundary != UTC_DAY_BOUNDARY:
            raise ValueError("quantile mapping fit requires UTC day boundary")
        if canonical_valid_date_column != "valid_date_utc":
            raise ValueError("quantile mapping fit requires valid_date_utc")
        if upper_tail_policy != UPPER_TAIL_CONSTANT_ADDITIVE:
            raise ValueError("quantile mapping fit requires constant additive tail")
    elif upper_tail_policy != UPPER_TAIL_MULTIPLICATIVE:
        raise ValueError("v1 quantile mapping fit requires multiplicative tail")
    group_keys = tuple(str(column) for column in group_keys)
    if len(group_keys) != len(set(group_keys)):
        raise ValueError("quantile mapping group_keys contain duplicates")
    required = {
        "site_id",
        "decision_date",
        canonical_valid_date_column,
        "lead_day",
        "gefs_member",
        "precipitation_mm_raw",
        "precipitation_mm_reference",
    }
    required.update(group_keys)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing quantile mapping fit columns: {sorted(missing)}")
    data = frame.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data[canonical_valid_date_column] = pd.to_datetime(
        data[canonical_valid_date_column]
    )
    actual_years = set(data["decision_date"].dt.year.astype(int))
    allowed_years = {int(year) for year in fit_years}
    unexpected_years = sorted(actual_years.difference(allowed_years))
    if unexpected_years:
        raise ValueError(f"non-fit years present in quantile mapping data: {unexpected_years}")
    if {2019, 2024}.intersection(actual_years):
        raise ValueError("2019 or 2024 data cannot be used to fit quantile mapping")
    if data.duplicated(
        ["site_id", "decision_date", canonical_valid_date_column, "gefs_member"]
    ).any():
        raise ValueError("duplicate member observations in quantile mapping fit data")
    expected_member_set = set(expected_members)
    for key, group in data.groupby(
        ["site_id", "decision_date", canonical_valid_date_column],
        sort=False,
        dropna=False,
    ):
        if set(group["gefs_member"]) != expected_member_set:
            raise ValueError(f"incomplete fit member set for {key}")
        if group["precipitation_mm_reference"].nunique(dropna=False) != 1:
            raise ValueError(f"reference precipitation differs across members for {key}")

    groups: dict[str, dict[str, Any]] = {}
    grouped = (
        [((), data)]
        if not group_keys
        else data.groupby(list(group_keys), sort=True, dropna=False)
    )
    for raw_values, group in grouped:
        values = raw_values if isinstance(raw_values, tuple) else (raw_values,)
        key = _group_artifact_key(group_keys, values)
        groups[key] = _fit_group_mapping(
            group["precipitation_mm_raw"].to_numpy(dtype=float),
            group["precipitation_mm_reference"].to_numpy(dtype=float),
            wet_day_reference_threshold_mm=wet_day_reference_threshold_mm,
            include_additive_offset=contract_id
            in {CONTRACT_ID_V2, CONTRACT_ID_TRAINING_CV},
            occurrence_correction=occurrence_correction,
        )
        normalised_values = {
            column: _normalise_group_value(column, value)
            for column, value in zip(group_keys, values, strict=True)
        }
        if list(group_keys) == ["site_id", "lead_day"]:
            groups[key]["site_id"] = normalised_values["site_id"]
            groups[key]["lead_day"] = normalised_values["lead_day"]
        else:
            groups[key]["group_values"] = normalised_values

    hash_columns = [
        "site_id",
        "decision_date",
        canonical_valid_date_column,
        "lead_day",
        "gefs_member",
        "precipitation_mm_raw",
        "precipitation_mm_reference",
    ]
    artifact: dict[str, Any] = {
        "contract_id": contract_id,
        "contract_version": int(contract_version),
        "fit_years": sorted(allowed_years),
        "site_ids": sorted(data["site_id"].astype(str).unique().tolist()),
        "group_keys": list(group_keys),
        "group_sample_counts": {
            key: value["sample_count"] for key, value in groups.items()
        },
        "wet_day_reference_threshold_mm": float(wet_day_reference_threshold_mm),
        "fitted_forecast_wet_thresholds_mm": {
            key: value["forecast_wet_threshold_mm"] for key, value in groups.items()
        },
        "forecast_quantile_nodes": {
            key: value["forecast_quantile_nodes"] for key, value in groups.items()
        },
        "reference_quantile_nodes": {
            key: value["reference_quantile_nodes"] for key, value in groups.items()
        },
        "upper_tail_policy": upper_tail_policy,
        "mapping_scope": "shared_across_exchangeable_members",
        "reference_weighting": "repeat_each_reference_value_by_member_count",
        "training_input_sha256": _stable_frame_hash(data, hash_columns),
        "groups": groups,
    }
    if contract_id in {CONTRACT_ID_V2, CONTRACT_ID_TRAINING_CV}:
        artifact.update(
            {
                "aggregation_day_boundary": aggregation_day_boundary,
                "canonical_valid_date_column": canonical_valid_date_column,
                "training_forecast_maxima_mm": {
                    key: value["training_forecast_maximum_mm"]
                    for key, value in groups.items()
                },
                "training_reference_maxima_mm": {
                    key: value["training_reference_maximum_mm"]
                    for key, value in groups.items()
                },
                "upper_tail_additive_offsets_mm": {
                    key: value["upper_tail_additive_offset_mm"]
                    for key, value in groups.items()
                },
            }
        )
    if artifact_context is not None:
        artifact["artifact_context"] = artifact_context
    if not occurrence_correction:
        artifact["occurrence_correction"] = False
    artifact["artifact_sha256"] = _canonical_artifact_hash(artifact)
    verify_quantile_mapping_artifact(artifact)
    return artifact


def _apply_group_mapping(
    values: np.ndarray,
    group: dict[str, Any],
    *,
    upper_tail_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(group["forecast_wet_threshold_mm"])
    forecast_nodes = np.asarray(group["forecast_quantile_nodes"], dtype=float)
    reference_nodes = np.asarray(group["reference_quantile_nodes"], dtype=float)
    if len(forecast_nodes) != len(reference_nodes) or len(forecast_nodes) == 0:
        raise ValueError("invalid quantile mapping node arrays")
    output = np.zeros(len(values), dtype=float)
    upper = values > float(forecast_nodes[-1])
    positive = values > threshold
    in_range = positive & ~upper
    if len(forecast_nodes) == 1:
        output[in_range] = float(reference_nodes[0])
    else:
        output[in_range] = np.interp(
            values[in_range], forecast_nodes, reference_nodes
        )
    if upper_tail_policy == UPPER_TAIL_MULTIPLICATIVE:
        output[upper] = values[upper] * float(
            group["upper_tail_multiplicative_ratio"]
        )
    elif upper_tail_policy == UPPER_TAIL_CONSTANT_ADDITIVE:
        output[upper] = values[upper] + float(
            group["upper_tail_additive_offset_mm"]
        )
    else:
        raise ValueError(f"unsupported upper-tail policy: {upper_tail_policy}")
    return np.maximum(output, 0.0), upper


def apply_empirical_precipitation_qm(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    split: str,
) -> pd.DataFrame:
    verify_quantile_mapping_artifact(artifact)
    group_keys = tuple(artifact["group_keys"])
    required = {
        "precipitation_mm_raw",
        "precipitation_mm_reference",
    }.union(group_keys)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing quantile mapping apply columns: {sorted(missing)}")
    data = frame.copy()
    data["precipitation_mm_qm"] = np.nan
    data["qm_forecast_wet_threshold_mm"] = np.nan
    data["qm_extrapolated_upper"] = False
    grouped_indices = (
        [((), data.index)]
        if not group_keys
        else data.groupby(list(group_keys), sort=False, dropna=False).groups.items()
    )
    group_key_by_index: dict[Any, str] = {}
    for raw_values, indices in grouped_indices:
        values = raw_values if isinstance(raw_values, tuple) else (raw_values,)
        key = _group_artifact_key(group_keys, values)
        if key not in artifact["groups"]:
            raise ValueError(f"no frozen quantile mapping group for {key}")
        group = artifact["groups"][key]
        values = data.loc[indices, "precipitation_mm_raw"].to_numpy(dtype=float)
        corrected, upper = _apply_group_mapping(
            values,
            group,
            upper_tail_policy=str(artifact["upper_tail_policy"]),
        )
        data.loc[indices, "precipitation_mm_qm"] = corrected
        data.loc[indices, "qm_forecast_wet_threshold_mm"] = float(
            group["forecast_wet_threshold_mm"]
        )
        data.loc[indices, "qm_extrapolated_upper"] = upper
        for index in indices:
            group_key_by_index[index] = key
    if artifact["contract_id"] in {CONTRACT_ID_V2, CONTRACT_ID_TRAINING_CV}:
        data["aggregation_day_boundary"] = artifact["aggregation_day_boundary"]
        data["qm_upper_tail_policy"] = artifact["upper_tail_policy"]
        data["qm_upper_tail_offset_mm"] = data.index.map(
            lambda index: float(
                artifact["groups"][group_key_by_index[index]][
                    "upper_tail_additive_offset_mm"
                ]
            )
        )
    if data["precipitation_mm_qm"].isna().any():
        raise ValueError("quantile mapping left missing corrected values")
    data["split"] = split
    data["qm_artifact_sha256"] = artifact["artifact_sha256"]
    return data


def write_quantile_mapping_artifact(path: Path, artifact: dict[str, Any]) -> None:
    verify_quantile_mapping_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_quantile_mapping_artifact(
    path: Path, *, expected_contract_id: str | None = None
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    verify_quantile_mapping_artifact(
        artifact, expected_contract_id=expected_contract_id
    )
    return artifact


def cycle_valid_dates(cycle_date: str, horizon_days: int = 7) -> list[pd.Timestamp]:
    start = pd.Timestamp(cycle_date).normalize()
    return [start + timedelta(days=offset) for offset in range(int(horizon_days))]


def pair_member_and_reference(
    member_daily: pd.DataFrame,
    reference_daily: pd.DataFrame,
    *,
    date_column: str = "local_date",
) -> pd.DataFrame:
    paired = member_daily.merge(
        reference_daily[
            ["site_id", date_column, "precipitation_mm_reference"]
        ],
        on=["site_id", date_column],
        how="left",
        validate="many_to_one",
    )
    if paired["precipitation_mm_reference"].isna().any():
        raise ValueError("missing ERA5 reference values after pairing")
    return paired

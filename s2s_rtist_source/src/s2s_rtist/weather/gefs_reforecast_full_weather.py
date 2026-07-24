"""Discover and extract full-variable GEFSv12 reforecast weather."""

from __future__ import annotations

import json
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .gefs_gridmet_bias import (
    GribIndexRecord,
    aggregate_gefs_point_records,
    decode_gefs_minigrib_points,
    fetch_selected_byte_ranges,
    merge_contiguous_ranges,
    parse_gefs_index,
)
from .gefs_quantile_mapping import (
    GEFS_REFORECAST_BASE,
    GEFS_REFORECAST_MEMBERS,
    _request,
    _sha256_file,
)


@dataclass(frozen=True)
class ReforecastProductSpec:
    product_id: str
    short_name: str
    level: str
    kind: str
    required: bool = True


@dataclass(frozen=True)
class ReforecastObject:
    key: str
    size: int
    etag: str
    last_modified: str


@dataclass(frozen=True)
class ProductObjectPair:
    spec: ReforecastProductSpec
    product: ReforecastObject
    index: ReforecastObject


REQUIRED_PRODUCT_SPECS = (
    ReforecastProductSpec("apcp_sfc", "APCP", "surface", "acc"),
    ReforecastProductSpec("tmp_2m", "TMP", "2 m above ground", "instant"),
    ReforecastProductSpec("spfh_2m", "SPFH", "2 m above ground", "instant"),
    ReforecastProductSpec("pres_sfc", "PRES", "surface", "instant"),
    ReforecastProductSpec("ugrd_hgt", "UGRD", "10 m above ground", "instant"),
    ReforecastProductSpec("vgrd_hgt", "VGRD", "10 m above ground", "instant"),
    ReforecastProductSpec("dswrf_sfc", "DSWRF", "surface", "ave"),
)
OPTIONAL_PRODUCT_SPECS = (
    ReforecastProductSpec(
        "tmax_2m", "TMAX", "2 m above ground", "max", required=False
    ),
    ReforecastProductSpec(
        "tmin_2m", "TMIN", "2 m above ground", "min", required=False
    ),
)
CANONICAL_WEATHER_COLUMNS = (
    "precipitation_mm_raw",
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)


def reforecast_member_prefix(cycle_date: str, member: str) -> str:
    if member not in GEFS_REFORECAST_MEMBERS:
        raise ValueError(f"unsupported GEFS reforecast member: {member!r}")
    cycle = pd.Timestamp(cycle_date).strftime("%Y%m%d00")
    return f"GEFSv12/reforecast/{cycle[:4]}/{cycle}/{member}/Days:1-10/"


def reforecast_inventory_url(cycle_date: str, member: str) -> str:
    query = urllib.parse.urlencode(
        {
            "list-type": "2",
            "prefix": reforecast_member_prefix(cycle_date, member),
            "max-keys": "1000",
        }
    )
    return f"{GEFS_REFORECAST_BASE}/?{query}"


def parse_reforecast_inventory(xml_payload: bytes | str) -> list[ReforecastObject]:
    payload = xml_payload.encode("utf-8") if isinstance(xml_payload, str) else xml_payload
    root = ET.fromstring(payload)
    truncated = root.findtext("{*}IsTruncated", default="false").strip().lower()
    if truncated == "true":
        raise ValueError("reforecast inventory is truncated")
    objects: list[ReforecastObject] = []
    for contents in root.findall("{*}Contents"):
        key = contents.findtext("{*}Key")
        size = contents.findtext("{*}Size")
        if key is None or size is None:
            raise ValueError("reforecast inventory object is missing key or size")
        objects.append(
            ReforecastObject(
                key=key,
                size=int(size),
                etag=(contents.findtext("{*}ETag") or "").strip('"'),
                last_modified=contents.findtext("{*}LastModified") or "",
            )
        )
    if not objects:
        raise ValueError("reforecast inventory contains no objects")
    return objects


def object_url(key: str) -> str:
    return f"{GEFS_REFORECAST_BASE}/{urllib.parse.quote(key, safe='/:')}"


def select_product_objects(
    objects: Sequence[ReforecastObject],
    *,
    specs: Sequence[ReforecastProductSpec] = REQUIRED_PRODUCT_SPECS,
) -> list[ProductObjectPair]:
    by_name = {Path(item.key).name: item for item in objects}
    pairs: list[ProductObjectPair] = []
    missing: list[str] = []
    for spec in specs:
        product_matches = [
            item
            for name, item in by_name.items()
            if name.startswith(f"{spec.product_id}_") and name.endswith(".grib2")
        ]
        if not product_matches:
            if spec.required:
                missing.append(spec.product_id)
            continue
        if len(product_matches) != 1:
            raise ValueError(
                f"multiple reforecast objects matched {spec.product_id}: "
                f"{[item.key for item in product_matches]}"
            )
        product = product_matches[0]
        index = by_name.get(Path(product.key).name + ".idx")
        if index is None:
            raise ValueError(f"missing index object for {product.key}")
        pairs.append(ProductObjectPair(spec=spec, product=product, index=index))
    if missing:
        available = sorted(
            name for name in by_name if name.endswith((".grib2", ".grib2.idx"))
        )
        raise ValueError(
            f"missing required reforecast products {missing}; available={available}"
        )
    return pairs


def select_product_records(
    index_text: str,
    *,
    spec: ReforecastProductSpec,
    maximum_end_hour: int = 174,
) -> list[GribIndexRecord]:
    records = parse_gefs_index(index_text)
    selected = [
        item
        for item in records
        if item.short_name == spec.short_name
        and item.level == spec.level
        and item.step.kind == spec.kind
        and 0 < item.step.end_hour <= maximum_end_hour
    ]
    if not selected:
        raise ValueError(
            f"no {spec.short_name}/{spec.level}/{spec.kind} records were selected"
        )
    end_hours = [item.step.end_hour for item in selected]
    if end_hours != sorted(set(end_hours)):
        raise ValueError(f"{spec.short_name} has duplicate or unordered end hours")
    if end_hours[-1] != maximum_end_hour:
        raise ValueError(
            f"{spec.short_name} ends at {end_hours[-1]} h, expected {maximum_end_hour} h"
        )
    if spec.required and end_hours != list(range(3, maximum_end_hour + 1, 3)):
        raise ValueError(f"{spec.short_name} does not provide complete three-hour coverage")
    if any(item.range_end is None for item in selected):
        raise ValueError(f"{spec.short_name} selected record has no byte-range end")
    return selected


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _product_cache_paths(
    *,
    cycle_date: str,
    member: str,
    product_id: str,
    cache_dir: Path,
    maximum_end_hour: int,
) -> dict[str, Path]:
    cycle = pd.Timestamp(cycle_date).strftime("%Y%m%d")
    stem = f"{cycle}_{member}_{product_id}_f003-f{maximum_end_hour:03d}"
    return {
        "index": cache_dir / "indices" / f"{stem}.idx",
        "grib": cache_dir / "minigrib" / f"{stem}.grib2",
        "points": cache_dir / "point_records" / f"{stem}.csv",
        "metadata": cache_dir / "metadata" / f"{stem}.json",
    }


def preflight_product(
    *,
    cycle_date: str,
    member: str,
    pair: ProductObjectPair,
    cache_dir: Path,
    timeout: int,
    retries: int,
    maximum_end_hour: int = 174,
) -> dict[str, object]:
    paths = _product_cache_paths(
        cycle_date=cycle_date,
        member=member,
        product_id=pair.spec.product_id,
        cache_dir=cache_dir,
        maximum_end_hour=maximum_end_hour,
    )
    index_path = paths["index"]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_url = object_url(pair.index.key)
    if index_path.exists():
        index_payload = index_path.read_bytes()
        network_bytes = 0
        status = "cached_index"
    else:
        index_payload, _ = _request(index_url, timeout=timeout, retries=retries)
        index_path.write_bytes(index_payload)
        network_bytes = len(index_payload)
        status = "downloaded_index"
    records = select_product_records(
        index_payload.decode("utf-8"),
        spec=pair.spec,
        maximum_end_hour=maximum_end_hour,
    )
    ranges = merge_contiguous_ranges(records)
    selected_range_bytes = sum(item.end - item.start + 1 for item in ranges)
    return {
        "status": status,
        "cycle_date": pd.Timestamp(cycle_date).strftime("%Y-%m-%d"),
        "gefs_member": member,
        "product_id": pair.spec.product_id,
        "short_name": pair.spec.short_name,
        "level": pair.spec.level,
        "kind": pair.spec.kind,
        "source_key": pair.product.key,
        "source_etag": pair.product.etag,
        "source_content_length": pair.product.size,
        "index_key": pair.index.key,
        "index_etag": pair.index.etag,
        "index_bytes": len(index_payload),
        "index_network_bytes_this_run": network_bytes,
        "selected_message_count": len(records),
        "selected_start_step": min(item.step.start_hour for item in records),
        "selected_end_step": max(item.step.end_hour for item in records),
        "range_count": len(ranges),
        "selected_range_bytes": selected_range_bytes,
    }


def load_or_download_inventory(
    *,
    cycle_date: str,
    member: str,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[list[ReforecastObject], dict[str, object]]:
    cycle = pd.Timestamp(cycle_date).strftime("%Y%m%d")
    path = cache_dir / "inventories" / f"{cycle}_{member}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = reforecast_inventory_url(cycle_date, member)
    if path.exists():
        payload = path.read_bytes()
        status = "cached_inventory"
        network_bytes = 0
    else:
        payload, _ = _request(url, timeout=timeout, retries=retries)
        path.write_bytes(payload)
        status = "downloaded_inventory"
        network_bytes = len(payload)
    objects = parse_reforecast_inventory(payload)
    return objects, {
        "status": status,
        "cycle_date": pd.Timestamp(cycle_date).strftime("%Y-%m-%d"),
        "gefs_member": member,
        "inventory_url": url,
        "inventory_object_count": len(objects),
        "inventory_bytes": len(payload),
        "network_bytes_this_run": network_bytes,
        "inventory_cache_path": str(path),
    }


def download_product_points(
    *,
    cycle_date: str,
    member: str,
    pair: ProductObjectPair,
    sites: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    retries: int,
    maximum_end_hour: int = 174,
    range_workers: int = 1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    paths = _product_cache_paths(
        cycle_date=cycle_date,
        member=member,
        product_id=pair.spec.product_id,
        cache_dir=cache_dir,
        maximum_end_hour=maximum_end_hour,
    )
    index_path = paths["index"]
    grib_path = paths["grib"]
    points_path = paths["points"]
    metadata_path = paths["metadata"]
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)

    if points_path.exists() and metadata_path.exists():
        points = pd.read_csv(points_path, parse_dates=["cycle_init_utc"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "cached_point_records"
        metadata["network_bytes_this_run"] = 0
        return points, metadata

    index_url = object_url(pair.index.key)
    product_url = object_url(pair.product.key)
    network_bytes = 0
    if index_path.exists():
        index_payload = index_path.read_bytes()
    else:
        index_payload, _ = _request(index_url, timeout=timeout, retries=retries)
        network_bytes += len(index_payload)
        index_path.write_bytes(index_payload)
    records = select_product_records(
        index_payload.decode("utf-8"),
        spec=pair.spec,
        maximum_end_hour=maximum_end_hour,
    )
    ranges = merge_contiguous_ranges(records)
    selected_bytes = 0

    def fetch(url: str, start: int, end: int) -> bytes:
        payload, _ = _request(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=timeout,
            retries=retries,
        )
        return payload

    if not grib_path.exists():
        payload = fetch_selected_byte_ranges(
            product_url,
            ranges,
            fetcher=fetch,
            workers=range_workers,
        )
        selected_bytes = len(payload)
        network_bytes += selected_bytes
        grib_path.write_bytes(payload)
    else:
        selected_bytes = grib_path.stat().st_size
    try:
        cycle_init = pd.Timestamp(f"{cycle_date}T00:00:00Z")
        points = decode_gefs_minigrib_points(
            grib_path,
            selected_records=records,
            sites=sites,
            cycle_init_utc=cycle_init,
            lead_hour=0,
        )
        points["gefs_member"] = member
        points["source_product_id"] = pair.spec.product_id
        points["source_key"] = pair.product.key
        points.to_csv(points_path, index=False, encoding="utf-8-sig")
        metadata: dict[str, object] = {
            "status": "downloaded",
            "cycle_date": pd.Timestamp(cycle_date).strftime("%Y-%m-%d"),
            "forecast_init_utc": cycle_init.isoformat(),
            "gefs_member": member,
            "product_id": pair.spec.product_id,
            "short_name": pair.spec.short_name,
            "level": pair.spec.level,
            "kind": pair.spec.kind,
            "source_key": pair.product.key,
            "source_etag": pair.product.etag,
            "source_content_length": pair.product.size,
            "index_key": pair.index.key,
            "index_etag": pair.index.etag,
            "selected_message_count": len(records),
            "selected_start_step": min(item.step.start_hour for item in records),
            "selected_end_step": max(item.step.end_hour for item in records),
            "range_count": len(ranges),
            "selected_range_bytes": grib_path.stat().st_size,
            "selected_range_sha256": _sha256_file(grib_path),
            "index_bytes": len(index_payload),
            "network_bytes_this_run": network_bytes,
            "point_rows": len(points),
        }
        _write_json(metadata_path, metadata)
    finally:
        grib_path.unlink(missing_ok=True)
    return points, metadata


def specific_humidity_to_vapor_pressure_kpa(
    specific_humidity_kg_kg: pd.Series,
    surface_pressure_pa: pd.Series,
) -> pd.Series:
    humidity = specific_humidity_kg_kg.astype(float)
    pressure = surface_pressure_pa.astype(float)
    if humidity.lt(0.0).any() or humidity.ge(1.0).any():
        raise ValueError("specific humidity must be in [0, 1) kg/kg")
    if pressure.le(0.0).any():
        raise ValueError("surface pressure must be positive")
    return (humidity * pressure / (0.622 + 0.378 * humidity)) / 1000.0


def _aggregate_humidity_pressure(points: pd.DataFrame) -> pd.DataFrame:
    selected = points.loc[points["short_name"].isin(["SPFH", "PRES"])].copy()
    index_columns = ["site", "timezone", "cycle_init_utc", "lead_hour"]
    pivot = selected.pivot_table(
        index=index_columns,
        columns="short_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    if not {"SPFH", "PRES"}.issubset(pivot.columns):
        raise ValueError("SPFH and PRES are both required for vapor pressure")
    pivot["valid_time_utc"] = pd.to_datetime(
        pivot["cycle_init_utc"], utc=True
    ) + pd.to_timedelta(pivot["lead_hour"], unit="h")
    pivot["local_date"] = [
        timestamp.tz_convert(ZoneInfo(timezone_name)).tz_localize(None).normalize()
        for timestamp, timezone_name in zip(pivot["valid_time_utc"], pivot["timezone"])
    ]
    pivot["actual_vapor_pressure_kpa"] = specific_humidity_to_vapor_pressure_kpa(
        pivot["SPFH"], pivot["PRES"]
    )
    return pivot.groupby(
        ["site", "timezone", "cycle_init_utc", "local_date"],
        as_index=False,
    ).agg(
        specific_humidity_kg_kg=("SPFH", "mean"),
        surface_pressure_kpa=("PRES", lambda values: float(values.mean()) / 1000.0),
        actual_vapor_pressure_kpa=("actual_vapor_pressure_kpa", "mean"),
    )


def aggregate_member_weather(
    points: pd.DataFrame,
    *,
    member: str,
    product_manifest: pd.DataFrame,
) -> pd.DataFrame:
    points = points.copy()
    points["lead_hour"] = points["end_hour"].astype(int)
    daily = aggregate_gefs_point_records(points)
    if daily.empty:
        raise ValueError("full-weather aggregation produced no daily rows")
    humidity = _aggregate_humidity_pressure(points)
    daily = daily.merge(
        humidity,
        on=["site", "timezone", "cycle_init_utc", "local_date"],
        how="left",
        validate="one_to_one",
    )
    daily["gefs_member"] = member
    mean_temperature = (
        daily["temperature_min_c"].astype(float)
        + daily["temperature_max_c"].astype(float)
    ) / 2.0
    saturation = 0.6108 * np.exp(
        (17.27 * mean_temperature) / (mean_temperature + 237.3)
    )
    daily["vpd_kpa"] = (
        saturation - daily["actual_vapor_pressure_kpa"].astype(float)
    ).clip(lower=0.0)
    daily["solar_kj_m2_day"] = daily["shortwave_w_m2"].astype(float) * 86.4
    daily = daily.rename(
        columns={
            "site": "site_id",
            "timezone": "site_timezone",
            "cycle_init_utc": "forecast_init_utc",
            "precipitation_mm": "precipitation_mm_raw",
        }
    )
    source_keys = ";".join(sorted(product_manifest["source_key"].astype(str)))
    source_etags = ";".join(sorted(product_manifest["source_etag"].astype(str)))
    daily["source_product_keys"] = source_keys
    daily["source_product_etags"] = source_etags
    columns = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        "local_date",
        "lead_day",
        *CANONICAL_WEATHER_COLUMNS,
        "vpd_kpa",
        "specific_humidity_kg_kg",
        "surface_pressure_kpa",
        "shortwave_w_m2",
        "source_product_keys",
        "source_product_etags",
    ]
    return daily[columns].sort_values(["site_id", "local_date"]).reset_index(drop=True)


def validate_full_weather(
    frame: pd.DataFrame,
    *,
    expected_cycles: Sequence[str],
    expected_sites: Sequence[str],
    expected_members: Sequence[str],
) -> dict[str, object]:
    required = {
        "site_id",
        "decision_date",
        "gefs_member",
        "local_date",
        "lead_day",
        *CANONICAL_WEATHER_COLUMNS,
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing full-weather columns: {sorted(missing_columns)}")
    work = frame.copy()
    work["decision_date"] = pd.to_datetime(work["decision_date"]).dt.normalize()
    work["local_date"] = pd.to_datetime(work["local_date"]).dt.normalize()
    expected_cycle_values = {pd.Timestamp(item).normalize() for item in expected_cycles}
    actual_cycle_values = set(work["decision_date"])
    if actual_cycle_values != expected_cycle_values:
        raise ValueError(
            f"full-weather cycles={sorted(actual_cycle_values)}, "
            f"expected={sorted(expected_cycle_values)}"
        )
    if 2024 in set(work["decision_date"].dt.year):
        raise ValueError("2024 is forbidden in the 2015-2019 pilot extraction")
    if set(work["site_id"].astype(str)) != set(expected_sites):
        raise ValueError("full-weather site set does not match the contract")
    if set(work["gefs_member"].astype(str)) != set(expected_members):
        raise ValueError("full-weather member set does not match the contract")
    expected_rows = (
        len(expected_cycles) * len(expected_sites) * len(expected_members) * 7
    )
    if len(work) != expected_rows:
        raise ValueError(f"full-weather rows={len(work)}, expected={expected_rows}")
    key = ["decision_date", "site_id", "gefs_member", "local_date"]
    duplicate_count = int(work.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"full-weather duplicate key count={duplicate_count}")
    missing = work[list(CANONICAL_WEATHER_COLUMNS)].isna().sum()
    if int(missing.sum()):
        raise ValueError(f"missing canonical weather values: {missing.to_dict()}")
    numeric = work[list(CANONICAL_WEATHER_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("canonical weather contains non-finite values")
    if work["lead_day"].astype(int).min() != 1 or work["lead_day"].astype(int).max() != 7:
        raise ValueError("lead_day must span 1 through 7")
    group_keys = ["decision_date", "site_id", "gefs_member"]
    bad_leads = [
        keys
        for keys, group in work.groupby(group_keys, sort=False)
        if group["lead_day"].astype(int).tolist() != list(range(1, 8))
    ]
    if bad_leads:
        raise ValueError(f"incomplete lead-day groups: {bad_leads[:5]}")
    if (work["temperature_min_c"] > work["temperature_max_c"]).any():
        raise ValueError("temperature minimum exceeds maximum")
    nonnegative = (
        "precipitation_mm_raw",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "solar_kj_m2_day",
    )
    negatives = {column: int(work[column].lt(0.0).sum()) for column in nonnegative}
    if any(negatives.values()):
        raise ValueError(f"negative canonical weather values: {negatives}")
    return {
        "row_count": len(work),
        "expected_row_count": expected_rows,
        "cycle_count": work["decision_date"].nunique(),
        "site_count": work["site_id"].nunique(),
        "member_count": work["gefs_member"].nunique(),
        "duplicate_sample_key_count": duplicate_count,
        "canonical_missing_value_count": int(missing.sum()),
        "canonical_nonfinite_value_count": int((~np.isfinite(numeric)).sum()),
        "negative_value_count_by_column": negatives,
        "contains_2024": False,
        "status": "passed",
    }

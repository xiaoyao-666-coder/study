#!/usr/bin/env python3
"""Prepare and analyze GEFS ensemble-mean weather against gridMET."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from dataclasses import dataclass, replace
from math import exp, sqrt
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepWindow:
    start_hour: int
    end_hour: int
    kind: str


@dataclass(frozen=True)
class GribIndexRecord:
    message_number: int
    offset: int
    init_text: str
    short_name: str
    level: str
    step_text: str
    ensemble_text: str
    step: StepWindow
    range_end: int | None = None


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    short_names: tuple[str, ...]


REQUIRED_MESSAGES = (
    ("TMP", "2 m above ground"),
    ("DPT", "2 m above ground"),
    ("TMAX", "2 m above ground"),
    ("TMIN", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("APCP", "surface"),
    ("DSWRF", "surface"),
)

GEFS_BUCKET_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"


def gefs_members() -> tuple[str, ...]:
    """Return the formal GEFS control plus 30 perturbed member names."""

    return ("gec00", *(f"gep{number:02d}" for number in range(1, 31)))


def packing_resolution(
    *, binary_scale_factor: int, decimal_scale_factor: int
) -> float:
    return (2.0 ** int(binary_scale_factor)) * (
        10.0 ** (-int(decimal_scale_factor))
    )


def parse_step_window(text: str) -> StepWindow:
    interval = re.fullmatch(
        r"(\d+)-(\d+) hour (acc|ave|max|min) fcst", text.strip()
    )
    if interval:
        return StepWindow(
            start_hour=int(interval.group(1)),
            end_hour=int(interval.group(2)),
            kind=interval.group(3),
        )

    instant = re.fullmatch(r"(\d+) hour fcst", text.strip())
    if instant:
        hour = int(instant.group(1))
        return StepWindow(start_hour=hour, end_hour=hour, kind="instant")

    raise ValueError(f"unsupported GEFS step text: {text!r}")


def parse_gefs_index(text: str) -> list[GribIndexRecord]:
    records: list[GribIndexRecord] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 7:
            raise ValueError(f"invalid GEFS index line: {line!r}")
        message_number, offset = int(parts[0]), int(parts[1])
        init_text, short_name, level, step_text = parts[2:6]
        ensemble_text = ":".join(parts[6:])
        records.append(
            GribIndexRecord(
                message_number=message_number,
                offset=offset,
                init_text=init_text,
                short_name=short_name,
                level=level,
                step_text=step_text,
                ensemble_text=ensemble_text,
                step=parse_step_window(step_text),
            )
        )

    if not records:
        raise ValueError("GEFS index is empty")
    if any(right.offset <= left.offset for left, right in zip(records, records[1:])):
        raise ValueError("GEFS index offsets must be strictly increasing")

    bounded = [
        replace(record, range_end=records[index + 1].offset - 1)
        if index + 1 < len(records)
        else record
        for index, record in enumerate(records)
    ]
    return bounded


def select_gefs_messages(
    records: Sequence[GribIndexRecord],
    required_messages: Sequence[tuple[str, str]] = REQUIRED_MESSAGES,
) -> list[GribIndexRecord]:
    selected: list[GribIndexRecord] = []
    for short_name, level in required_messages:
        matches = [
            record
            for record in records
            if record.short_name == short_name and record.level == level
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {short_name}/{level} message, found {len(matches)}"
            )
        if matches[0].range_end is None:
            raise ValueError(f"cannot determine byte range for {short_name}/{level}")
        selected.append(matches[0])
    return selected


def merge_contiguous_ranges(
    records: Sequence[GribIndexRecord],
) -> list[ByteRange]:
    ranges: list[ByteRange] = []
    for record in sorted(records, key=lambda item: item.offset):
        if record.range_end is None:
            raise ValueError(f"missing byte-range end for {record.short_name}")
        if ranges and record.offset == ranges[-1].end + 1:
            previous = ranges[-1]
            ranges[-1] = ByteRange(
                start=previous.start,
                end=record.range_end,
                short_names=(*previous.short_names, record.short_name),
            )
        else:
            ranges.append(
                ByteRange(
                    start=record.offset,
                    end=record.range_end,
                    short_names=(record.short_name,),
                )
            )
    return ranges


def fetch_selected_byte_ranges(
    product_url: str,
    ranges: Sequence[ByteRange],
    *,
    fetcher: Callable[[str, int, int], bytes],
) -> bytes:
    chunks: list[bytes] = []
    for byte_range in ranges:
        payload = fetcher(product_url, byte_range.start, byte_range.end)
        expected_length = byte_range.end - byte_range.start + 1
        if len(payload) != expected_length:
            raise ValueError(
                f"range {byte_range.start}-{byte_range.end} returned "
                f"{len(payload)} bytes, expected {expected_length}"
            )
        chunks.append(payload)
    return b"".join(chunks)


def build_gefs_product_url(
    cycle_date: str,
    *,
    cycle_hour: int,
    lead_hour: int,
    product: str = "geavg",
    index: bool = False,
) -> str:
    supported_products = {"geavg", *gefs_members()}
    if product not in supported_products:
        raise ValueError(f"unsupported GEFS product: {product!r}")
    date_text = pd.Timestamp(cycle_date).strftime("%Y%m%d")
    cycle_text = f"{int(cycle_hour):02d}"
    lead_text = f"{int(lead_hour):03d}"
    suffix = ".idx" if index else ""
    return (
        f"{GEFS_BUCKET_BASE}/gefs.{date_text}/{cycle_text}/atmos/pgrb2sp25/"
        f"{product}.t{cycle_text}z.pgrb2s.0p25.f{lead_text}{suffix}"
    )


def decumulate_reset_intervals(
    frame: pd.DataFrame,
    *,
    kind: str,
    negative_tolerance: float = 1.0e-6,
) -> pd.DataFrame:
    if kind not in {"acc", "ave"}:
        raise ValueError("kind must be 'acc' or 'ave'")
    required = {"start_hour", "end_hour", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing interval columns: {sorted(missing)}")

    output: list[dict[str, object]] = []
    ordered = frame.sort_values(["start_hour", "end_hour"], kind="stable")
    for reset_start, group in ordered.groupby("start_hour", sort=True):
        previous_end = int(reset_start)
        previous_total = 0.0
        for row in group.itertuples(index=False):
            end_hour = int(row.end_hour)
            if end_hour <= previous_end:
                raise ValueError("interval endpoints must increase within each reset block")
            duration = end_hour - int(reset_start)
            cumulative_total = (
                float(row.value) if kind == "acc" else float(row.value) * duration
            )
            interval_total = cumulative_total - previous_total
            if interval_total < -float(negative_tolerance):
                raise ValueError(
                    f"negative interval reconstructed for {reset_start}-{end_hour}: "
                    f"{interval_total}"
                )
            interval_total = max(0.0, interval_total)
            interval_duration = end_hour - previous_end
            interval_value = (
                interval_total
                if kind == "acc"
                else interval_total / float(interval_duration)
            )
            result = row._asdict()
            result.update(
                {
                    "interval_start_hour": previous_end,
                    "interval_end_hour": end_hour,
                    "interval_value": interval_value,
                }
            )
            output.append(result)
            previous_end = end_hour
            previous_total = cumulative_total

    return pd.DataFrame(output)


def allocate_interval_to_local_days(
    *,
    start_utc: datetime,
    end_utc: datetime,
    value: float,
    kind: str,
    timezone_name: str,
) -> pd.DataFrame:
    if kind not in {"acc", "ave"}:
        raise ValueError("kind must be 'acc' or 'ave'")
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("interval endpoints must be timezone-aware")
    start = start_utc.astimezone(timezone.utc)
    end = end_utc.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("interval end must be after interval start")

    tz = ZoneInfo(timezone_name)
    total_hours = (end - start).total_seconds() / 3600.0
    rows: list[dict[str, object]] = []
    cursor = start
    while cursor < end:
        local_cursor = cursor.astimezone(tz)
        next_local_date = local_cursor.date() + timedelta(days=1)
        next_midnight_local = datetime.combine(next_local_date, time.min, tzinfo=tz)
        segment_end = min(end, next_midnight_local.astimezone(timezone.utc))
        overlap_hours = (segment_end - cursor).total_seconds() / 3600.0
        allocated = (
            float(value) * overlap_hours / total_hours
            if kind == "acc"
            else float(value)
        )
        rows.append(
            {
                "local_date": pd.Timestamp(local_cursor.date()),
                "overlap_hours": overlap_hours,
                "allocated_value": allocated,
                "weighted_value_hours": float(value) * overlap_hours,
            }
        )
        cursor = segment_end
    return pd.DataFrame(rows)


def vapor_pressure_deficit_kpa(temperature_k: float, dewpoint_k: float) -> float:
    temperature_c = float(temperature_k) - 273.15
    dewpoint_c = float(dewpoint_k) - 273.15

    def saturation_vapor_pressure(value_c: float) -> float:
        return 0.6108 * exp(17.27 * value_c / (value_c + 237.3))

    return max(
        0.0,
        saturation_vapor_pressure(temperature_c)
        - saturation_vapor_pressure(dewpoint_c),
    )


def _attribute_scalar(dataset: h5py.Dataset, name: str, default: float) -> float:
    if name not in dataset.attrs:
        return float(default)
    values = np.asarray(dataset.attrs[name]).reshape(-1)
    return float(values[0])


def _gridmet_data_variable(handle: h5py.File) -> str:
    coordinate_names = {"lon", "lat", "day", "crs"}
    names = [name for name in handle.keys() if name not in coordinate_names]
    if len(names) != 1:
        raise ValueError(f"expected one gridMET data variable, found {names}")
    return names[0]


def read_gridmet_variable_points(
    path: str | Path,
    *,
    sites: pd.DataFrame,
    dates: Sequence[str],
    output_variable: str,
) -> pd.DataFrame:
    required_site_columns = {"site", "latitude", "longitude"}
    missing = required_site_columns.difference(sites.columns)
    if missing:
        raise ValueError(f"missing site columns: {sorted(missing)}")

    requested_dates = [pd.Timestamp(value).normalize() for value in dates]
    rows: list[dict[str, object]] = []
    with h5py.File(Path(path), "r") as handle:
        longitudes = np.asarray(handle["lon"][:], dtype=float)
        latitudes = np.asarray(handle["lat"][:], dtype=float)
        day_values = np.asarray(handle["day"][:], dtype=float)
        units = handle["day"].attrs.get("units", b"")
        if isinstance(units, bytes):
            units = units.decode("ascii")
        match = re.match(r"days since (\d{4}-\d{2}-\d{2})", str(units))
        if not match:
            raise ValueError(f"unsupported gridMET day units: {units!r}")
        origin = pd.Timestamp(match.group(1))
        available_dates = origin + pd.to_timedelta(day_values, unit="D")
        date_index = {
            timestamp.normalize(): index
            for index, timestamp in enumerate(available_dates)
        }

        data = handle[_gridmet_data_variable(handle)]
        scale = _attribute_scalar(data, "scale_factor", 1.0)
        offset = _attribute_scalar(data, "add_offset", 0.0)
        fill_value = _attribute_scalar(data, "_FillValue", np.nan)

        for site in sites.itertuples(index=False):
            latitude_index = int(np.abs(latitudes - float(site.latitude)).argmin())
            longitude_index = int(np.abs(longitudes - float(site.longitude)).argmin())
            for requested_date in requested_dates:
                value = np.nan
                if requested_date in date_index:
                    raw = float(
                        data[
                            date_index[requested_date],
                            latitude_index,
                            longitude_index,
                        ]
                    )
                    if not np.isclose(raw, fill_value, equal_nan=True):
                        value = raw * scale + offset
                rows.append(
                    {
                        "site": str(site.site),
                        "local_date": requested_date,
                        "variable": output_variable,
                        "reference_value": value,
                        "grid_latitude": float(latitudes[latitude_index]),
                        "grid_longitude": float(longitudes[longitude_index]),
                    }
                )
    return pd.DataFrame(rows)


def required_valid_dates(
    decision_dates: Sequence[str], *, horizon_days: int = 7
) -> list[str]:
    values = {
        (pd.Timestamp(decision_date).normalize() + pd.Timedelta(days=offset)).strftime(
            "%Y-%m-%d"
        )
        for decision_date in decision_dates
        for offset in range(int(horizon_days))
    }
    return sorted(values)


def validate_reference_coverage(
    frame: pd.DataFrame,
    *,
    sites: Sequence[str],
    variables: Sequence[str],
    dates: Sequence[str],
) -> None:
    required = {
        (str(site), pd.Timestamp(day).normalize(), str(variable))
        for site in sites
        for day in dates
        for variable in variables
    }
    available = {
        (str(row.site), pd.Timestamp(row.local_date).normalize(), str(row.variable))
        for row in frame.itertuples(index=False)
        if pd.notna(row.reference_value)
    }
    missing = required.difference(available)
    if missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"missing reference coverage for {len(missing)} site/date/variable rows; "
            f"examples={examples}"
        )


def convert_gridmet_reference_units(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"variable", "reference_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing gridMET unit columns: {sorted(missing)}")
    work = frame.copy()
    temperature = work["variable"].isin(
        ["temperature_min_c", "temperature_max_c"]
    )
    work.loc[temperature, "reference_value"] = (
        work.loc[temperature, "reference_value"] - 273.15
    )
    return work


FORECAST_DAILY_VARIABLES = (
    "precipitation_mm",
    "temperature_min_c",
    "temperature_max_c",
    "shortwave_w_m2",
    "wind_speed_m_s",
    "vpd_kpa",
)


def forecast_daily_to_long(
    frame: pd.DataFrame,
    *,
    variables: Sequence[str] = FORECAST_DAILY_VARIABLES,
) -> pd.DataFrame:
    selected_variables = tuple(variables)
    unsupported = set(selected_variables).difference(FORECAST_DAILY_VARIABLES)
    if unsupported:
        raise ValueError(f"unsupported daily forecast variables: {sorted(unsupported)}")
    if not selected_variables:
        raise ValueError("at least one daily forecast variable is required")
    missing = set(selected_variables).difference(frame.columns)
    if missing:
        raise ValueError(f"missing daily forecast variables: {sorted(missing)}")
    id_columns = [
        column
        for column in (
            "site",
            "timezone",
            "cycle_init_utc",
            "decision_date",
            "local_date",
            "lead_day",
            "gefs_product",
            "gefs_member",
        )
        if column in frame.columns
    ]
    return (
        frame.melt(
            id_vars=id_columns,
            value_vars=list(selected_variables),
            var_name="variable",
            value_name="forecast_value",
        )
        .dropna(subset=["forecast_value"])
        .reset_index(drop=True)
    )


def pair_forecast_and_reference(
    forecast: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    keys = ["site", "local_date", "variable"]
    if reference.duplicated(keys).any():
        duplicates = reference.loc[reference.duplicated(keys, keep=False), keys]
        raise ValueError(
            "duplicate reference values: "
            + duplicates.head(5).to_dict(orient="records").__repr__()
        )
    paired = forecast.merge(
        reference[keys + ["reference_value"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if paired["reference_value"].isna().any():
        examples = paired.loc[
            paired["reference_value"].isna(), keys
        ].head(5).to_dict(orient="records")
        raise ValueError(f"missing reference values for forecast rows: {examples}")
    paired["error"] = paired["forecast_value"] - paired["reference_value"]
    paired["absolute_error"] = paired["error"].abs()
    return paired


def decode_gefs_minigrib_points(
    path: str | Path,
    *,
    selected_records: Sequence[GribIndexRecord],
    sites: pd.DataFrame,
    cycle_init_utc: str | pd.Timestamp,
    lead_hour: int,
) -> pd.DataFrame:
    required_site_columns = {"site", "latitude", "longitude", "timezone"}
    missing = required_site_columns.difference(sites.columns)
    if missing:
        raise ValueError(f"missing GEFS site columns: {sorted(missing)}")
    try:
        import eccodes
    except ImportError as exc:
        raise RuntimeError(
            "ecCodes is required to decode GEFS GRIB2 files; install requirements_gefs_gridmet_bias_validation_v1.txt"
        ) from exc

    cycle = pd.Timestamp(cycle_init_utc)
    cycle = cycle.tz_localize("UTC") if cycle.tzinfo is None else cycle.tz_convert("UTC")
    ordered_records = sorted(selected_records, key=lambda item: item.offset)
    rows: list[dict[str, object]] = []
    message_index = 0
    with Path(path).open("rb") as handle:
        while True:
            message = eccodes.codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                if message_index >= len(ordered_records):
                    raise ValueError("mini-GRIB contains more messages than selected index records")
                record = ordered_records[message_index]
                decoded_short_name = str(eccodes.codes_get(message, "shortName"))
                binary_scale = int(eccodes.codes_get(message, "binaryScaleFactor"))
                decimal_scale = int(eccodes.codes_get(message, "decimalScaleFactor"))
                resolution = packing_resolution(
                    binary_scale_factor=binary_scale,
                    decimal_scale_factor=decimal_scale,
                )
                for site in sites.itertuples(index=False):
                    nearest = eccodes.codes_grib_find_nearest(
                        message,
                        float(site.latitude),
                        float(site.longitude),
                        npoints=1,
                    )[0]
                    grid_longitude = float(nearest["lon"])
                    if grid_longitude > 180.0:
                        grid_longitude -= 360.0
                    rows.append(
                        {
                            "site": str(site.site),
                            "latitude": float(site.latitude),
                            "longitude": float(site.longitude),
                            "timezone": str(site.timezone),
                            "cycle_init_utc": cycle,
                            "lead_hour": int(lead_hour),
                            "short_name": record.short_name,
                            "decoded_short_name": decoded_short_name,
                            "value": float(nearest["value"]),
                            "grid_latitude": float(nearest["lat"]),
                            "grid_longitude": grid_longitude,
                            "grid_distance_km": float(nearest["distance"]),
                            "start_hour": record.step.start_hour,
                            "end_hour": record.step.end_hour,
                            "kind": record.step.kind,
                            "packing_resolution": resolution,
                            "binary_scale_factor": binary_scale,
                            "decimal_scale_factor": decimal_scale,
                        }
                    )
                message_index += 1
            finally:
                eccodes.codes_release(message)
    if message_index != len(ordered_records):
        raise ValueError(
            f"mini-GRIB decoded {message_index} messages, expected {len(ordered_records)}"
        )
    return pd.DataFrame(rows)


def compute_bias_metrics(
    frame: pd.DataFrame, *, group_columns: Sequence[str]
) -> pd.DataFrame:
    required = {"forecast_value", "reference_value", *group_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing metric columns: {sorted(missing)}")
    work = frame.dropna(subset=["forecast_value", "reference_value"]).copy()
    work["error"] = work["forecast_value"] - work["reference_value"]
    work["absolute_error"] = work["error"].abs()
    work["squared_error"] = work["error"] ** 2

    rows: list[dict[str, object]] = []
    grouped = work.groupby(list(group_columns), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        correlation_defined = (
            len(group) >= 2
            and group["forecast_value"].nunique(dropna=True) >= 2
            and group["reference_value"].nunique(dropna=True) >= 2
        )
        correlation = (
            float(group["forecast_value"].corr(group["reference_value"]))
            if correlation_defined
            else np.nan
        )
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n": int(len(group)),
                "bias": float(group["error"].mean()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": sqrt(float(group["squared_error"].mean())),
                "correlation": correlation,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_reference_condition(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"variable", "reference_value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing condition columns: {sorted(missing)}")
    work = frame.copy()
    work["reference_condition"] = "unclassified"

    precipitation = work["variable"].eq("precipitation_mm")
    values = work.loc[precipitation, "reference_value"]
    work.loc[precipitation & values.lt(0.1), "reference_condition"] = "dry"
    work.loc[
        precipitation & values.ge(0.1) & values.lt(5.0), "reference_condition"
    ] = "light"
    work.loc[
        precipitation & values.ge(5.0) & values.lt(20.0), "reference_condition"
    ] = "moderate"
    work.loc[precipitation & values.ge(20.0), "reference_condition"] = "heavy"

    for variable, group in work.loc[~precipitation].groupby("variable"):
        valid = group["reference_value"].dropna()
        if valid.empty:
            continue
        lower, upper = valid.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
        mask = work["variable"].eq(variable)
        work.loc[mask & work["reference_value"].le(lower), "reference_condition"] = "low"
        work.loc[
            mask
            & work["reference_value"].gt(lower)
            & work["reference_value"].lt(upper),
            "reference_condition",
        ] = "middle"
        work.loc[mask & work["reference_value"].ge(upper), "reference_condition"] = "high"
    return work


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else np.nan


def compute_precipitation_event_metrics(
    frame: pd.DataFrame,
    *,
    thresholds_mm: Sequence[float] = (1.0, 10.0, 20.0),
    group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    required = {"forecast_value", "reference_value", *group_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing precipitation columns: {sorted(missing)}")
    work = frame.dropna(subset=["forecast_value", "reference_value"]).copy()
    groupers = list(group_columns)
    grouped = [((), work)] if not groupers else work.groupby(groupers, dropna=False)
    rows: list[dict[str, object]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        for threshold in thresholds_mm:
            forecast_event = group["forecast_value"].ge(float(threshold))
            reference_event = group["reference_value"].ge(float(threshold))
            hits = int((forecast_event & reference_event).sum())
            misses = int((~forecast_event & reference_event).sum())
            false_alarms = int((forecast_event & ~reference_event).sum())
            correct_negatives = int((~forecast_event & ~reference_event).sum())
            row = dict(zip(groupers, keys))
            row.update(
                {
                    "threshold_mm": float(threshold),
                    "n": int(len(group)),
                    "hits": hits,
                    "misses": misses,
                    "false_alarms": false_alarms,
                    "correct_negatives": correct_negatives,
                    "probability_of_detection": _safe_ratio(hits, hits + misses),
                    "false_alarm_ratio": _safe_ratio(
                        false_alarms, hits + false_alarms
                    ),
                    "critical_success_index": _safe_ratio(
                        hits, hits + misses + false_alarms
                    ),
                    "frequency_bias": _safe_ratio(
                        hits + false_alarms, hits + misses
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_instantaneous_records(frame: pd.DataFrame) -> pd.DataFrame:
    instant = frame.loc[frame["kind"].eq("instant")].copy()
    if instant.empty:
        return pd.DataFrame()
    index_columns = [
        "site",
        "timezone",
        "cycle_init_utc",
        "lead_hour",
    ]
    pivot = instant.pivot_table(
        index=index_columns,
        columns="short_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot["valid_time_utc"] = pd.to_datetime(
        pivot["cycle_init_utc"], utc=True
    ) + pd.to_timedelta(pivot["lead_hour"], unit="h")
    pivot["local_date"] = [
        timestamp.tz_convert(ZoneInfo(timezone_name)).tz_localize(None).normalize()
        for timestamp, timezone_name in zip(pivot["valid_time_utc"], pivot["timezone"])
    ]
    if "TMP" in pivot:
        pivot["temperature_c"] = pivot["TMP"] - 273.15
    if {"TMP", "DPT"}.issubset(pivot.columns):
        pivot["vpd_kpa"] = [
            vapor_pressure_deficit_kpa(temperature, dewpoint)
            for temperature, dewpoint in zip(pivot["TMP"], pivot["DPT"])
        ]
    if {"UGRD", "VGRD"}.issubset(pivot.columns):
        pivot["wind_speed_m_s"] = np.hypot(pivot["UGRD"], pivot["VGRD"])

    aggregations: dict[str, tuple[str, str]] = {}
    if "temperature_c" in pivot:
        aggregations["temperature_min_sampled_c"] = ("temperature_c", "min")
        aggregations["temperature_max_sampled_c"] = ("temperature_c", "max")
    if "vpd_kpa" in pivot:
        aggregations["vpd_kpa"] = ("vpd_kpa", "mean")
    if "wind_speed_m_s" in pivot:
        aggregations["wind_speed_m_s"] = ("wind_speed_m_s", "mean")
    return (
        pivot.groupby(
            ["site", "timezone", "cycle_init_utc", "local_date"],
            as_index=False,
        )
        .agg(**aggregations)
    )


def _aggregate_extreme_records(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["short_name"].isin(["TMAX", "TMIN"])].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["midpoint_hour"] = (
        selected["start_hour"].astype(float) + selected["end_hour"].astype(float)
    ) / 2.0
    selected["midpoint_utc"] = pd.to_datetime(
        selected["cycle_init_utc"], utc=True
    ) + pd.to_timedelta(selected["midpoint_hour"], unit="h")
    selected["local_date"] = [
        timestamp.tz_convert(ZoneInfo(timezone_name)).tz_localize(None).normalize()
        for timestamp, timezone_name in zip(
            selected["midpoint_utc"], selected["timezone"]
        )
    ]
    selected["value_c"] = selected["value"].astype(float) - 273.15
    keys = ["site", "timezone", "cycle_init_utc", "local_date"]
    maximum = (
        selected.loc[selected["short_name"].eq("TMAX")]
        .groupby(keys, as_index=False)["value_c"]
        .max()
        .rename(columns={"value_c": "temperature_max_c"})
    )
    minimum = (
        selected.loc[selected["short_name"].eq("TMIN")]
        .groupby(keys, as_index=False)["value_c"]
        .min()
        .rename(columns={"value_c": "temperature_min_c"})
    )
    if maximum.empty:
        return minimum
    if minimum.empty:
        return maximum
    return maximum.merge(minimum, on=keys, how="outer", validate="one_to_one")


def _aggregate_interval_records(
    frame: pd.DataFrame, *, short_name: str, kind: str, output_column: str
) -> pd.DataFrame:
    selected = frame.loc[frame["short_name"].eq(short_name)].copy()
    if selected.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["site", "timezone", "cycle_init_utc"]
    for keys, group in selected.groupby(group_columns, sort=False):
        site, timezone_name, cycle_init = keys
        tolerance = 1.0e-6
        if "packing_resolution" in group.columns:
            resolution = float(group["packing_resolution"].max())
            if kind == "acc":
                tolerance = 2.0 * resolution
            else:
                longest_duration = float(
                    (group["end_hour"] - group["start_hour"]).max()
                )
                tolerance = 2.0 * resolution * longest_duration
        intervals = decumulate_reset_intervals(
            group[["start_hour", "end_hour", "value"]],
            kind=kind,
            negative_tolerance=tolerance,
        )
        cycle_timestamp = pd.Timestamp(cycle_init)
        if cycle_timestamp.tzinfo is None:
            cycle_timestamp = cycle_timestamp.tz_localize("UTC")
        else:
            cycle_timestamp = cycle_timestamp.tz_convert("UTC")
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
                kind=kind,
                timezone_name=str(timezone_name),
            )
            for part in allocated.itertuples(index=False):
                rows.append(
                    {
                        "site": site,
                        "timezone": timezone_name,
                        "cycle_init_utc": cycle_timestamp,
                        "local_date": part.local_date,
                        "overlap_hours": part.overlap_hours,
                        "allocated_value": part.allocated_value,
                        "weighted_value_hours": part.weighted_value_hours,
                    }
                )
    allocated_frame = pd.DataFrame(rows)
    keys = ["site", "timezone", "cycle_init_utc", "local_date"]
    if kind == "acc":
        return (
            allocated_frame.groupby(keys, as_index=False)["allocated_value"]
            .sum()
            .rename(columns={"allocated_value": output_column})
        )
    totals = allocated_frame.groupby(keys, as_index=False).agg(
        weighted_value_hours=("weighted_value_hours", "sum"),
        overlap_hours=("overlap_hours", "sum"),
    )
    totals[output_column] = totals["weighted_value_hours"] / totals["overlap_hours"]
    return totals[keys + [output_column]]


def aggregate_gefs_point_records(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "site",
        "timezone",
        "cycle_init_utc",
        "lead_hour",
        "short_name",
        "value",
        "start_hour",
        "end_hour",
        "kind",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing GEFS point columns: {sorted(missing)}")
    components = [
        _aggregate_instantaneous_records(frame),
        _aggregate_extreme_records(frame),
        _aggregate_interval_records(
            frame,
            short_name="APCP",
            kind="acc",
            output_column="precipitation_mm",
        ),
        _aggregate_interval_records(
            frame,
            short_name="DSWRF",
            kind="ave",
            output_column="shortwave_w_m2",
        ),
    ]
    components = [component for component in components if not component.empty]
    if not components:
        return pd.DataFrame()
    keys = ["site", "timezone", "cycle_init_utc", "local_date"]
    daily = components[0]
    for component in components[1:]:
        daily = daily.merge(component, on=keys, how="outer", validate="one_to_one")
    if "temperature_min_c" not in daily:
        daily["temperature_min_c"] = daily.get("temperature_min_sampled_c")
    elif "temperature_min_sampled_c" in daily:
        daily["temperature_min_c"] = daily["temperature_min_c"].fillna(
            daily["temperature_min_sampled_c"]
        )
    if "temperature_max_c" not in daily:
        daily["temperature_max_c"] = daily.get("temperature_max_sampled_c")
    elif "temperature_max_sampled_c" in daily:
        daily["temperature_max_c"] = daily["temperature_max_c"].fillna(
            daily["temperature_max_sampled_c"]
        )
    daily["decision_date"] = pd.to_datetime(
        daily["cycle_init_utc"], utc=True
    ).dt.tz_localize(None).dt.normalize()
    daily["lead_day"] = (
        daily["local_date"] - daily["decision_date"]
    ).dt.days + 1
    return daily.loc[daily["lead_day"].between(1, 7)].sort_values(
        ["site", "cycle_init_utc", "local_date"]
    ).reset_index(drop=True)

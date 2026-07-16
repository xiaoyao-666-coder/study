#!/usr/bin/env python3
"""Run the GEFS ensemble-mean versus gridMET weather bias diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_gridmet_bias import (
    FORECAST_DAILY_VARIABLES,
    add_reference_condition,
    aggregate_gefs_point_records,
    build_gefs_product_url,
    compute_bias_metrics,
    compute_precipitation_event_metrics,
    convert_gridmet_reference_units,
    decode_gefs_minigrib_points,
    fetch_selected_byte_ranges,
    forecast_daily_to_long,
    merge_contiguous_ranges,
    pair_forecast_and_reference,
    parse_gefs_index,
    read_gridmet_variable_points,
    required_valid_dates,
    select_gefs_messages,
    validate_reference_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_GRIDMET_DIR = PROJECT_ROOT / "model3_opt_sto_upload" / "data" / "gridmet"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "site_general_surrogate_eval" / "gefs_gridmet_bias_validation_v1"
)
DEFAULT_DECISION_DATES = (
    "2024-07-16",
    "2024-07-20",
    "2024-07-24",
    "2024-07-28",
    "2024-08-01",
    "2024-08-05",
    "2024-08-09",
    "2024-08-13",
    "2024-08-17",
    "2024-08-21",
)
DEFAULT_SITES = (
    ("P1", "N1", 42.015928, -98.224144, "America/Chicago"),
    ("P2", "N2", 40.595000, -88.415000, "America/Chicago"),
    ("P3", "N3", 46.321000, -96.877000, "America/Chicago"),
    ("P4", "N4", 42.681600, -94.668600, "America/Chicago"),
    ("P15", "coord_12", 41.735000, -112.265000, "America/Denver"),
)
GRIDMET_SPECS = (
    ("pr_2024.nc", "precipitation_mm"),
    ("tmmn_2024.nc", "temperature_min_c"),
    ("tmmx_2024.nc", "temperature_max_c"),
    ("srad_2024.nc", "shortwave_w_m2"),
    ("vs_2024.nc", "wind_speed_m_s"),
    ("vpd_2024.nc", "vpd_kpa"),
)
GRIDMET_BASE_URL = "https://www.northwestknowledge.net/metdata/data"


def site_frame(site_names: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        DEFAULT_SITES,
        columns=["site", "code_site", "latitude", "longitude", "timezone"],
    )
    if site_names:
        unknown = sorted(set(site_names).difference(frame["site"]))
        if unknown:
            raise ValueError(f"unknown sites: {unknown}")
        frame = frame.loc[frame["site"].isin(site_names)].copy()
    return frame.reset_index(drop=True)


def _request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int,
    retries: int,
) -> bytes:
    request_headers = {"User-Agent": "s2s-rtist-gefs-bias-validation/1.0"}
    request_headers.update(headers or {})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_error


def _range_fetcher(timeout: int, retries: int) -> Callable[[str, int, int], bytes]:
    def fetch(url: str, start: int, end: int) -> bytes:
        return _request_bytes(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=timeout,
            retries=retries,
        )

    return fetch


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, path: Path, *, timeout: int, retries: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "s2s-rtist-gefs-bias-validation/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to download {url} after {retries} attempts") from last_error


def _gridmet_last_date(path: Path) -> pd.Timestamp:
    with h5py.File(path, "r") as handle:
        units = handle["day"].attrs.get("units", b"")
        if isinstance(units, bytes):
            units = units.decode("ascii")
        origin_text = str(units).split("days since ", 1)[-1].split()[0]
        origin = pd.Timestamp(origin_text)
        return (origin + pd.to_timedelta(float(handle["day"][-1]), unit="D")).normalize()


def ensure_gridmet_reference(
    source_dir: Path,
    *,
    run_dir: Path,
    required_last_date: pd.Timestamp,
    download_complete: bool,
    timeout: int,
    retries: int,
    workers: int,
) -> Path:
    source_paths = [source_dir / filename for filename, _ in GRIDMET_SPECS]
    complete = all(path.exists() for path in source_paths)
    if complete:
        complete = all(_gridmet_last_date(path) >= required_last_date for path in source_paths)
    if complete:
        return source_dir
    if not download_complete:
        coverage = {
            path.name: _gridmet_last_date(path).strftime("%Y-%m-%d")
            if path.exists()
            else "missing"
            for path in source_paths
        }
        raise ValueError(
            f"gridMET does not cover required date {required_last_date.date()}: {coverage}; "
            "rerun with --download-gridmet"
        )

    target_dir = run_dir / "gridmet_complete_2024"
    target_dir.mkdir(parents=True, exist_ok=True)

    def download(filename: str) -> Path:
        target = target_dir / filename
        if target.exists() and _gridmet_last_date(target) >= required_last_date:
            return target
        print(f"[gridmet] downloading {filename}", flush=True)
        _download_file(
            f"{GRIDMET_BASE_URL}/{filename}",
            target,
            timeout=timeout,
            retries=retries,
        )
        return target

    with ThreadPoolExecutor(max_workers=min(workers, len(GRIDMET_SPECS))) as executor:
        futures = [executor.submit(download, filename) for filename, _ in GRIDMET_SPECS]
        for future in as_completed(futures):
            path = future.result()
            print(
                f"[gridmet] ready {path.name} last_date={_gridmet_last_date(path).date()}",
                flush=True,
            )
    return target_dir


def build_gridmet_reference(
    gridmet_dir: Path, *, sites: pd.DataFrame, dates: Sequence[str]
) -> pd.DataFrame:
    parts = [
        read_gridmet_variable_points(
            gridmet_dir / filename,
            sites=sites,
            dates=dates,
            output_variable=variable,
        )
        for filename, variable in GRIDMET_SPECS
    ]
    reference = convert_gridmet_reference_units(pd.concat(parts, ignore_index=True))
    validate_reference_coverage(
        reference,
        sites=sites["site"].tolist(),
        variables=list(FORECAST_DAILY_VARIABLES),
        dates=dates,
    )
    return reference


def _download_lead_minigrib(
    *,
    cycle_date: str,
    lead_hour: int,
    cache_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[Path, list, dict[str, object]]:
    product_url = build_gefs_product_url(
        cycle_date, cycle_hour=0, lead_hour=lead_hour
    )
    index_url = product_url + ".idx"
    stem = f"geavg_{pd.Timestamp(cycle_date).strftime('%Y%m%d')}_f{lead_hour:03d}"
    index_path = cache_dir / "indices" / f"{stem}.idx"
    grib_path = cache_dir / "minigrib" / f"{stem}.grib2"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    grib_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        index_text = index_path.read_text(encoding="utf-8")
    else:
        index_text = _request_bytes(
            index_url, timeout=timeout, retries=retries
        ).decode("utf-8")
        index_path.write_text(index_text, encoding="utf-8")
    selected = select_gefs_messages(parse_gefs_index(index_text))
    ranges = merge_contiguous_ranges(selected)
    if not grib_path.exists():
        payload = fetch_selected_byte_ranges(
            product_url,
            ranges,
            fetcher=_range_fetcher(timeout, retries),
        )
        grib_path.write_bytes(payload)
    manifest = {
        "cycle_date": cycle_date,
        "cycle_hour_utc": 0,
        "lead_hour": lead_hour,
        "product_url": product_url,
        "index_url": index_url,
        "selected_variables": ",".join(record.short_name for record in selected),
        "range_count": len(ranges),
        "downloaded_bytes": grib_path.stat().st_size,
        "sha256": _sha256_file(grib_path),
    }
    return grib_path, selected, manifest


def build_gefs_point_records(
    *,
    decision_dates: Sequence[str],
    sites: pd.DataFrame,
    cache_dir: Path,
    timeout: int,
    retries: int,
    workers: int,
    keep_grib: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_dir = cache_dir / "point_records"
    point_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, int]] = []
    point_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    for cycle_date in decision_dates:
        for lead_hour in range(3, 181, 3):
            point_path = point_dir / (
                f"geavg_{pd.Timestamp(cycle_date).strftime('%Y%m%d')}_f{lead_hour:03d}.csv"
            )
            if point_path.exists():
                cached = pd.read_csv(point_path, parse_dates=["cycle_init_utc"])
                point_frames.append(cached)
                manifest_rows.append(
                    {
                        "cycle_date": cycle_date,
                        "cycle_hour_utc": 0,
                        "lead_hour": lead_hour,
                        "status": "cached_point_records",
                        "point_rows": len(cached),
                    }
                )
            else:
                tasks.append((cycle_date, lead_hour))

    def download(task: tuple[str, int]):
        cycle_date, lead_hour = task
        return task, _download_lead_minigrib(
            cycle_date=cycle_date,
            lead_hour=lead_hour,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            (cycle_date, lead_hour), (grib_path, selected, manifest) = future.result()
            cycle_init = pd.Timestamp(f"{cycle_date}T00:00:00Z")
            points = decode_gefs_minigrib_points(
                grib_path,
                selected_records=selected,
                sites=sites,
                cycle_init_utc=cycle_init,
                lead_hour=lead_hour,
            )
            point_path = point_dir / (
                f"geavg_{cycle_init.strftime('%Y%m%d')}_f{lead_hour:03d}.csv"
            )
            points.to_csv(point_path, index=False)
            point_frames.append(points)
            manifest.update({"status": "downloaded", "point_rows": len(points)})
            manifest_rows.append(manifest)
            if not keep_grib:
                grib_path.unlink(missing_ok=True)
            print(
                f"[gefs] {cycle_date} f{lead_hour:03d} ready "
                f"({completed}/{len(tasks)} downloads)",
                flush=True,
            )
            pd.DataFrame(manifest_rows).sort_values(
                ["cycle_date", "lead_hour"]
            ).to_csv(cache_dir / "gefs_download_manifest_partial.csv", index=False)

    if not point_frames:
        raise ValueError("no GEFS point records were produced")
    points = pd.concat(point_frames, ignore_index=True)
    expected_rows = len(decision_dates) * 60 * len(sites) * 8
    if len(points) != expected_rows:
        raise ValueError(
            f"GEFS point row count is {len(points)}, expected {expected_rows}"
        )
    manifest_frame = pd.DataFrame(manifest_rows).sort_values(
        ["cycle_date", "lead_hour"]
    ).reset_index(drop=True)
    return points, manifest_frame


def _validate_daily_forecast(
    daily: pd.DataFrame, *, decision_dates: Sequence[str], sites: pd.DataFrame
) -> None:
    expected_rows = len(decision_dates) * len(sites) * 7
    if len(daily) != expected_rows:
        raise ValueError(f"daily GEFS rows={len(daily)}, expected={expected_rows}")
    if daily.duplicated(["site", "decision_date", "local_date"]).any():
        raise ValueError("duplicate daily GEFS site/cycle/date rows")
    missing = daily[list(FORECAST_DAILY_VARIABLES)].isna().sum()
    missing = missing.loc[missing.gt(0)]
    if not missing.empty:
        raise ValueError(f"missing daily GEFS values: {missing.to_dict()}")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    selected = frame[list(columns)].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("nan" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    path: Path,
    *,
    paired: pd.DataFrame,
    overall: pd.DataFrame,
    by_condition: pd.DataFrame,
    event_metrics: pd.DataFrame,
    decision_dates: Sequence[str],
    sites: pd.DataFrame,
) -> None:
    lines = [
        "# GEFS Ensemble-Mean vs gridMET Bias Validation",
        "",
        "## Scope",
        "",
        f"- GEFS cycles: `{len(decision_dates)}` same-date 00 UTC cycles",
        f"- Sites: `{len(sites)}` ({', '.join(sites['site'])})",
        "- Horizon: local decision day D through D+6",
        f"- Paired site/cycle/day/variable rows: `{len(paired)}`",
        "- Forecast product: official GEFS `geavg` ensemble mean",
        "- Reference product: gridMET 2024 gridded weather",
        "",
        "## Overall Metrics",
        "",
        *_markdown_table(
            overall,
            ["variable", "n", "bias", "mae", "rmse", "correlation"],
        ),
        "",
        "## Conditional Metrics",
        "",
        *_markdown_table(
            by_condition,
            [
                "variable",
                "reference_condition",
                "n",
                "bias",
                "mae",
                "rmse",
            ],
        ),
        "",
        "## Precipitation Event Metrics",
        "",
        *_markdown_table(
            event_metrics,
            [
                "threshold_mm",
                "n",
                "hits",
                "misses",
                "false_alarms",
                "probability_of_detection",
                "false_alarm_ratio",
                "critical_success_index",
                "frequency_bias",
            ],
        ),
        "",
        "## Interpretation Limits",
        "",
        "- gridMET is a gridded reference dataset, not station-observed truth.",
        "- GEFS and gridMET have different grid resolutions; representativeness error is part of the reported difference.",
        "- TMAX/TMIN intervals are assigned by interval midpoint when an interval overlaps local midnight.",
        "- Precipitation and shortwave intervals crossing local midnight are allocated uniformly by overlap duration.",
        "- VPD is derived from ensemble-mean temperature and dew point; it is not the mean of member-level VPD.",
        "- This diagnostic evaluates `geavg`; it does not evaluate ensemble spread or individual-member calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="gefs_gridmet_bias_10cycle_5site_20260715_v1"
    )
    parser.add_argument("--decision-dates", nargs="+", default=list(DEFAULT_DECISION_DATES))
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--gridmet-dir", type=Path, default=DEFAULT_GRIDMET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download-gridmet", action="store_true")
    parser.add_argument("--keep-grib", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in args.decision_dates]
    sites = site_frame(args.sites)
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sites.to_csv(run_dir / "sites.csv", index=False)
    pd.DataFrame({"decision_date": dates}).to_csv(
        run_dir / "decision_dates.csv", index=False
    )

    valid_dates = required_valid_dates(dates, horizon_days=7)
    reference_dir = ensure_gridmet_reference(
        args.gridmet_dir,
        run_dir=run_dir,
        required_last_date=pd.Timestamp(valid_dates[-1]),
        download_complete=args.download_gridmet,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
    )
    print(f"[reference] using {reference_dir}", flush=True)
    reference = build_gridmet_reference(
        reference_dir, sites=sites, dates=valid_dates
    )
    reference.to_csv(run_dir / "gridmet_reference_daily_long.csv", index=False)

    points, manifest = build_gefs_point_records(
        decision_dates=dates,
        sites=sites,
        cache_dir=cache_dir,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
        keep_grib=args.keep_grib,
    )
    points.to_csv(run_dir / "gefs_point_records.csv", index=False)
    manifest.to_csv(run_dir / "gefs_download_manifest.csv", index=False)

    daily = aggregate_gefs_point_records(points)
    _validate_daily_forecast(daily, decision_dates=dates, sites=sites)
    daily.to_csv(run_dir / "gefs_daily_weather.csv", index=False)
    forecast_long = forecast_daily_to_long(daily)
    paired = pair_forecast_and_reference(forecast_long, reference)
    paired = add_reference_condition(paired)
    paired.to_csv(run_dir / "gefs_gridmet_paired_daily.csv", index=False)

    overall = compute_bias_metrics(paired, group_columns=["variable"])
    by_lead = compute_bias_metrics(paired, group_columns=["variable", "lead_day"])
    by_site = compute_bias_metrics(paired, group_columns=["variable", "site"])
    by_condition = compute_bias_metrics(
        paired, group_columns=["variable", "reference_condition"]
    )
    precipitation = paired.loc[paired["variable"].eq("precipitation_mm")]
    event_metrics = compute_precipitation_event_metrics(precipitation)
    event_by_lead = compute_precipitation_event_metrics(
        precipitation, group_columns=["lead_day"]
    )
    overall.to_csv(run_dir / "bias_metrics_overall.csv", index=False)
    by_lead.to_csv(run_dir / "bias_metrics_by_lead_day.csv", index=False)
    by_site.to_csv(run_dir / "bias_metrics_by_site.csv", index=False)
    by_condition.to_csv(run_dir / "bias_metrics_by_reference_condition.csv", index=False)
    event_metrics.to_csv(run_dir / "precipitation_event_metrics.csv", index=False)
    event_by_lead.to_csv(
        run_dir / "precipitation_event_metrics_by_lead_day.csv", index=False
    )
    write_report(
        run_dir / "gefs_gridmet_bias_validation_v1.md",
        paired=paired,
        overall=overall,
        by_condition=by_condition,
        event_metrics=event_metrics,
        decision_dates=dates,
        sites=sites,
    )
    metadata = {
        "run_id": args.run_id,
        "decision_dates": dates,
        "sites": sites["site"].tolist(),
        "forecast_product": "GEFS geavg pgrb2s 0p25",
        "cycle_hour_utc": 0,
        "required_leads": "f003-f180 at 3-hour intervals",
        "reference_product": "gridMET 2024",
        "reference_directory": str(reference_dir.resolve()),
        "paired_rows": len(paired),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"run_dir: {run_dir}", flush=True)
    print(f"report: {run_dir / 'gefs_gridmet_bias_validation_v1.md'}", flush=True)


if __name__ == "__main__":
    main()

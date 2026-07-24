#!/usr/bin/env python3
"""Download an expanded GEFS reforecast period and pair fixed GHCN stations."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    aggregate_reforecast_member_daily_utc,
    cycle_valid_dates,
    download_reforecast_member_points,
    reforecast_site_frame,
    validate_member_daily_precipitation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_IDS = ("P1", "P2", "P3", "P4", "P15")
MONTH_DAYS = ("06-01", "06-15", "07-01", "07-15", "08-01", "08-15")
DEFAULT_REFERENCE_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "selected_station_reference"
    / "ghcnd_selected_station_daily_precipitation_2000_2019_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_reforecast_2000_2002_smoke_v1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cycle_dates(first_year: int, last_year: int) -> tuple[str, ...]:
    if int(first_year) > int(last_year):
        raise ValueError("first year must not exceed last year")
    return tuple(
        f"{year}-{month_day}"
        for year in range(int(first_year), int(last_year) + 1)
        for month_day in MONTH_DAYS
    )


def append_log(path: Path | None, message: str) -> None:
    print(message, flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def extract_downloads(
    *,
    dates: tuple[str, ...],
    output_dir: Path,
    workers: int,
    timeout: int,
    retries: int,
    log_file: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = reforecast_site_frame(SITE_IDS)
    tasks = [(date, member) for date in dates for member in GEFS_REFORECAST_MEMBERS]
    point_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []
    partial_path = output_dir / "gefs_reforecast_download_manifest.partial.csv"

    def run_task(task: tuple[str, str]):
        date, member = task
        points, metadata = download_reforecast_member_points(
            cycle_date=date,
            member=member,
            sites=sites,
            cache_dir=output_dir / "cache",
            timeout=timeout,
            retries=retries,
            keep_grib=False,
        )
        return date, member, points, metadata

    errors: list[tuple[tuple[str, str], Exception]] = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                date, member, points, metadata = future.result()
            except Exception as exc:
                errors.append((task, exc))
                append_log(log_file, f"[expanded] {task[0]} {task[1]} failed: {exc}")
                continue
            point_parts.append(points)
            manifest_rows.append(metadata)
            manifest = pd.DataFrame(manifest_rows).sort_values(
                ["cycle_date", "gefs_member"]
            )
            manifest.to_csv(partial_path, index=False, encoding="utf-8-sig")
            append_log(
                log_file,
                f"[expanded] {date} {member} ready ({completed}/{len(tasks)})",
            )
    if errors:
        examples = "; ".join(
            f"{date}/{member}: {error}" for (date, member), error in errors[:5]
        )
        raise RuntimeError(f"{len(errors)} GEFS download tasks failed: {examples}")
    points = pd.concat(point_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["cycle_date", "gefs_member"]
    ).reset_index(drop=True)
    return points, manifest


def load_reference(
    path: Path,
    *,
    valid_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    reference = pd.read_csv(path)
    reference["reference_valid_unflagged"] = reference[
        "reference_valid_unflagged"
    ].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )
    reference["valid_date_utc"] = pd.to_datetime(reference["station_record_date"])
    valid_set = {pd.Timestamp(value).normalize() for value in valid_dates}
    reference = reference.loc[
        reference["valid_date_utc"].dt.normalize().isin(valid_set)
    ].copy()
    reference["aggregation_day_boundary"] = "GHCND_station_record_date_offset_0"
    columns = [
        "site_id",
        "ghcnd_station_id",
        "valid_date_utc",
        "aggregation_day_boundary",
        "precipitation_mm_reference",
        "m_flag",
        "q_flag",
        "s_flag",
        "observation_time",
        "reference_present",
        "reference_valid_unflagged",
        "date_offset_days_applied",
        "date_boundary_status",
    ]
    reference = reference[columns].sort_values(["site_id", "valid_date_utc"])
    expected = len(SITE_IDS) * len(valid_set)
    if len(reference) != expected:
        raise ValueError(
            f"GHCN reference rows={len(reference)}, expected={expected}"
        )
    if reference.duplicated(["site_id", "valid_date_utc"]).any():
        raise ValueError("duplicate selected GHCN site-date rows")
    return reference.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    dates = cycle_dates(args.first_year, args.last_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text("", encoding="utf-8")
    append_log(
        args.log_file,
        f"[expanded] years={args.first_year}-{args.last_year} "
        f"cycles={len(dates)} tasks={len(dates) * len(GEFS_REFORECAST_MEMBERS)}",
    )
    points, manifest = extract_downloads(
        dates=dates,
        output_dir=args.output_dir,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        log_file=args.log_file,
    )
    member_daily = aggregate_reforecast_member_daily_utc(points, manifest=manifest)
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=SITE_IDS,
        expected_members=GEFS_REFORECAST_MEMBERS,
        expected_cycles=dates,
        date_column="valid_date_utc",
    )
    valid_dates = sorted(
        {valid_date for date in dates for valid_date in cycle_valid_dates(date)}
    )
    reference = load_reference(args.reference_file, valid_dates=valid_dates)
    paired = member_daily.merge(
        reference,
        on=["site_id", "valid_date_utc"],
        how="left",
        validate="many_to_one",
    )
    if paired["ghcnd_station_id"].isna().any():
        raise ValueError("missing selected GHCN station row after pairing")

    tag = f"{args.first_year}_{args.last_year}"
    member_path = args.output_dir / f"gefs_member_daily_precipitation_utc_{tag}_v1.csv"
    manifest_path = args.output_dir / f"gefs_download_manifest_{tag}_v1.csv"
    reference_path = args.output_dir / f"ghcnd_reference_daily_precipitation_{tag}_v1.csv"
    paired_path = args.output_dir / f"gefs_ghcnd_paired_member_daily_{tag}_v1.csv"
    extraction_path = args.output_dir / f"gefs_ghcnd_extraction_manifest_{tag}_v1.json"
    member_daily.to_csv(member_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")

    expected_member_rows = (
        len(dates)
        * len(SITE_IDS)
        * len(GEFS_REFORECAST_MEMBERS)
        * 7
    )
    expected_reference_rows = len(dates) * len(SITE_IDS) * 7
    if len(member_daily) != expected_member_rows:
        raise ValueError("expanded GEFS member row count mismatch")
    if len(reference) != expected_reference_rows:
        raise ValueError("expanded GHCN reference row count mismatch")
    retained_grib_count = len(list((args.output_dir / "cache" / "minigrib").glob("*.grib2")))
    extraction = {
        "contract_id": "gefs-qdm-expanded-period-extraction-v1",
        "first_year": int(args.first_year),
        "last_year": int(args.last_year),
        "cycle_count": int(len(dates)),
        "download_task_count": int(len(manifest)),
        "site_ids": list(SITE_IDS),
        "members": list(GEFS_REFORECAST_MEMBERS),
        "member_daily_rows": int(len(member_daily)),
        "reference_daily_rows": int(len(reference)),
        "paired_member_rows": int(len(paired)),
        "valid_unflagged_reference_rows": int(
            reference["reference_valid_unflagged"].sum()
        ),
        "missing_or_flagged_reference_rows": int(
            (~reference["reference_valid_unflagged"]).sum()
        ),
        "station_date_offset_from_gefs_valid_date_days": 0,
        "retained_grib_file_count": retained_grib_count,
        "member_file_sha256": sha256_file(member_path),
        "download_manifest_sha256": sha256_file(manifest_path),
        "reference_file_sha256": sha256_file(reference_path),
        "paired_file_sha256": sha256_file(paired_path),
        "source_reference_file_sha256": sha256_file(args.reference_file),
    }
    if retained_grib_count != 0:
        raise ValueError("temporary GRIB files were retained unexpectedly")
    extraction_path.write_text(
        json.dumps(extraction, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    append_log(args.log_file, json.dumps({
        "member_daily": str(member_path),
        "download_manifest": str(manifest_path),
        "reference": str(reference_path),
        "paired": str(paired_path),
        "extraction_manifest": str(extraction_path),
    }, indent=2))
    return {
        "member_daily": member_path,
        "download_manifest": manifest_path,
        "reference": reference_path,
        "paired": paired_path,
        "extraction_manifest": extraction_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-year", type=int, default=2000)
    parser.add_argument("--last-year", type=int, default=2002)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception:
        if args.log_file is not None:
            append_log(args.log_file, traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

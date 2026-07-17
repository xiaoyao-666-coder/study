#!/usr/bin/env python3
"""Validate all GEFS members against gridMET with probabilistic metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_gefs_gridmet_bias_validation_v1 import (  # noqa: E402
    DEFAULT_GRIDMET_DIR,
    _validate_daily_forecast,
    aggregate_gefs_point_records,
    build_gefs_point_records,
    build_gridmet_reference,
    ensure_gridmet_reference,
    site_frame,
)
from s2s_rtist.weather.gefs_ensemble_validation import (  # noqa: E402
    aggregate_probabilistic_metrics,
    compute_precipitation_probability_metrics,
    summarize_ensemble_observations,
)
from s2s_rtist.weather.gefs_gridmet_bias import (  # noqa: E402
    FORECAST_DAILY_VARIABLES,
    REQUIRED_MESSAGES,
    add_reference_condition,
    forecast_daily_to_long,
    gefs_members,
    pair_forecast_and_reference,
    required_valid_dates,
)


DEFAULT_DECISION_DATES = ("2024-07-16",)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_member_gridmet_validation_v1"
)


def validate_member_daily_weather(
    daily: pd.DataFrame,
    *,
    decision_dates: Sequence[str],
    site_names: Sequence[str],
    expected_members: Sequence[str],
    variables: Sequence[str] = FORECAST_DAILY_VARIABLES,
    horizon_days: int = 7,
) -> None:
    required = {
        "site",
        "decision_date",
        "local_date",
        "lead_day",
        "gefs_member",
        *variables,
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"missing member daily columns: {sorted(missing)}")

    expected_rows = (
        len(decision_dates)
        * len(site_names)
        * len(expected_members)
        * int(horizon_days)
    )
    if len(daily) != expected_rows:
        raise ValueError(
            f"member daily rows={len(daily)}, expected={expected_rows}"
        )
    duplicate_keys = ["site", "decision_date", "local_date", "gefs_member"]
    if daily.duplicated(duplicate_keys).any():
        raise ValueError("duplicate GEFS member daily rows")
    missing_values = daily[list(variables)].isna().sum()
    missing_values = missing_values.loc[missing_values.gt(0)]
    if not missing_values.empty:
        raise ValueError(f"missing member daily values: {missing_values.to_dict()}")

    expected_set = set(expected_members)
    for key, group in daily.groupby(
        ["site", "decision_date", "local_date"], sort=False, dropna=False
    ):
        actual_set = set(group["gefs_member"])
        if actual_set != expected_set:
            missing_members = sorted(expected_set.difference(actual_set))
            extra_members = sorted(actual_set.difference(expected_set))
            raise ValueError(
                "incomplete GEFS member set for daily weather "
                f"{key}: missing={missing_members}, extra={extra_members}"
            )


def analyze_member_daily_weather(
    daily: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_members: Sequence[str],
    variables: Sequence[str] = FORECAST_DAILY_VARIABLES,
) -> dict[str, pd.DataFrame]:
    forecast_long = forecast_daily_to_long(daily, variables=variables)
    paired = pair_forecast_and_reference(forecast_long, reference)
    paired = add_reference_condition(paired)
    observations = summarize_ensemble_observations(
        paired, expected_members=expected_members
    )
    observations = add_reference_condition(observations)
    return {
        "paired_members": paired,
        "ensemble_observations": observations,
        "probabilistic_metrics_overall": aggregate_probabilistic_metrics(
            observations, group_columns=("variable",)
        ),
        "probabilistic_metrics_by_lead": aggregate_probabilistic_metrics(
            observations, group_columns=("variable", "lead_day")
        ),
        "probabilistic_metrics_by_site": aggregate_probabilistic_metrics(
            observations, group_columns=("variable", "site")
        ),
        "probabilistic_metrics_by_condition": aggregate_probabilistic_metrics(
            observations, group_columns=("variable", "reference_condition")
        ),
        "precipitation_probability_overall": (
            compute_precipitation_probability_metrics(
                paired, expected_members=expected_members
            )
        ),
        "precipitation_probability_by_lead": (
            compute_precipitation_probability_metrics(
                paired,
                expected_members=expected_members,
                group_columns=("lead_day",),
            )
        ),
    }


def download_member_daily_weather(
    *,
    decision_dates: Sequence[str],
    sites: pd.DataFrame,
    members: Sequence[str],
    cache_dir: Path,
    member_output_dir: Path,
    timeout: int,
    retries: int,
    workers: int,
    keep_grib: bool,
    required_messages: Sequence[tuple[str, str]] = REQUIRED_MESSAGES,
    variables: Sequence[str] = FORECAST_DAILY_VARIABLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_parts: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []
    member_output_dir.mkdir(parents=True, exist_ok=True)
    for index, member in enumerate(members, start=1):
        print(f"[member] starting {member} ({index}/{len(members)})", flush=True)
        points, manifest = build_gefs_point_records(
            decision_dates=decision_dates,
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
            workers=workers,
            keep_grib=keep_grib,
            product=member,
            required_messages=required_messages,
        )
        daily = aggregate_gefs_point_records(points)
        _validate_daily_forecast(
            daily,
            decision_dates=decision_dates,
            sites=sites,
            variables=variables,
        )
        daily["gefs_member"] = member
        daily.to_csv(member_output_dir / f"{member}_daily_weather.csv", index=False)
        daily_parts.append(daily)
        manifest_parts.append(manifest)
        print(f"[member] completed {member}", flush=True)
    return (
        pd.concat(daily_parts, ignore_index=True),
        pd.concat(manifest_parts, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="gefs_31member_1cycle_5site_20260716_v1"
    )
    parser.add_argument(
        "--decision-dates", nargs="+", default=list(DEFAULT_DECISION_DATES)
    )
    parser.add_argument("--sites", nargs="+", default=None)
    parser.add_argument("--members", nargs="+", default=list(gefs_members()))
    parser.add_argument("--gridmet-dir", type=Path, default=DEFAULT_GRIDMET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--download-gridmet", action="store_true")
    parser.add_argument(
        "--precipitation-only",
        action="store_true",
        help="download APCP only and compute precipitation ensemble metrics",
    )
    parser.add_argument("--keep-grib", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in args.decision_dates]
    members = tuple(args.members)
    official_members = set(gefs_members())
    unknown_members = sorted(set(members).difference(official_members))
    if unknown_members:
        raise ValueError(f"unsupported GEFS members: {unknown_members}")
    if len(members) != len(set(members)):
        raise ValueError("GEFS members must be unique")
    if args.precipitation_only:
        required_messages = (("APCP", "surface"),)
        forecast_variables = ("precipitation_mm",)
    else:
        required_messages = REQUIRED_MESSAGES
        forecast_variables = FORECAST_DAILY_VARIABLES

    sites = site_frame(args.sites)
    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sites.to_csv(run_dir / "sites.csv", index=False)
    pd.DataFrame({"decision_date": dates}).to_csv(
        run_dir / "decision_dates.csv", index=False
    )
    pd.DataFrame({"gefs_member": members}).to_csv(
        run_dir / "gefs_members.csv", index=False
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
    reference = build_gridmet_reference(
        reference_dir, sites=sites, dates=valid_dates
    )
    reference.to_csv(run_dir / "gridmet_reference_daily_long.csv", index=False)

    daily, manifest = download_member_daily_weather(
        decision_dates=dates,
        sites=sites,
        members=members,
        cache_dir=cache_dir,
        member_output_dir=run_dir / "member_daily_weather",
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
        keep_grib=args.keep_grib,
        required_messages=required_messages,
        variables=forecast_variables,
    )
    validate_member_daily_weather(
        daily,
        decision_dates=dates,
        site_names=sites["site"].tolist(),
        expected_members=members,
        variables=forecast_variables,
    )
    daily.to_csv(run_dir / "gefs_member_daily_weather.csv", index=False)
    manifest.to_csv(run_dir / "gefs_member_download_manifest.csv", index=False)

    analysis = analyze_member_daily_weather(
        daily,
        reference,
        expected_members=members,
        variables=forecast_variables,
    )
    for name, frame in analysis.items():
        frame.to_csv(run_dir / f"{name}.csv", index=False)

    metadata = {
        "run_id": args.run_id,
        "decision_dates": dates,
        "sites": sites["site"].tolist(),
        "members": list(members),
        "member_count": len(members),
        "cycle_hour_utc": 0,
        "required_leads": "f003-f180 at 3-hour intervals",
        "forecast_variables": list(forecast_variables),
        "required_grib_messages": [
            {"short_name": short_name, "level": level}
            for short_name, level in required_messages
        ],
        "precipitation_only": bool(args.precipitation_only),
        "horizon": "local decision date D through D+6",
        "reference_product": "gridMET 2024",
        "daily_rows": len(daily),
        "paired_member_rows": len(analysis["paired_members"]),
        "ensemble_observation_rows": len(analysis["ensemble_observations"]),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"run_dir: {run_dir}", flush=True)
    print(
        f"member_daily_rows: {len(daily)}; "
        f"ensemble_observations: {len(analysis['ensemble_observations'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()

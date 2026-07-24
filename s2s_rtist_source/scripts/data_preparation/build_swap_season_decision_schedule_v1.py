#!/usr/bin/env python3
"""Build per-site seasonal decision dates from full-season SWAP crop outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from s2s_rtist.pipelines.season_decision_schedule import (
    build_decision_schedule,
    read_crop_trajectory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_full_season_trunk_branch_v1"
    / "decision_schedule_v1"
)
REQUIRED_MANIFEST_COLUMNS = {"site_id", "target_year", "crop_output_path"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_year(year: int) -> str:
    if year in {2015, 2016, 2017, 2018}:
        return "training"
    if year == 2019:
        return "validation"
    if year == 2024:
        return "test_tta"
    return "unspecified"


def resolve_input_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_from_manifest(
    manifest: pd.DataFrame,
    *,
    dvs_threshold: float,
    interval_days: int,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest[["site_id", "target_year"]].duplicated().any():
        raise ValueError("manifest contains duplicate site-year rows")

    schedules: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for item in manifest.itertuples(index=False):
        year = int(item.target_year)
        source = resolve_input_path(item.crop_output_path)
        crop = read_crop_trajectory(source)
        crop_years = set(crop["Date"].dt.year.astype(int))
        if crop_years != {year}:
            raise ValueError(
                f"crop output year mismatch for {item.site_id}/{year}: {sorted(crop_years)}"
            )
        split = str(getattr(item, "split", "") or split_for_year(year))
        harvest_value = getattr(item, "harvest_date", None)
        if pd.isna(harvest_value) or str(harvest_value).strip() == "":
            harvest_value = None
        schedule = build_decision_schedule(
            crop,
            site_id=str(item.site_id),
            target_year=year,
            split=split,
            dvs_threshold=dvs_threshold,
            interval_days=interval_days,
            horizon_days=horizon_days,
            harvest_date=harvest_value,
        )
        schedules.append(schedule)
        sources.append(
            {
                "site_id": str(item.site_id),
                "target_year": year,
                "crop_output_path": str(source),
                "crop_output_sha256": sha256_file(source),
                "crop_daily_rows": int(len(crop)),
                "first_crop_date": crop["Date"].min().strftime("%Y-%m-%d"),
                "last_crop_date": crop["Date"].max().strftime("%Y-%m-%d"),
                "decision_rows": int(len(schedule)),
            }
        )
    combined = pd.concat(schedules, ignore_index=True).sort_values(
        ["target_year", "site_id", "decision_date"]
    ).reset_index(drop=True)
    return combined, pd.DataFrame(sources).sort_values(
        ["target_year", "site_id"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dvs-threshold", type=float, default=0.1)
    parser.add_argument("--interval-days", type=int, default=7)
    parser.add_argument("--horizon-days", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.trunk_manifest)
    schedule, sources = build_from_manifest(
        manifest,
        dvs_threshold=args.dvs_threshold,
        interval_days=args.interval_days,
        horizon_days=args.horizon_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "swap_season_decision_schedule_v1.csv"
    sources_path = args.output_dir / "swap_season_trunk_source_manifest_v1.csv"
    audit_path = args.output_dir / "swap_season_decision_schedule_audit_v1.json"
    schedule.to_csv(schedule_path, index=False)
    sources.to_csv(sources_path, index=False)
    audit = {
        "status": "season_decision_schedule_passed",
        "site_year_rows": int(len(sources)),
        "decision_rows": int(len(schedule)),
        "minimum_state_dvs": float(schedule["state_dvs"].min()),
        "dvs_threshold": float(args.dvs_threshold),
        "sampling_interval_days": int(args.interval_days),
        "horizon_days": int(args.horizon_days),
        "incomplete_horizon_rows": int(
            (
                pd.to_datetime(schedule["horizon_end_date"])
                > pd.to_datetime(schedule["harvest_date"])
            ).sum()
        ),
        "cross_site_date_alignment_required": False,
        "branch_weather_requirement": "all_six_swap_fields_bias_corrected_gefs",
        "branch_generation_allowed": False,
        "branch_generation_blocker": (
            "all-variable correction and daily restart checkpoint smoke not yet passed"
        ),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"schedule": str(schedule_path), "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()

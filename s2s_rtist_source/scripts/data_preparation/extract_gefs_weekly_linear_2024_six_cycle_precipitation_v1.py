#!/usr/bin/env python3
"""Extract the prelocked six-cycle 31-member GEFS precipitation sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.diagnostics.run_gefs_gridmet_bias_validation_v1 import site_frame
from scripts.diagnostics.run_gefs_member_gridmet_validation_v1 import (
    download_member_daily_weather,
    validate_member_daily_weather,
)
from s2s_rtist.weather.gefs_gridmet_bias import gefs_members


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_2024_six_cycle_confirmation_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_2024_six_cycle_extraction_v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-2024-six-cycle-confirmation-v1":
        raise ValueError("six-cycle confirmation contract id mismatch")
    if not contract.get("precipitation_only"):
        raise ValueError("six-cycle extraction must remain precipitation-only")
    current = set(contract["decision_dates"])
    previous = set(contract["previously_scored_decision_dates"])
    if current.intersection(previous):
        raise ValueError("six-cycle decision dates overlap previously scored cycles")
    if len(current) != 6:
        raise ValueError("six-cycle contract must contain six unique dates")
    scope = contract["scope"]
    if scope["network_download_location"] != "local_workstation_only":
        raise ValueError("GEFS network extraction must remain local-only")
    if any(
        scope[key]
        for key in (
            "artifact_refit_allowed",
            "candidate_reselection_allowed",
            "hyperparameter_tuning_allowed",
            "station_reselection_allowed",
            "reference_used_for_application_factor",
            "surrogate_training_allowed",
        )
    ):
        raise ValueError("six-cycle contract permits a forbidden operation")
    return contract


def expected_row_count(contract: dict[str, Any]) -> int:
    return (
        len(contract["decision_dates"])
        * len(contract["expected_sites"])
        * int(contract["expected_member_count"])
        * len(contract["expected_lead_days"])
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    dates = list(contract["decision_dates"])
    sites = site_frame(contract["expected_sites"])
    members = tuple(gefs_members())
    if len(members) != int(contract["expected_member_count"]):
        raise ValueError("official GEFS member count changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"
    member_dir = args.output_dir / "member_daily_weather"
    daily, manifest = download_member_daily_weather(
        decision_dates=dates,
        sites=sites,
        members=members,
        cache_dir=cache_dir,
        member_output_dir=member_dir,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
        keep_grib=False,
        required_messages=(("APCP", "surface"),),
        variables=("precipitation_mm",),
    )
    validate_member_daily_weather(
        daily,
        decision_dates=dates,
        site_names=sites["site"].tolist(),
        expected_members=members,
        variables=("precipitation_mm",),
    )
    expected = expected_row_count(contract)
    if expected != int(contract["expected_member_daily_rows"]):
        raise ValueError("contract expected row count is internally inconsistent")
    if len(daily) != expected:
        raise ValueError(f"six-cycle member rows={len(daily)}, expected={expected}")
    daily_path = args.output_dir / "gefs_member_daily_precipitation_2024_six_cycle_v1.csv"
    manifest_path = args.output_dir / "gefs_member_download_manifest_2024_six_cycle_v1.csv"
    metadata_path = args.output_dir / "gefs_six_cycle_extraction_metadata_v1.json"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    metadata = {
        "contract_id": contract["contract_id"],
        "decision_dates": dates,
        "previously_scored_decision_dates": contract["previously_scored_decision_dates"],
        "decision_cycle_initializations_disjoint": True,
        "sites": contract["expected_sites"],
        "members": list(members),
        "member_count": len(members),
        "member_daily_rows": int(len(daily)),
        "precipitation_only": True,
        "required_grib_messages": [{"short_name": "APCP", "level": "surface"}],
        "network_download_location": "local_workstation_only",
        "retained_grib_file_count": 0,
        "status": "six_cycle_precipitation_extraction_completed",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = {"member_daily": daily_path, "manifest": manifest_path, "metadata": metadata_path}
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble the crop-gate-reselected five-cycle GEFS member weather table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from s2s_rtist.weather.gefs_quantile_mapping import GEFS_REFORECAST_MEMBERS
from s2s_rtist.weather.gefs_reforecast_full_weather import validate_full_weather


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_full_weather_pilot_local_v1"
) / "gefs_2015_2019_full_weather_member_daily_v1.csv"
DEFAULT_REPLACEMENT = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2017_july_replacement_full_weather_v1"
) / "gefs_2015_2019_full_weather_member_daily_v1.csv"
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_scenario_consistent_pilot_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_reselected_full_weather_v2"
)
ROW_KEYS = ["decision_date", "site_id", "gefs_member", "lead_day"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_cycle_dates(contract: dict[str, Any]) -> list[str]:
    dates = [
        pd.Timestamp(row["decision_date"]).strftime("%Y-%m-%d")
        for row in contract["selected_cycles"]
    ]
    if len(dates) != 5 or len(set(dates)) != 5:
        raise ValueError("contract must select exactly five unique cycles")
    if {pd.Timestamp(date).year for date in dates} != set(range(2015, 2020)):
        raise ValueError("contract must select one cycle per year from 2015-2019")
    return dates


def assemble_selected_weather(
    existing: pd.DataFrame,
    replacement: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    dates = selected_cycle_dates(contract)
    old = existing.copy()
    new = replacement.copy()
    old["decision_date"] = pd.to_datetime(old["decision_date"]).dt.strftime("%Y-%m-%d")
    new["decision_date"] = pd.to_datetime(new["decision_date"]).dt.strftime("%Y-%m-%d")
    old_selected = old.loc[old["decision_date"].isin(dates)].copy()
    new_selected = new.loc[new["decision_date"].isin(dates)].copy()
    combined = pd.concat([old_selected, new_selected], ignore_index=True)
    if combined[ROW_KEYS].duplicated().any():
        raise ValueError("selected weather contains duplicate member-site-lead rows")
    if set(combined["decision_date"]) != set(dates):
        missing = sorted(set(dates).difference(combined["decision_date"]))
        raise ValueError(f"selected weather is missing cycles: {missing}")
    combined = combined.sort_values(ROW_KEYS).reset_index(drop=True)
    counts = {
        "existing_selected_rows": int(len(old_selected)),
        "replacement_selected_rows": int(len(new_selected)),
        "combined_rows": int(len(combined)),
    }
    return combined, counts


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dates = selected_cycle_dates(contract)
    existing = pd.read_csv(args.existing_weather)
    replacement = pd.read_csv(args.replacement_weather)
    combined, counts = assemble_selected_weather(existing, replacement, contract)
    validation = validate_full_weather(
        combined,
        expected_cycles=dates,
        expected_sites=tuple(contract["sites"]),
        expected_members=tuple(GEFS_REFORECAST_MEMBERS),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "weather": args.output_dir / "gefs_2015_2019_full_weather_member_daily_v1.csv",
        "audit": args.output_dir / "gefs_2015_2019_reselected_weather_audit_v1.json",
        "manifest": args.output_dir
        / "gefs_2015_2019_reselected_weather_manifest_v1.json",
    }
    combined.to_csv(outputs["weather"], index=False, encoding="utf-8-sig")
    audit = {
        **validation,
        **counts,
        "status": "crop_gate_reselected_full_weather_assembled",
        "selected_cycles": dates,
        "surrogate_training_performed": False,
        "swap_simulation_performed": False,
        "tta_performed": False,
    }
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "existing_weather_sha256": sha256_file(args.existing_weather),
        "replacement_weather_sha256": sha256_file(args.replacement_weather),
        "contract_sha256": sha256_file(args.contract),
        "output_weather_sha256": sha256_file(outputs["weather"]),
        "audit_sha256": sha256_file(outputs["audit"]),
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-weather", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--replacement-weather", type=Path, default=DEFAULT_REPLACEMENT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

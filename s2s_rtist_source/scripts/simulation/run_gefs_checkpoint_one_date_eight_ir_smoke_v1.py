#!/usr/bin/env python3
"""Run one verified-checkpoint SWAP branch smoke with frozen GEFS weather."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2s_rtist.validation.three_output_smoke import (
    validate_smoke_dataset,
    write_validation_outputs,
)


IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]
NONPRECIP_VARIABLES = [
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]
PRIMARY_OUTPUTS = [
    "net_gain_7d",
    "aet_7d_mm",
    "soil_vwc_0_100cm_day01",
    "soil_vwc_0_100cm_day02",
    "soil_vwc_0_100cm_day03",
    "soil_vwc_0_100cm_day04",
    "soil_vwc_0_100cm_day05",
    "soil_vwc_0_100cm_day06",
    "soil_vwc_0_100cm_day07",
]


def swap_weather_filenames(year: int) -> tuple[str, str]:
    suffix = f".{int(year) % 100:03d}"
    return f"weather{suffix}", f"WeatherOriginal{suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_swap_weather_record(line: str) -> dict[str, Any] | None:
    try:
        fields = shlex.split(line.strip())
    except ValueError:
        return None
    if len(fields) < 11 or fields[0].lower() != "weather":
        return None
    try:
        date = pd.Timestamp(year=int(fields[3]), month=int(fields[2]), day=int(fields[1]))
        values = [float(value) for value in fields[4:11]]
    except (TypeError, ValueError):
        return None
    return {
        "date": date.strftime("%Y-%m-%d"),
        "solar_kj_m2_day": values[0],
        "temperature_min_c": values[1],
        "temperature_max_c": values[2],
        "actual_vapor_pressure_kpa": values[3],
        "wind_speed_m_s": values[4],
        "precipitation_mm": values[5],
        "etref": values[6],
    }


def format_swap_weather_record(date_text: str, row: pd.Series, etref: float) -> str:
    date = pd.Timestamp(date_text)
    return (
        f" 'Weather' {date.day:9d} {date.month:7d} {date.year:7d}"
        f" {float(row.solar_kj_m2_day_mean):14.6f}"
        f" {float(row.temperature_min_c_mean):12.6f}"
        f" {float(row.temperature_max_c_mean):12.6f}"
        f" {float(row.actual_vapor_pressure_kpa_mean):12.6f}"
        f" {float(row.wind_speed_m_s_mean):12.6f}"
        f" {float(row.precipitation_mm_corrected_mean):12.6f}"
        f" {float(etref):12.6f}\n"
    )


def patch_swap_weather_file(path: Path, daily: pd.DataFrame) -> pd.DataFrame:
    replacements = daily.set_index("local_date")
    expected_dates = set(replacements.index.astype(str))
    original_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    output_lines = []
    audit_rows = []
    for line in original_lines:
        old = parse_swap_weather_record(line)
        if old is None or old["date"] not in expected_dates:
            output_lines.append(line)
            continue
        row = replacements.loc[old["date"]]
        output_lines.append(format_swap_weather_record(old["date"], row, old["etref"]))
        audit_rows.append(
            {
                "patched_file": path.name,
                "local_date": old["date"],
                "old_solar_kj_m2_day": old["solar_kj_m2_day"],
                "new_solar_kj_m2_day": float(row.solar_kj_m2_day_mean),
                "old_temperature_min_c": old["temperature_min_c"],
                "new_temperature_min_c": float(row.temperature_min_c_mean),
                "old_temperature_max_c": old["temperature_max_c"],
                "new_temperature_max_c": float(row.temperature_max_c_mean),
                "old_actual_vapor_pressure_kpa": old[
                    "actual_vapor_pressure_kpa"
                ],
                "new_actual_vapor_pressure_kpa": float(
                    row.actual_vapor_pressure_kpa_mean
                ),
                "old_wind_speed_m_s": old["wind_speed_m_s"],
                "new_wind_speed_m_s": float(row.wind_speed_m_s_mean),
                "old_precipitation_mm": old["precipitation_mm"],
                "new_precipitation_mm": float(
                    row.precipitation_mm_corrected_mean
                ),
                "old_etref": old["etref"],
                "new_etref": old["etref"],
            }
        )
    actual_dates = {row["local_date"] for row in audit_rows}
    if actual_dates != expected_dates or len(audit_rows) != len(expected_dates):
        raise ValueError(
            f"{path} patched dates={sorted(actual_dates)}, expected={sorted(expected_dates)}"
        )
    before_non_target = [
        line
        for line in original_lines
        if (record := parse_swap_weather_record(line)) is not None
        and record["date"] not in expected_dates
    ]
    after_non_target = [
        line
        for line in output_lines
        if (record := parse_swap_weather_record(line)) is not None
        and record["date"] not in expected_dates
    ]
    if before_non_target != after_non_target:
        raise ValueError(f"{path} changed a non-target weather record")
    path.write_text("".join(output_lines), encoding="utf-8")
    return pd.DataFrame(audit_rows)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_generator(workspace: Path):
    module_path = workspace / "generate_restart_decision_dataset.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"missing restart generator: {module_path}")
    sys.path.insert(0, str(workspace))
    spec = importlib.util.spec_from_file_location(
        "gefs_checkpoint_restart_generator", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import restart generator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_checkpoint(
    checkpoint_dir: Path,
    checkpoint_audit_csv: Path,
    decision_date: str,
) -> dict[str, Any]:
    date = pd.Timestamp(decision_date)
    expected_checkpoint = (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    required = ["result_forec.end", "result_forec.crp", "result_forec.vap"]
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is missing files: {missing}")
    audit = pd.read_csv(checkpoint_audit_csv)
    audit["decision_date"] = pd.to_datetime(audit["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    rows = audit.loc[audit["decision_date"].eq(date.strftime("%Y-%m-%d"))]
    if len(rows) != 1:
        raise ValueError("checkpoint audit does not contain exactly one decision row")
    row = rows.iloc[0]
    equivalence_value = row["checkpoint_equivalence_passed"]
    equivalence_passed = (
        bool(equivalence_value)
        if isinstance(equivalence_value, (bool, np.bool_))
        else str(equivalence_value).strip().lower() == "true"
    )
    if not equivalence_passed:
        raise ValueError("checkpoint did not pass equivalence audit")
    if str(row["checkpoint_date"]) != expected_checkpoint:
        raise ValueError("checkpoint date does not equal decision date minus one day")
    if float(row["maximum_absolute_crop_state_error"]) > 1e-6:
        raise ValueError("checkpoint crop state error exceeds tolerance")
    if float(row["maximum_absolute_profile_state_error"]) > 1e-6:
        raise ValueError("checkpoint profile state error exceeds tolerance")
    return {
        "decision_date": date.strftime("%Y-%m-%d"),
        "checkpoint_date": expected_checkpoint,
        "maximum_absolute_crop_state_error": float(
            row["maximum_absolute_crop_state_error"]
        ),
        "maximum_absolute_profile_state_error": float(
            row["maximum_absolute_profile_state_error"]
        ),
    }


def build_ensemble_mean_weather(
    weather: pd.DataFrame, *, site_id: str, decision_date: str
) -> pd.DataFrame:
    data = weather.copy()
    for column in ("decision_date", "local_date"):
        data[column] = pd.to_datetime(data[column]).dt.strftime("%Y-%m-%d")
    selected = data.loc[
        data["site_id"].astype(str).eq(str(site_id))
        & data["decision_date"].eq(pd.Timestamp(decision_date).strftime("%Y-%m-%d"))
    ].copy()
    required = {
        "gefs_member",
        "lead_day",
        "local_date",
        "precipitation_mm",
        *NONPRECIP_VARIABLES,
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"all-variable weather is missing fields: {missing}")
    key = ["gefs_member", "local_date", "lead_day"]
    if len(selected) != 35 or selected[key].duplicated().any():
        raise ValueError("one-site weather must contain 35 unique member-day rows")
    member_counts = selected.groupby(["local_date", "lead_day"])[
        "gefs_member"
    ].nunique()
    if not member_counts.eq(5).all():
        raise ValueError("each future day must contain five GEFS members")
    expected_dates = pd.date_range(decision_date, periods=7, freq="D").strftime(
        "%Y-%m-%d"
    ).tolist()
    if selected.sort_values("lead_day")["lead_day"].unique().tolist() != list(
        range(1, 8)
    ):
        raise ValueError("future lead days are incomplete")
    daily = selected.groupby(["local_date", "lead_day"], as_index=False)[
        [*NONPRECIP_VARIABLES, "precipitation_mm"]
    ].mean()
    daily = daily.sort_values("lead_day").reset_index(drop=True)
    if daily["local_date"].tolist() != expected_dates:
        raise ValueError("future local dates are incomplete")
    numeric = daily[[*NONPRECIP_VARIABLES, "precipitation_mm"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("ensemble mean weather contains nonfinite values")
    if (daily["temperature_min_c"] > daily["temperature_max_c"]).any():
        raise ValueError("ensemble mean Tmin exceeds Tmax")
    positive = [
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "solar_kj_m2_day",
        "precipitation_mm",
    ]
    if (daily[positive] < 0.0).any().any():
        raise ValueError("ensemble mean weather contains negative physical values")
    daily = daily.rename(
        columns={
            **{field: f"{field}_mean" for field in NONPRECIP_VARIABLES},
            "precipitation_mm": "precipitation_mm_corrected_mean",
        }
    )
    daily["member_count"] = 5
    daily["artifact_sha256"] = "frozen_all_variable_weather_v1"
    return daily


def inject_future_weather(
    workspace: Path, daily: pd.DataFrame, *, year: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    audit_frames = []
    hashes: dict[str, str] = {}
    for filename in swap_weather_filenames(year):
        path = workspace / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing SWAP weather file: {path}")
        hashes[f"{filename}_before_sha256"] = sha256_file(path)
        audit_frames.append(patch_swap_weather_file(path, daily))
        hashes[f"{filename}_after_sha256"] = sha256_file(path)
    audit = pd.concat(audit_frames, ignore_index=True)
    if len(audit) != 14:
        raise ValueError("weather injection must patch seven rows in each SWAP file")
    if set(audit["local_date"]) != set(daily["local_date"]):
        raise ValueError("weather injection dates differ from the frozen GEFS horizon")
    return audit, hashes


def copy_checkpoint(checkpoint_dir: Path, workspace: Path) -> None:
    for name in ("result_forec.end", "result_forec.crp", "result_forec.vap"):
        shutil.copy2(checkpoint_dir / name, workspace / name)


def run_checkpoint_branches(
    *,
    workspace: Path,
    checkpoint_dir: Path,
    decision_date: str,
    year: int,
    sowing_month_day: str,
) -> pd.DataFrame:
    generator = load_generator(workspace)
    decision = pd.Timestamp(decision_date)
    decision_doy = int(decision.dayofyear)
    generator.YEAR = int(year)
    generator.START_DOY = int(pd.Timestamp(f"{year}-{sowing_month_day}").dayofyear)
    generator.HORIZON_DAYS = 7
    generator.RESTART_NPRINTDAY = 24

    checkpoint_used = {"count": 0}

    def use_verified_checkpoint(log_name: str, requested_doy: int) -> None:
        if int(requested_doy) != decision_doy:
            raise ValueError("restart generator requested an unexpected decision day")
        copy_checkpoint(checkpoint_dir, workspace)
        Path(log_name).write_text(
            "verified seasonal checkpoint reused; pre-state SWAP was not rerun\n",
            encoding="utf-8",
        )
        checkpoint_used["count"] += 1

    generator.run_pre_state = use_verified_checkpoint
    date_t = decision.strftime("%d-%b-%Y")
    with working_directory(workspace):
        frame = generator.run_one_date(
            date_t,
            decision_doy,
            irrigation_options_mm=IRRIGATION_OPTIONS_MM,
        )
    if checkpoint_used["count"] != 1:
        raise RuntimeError("verified checkpoint was not used exactly once")
    return frame


def build_audit(
    *,
    candidates: pd.DataFrame,
    daily: pd.DataFrame,
    injection: pd.DataFrame,
    checkpoint: dict[str, Any],
    site_id: str = "P1",
    target_year: int = 2015,
    next_gate: str = "expand_verified_checkpoint_branch_smoke_to_five_sites",
) -> dict[str, Any]:
    expected_rain = float(daily["precipitation_mm_corrected_mean"].sum())
    rain_error = float(
        (candidates["rain_7d_mm"].astype(float) - expected_rain).abs().max()
    )
    maximum_residual = float(
        candidates["water_balance_residual_0_100cm_7d_mm"].astype(float).abs().max()
    )
    missing_primary = int(candidates[PRIMARY_OUTPUTS].isna().sum().sum())
    duplicate_count = int(candidates[["date_t", "ir"]].duplicated().sum())
    irrigation_values = sorted(candidates["ir"].astype(float).tolist())
    passed = all(
        [
            len(candidates) == 8,
            irrigation_values == IRRIGATION_OPTIONS_MM,
            duplicate_count == 0,
            missing_primary == 0,
            rain_error <= 0.01,
            maximum_residual <= 0.5,
            len(injection) == 14,
        ]
    )
    return {
        "status": (
            "verified_checkpoint_one_date_eight_ir_swap_smoke_passed"
            if passed
            else "verified_checkpoint_one_date_eight_ir_swap_smoke_failed"
        ),
        "mandatory_gate_passed": passed,
        "site_id": str(site_id),
        "target_year": int(target_year),
        "decision_date": checkpoint["decision_date"],
        "checkpoint_date": checkpoint["checkpoint_date"],
        "checkpoint_equivalence_passed": True,
        "maximum_absolute_checkpoint_crop_state_error": checkpoint[
            "maximum_absolute_crop_state_error"
        ],
        "maximum_absolute_checkpoint_profile_state_error": checkpoint[
            "maximum_absolute_profile_state_error"
        ],
        "prestate_swap_rerun_count": 0,
        "candidate_rows": int(len(candidates)),
        "irrigation_values_mm": irrigation_values,
        "duplicate_candidate_key_count": duplicate_count,
        "weather_member_count": int(daily["member_count"].iloc[0]),
        "weather_ensemble_daily_rows": int(len(daily)),
        "weather_patch_rows": int(len(injection)),
        "predecision_weather_rows_modified": 0,
        "expected_corrected_gefs_rain_7d_mm": expected_rain,
        "maximum_absolute_swap_rain_error_mm": rain_error,
        "maximum_absolute_water_balance_residual_mm": maximum_residual,
        "primary_output_missing_value_count": missing_primary,
        "weather_driver_source": "frozen_corrected_GEFS_5member_ensemble_mean",
        "weather_label_scenario_consistent": True,
        "training_eligible": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            next_gate
            if passed
            else "repair_one_date_checkpoint_branch_smoke"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if pd.Timestamp(args.decision_date).year != int(args.year):
        raise ValueError("decision date year does not match --year")
    checkpoint = validate_checkpoint(
        args.checkpoint_dir, args.checkpoint_audit_csv, args.decision_date
    )
    daily = build_ensemble_mean_weather(
        pd.read_csv(args.all_variable_weather),
        site_id=args.site_id,
        decision_date=args.decision_date,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    workspace = args.output_dir / "workspace"
    shutil.copytree(args.source_workspace, workspace)
    injection, weather_hashes = inject_future_weather(
        workspace, daily, year=args.year
    )
    copy_checkpoint(args.checkpoint_dir, workspace)
    candidates = run_checkpoint_branches(
        workspace=workspace,
        checkpoint_dir=args.checkpoint_dir,
        decision_date=args.decision_date,
        year=args.year,
        sowing_month_day=args.sowing_month_day,
    )
    candidates.insert(0, "site", args.site_id)
    candidates["target_year"] = int(args.year)
    candidates["weather_member_count"] = 5
    candidates["weather_driver_source"] = (
        "frozen_corrected_GEFS_5member_ensemble_mean"
    )
    candidates["weather_label_scenario_consistent"] = True
    candidates["training_eligible"] = False
    candidates["checkpoint_date"] = checkpoint["checkpoint_date"]
    candidates["prestate_swap_rerun"] = False
    formal = validate_smoke_dataset(
        candidates, expected_irrigation_options_mm=IRRIGATION_OPTIONS_MM
    )
    formal_summary, formal_report = write_validation_outputs(formal, args.output_dir)
    audit = build_audit(
        candidates=candidates,
        daily=daily,
        injection=injection,
        checkpoint=checkpoint,
        site_id=args.site_id,
        target_year=args.year,
    )
    outputs = {
        "candidates": args.output_dir / "gefs_checkpoint_one_date_eight_ir_candidates_v1.csv",
        "daily_weather": args.output_dir / "gefs_checkpoint_ensemble_mean_weather_v1.csv",
        "injection": args.output_dir / "gefs_checkpoint_weather_injection_audit_v1.csv",
        "audit": args.output_dir / "gefs_checkpoint_one_date_eight_ir_audit_v1.json",
        "manifest": args.output_dir / "gefs_checkpoint_one_date_eight_ir_manifest_v1.json",
    }
    candidates.to_csv(outputs["candidates"], index=False)
    daily.to_csv(outputs["daily_weather"], index=False)
    injection.to_csv(outputs["injection"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "inputs": {
            "all_variable_weather_sha256": sha256_file(args.all_variable_weather),
            "checkpoint_end_sha256": sha256_file(
                args.checkpoint_dir / "result_forec.end"
            ),
            "checkpoint_audit_csv_sha256": sha256_file(args.checkpoint_audit_csv),
        },
        "weather_file_hashes": weather_hashes,
        "outputs": {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in outputs.items()
            if key != "manifest"
        },
        "formal_validation_summary": str(formal_summary),
        "formal_validation_report": str(formal_report),
        "network_download_performed": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"checkpoint branch smoke failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-audit-csv", type=Path, required=True)
    parser.add_argument("--all-variable-weather", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-id", default="P1")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--decision-date", default="2015-07-06")
    parser.add_argument("--sowing-month-day", default="04-26")
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

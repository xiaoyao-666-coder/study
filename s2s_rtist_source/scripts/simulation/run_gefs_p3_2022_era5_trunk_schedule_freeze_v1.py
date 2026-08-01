#!/usr/bin/env python3
"""Freeze the P3 2022 ERA5 zero-irrigation trunk and decision schedule."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_hybrid_weather_pilot_v1 import (
    VARIABLES as ERA5_VARIABLES,
    convert_era5_to_swap,
    extract_era5_variable,
)
from scripts.simulation.run_gefs_checkpoint_2015_2019_bounded_pilot_v1 import (
    boolean_mask,
)
from scripts.simulation.run_gefs_checkpoint_five_site_eight_ir_smoke_v1 import (
    choose_swap_executable,
    copy_formal_dependencies,
    run_logged,
)
from scripts.training.train_gefs_exact_schedule_no_tta_baseline_v1 import (
    sha256_file,
    write_json,
)
from s2s_rtist.pipelines.season_decision_schedule import read_crop_trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SITE = "P3"
TARGET_YEAR = 2022
EXPECTED_DAYS = 364
EXPECTED_RASTERS = EXPECTED_DAYS * len(ERA5_VARIABLES)
EXPECTED_PARITY_ROWS = 364 * len(ERA5_VARIABLES)
EXPECTED_COMPONENT_HASHES = {
    "P1": "61172cbdee69c1e43c66570baa209287a1ef73ed658c48fd144ca035d7fc683b",
    "P15": "b2ad3cd245d0dd88689f316a29037ceb1fd0b4d59ef915d2a903155dbec15363",
    "B2_gate": "cfd509a41141f64abe7cb8d51a42a53d9e9a3b89893530377d30044e99052056",
    "B2_model_payload": "6d6c4bd4e3ab9b5a7038a4e13b066eced13880829d149be5d584bf76c7bb503f",
}
INDEPENDENT_PROTOCOL_NAMES = {
    "protocol": "gefs_p3_2022_b2_independent_protocol_v1.json",
    "components": "gefs_p3_2022_b2_component_registry_v1.csv",
    "stages": "gefs_p3_2022_b2_stage_registry_v1.csv",
    "audit": "gefs_p3_2022_b2_independent_protocol_audit_v1.json",
    "manifest": "gefs_p3_2022_b2_independent_protocol_manifest_v1.csv",
}
EXPECTED_INDEPENDENT_PROTOCOL_SHA256 = {
    "protocol": "eadecbcc25863642ce355a853b5d390fb7534f72deae1bc30a671f7f2102a255",
    "components": "c7a7f06bbccef0c51eae690c8f27443b01574b4ba1f4bfa2edbb3218f23b8911",
    "stages": "9ce72a98aa8557adf1574d2cc7098c8cdb2489a2325bfb524db322fa5164a853",
    "audit": "1f021c5b80ccf48339c47d0cc1b9ad9ec8ec5e7fcbbfa14835ed92d4bf4013fb",
    "manifest": "25e731ab750e3b19d0d7ce835d0977aa435c3d494cd184f05d0b25abb9162d34",
}
SCHEDULE_RULE = {
    "target_year": TARGET_YEAR,
    "site_id": TARGET_SITE,
    "trunk_weather_source": "ERA5_2022_daily_weather",
    "trunk_irrigation_policy": "zero_irrigation",
    "sowing_month_day": "04-26",
    "harvest_month_day": "10-10",
    "dvs_threshold": 0.1,
    "mature_dvs_excluded": True,
    "interval_days": 7,
    "horizon_days": 7,
    "state_checkpoint_offset_days": -1,
    "first_decision_rule": "first_day_after_checkpoint_DVS_reaches_0.1",
    "last_decision_rule": "retain_complete_D_through_D_plus_6_pre_maturity_cycles",
    "all_eligible_cycles_retained": True,
    "manual_date_selection_allowed": False,
    "GEFS_values_used_to_select_dates": False,
    "SWAP_candidate_labels_used_to_select_dates": False,
}


def strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_independent_protocol(root: Path) -> dict[str, Any]:
    paths = {key: root / name for key, name in INDEPENDENT_PROTOCOL_NAMES.items()}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"independent protocol {key} is missing: {path}")
        if sha256_file(path) != EXPECTED_INDEPENDENT_PROTOCOL_SHA256[key]:
            raise ValueError(f"independent protocol {key} SHA256 changed")
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if not (
        protocol.get("status")
        == "p3_2022_b2_independent_protocol_frozen_before_target_data_access"
        and audit.get("status")
        == "p3_2022_b2_independent_protocol_frozen_before_target_data_access"
        and audit.get("mandatory_gate_passed") is True
        and protocol.get("target", {}).get("site_id") == TARGET_SITE
        and int(protocol.get("target", {}).get("target_year", -1)) == TARGET_YEAR
        and protocol.get("schedule_rule", {}).get("trunk_weather_source")
        == "ERA5_2022_observed_daily_weather"
    ):
        raise ValueError("independent protocol is not eligible for trunk generation")
    for key, expected in SCHEDULE_RULE.items():
        protocol_key = "target_site" if key == "site_id" else key
        protocol_value = protocol.get("schedule_rule", {}).get(protocol_key)
        if key == "trunk_weather_source":
            protocol_value = str(protocol_value).replace("observed_", "")
        if key == "harvest_month_day":
            continue
        if protocol_value != expected:
            raise ValueError(f"frozen schedule rule changed: {key}")
    components = pd.read_csv(paths["components"])
    observed = dict(zip(components["component_id"], components["sha256"]))
    if not (
        observed.get("P1_full_history_composite") == EXPECTED_COMPONENT_HASHES["P1"]
        and observed.get("P15_full_history_composite")
        == EXPECTED_COMPONENT_HASHES["P15"]
        and observed.get("B2_regret_sensitive_router")
        == EXPECTED_COMPONENT_HASHES["B2_gate"]
        and observed.get("B2_regret_sensitive_router_payload")
        == EXPECTED_COMPONENT_HASHES["B2_model_payload"]
    ):
        raise ValueError("independent protocol component hashes changed")
    return {"paths": paths, "audit": audit, "protocol": protocol}


def validate_acquisition_protocol(root: Path) -> dict[str, Any]:
    contract_path = root / "gefs_p3_2022_era5_local_acquisition_contract_v1.json"
    audit_path = root / "gefs_p3_2022_era5_local_acquisition_protocol_audit_v1.json"
    manifest_path = root / "gefs_p3_2022_era5_local_acquisition_protocol_manifest_v1.csv"
    for path in (contract_path, audit_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"ERA5 acquisition protocol file is missing: {path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(manifest_path)
    bindings = dict(zip(manifest["role"], manifest["sha256"]))
    if not (
        contract.get("contract_id") == "gefs-p3-2022-era5-local-acquisition-v1"
        and contract.get("target_site") == TARGET_SITE
        and int(contract.get("target_year", -1)) == TARGET_YEAR
        and contract.get("variables") == list(ERA5_VARIABLES.values())
        and contract.get("source_product", {}).get("expected_day_count") == EXPECTED_DAYS
        and audit.get("mandatory_gate_passed") is True
        and audit.get("P3_2022_weather_rows_read") == 0
        and audit.get("P3_2022_network_requests") == 0
        and audit.get("P3_2022_target_labels_read") == 0
        and sha256_file(contract_path) == bindings.get("output_contract")
        and sha256_file(audit_path) == bindings.get("output_audit")
    ):
        raise ValueError("ERA5 acquisition protocol failed validation")
    return {"contract": contract_path, "audit": audit_path, "manifest": manifest_path}


def validate_download_archive(root: Path) -> dict[str, Any]:
    audit_path = root / "gefs_p3_2022_era5_local_acquisition_audit_v1.json"
    manifest_path = root / "gefs_p3_2022_era5_local_file_manifest_v1.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not (
        audit.get("status")
        == "p3_2022_era5_local_acquisition_complete_pending_server_transfer"
        and audit.get("mandatory_gate_passed") is True
        and int(audit.get("historical_parity_rows", -1)) == EXPECTED_PARITY_ROWS
        and audit.get("historical_parity_all_passed") is True
        and int(audit.get("P3_2022_weather_rows_read", -1)) == EXPECTED_DAYS
        and int(audit.get("P3_2022_target_labels_read", -1)) == 0
        and int(audit.get("P3_2023_rows_or_labels_read", -1)) == 0
        and int(audit.get("P3_2024_rows_or_labels_read", -1)) == 0
        and int(audit.get("raster_file_count", -1)) == EXPECTED_RASTERS
        and audit.get("spatial_subset") == "P3_exact_historical_export_grid_pixel"
        and audit.get("model_training_performed") is False
        and audit.get("decision_dates_resolved") is False
        and audit.get("swap_simulation_performed") is False
    ):
        raise ValueError("transferred ERA5 acquisition audit failed validation")
    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "bytes", "sha256"}
    if set(manifest.columns) != required or manifest["relative_path"].duplicated().any():
        raise ValueError("ERA5 download manifest schema or uniqueness changed")
    for item in manifest.itertuples(index=False):
        relative = PurePosixPath(str(item.relative_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe ERA5 manifest path: {relative}")
        path = root.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.stat().st_size != int(item.bytes)
            or sha256_file(path) != str(item.sha256)
        ):
            raise ValueError(f"ERA5 transferred file changed: {path}")
    raster_entries = manifest["relative_path"].astype(str).str.endswith(".tif")
    if int(raster_entries.sum()) != EXPECTED_RASTERS:
        raise ValueError("ERA5 manifest raster count changed")
    return {"audit": audit_path, "manifest": manifest_path, "audit_value": audit}


def extract_p3_raw_weather(era5_year_root: Path, year: int) -> pd.DataFrame:
    keys = ["target_year", "site_id", "local_date"]
    merged: pd.DataFrame | None = None
    for variable, directory_name in ERA5_VARIABLES.items():
        frame = extract_era5_variable(
            era5_year_root / directory_name, int(year), variable
        )
        frame = frame.loc[frame["site_id"].astype(str).eq(TARGET_SITE)].copy()
        if len(frame) != EXPECTED_DAYS or frame[keys].duplicated().any():
            raise ValueError(f"P3 ERA5 extraction failed for {directory_name}")
        merged = frame if merged is None else merged.merge(
            frame, on=keys, how="inner", validate="one_to_one"
        )
    if merged is None:
        raise RuntimeError("no ERA5 variables extracted")
    expected_count = 365 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 364
    expected_dates = pd.date_range(f"{year}-01-01", periods=expected_count, freq="D")
    if not pd.to_datetime(merged["local_date"]).sort_values().reset_index(drop=True).equals(
        pd.Series(expected_dates)
    ):
        raise ValueError("P3 ERA5 weather dates differ from the frozen acquisition")
    if len(merged) != expected_count or set(merged["site_id"]) != {TARGET_SITE}:
        raise ValueError("P3 ERA5 converted weather identity changed")
    return merged.sort_values("local_date").reset_index(drop=True)


def convert_era5_legacy_original_to_swap(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the original model3 ERA5-to-SWAP weather preparation."""
    data = frame.copy()
    temperature_c = data["temperature_mean_k"].astype(float) - 273.15
    dewpoint_c = data["dewpoint_k"].astype(float) - 273.15
    tmin = data["temperature_min_k"].astype(float) - 273.15
    tmax = data["temperature_max_k"].astype(float) - 273.15
    alpha = (
        17.27 * temperature_c / (237.7 + temperature_c)
        + np.log(temperature_c + 273.15)
    )
    beta = (
        17.27 * dewpoint_c / (237.7 + dewpoint_c)
        + np.log(dewpoint_c + 273.15)
    )
    relative_humidity_percent = np.exp(beta - alpha) * 100.0
    saturation_vapor_pressure = 0.61078 * np.exp(
        temperature_c
        / (temperature_c + 238.3)
        * 17.2694
    )
    result = data[["target_year", "site_id", "local_date"]].copy()
    result["solar_kj_m2_day"] = data["solar_j_m2_day"].astype(float) / 1000.0
    result["temperature_min_c"] = np.minimum(tmin, tmax * 0.95)
    result["temperature_max_c"] = tmax
    result["actual_vapor_pressure_kpa"] = np.maximum(
        0.0,
        saturation_vapor_pressure * (1.0 - relative_humidity_percent / 100.0),
    )
    result["wind_speed_m_s"] = np.maximum(
        0.0,
        np.hypot(data["wind_u_m_s"].astype(float), data["wind_v_m_s"].astype(float)),
    )
    result["precipitation_mm"] = np.maximum(
        0.0, data["precipitation_m"].astype(float) * 1000.0
    )
    result["etref_mm"] = np.maximum(
        0.0, -data["potential_evaporation_m"].astype(float) * 1000.0
    )
    result["weather_source"] = "ERA5_Land_corresponding_year_predecision"
    return result


def convert_raw_weather(frame: pd.DataFrame, conversion_id: str) -> pd.DataFrame:
    if conversion_id == "formal_dewpoint_actual_vapor_pressure_v1":
        result = convert_era5_to_swap(frame)
    elif conversion_id == "model3_original_humidity_temperature_mean_pipeline_v1":
        result = convert_era5_legacy_original_to_swap(frame)
    else:
        raise ValueError(f"unknown ERA5-to-SWAP conversion: {conversion_id}")
    numeric = [
        "solar_kj_m2_day",
        "temperature_min_c",
        "temperature_max_c",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "precipitation_mm",
        "etref_mm",
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"nonfinite weather values from {conversion_id}")
    return result.sort_values("local_date").reset_index(drop=True)


def read_legacy_swap_weather(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 11 or parts[0].strip("'") != "Weather":
            continue
        rows.append(
            {
                "local_date": pd.Timestamp(
                    year=int(parts[3]), month=int(parts[2]), day=int(parts[1])
                ).strftime("%Y-%m-%d"),
                "solar_kj_m2_day": float(parts[4]),
                "temperature_min_c": float(parts[5]),
                "temperature_max_c": float(parts[6]),
                "actual_vapor_pressure_kpa": float(parts[7]),
                "wind_speed_m_s": float(parts[8]),
                "precipitation_mm": float(parts[9]),
                "etref_mm": float(parts[10]),
            }
        )
    if not rows:
        raise ValueError(f"no SWAP weather rows found in {path}")
    return pd.DataFrame(rows)


def rounded_swap_values(weather: pd.DataFrame) -> pd.DataFrame:
    result = weather[
        [
            "local_date",
            "solar_kj_m2_day",
            "temperature_min_c",
            "temperature_max_c",
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "precipitation_mm",
            "etref_mm",
        ]
    ].copy()
    for column in result.columns[1:]:
        decimals = 2 if column == "actual_vapor_pressure_kpa" else 1
        result[column] = result[column].map(lambda value: float(f"{float(value):.{decimals}f}"))
    return result.sort_values("local_date").reset_index(drop=True)


def select_historical_conversion(
    historical_era5_root: Path, source_weather_path: Path
) -> tuple[str, dict[str, Any]]:
    raw = extract_p3_raw_weather(historical_era5_root / "era5_2019", 2019)
    observed = read_legacy_swap_weather(source_weather_path)
    observed = observed.loc[
        observed["local_date"].isin(raw["local_date"].astype(str))
    ].sort_values("local_date").reset_index(drop=True)
    if len(observed) != 364:
        raise ValueError("source P3 weather.019 does not contain the 364 historical days")
    candidates = (
        "formal_dewpoint_actual_vapor_pressure_v1",
        "model3_original_humidity_temperature_mean_pipeline_v1",
    )
    comparisons: dict[str, Any] = {}
    passed = []
    numeric = list(observed.columns[1:])
    for conversion_id in candidates:
        expected = rounded_swap_values(convert_raw_weather(raw, conversion_id))
        dates_match = expected["local_date"].equals(observed["local_date"])
        errors = {
            column: float((expected[column] - observed[column]).abs().max())
            for column in numeric
        }
        exact = bool(
            dates_match
            and all(
                np.array_equal(
                    expected[column].to_numpy(dtype=float),
                    observed[column].to_numpy(dtype=float),
                )
                for column in numeric
            )
        )
        comparisons[conversion_id] = {
            "dates_match": dates_match,
            "maximum_absolute_errors_at_swap_file_precision": errors,
            "exact_match": exact,
        }
        if exact:
            passed.append(conversion_id)
    if len(passed) != 1:
        raise ValueError(
            "historical ERA5-to-SWAP conversion parity did not identify exactly one "
            f"method: passed={passed} comparisons={comparisons}"
        )
    return passed[0], {
        "historical_year": 2019,
        "historical_rows": 364,
        "source_weather_path": str(source_weather_path),
        "selected_conversion_id": passed[0],
        "candidate_comparisons": comparisons,
        "exact_match_at_swap_file_precision": True,
    }


def legacy_weather_header(source_workspace: Path) -> list[str]:
    for name in ("WeatherOriginal.019", "weather.019"):
        path = source_workspace / name
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(
                keepends=True
            )[:11]
            if len(lines) == 11:
                return lines
    raise FileNotFoundError("P3 source workspace has no 2019 ERA5 weather header")


def write_legacy_swap_weather(
    path: Path, weather: pd.DataFrame, header: Sequence[str]
) -> None:
    lines = list(header)
    for row in weather.sort_values("local_date").itertuples(index=False):
        date = pd.Timestamp(row.local_date)
        lines.append(
            f" 'Weather'         {date.day}       {date.month}    {date.year}"
            f"      {float(row.solar_kj_m2_day):.1f}"
            f"      {float(row.temperature_min_c):.1f}"
            f"      {float(row.temperature_max_c):.1f}"
            f"      {float(row.actual_vapor_pressure_kpa):.2f}"
            f"      {float(row.wind_speed_m_s):.1f}"
            f"      {float(row.precipitation_mm):.1f}"
            f"      {float(row.etref_mm):.1f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def validate_source_workspace(path: Path) -> dict[str, Path]:
    required = {
        "irrigation_helper": path / "real_ir_update.py",
        "historical_weather": path / "weather.019",
    }
    for label, item in required.items():
        if not item.is_file():
            raise FileNotFoundError(f"P3 source workspace {label} is missing: {item}")
    pristine_swp = path / "Swap1.pre_trunk_smoke.swp"
    if not pristine_swp.is_file():
        pristine_swp = path / "Swap1.swp"
    if not pristine_swp.is_file():
        raise FileNotFoundError(f"P3 pristine SWAP template is missing: {pristine_swp}")
    executable = choose_swap_executable(path)
    config_path = path / "site_config.json"
    formal_identity = tuple(part.lower() for part in path.parts[-3:])
    valid_formal_identity = formal_identity == ("y2019", "p3", "workspace")
    formal_setup_audit = path.parent / "setup_audit_v1.json"
    if valid_formal_identity:
        if not formal_setup_audit.is_file():
            raise FileNotFoundError(
                f"frozen P3 formal setup audit is missing: {formal_setup_audit}"
            )
        setup = json.loads(formal_setup_audit.read_text(encoding="utf-8"))
        if not (
            setup.get("mandatory_gate_passed") is True
            and setup.get("site_id") == TARGET_SITE
            and int(setup.get("target_year", -1)) == 2019
        ):
            raise ValueError("frozen formal setup audit is not the passed P3/2019 unit")
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("paper_site_id") != TARGET_SITE and not valid_formal_identity:
            raise ValueError("source workspace site_config is not P3")
    else:
        if path.name != "code_N3_Maize" and not valid_formal_identity:
            raise ValueError(
                "source workspace without site_config must be code_N3_Maize or "
                "the frozen units/Y2019/P3/workspace"
            )
    result = {**required, "swp": pristine_swp, "executable": executable}
    if valid_formal_identity:
        result["formal_setup_audit"] = formal_setup_audit
    return result


def validate_schedule(schedule: pd.DataFrame, crop: pd.DataFrame) -> dict[str, Any]:
    required = {
        "site_id",
        "target_year",
        "split",
        "schedule_index",
        "state_checkpoint_date",
        "state_dvs",
        "decision_date",
        "horizon_end_date",
        "harvest_date",
        "dvs_threshold",
        "sampling_interval_days",
        "horizon_days",
    }
    if missing := sorted(required - set(schedule.columns)):
        raise ValueError(f"P3 schedule is missing columns: {missing}")
    ordered = schedule.sort_values("schedule_index").reset_index(drop=True)
    decision = pd.to_datetime(ordered["decision_date"], errors="raise")
    checkpoint = pd.to_datetime(ordered["state_checkpoint_date"], errors="raise")
    horizon_end = pd.to_datetime(ordered["horizon_end_date"], errors="raise")
    harvest = pd.to_datetime(ordered["harvest_date"], errors="raise")
    state_dvs = pd.to_numeric(ordered["state_dvs"], errors="raise")
    gate = bool(
        len(ordered) >= 8
        and set(ordered["site_id"].astype(str)) == {TARGET_SITE}
        and set(ordered["target_year"].astype(int)) == {TARGET_YEAR}
        and set(ordered["split"].astype(str)) == {"independent_test"}
        and ordered["schedule_index"].astype(int).tolist() == list(range(len(ordered)))
        and decision.dt.year.eq(TARGET_YEAR).all()
        and (decision - checkpoint).dt.days.eq(1).all()
        and (horizon_end - decision).dt.days.eq(6).all()
        and horizon_end.le(harvest).all()
        and state_dvs.ge(float(SCHEDULE_RULE["dvs_threshold"])).all()
        and state_dvs.lt(2.0).all()
        and ordered["dvs_threshold"].astype(float).eq(0.1).all()
        and ordered["sampling_interval_days"].astype(int).eq(7).all()
        and ordered["horizon_days"].astype(int).eq(7).all()
        and (decision.diff().dropna().dt.days == 7).all()
    )
    daily = crop[["Date", "DVS"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["DVS"] = pd.to_numeric(daily["DVS"], errors="raise")
    first_eligible = daily.loc[daily["DVS"].ge(0.1), "Date"].iloc[0]
    if not gate or decision.iloc[0] != first_eligible + pd.Timedelta(days=1):
        raise ValueError("P3 schedule differs from the frozen DVS rule")
    return {
        "decision_rows": int(len(ordered)),
        "first_decision_date": decision.iloc[0].strftime("%Y-%m-%d"),
        "last_decision_date": decision.iloc[-1].strftime("%Y-%m-%d"),
        "minimum_state_dvs": float(state_dvs.min()),
        "maximum_state_dvs": float(state_dvs.max()),
        "all_decisions_pre_maturity": bool(state_dvs.lt(2.0).all()),
    }


def build_manifest_rows(paths: Sequence[tuple[str, Path]], root: Path) -> pd.DataFrame:
    rows = []
    for role, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"manifest output is missing: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = str(path)
        rows.append(
            {
                "role": role,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    era5_root = args.era5_root.resolve()
    historical_era5_root = args.historical_era5_root.resolve()
    acquisition_protocol_dir = args.acquisition_protocol_dir.resolve()
    independent_protocol_dir = args.independent_protocol_dir.resolve()
    source_workspace = args.source_workspace.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen trunk output: {output_dir}")

    independent = validate_independent_protocol(independent_protocol_dir)
    acquisition_protocol = validate_acquisition_protocol(acquisition_protocol_dir)
    acquisition = validate_download_archive(era5_root)
    source_files = validate_source_workspace(source_workspace)
    conversion_id, historical_conversion_parity = select_historical_conversion(
        historical_era5_root, source_files["historical_weather"]
    )
    raw_2022_weather = extract_p3_raw_weather(era5_root, TARGET_YEAR)
    weather = convert_raw_weather(raw_2022_weather, conversion_id)

    unit_root = output_dir / "units" / "Y2022" / TARGET_SITE
    workspace = unit_root / "workspace"
    trunk_dir = unit_root / "trunk"
    checkpoint_dir = unit_root / "all_checkpoints_v1"
    unit_root.mkdir(parents=True)
    shutil.copytree(source_workspace, workspace)
    shutil.copy2(source_files["swp"], workspace / "Swap1.swp")
    copied_backup = workspace / "Swap1.pre_trunk_smoke.swp"
    if copied_backup.exists():
        copied_backup.unlink()
    copy_formal_dependencies(workspace)

    header = legacy_weather_header(source_workspace)
    weather_path = workspace / "weather.022"
    original_weather_path = workspace / "WeatherOriginal.022"
    write_legacy_swap_weather(weather_path, weather, header)
    shutil.copy2(weather_path, original_weather_path)
    trunk_dir.mkdir()
    converted_weather_path = trunk_dir / "gefs_p3_2022_era5_swap_weather_v1.csv"
    weather.to_csv(converted_weather_path, index=False)

    run_logged(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "run_swap_season_trunk_smoke_v1.py"),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(trunk_dir),
            "--site-id",
            TARGET_SITE,
            "--year",
            str(TARGET_YEAR),
            "--split",
            "independent_test",
            "--sowing-month-day",
            str(SCHEDULE_RULE["sowing_month_day"]),
            "--harvest-month-day",
            str(SCHEDULE_RULE["harvest_month_day"]),
            "--output-prefix",
            "trunk2022",
            "--dvs-threshold",
            str(SCHEDULE_RULE["dvs_threshold"]),
            "--interval-days",
            str(SCHEDULE_RULE["interval_days"]),
            "--horizon-days",
            str(SCHEDULE_RULE["horizon_days"]),
        ],
        unit_root / "trunk_runner.log",
    )
    schedule_path = trunk_dir / "swap_season_decision_schedule_v1.csv"
    crop_path = workspace / "trunk2022.crp"
    schedule = pd.read_csv(schedule_path)
    crop = read_crop_trajectory(crop_path)
    schedule_summary = validate_schedule(schedule, crop)

    run_logged(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "simulation" / "audit_swap_season_checkpoint_equivalence_v1.py"),
            "--workspace",
            str(workspace),
            "--schedule",
            str(schedule_path),
            "--output-dir",
            str(checkpoint_dir),
            "--year",
            str(TARGET_YEAR),
            "--sowing-month-day",
            str(SCHEDULE_RULE["sowing_month_day"]),
            "--trunk-prefix",
            "trunk2022",
            "--all-checkpoints",
        ],
        unit_root / "checkpoint_runner.log",
    )
    checkpoint_csv = checkpoint_dir / "swap_season_checkpoint_equivalence_v1.csv"
    checkpoint_audit_path = (
        checkpoint_dir / "swap_season_checkpoint_equivalence_audit_v1.json"
    )
    checkpoints = pd.read_csv(checkpoint_csv)
    checkpoint_audit = json.loads(checkpoint_audit_path.read_text(encoding="utf-8"))
    checkpoint_gate = bool(
        len(checkpoints) == len(schedule)
        and boolean_mask(checkpoints["checkpoint_equivalence_passed"]).all()
        and float(checkpoints["maximum_absolute_crop_state_error"].max()) <= 1.0e-6
        and float(checkpoints["maximum_absolute_profile_state_error"].max()) <= 1.0e-6
        and checkpoint_audit.get("all_checkpoints_passed") is True
        and checkpoint_audit.get("all_scheduled_checkpoints_saved") is True
    )
    if not checkpoint_gate:
        raise RuntimeError("P3 2022 all-checkpoint equivalence gate failed")

    audit_path = output_dir / "gefs_p3_2022_era5_trunk_schedule_freeze_audit_v1.json"
    manifest_path = output_dir / "gefs_p3_2022_era5_trunk_schedule_freeze_manifest_v1.csv"
    audit = {
        "status": "p3_2022_era5_zero_irrigation_trunk_and_schedule_frozen_pending_operational_gefs_acquisition",
        "mandatory_gate_passed": True,
        "target_site": TARGET_SITE,
        "target_year": TARGET_YEAR,
        "historical_trunk_weather_source": "ERA5_2015_2019",
        "test_trunk_weather_source": "ERA5_2022",
        "weather_conversion_and_rounding": "same_legacy_ERA5_to_SWAP_chain_as_2015_2019",
        "era5_to_swap_conversion_id": conversion_id,
        "historical_conversion_parity": historical_conversion_parity,
        "weather_rows": int(len(weather)),
        "weather_first_date": str(weather["local_date"].iloc[0]),
        "weather_last_date": str(weather["local_date"].iloc[-1]),
        "era5_raster_file_count": EXPECTED_RASTERS,
        "historical_parity_rows": EXPECTED_PARITY_ROWS,
        "historical_parity_all_passed": True,
        "trunk_irrigation_policy": "zero_irrigation",
        "swap_trunk_normal_completion": True,
        **schedule_summary,
        "all_eligible_cycles_retained": True,
        "manual_date_selection_performed": False,
        "decision_dates_selected_with_GEFS": False,
        "decision_dates_selected_with_SWAP_candidate_labels": False,
        "saved_checkpoint_count": int(len(checkpoints)),
        "all_checkpoint_equivalence_passed": True,
        "maximum_absolute_checkpoint_crop_state_error": float(
            checkpoints["maximum_absolute_crop_state_error"].max()
        ),
        "maximum_absolute_checkpoint_profile_state_error": float(
            checkpoints["maximum_absolute_profile_state_error"].max()
        ),
        "target_columns_loaded": [],
        "P3_2022_target_labels_read": 0,
        "GEFS_rows_read": 0,
        "GEFS_download_performed": False,
        "network_access_performed_by_this_stage": False,
        "SWAP_candidate_label_rows_read": 0,
        "SWAP_candidate_label_generation_performed": False,
        "model_checkpoint_files_loaded": 0,
        "model_training_performed": False,
        "model_inference_performed": False,
        "model_or_threshold_reselection_performed": False,
        "P3_2023_rows_or_labels_read": 0,
        "P3_2024_rows_or_labels_read": 0,
        "frozen_component_hashes": EXPECTED_COMPONENT_HASHES,
        "source_workspace": str(source_workspace),
        "source_workspace_identity": "passed_formal_units_Y2019_P3_setup_audit",
        "auxiliary_site_config_used_for_identity": False,
        "source_workspace_pristine_swp_sha256": sha256_file(source_files["swp"]),
        "weather_file_sha256": sha256_file(weather_path),
        "schedule_sha256": sha256_file(schedule_path),
        "checkpoint_equivalence_sha256": sha256_file(checkpoint_csv),
        "next_gate": "acquire_operational_GEFSv12_2022_for_frozen_decision_dates",
    }
    write_json(audit_path, audit)

    manifest_inputs = [
        ("input_independent_protocol", independent["paths"]["protocol"]),
        ("input_independent_components", independent["paths"]["components"]),
        ("input_independent_stages", independent["paths"]["stages"]),
        ("input_independent_audit", independent["paths"]["audit"]),
        ("input_independent_manifest", independent["paths"]["manifest"]),
        ("input_acquisition_contract", acquisition_protocol["contract"]),
        ("input_acquisition_protocol_audit", acquisition_protocol["audit"]),
        ("input_acquisition_protocol_manifest", acquisition_protocol["manifest"]),
        ("input_download_audit", acquisition["audit"]),
        ("input_download_manifest", acquisition["manifest"]),
        ("input_source_swap1", source_files["swp"]),
        ("input_source_irrigation_helper", source_files["irrigation_helper"]),
        ("input_source_swap_executable", source_files["executable"]),
        ("input_source_historical_weather", source_files["historical_weather"]),
        ("output_converted_weather", converted_weather_path),
        ("output_swap_weather", weather_path),
        ("output_swap_weather_original", original_weather_path),
        ("output_trunk_crop", crop_path),
        ("output_schedule", schedule_path),
        ("output_trunk_audit", trunk_dir / "swap_season_trunk_smoke_audit_v1.json"),
        ("output_checkpoint_equivalence", checkpoint_csv),
        ("output_checkpoint_audit", checkpoint_audit_path),
        ("output_stage_audit", audit_path),
    ]
    if "formal_setup_audit" in source_files:
        manifest_inputs.append(
            ("input_source_formal_setup_audit", source_files["formal_setup_audit"])
        )
    for restart in sorted((checkpoint_dir / "checkpoints").glob("*/result_forec.end")):
        manifest_inputs.append(("output_verified_restart_checkpoint", restart))
    manifest = build_manifest_rows(manifest_inputs, PROJECT_ROOT)
    manifest.to_csv(manifest_path, index=False)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return {
        "audit": audit_path,
        "manifest": manifest_path,
        "schedule": schedule_path,
        "checkpoint_equivalence": checkpoint_csv,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--historical-era5-root", type=Path, required=True)
    parser.add_argument("--acquisition-protocol-dir", type=Path, required=True)
    parser.add_argument("--independent-protocol-dir", type=Path, required=True)
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)

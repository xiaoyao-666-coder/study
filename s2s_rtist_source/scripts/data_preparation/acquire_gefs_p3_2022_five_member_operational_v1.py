#!/usr/bin/env python3
"""Acquire and freeze the five-member operational GEFS input for P3 2022."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scripts.data_preparation import (
    acquire_gefs_p3_2021_five_member_operational_v1 as legacy,
)
from scripts.simulation.run_gefs_p3_2022_era5_trunk_schedule_freeze_v1 import (
    validate_independent_protocol,
)
from scripts.simulation import (
    run_gefs_p3_2022_era5_trunk_schedule_freeze_v1 as trunk_validator,
)
from s2s_rtist.weather import gefs_gridmet_bias as weather_core


REQUIRED_MESSAGES = weather_core.REQUIRED_MESSAGES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_SITE = "P3"
TARGET_YEAR = 2022
EXPECTED_CYCLES = 13
OPERATIONAL_MEMBERS = ("gec00", "gep01", "gep02", "gep03", "gep04")
LEAD_HOURS = tuple(range(3, 181, 3))
MEMBER_DAY_ROWS_PER_CYCLE = len(OPERATIONAL_MEMBERS) * 7
DEFAULT_PROTOCOL_DIR = Path(
    "site_general_surrogate_eval/gefs_p3_2022_b2_independent_protocol_v1"
)
DEFAULT_TRUNK_DIR = Path(
    "site_general_surrogate_eval/gefs_p3_2022_era5_trunk_schedule_freeze_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "site_general_surrogate_eval/gefs_p3_2022_five_member_operational_acquisition_v1"
)
EXPECTED_TRUNK_SHA256 = {
    "audit": "1a8646bef9c7d2ebfafda9dfaabd6764681cf8c63af10be89055f86df4bdf480",
    "manifest": "abd9df75c161301a811bd377a9fcb3a4814da2f212a631176e6eafb54ce0a6f8",
    "schedule": "0c35285a77c55a467bebb9326e620e42a89e9c96d384c514ae24fb5491bf738a",
    "checkpoint_equivalence": "2845f88aa38b8d993dce961825c959cbb367297fa1b8ddc39fc3cc182a3c454a",
    "checkpoint_audit": "6d3e55c71cb365e90b0a88aad641239aa73ef91bff1ad142504828f0b076a169",
}

DownloadTask = legacy.DownloadTask
task_paths = legacy.task_paths
validate_cached_task = legacy.validate_cached_task
acquire_task = legacy.acquire_task
task_manifest_row = legacy.task_manifest_row
atomic_write_frame = legacy.atomic_write_frame
atomic_write_json = legacy.atomic_write_json
read_json = legacy.read_json
require_file = legacy.require_file
sha256_file = legacy.sha256_file
write_json = legacy.write_json


def strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def require_exact_hashes(paths: dict[str, Path], expected: dict[str, str]) -> None:
    if set(paths) != set(expected):
        raise ValueError("frozen predecessor hash roles changed")
    for role, path in paths.items():
        require_file(path, f"frozen predecessor {role}")
        if sha256_file(path) != expected[role]:
            raise ValueError(f"frozen predecessor SHA256 changed: {role}")


def validate_weather_contract(weather: dict[str, Any]) -> None:
    expected_variables = [
        "precipitation_mm",
        "temperature_min_c",
        "temperature_max_c",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "solar_kj_m2_day",
    ]
    gate = bool(
        weather.get("archive") == "NOAA_GEFS_operational_public_archive"
        and weather.get("contract_id")
        == "gefs-p3-2022-frozen-operational-weather-v1"
        and int(weather.get("cycle_hour_utc", -1)) == 0
        and weather.get("forecast_horizon")
        == "local_decision_date_D_through_D_plus_6"
        and int(weather.get("member_count", -1)) == len(OPERATIONAL_MEMBERS)
        and weather.get("members") == list(OPERATIONAL_MEMBERS)
        and weather.get("model_version") == "GEFSv12"
        and weather.get("product_grid") == "atmos/pgrb2sp25_0p25_degree"
        and weather.get("timezone_conversion") == "America/Chicago_local_day"
        and weather.get("server_external_weather_download_allowed") is False
        and weather.get("network_access_performed") is False
        and weather.get("derived_daily_variables") == expected_variables
        and weather.get("precipitation_correction", {}).get(
            "2022_reference_weather_used_for_fit"
        )
        is False
        and weather.get("precipitation_correction", {}).get("fit_period")
        == "2000-2019"
        and weather.get("nonprecipitation_correction", {}).get(
            "2022_reference_weather_used_for_fit"
        )
        is False
    )
    if not gate:
        raise ValueError("frozen operational weather contract changed")


def validate_protocol_for_acquisition(root: Path) -> dict[str, Any]:
    validated = validate_independent_protocol(root)
    protocol = validated["protocol"]
    validate_weather_contract(protocol.get("operational_weather", {}))
    stages = pd.read_csv(validated["paths"]["stages"])
    expected_stage_ids = [
        "protocol_freeze",
        "local_ERA5_2022_acquisition",
        "server_ERA5_zero_irrigation_trunk_and_schedule",
        "local_operational_GEFSv12_five_member_acquisition",
        "frozen_weather_correction",
        "B2_and_fixed_expert_recommendation_freeze",
        "shared_fixed_eight_SWAP_generation",
        "single_independent_evaluation",
    ]
    if not (
        stages["stage_order"].astype(int).tolist() == list(range(1, 9))
        and stages["stage_id"].astype(str).tolist() == expected_stage_ids
        and strict_bool(protocol.get("sealed_reserves", {}).get("P3_2023_replication"))
        and strict_bool(protocol.get("sealed_reserves", {}).get("P3_2024_TTA_workflow"))
    ):
        raise ValueError("frozen stage order or sealed reserve changed")
    return validated


def manifest_binds_file(
    manifest: pd.DataFrame,
    *,
    role: str,
    path: Path,
    relative_suffix: str,
) -> bool:
    matches = manifest.loc[manifest["role"].astype(str).eq(role)]
    expected_hash = sha256_file(path)
    expected_size = path.stat().st_size
    suffix = relative_suffix.replace("\\", "/")
    for row in matches.itertuples(index=False):
        row_path = str(row.path).replace("\\", "/")
        if (
            row_path.endswith(suffix)
            and int(row.bytes) == expected_size
            and str(row.sha256) == expected_hash
        ):
            return True
    return False


def validate_manifest_output_files(
    trunk_dir: Path, manifest: pd.DataFrame
) -> dict[str, Path]:
    root = trunk_dir.resolve()
    marker = f"{trunk_dir.name}/"
    outputs: dict[str, Path] = {}
    output_rows = manifest.loc[manifest["role"].astype(str).str.startswith("output_")]
    if output_rows.empty or output_rows[["role", "path"]].duplicated().any():
        raise ValueError("trunk output manifest is empty or duplicated")
    for index, row in enumerate(output_rows.itertuples(index=False)):
        manifest_path = str(row.path).replace("\\", "/")
        if marker not in manifest_path:
            raise ValueError(f"trunk output path is outside frozen root: {manifest_path}")
        relative = manifest_path.split(marker, 1)[1]
        if not relative or ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError(f"unsafe trunk output path: {manifest_path}")
        path = (root / Path(relative)).resolve()
        if root not in path.parents:
            raise ValueError(f"trunk output path escaped frozen root: {manifest_path}")
        label = (
            "restart checkpoint"
            if str(row.role) == "output_verified_restart_checkpoint"
            else "trunk manifest output"
        )
        require_file(path, label)
        if int(row.bytes) != path.stat().st_size or str(row.sha256) != sha256_file(path):
            raise ValueError(f"trunk manifest output hash changed: {manifest_path}")
        outputs[f"manifest_output_{index:02d}"] = path
    return outputs


def validate_schedule(trunk_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = {
        "audit": trunk_dir
        / "gefs_p3_2022_era5_trunk_schedule_freeze_audit_v1.json",
        "manifest": trunk_dir
        / "gefs_p3_2022_era5_trunk_schedule_freeze_manifest_v1.csv",
        "schedule": trunk_dir
        / "units/Y2022/P3/trunk/swap_season_decision_schedule_v1.csv",
        "checkpoint_equivalence": trunk_dir
        / "units/Y2022/P3/all_checkpoints_v1/swap_season_checkpoint_equivalence_v1.csv",
        "checkpoint_audit": trunk_dir
        / "units/Y2022/P3/all_checkpoints_v1/swap_season_checkpoint_equivalence_audit_v1.json",
    }
    require_exact_hashes(paths, EXPECTED_TRUNK_SHA256)
    audit = read_json(paths["audit"])
    checkpoint_audit = read_json(paths["checkpoint_audit"])
    manifest = pd.read_csv(paths["manifest"])
    validate_manifest_output_files(trunk_dir, manifest)
    schedule = pd.read_csv(paths["schedule"])
    checkpoints = pd.read_csv(paths["checkpoint_equivalence"])
    required_schedule = {
        "site_id",
        "target_year",
        "split",
        "schedule_index",
        "state_checkpoint_date",
        "state_dvs",
        "decision_date",
        "horizon_end_date",
        "horizon_days",
    }
    if missing := sorted(required_schedule - set(schedule.columns)):
        raise ValueError(f"P3 2022 schedule is missing columns: {missing}")
    ordered = schedule.sort_values("schedule_index").reset_index(drop=True)
    decisions = pd.to_datetime(ordered["decision_date"], errors="raise")
    checkpoints_at = pd.to_datetime(ordered["state_checkpoint_date"], errors="raise")
    horizons = pd.to_datetime(ordered["horizon_end_date"], errors="raise")
    state_dvs = pd.to_numeric(ordered["state_dvs"], errors="raise")
    checkpoint_passed = checkpoints["checkpoint_equivalence_passed"].map(strict_bool)
    gate = bool(
        audit.get("status")
        == "p3_2022_era5_zero_irrigation_trunk_and_schedule_frozen_pending_operational_gefs_acquisition"
        and audit.get("mandatory_gate_passed") is True
        and audit.get("target_site") == TARGET_SITE
        and int(audit.get("target_year", -1)) == TARGET_YEAR
        and int(audit.get("decision_rows", -1)) == EXPECTED_CYCLES
        and audit.get("first_decision_date") == decisions.iloc[0].strftime("%Y-%m-%d")
        and audit.get("last_decision_date") == decisions.iloc[-1].strftime("%Y-%m-%d")
        and audit.get("all_decisions_pre_maturity") is True
        and audit.get("all_eligible_cycles_retained") is True
        and audit.get("manual_date_selection_performed") is False
        and int(audit.get("saved_checkpoint_count", -1)) == EXPECTED_CYCLES
        and audit.get("all_checkpoint_equivalence_passed") is True
        and float(audit.get("maximum_absolute_checkpoint_crop_state_error", -1.0))
        == 0.0
        and float(audit.get("maximum_absolute_checkpoint_profile_state_error", -1.0))
        == 0.0
        and audit.get("historical_parity_all_passed") is True
        and audit.get("historical_conversion_parity", {}).get(
            "exact_match_at_swap_file_precision"
        )
        is True
        and int(audit.get("GEFS_rows_read", -1)) == 0
        and audit.get("GEFS_download_performed") is False
        and int(audit.get("SWAP_candidate_label_rows_read", -1)) == 0
        and audit.get("SWAP_candidate_label_generation_performed") is False
        and int(audit.get("P3_2022_target_labels_read", -1)) == 0
        and int(audit.get("P3_2023_rows_or_labels_read", -1)) == 0
        and int(audit.get("P3_2024_rows_or_labels_read", -1)) == 0
        and audit.get("target_columns_loaded") == []
        and int(audit.get("model_checkpoint_files_loaded", -1)) == 0
        and audit.get("model_inference_performed") is False
        and audit.get("model_training_performed") is False
        and audit.get("model_or_threshold_reselection_performed") is False
        and audit.get("network_access_performed_by_this_stage") is False
        and audit.get("next_gate")
        == "acquire_operational_GEFSv12_2022_for_frozen_decision_dates"
        and len(ordered) == EXPECTED_CYCLES
        and set(ordered["site_id"].astype(str)) == {TARGET_SITE}
        and set(ordered["target_year"].astype(int)) == {TARGET_YEAR}
        and set(ordered["split"].astype(str)) == {"independent_test"}
        and ordered["schedule_index"].astype(int).tolist()
        == list(range(EXPECTED_CYCLES))
        and decisions.dt.year.eq(TARGET_YEAR).all()
        and decisions.diff().dropna().dt.days.eq(7).all()
        and (decisions - checkpoints_at).dt.days.eq(1).all()
        and (horizons - decisions).dt.days.eq(6).all()
        and ordered["horizon_days"].astype(int).eq(7).all()
        and state_dvs.ge(0.1).all()
        and state_dvs.lt(2.0).all()
        and len(checkpoints) == EXPECTED_CYCLES
        and checkpoint_passed.all()
        and float(checkpoints["maximum_absolute_crop_state_error"].max()) == 0.0
        and float(checkpoints["maximum_absolute_profile_state_error"].max()) == 0.0
        and checkpoint_audit.get("status")
        == "all_season_checkpoints_generated_and_verified"
        and int(checkpoint_audit.get("target_year", -1)) == TARGET_YEAR
        and int(checkpoint_audit.get("scheduled_checkpoint_count", -1))
        == EXPECTED_CYCLES
        and int(checkpoint_audit.get("tested_checkpoint_count", -1))
        == EXPECTED_CYCLES
        and checkpoint_audit.get("all_checkpoints_passed") is True
        and checkpoint_audit.get("all_scheduled_checkpoints_saved") is True
        and checkpoint_audit.get("formal_label_generation_allowed") is False
        and audit.get("schedule_sha256") == sha256_file(paths["schedule"])
        and audit.get("checkpoint_equivalence_sha256")
        == sha256_file(paths["checkpoint_equivalence"])
    )
    if not gate:
        raise ValueError("P3 2022 frozen trunk or schedule contract changed")

    role_suffixes = {
        "output_schedule": "units/Y2022/P3/trunk/swap_season_decision_schedule_v1.csv",
        "output_checkpoint_equivalence": "units/Y2022/P3/all_checkpoints_v1/swap_season_checkpoint_equivalence_v1.csv",
        "output_checkpoint_audit": "units/Y2022/P3/all_checkpoints_v1/swap_season_checkpoint_equivalence_audit_v1.json",
        "output_stage_audit": "gefs_p3_2022_era5_trunk_schedule_freeze_audit_v1.json",
    }
    role_paths = {
        "output_schedule": paths["schedule"],
        "output_checkpoint_equivalence": paths["checkpoint_equivalence"],
        "output_checkpoint_audit": paths["checkpoint_audit"],
        "output_stage_audit": paths["audit"],
    }
    for role, path in role_paths.items():
        if not manifest_binds_file(
            manifest, role=role, path=path, relative_suffix=role_suffixes[role]
        ):
            raise ValueError(f"trunk manifest binding changed: {role}")

    for decision in decisions:
        compact = decision.strftime("%Y%m%d")
        relative = (
            f"units/Y2022/P3/all_checkpoints_v1/checkpoints/{compact}/"
            "result_forec.end"
        )
        restart = trunk_dir / relative
        require_file(restart, f"restart checkpoint {compact}")
        if not manifest_binds_file(
            manifest,
            role="output_verified_restart_checkpoint",
            path=restart,
            relative_suffix=relative,
        ):
            raise ValueError(f"restart checkpoint manifest binding changed: {compact}")
        paths[f"restart_{compact}"] = restart

    ordered["decision_date"] = decisions.dt.strftime("%Y-%m-%d")
    return ordered, paths


def build_tasks(decision_dates: Sequence[str]) -> list[DownloadTask]:
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in decision_dates]
    if (
        len(dates) != EXPECTED_CYCLES
        or len(set(dates)) != EXPECTED_CYCLES
        or any(pd.Timestamp(value).year != TARGET_YEAR for value in dates)
    ):
        raise ValueError("formal acquisition requires 13 unique decision dates in 2022")
    tasks = [
        DownloadTask(decision_date, member, lead_hour)
        for decision_date in dates
        for member in OPERATIONAL_MEMBERS
        for lead_hour in LEAD_HOURS
    ]
    expected = EXPECTED_CYCLES * len(OPERATIONAL_MEMBERS) * len(LEAD_HOURS)
    if len(tasks) != expected or len({task.task_id for task in tasks}) != expected:
        raise ValueError("formal GEFS download task matrix is incomplete")
    return tasks


def assemble_point_records(output_dir: Path, tasks: Sequence[DownloadTask]) -> pd.DataFrame:
    return legacy.assemble_point_records(output_dir, tasks)


def assemble_daily_member_weather(points: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for member, group in points.groupby("gefs_member", sort=False):
        daily = legacy.aggregate_gefs_point_records(group)
        daily["gefs_member"] = str(member)
        parts.append(daily)
    output = pd.concat(parts, ignore_index=True)
    output["decision_date"] = pd.to_datetime(output["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    output["local_date"] = pd.to_datetime(output["local_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    keys = ["decision_date", "gefs_member", "lead_day"]
    required_weather = {
        "precipitation_mm",
        "temperature_min_c",
        "temperature_max_c",
        "shortwave_w_m2",
        "wind_speed_m_s",
        "vpd_kpa",
    }
    if missing := sorted(required_weather - set(output.columns)):
        raise ValueError(f"daily GEFS weather is missing variables: {missing}")
    if output[list(required_weather)].isna().any().any():
        raise ValueError("daily GEFS weather contains missing values")
    counts = output.groupby("decision_date").size()
    expected_rows = EXPECTED_CYCLES * MEMBER_DAY_ROWS_PER_CYCLE
    if (
        len(output) != expected_rows
        or output.duplicated(keys).any()
        or not counts.eq(MEMBER_DAY_ROWS_PER_CYCLE).all()
        or set(output["gefs_member"].astype(str)) != set(OPERATIONAL_MEMBERS)
        or set(output["lead_day"].astype(int)) != set(range(1, 8))
    ):
        raise ValueError("daily GEFS member weather failed the 35-row cycle contract")
    return output.sort_values(keys).reset_index(drop=True)


def write_top_manifest(
    output_dir: Path,
    *,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
) -> None:
    rows = []
    for role, path in inputs.items():
        rows.append(
            {
                "role": f"input_{role}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for role, code_path in (
        ("code", Path(__file__).resolve()),
        ("code_legacy_acquisition_core", Path(legacy.__file__).resolve()),
        ("code_weather_core", Path(weather_core.__file__).resolve()),
        ("code_trunk_validator", Path(trunk_validator.__file__).resolve()),
    ):
        rows.append(
            {
                "role": role,
                "path": code_path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": code_path.stat().st_size,
                "sha256": sha256_file(code_path),
            }
        )
    for role, path in outputs.items():
        if role != "manifest":
            rows.append(
                {
                    "role": f"output_{role}",
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_write_frame(outputs["manifest"], pd.DataFrame(rows))


def run(args: argparse.Namespace) -> dict[str, Path]:
    if os.name != "nt":
        raise RuntimeError(
            "formal GEFS acquisition must run on the local Windows workstation; "
            "server-side external downloads are forbidden"
        )
    if args.workers < 1 or args.timeout < 1 or args.retries < 1:
        raise ValueError("workers, timeout, and retries must be positive")
    protocol = validate_protocol_for_acquisition(args.protocol_dir)
    schedule, trunk_paths = validate_schedule(args.trunk_dir)
    tasks = build_tasks(schedule["decision_date"].tolist())
    output_dir = args.output_dir
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists; use --resume: {output_dir}")
    if not output_dir.exists() and args.resume:
        raise FileNotFoundError(f"--resume output directory does not exist: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    outputs = {
        "contract": output_dir
        / "gefs_p3_2022_five_member_operational_acquisition_contract_v1.json",
        "task_manifest": output_dir
        / "gefs_p3_2022_five_member_download_task_manifest_v1.csv",
        "points": output_dir / "gefs_p3_2022_five_member_point_records_v1.csv",
        "daily": output_dir
        / "gefs_p3_2022_five_member_uncorrected_daily_weather_v1.csv",
        "audit": output_dir
        / "gefs_p3_2022_five_member_operational_acquisition_audit_v1.json",
        "manifest": output_dir
        / "gefs_p3_2022_five_member_operational_acquisition_manifest_v1.csv",
    }
    if outputs["audit"].exists():
        raise FileExistsError("formal acquisition is already complete; refusing to alter it")

    results: list[dict[str, Any]] = []
    pending: list[DownloadTask] = []
    for task in tasks:
        cached = validate_cached_task(output_dir, task)
        if cached is None:
            pending.append(task)
        else:
            results.append(cached)
    print(
        f"validated formal acquisition tasks={len(tasks)} cached={len(results)} "
        f"pending={len(pending)} members={len(OPERATIONAL_MEMBERS)} "
        f"leads={len(LEAD_HOURS)}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                acquire_task,
                output_dir,
                task,
                timeout=args.timeout,
                retries=args.retries,
                keep_grib=args.keep_grib,
            ): task
            for task in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            metadata = future.result()
            results.append(metadata)
            if completed == 1 or completed % 20 == 0 or completed == len(pending):
                partial = pd.DataFrame(task_manifest_row(item) for item in results)
                partial = partial.sort_values(
                    ["decision_date", "gefs_member", "lead_hour"]
                )
                atomic_write_frame(
                    output_dir
                    / "gefs_p3_2022_five_member_download_task_manifest_partial_v1.csv",
                    partial,
                )
                print(
                    f"acquisition_progress completed_new={completed}/{len(pending)} "
                    f"total_verified={len(results)}/{len(tasks)} last={task.task_id}",
                    flush=True,
                )

    task_manifest = pd.DataFrame(task_manifest_row(item) for item in results)
    task_manifest = task_manifest.sort_values(
        ["decision_date", "gefs_member", "lead_hour"]
    ).reset_index(drop=True)
    if (
        len(task_manifest) != len(tasks)
        or task_manifest["task_id"].duplicated().any()
        or set(task_manifest["gefs_member"].astype(str)) != set(OPERATIONAL_MEMBERS)
        or set(task_manifest["lead_hour"].astype(int)) != set(LEAD_HOURS)
    ):
        raise ValueError("final GEFS task manifest is incomplete")
    atomic_write_frame(outputs["task_manifest"], task_manifest)
    points = assemble_point_records(output_dir, tasks)
    atomic_write_frame(outputs["points"], points)
    daily = assemble_daily_member_weather(points)
    atomic_write_frame(outputs["daily"], daily)
    contract = {
        "contract_id": "gefs-p3-2022-five-member-operational-acquisition-v1",
        "source": "NOAA_GEFS_operational_AWS_open_data",
        "cycle_hour_utc": 0,
        "decision_dates": schedule["decision_date"].tolist(),
        "members": list(OPERATIONAL_MEMBERS),
        "lead_hours": list(LEAD_HOURS),
        "selected_messages": [list(item) for item in REQUIRED_MESSAGES],
        "target_site": legacy.P3_SITE.iloc[0].to_dict(),
        "logical_member_day_rows_per_cycle": MEMBER_DAY_ROWS_PER_CYCLE,
        "formal_weather_stage": "uncorrected_operational_GEFS_point_values",
        "post_acquisition_model_gate_threshold_or_date_changes_allowed": False,
        "weather_bias_correction_allowed_in_this_stage": False,
        "SWAP_candidate_label_access_allowed_in_this_stage": False,
        "P3_2023_access_allowed": False,
        "P3_2024_access_allowed": False,
    }
    write_json(outputs["contract"], contract)
    audit = {
        "status": "p3_2022_five_member_operational_GEFS_acquisition_complete_pending_frozen_weather_correction",
        "mandatory_gate_passed": True,
        "target_site": TARGET_SITE,
        "target_year": TARGET_YEAR,
        "decision_cycle_count": EXPECTED_CYCLES,
        "formal_members": list(OPERATIONAL_MEMBERS),
        "formal_member_count": len(OPERATIONAL_MEMBERS),
        "nonformal_member_count": 0,
        "lead_hours_per_member_cycle": len(LEAD_HOURS),
        "download_task_count": len(task_manifest),
        "index_file_count": len(tasks),
        "point_task_file_count": len(tasks),
        "selected_messages_per_task": len(REQUIRED_MESSAGES),
        "point_record_rows": len(points),
        "uncorrected_member_day_rows": len(daily),
        "member_day_rows_per_cycle": MEMBER_DAY_ROWS_PER_CYCLE,
        "total_payload_bytes_fetched": int(task_manifest["payload_bytes"].sum()),
        "payload_hash_count": int(task_manifest["payload_sha256"].nunique()),
        "raw_GRIB_retained": bool(task_manifest["grib_retained"].astype(bool).all()),
        "raw_GRIB_retained_task_count": int(
            task_manifest["grib_retained"].astype(bool).sum()
        ),
        "download_execution_environment": "local_Windows_workstation",
        "server_external_network_download_allowed": False,
        "network_access_performed_this_run": bool(pending),
        "acquisition_contains_network_data": True,
        "independent_protocol_hashes_read_but_not_modified": True,
        "frozen_trunk_schedule_checkpoint_hashes_verified": True,
        "ERA5_weather_rows_read_by_this_stage": 0,
        "model_checkpoint_files_loaded": 0,
        "model_inference_performed": False,
        "model_training_performed": False,
        "model_or_threshold_reselection_performed": False,
        "weather_bias_correction_performed": False,
        "P3_2022_target_labels_read": 0,
        "P3_2023_rows_or_labels_read": 0,
        "P3_2024_rows_or_labels_read": 0,
        "SWAP_candidate_labels_read": 0,
        "target_columns_loaded": [],
        "irrigation_recommendations_generated": False,
        "manifest_all_task_hashes_present": bool(
            task_manifest[["index_sha256", "payload_sha256", "point_sha256"]]
            .notna()
            .all()
            .all()
        ),
        "next_gate": "apply_only_frozen_pre_2022_weather_correction_without_target_reference",
    }
    write_json(outputs["audit"], audit)
    write_top_manifest(
        output_dir,
        inputs={
            **{f"protocol_{key}": value for key, value in protocol["paths"].items()},
            **{f"trunk_{key}": value for key, value in trunk_paths.items()},
        },
        outputs=outputs,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL_DIR)
    parser.add_argument("--trunk-dir", type=Path, default=DEFAULT_TRUNK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--keep-grib", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    generated = run(parse_args())
    for name, path in generated.items():
        print(f"{name}: {path}", flush=True)

#!/usr/bin/env python3
"""Run the bounded 2015-2019 scenario-consistent SWAP pilot on the server."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation import run_confirmed_5site_restart_generation_smoke_v1 as base
from s2s_rtist.validation.three_output_smoke import (
    validate_smoke_dataset,
    write_validation_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_scenario_consistent_pilot_contract_v1.json"
)
DEFAULT_HYBRID_WEATHER = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_pilot_server_inputs_v1"
    / "gefs_2015_2019_hybrid_weather_daily_v1.csv"
)
DEFAULT_CORRECTED_DAILY = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_pilot_server_inputs_v1"
    / "gefs_2015_2019_corrected_ensemble_daily_v1.csv"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_2015_2019_scenario_consistent_swap_pilot_v1"
)
WEATHER_FIELDS = [
    "solar_kj_m2_day",
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "precipitation_mm",
    "etref_mm",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weather_extension(year: int) -> str:
    return f".{int(year) % 1000:03d}"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != (
        "gefs-2015-2019-scenario-consistent-three-output-pilot-v1"
    ):
        raise ValueError("historical pilot contract id mismatch")
    if contract["split_policy"]["2024_use_allowed"] is not False:
        raise ValueError("historical pilot must forbid 2024")
    if contract["scope"]["surrogate_training_allowed"] is not False:
        raise ValueError("historical pilot must not train the surrogate")
    return contract


def selected_cycles(contract: dict[str, Any]) -> pd.DataFrame:
    cycles = pd.DataFrame(contract["selected_cycles"])
    cycles["target_year"] = cycles["target_year"].astype(int)
    cycles["decision_date"] = pd.to_datetime(cycles["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    cycles["decision_doy"] = pd.to_datetime(cycles["decision_date"]).dt.dayofyear
    cycles["date_t"] = pd.to_datetime(cycles["decision_date"]).dt.strftime("%d-%b-%Y")
    return cycles


def validate_crop_activity_gate(
    path: Path, contract: dict[str, Any]
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"crop activity gate CSV is missing: {path}")
    gate = pd.read_csv(path)
    required = {
        "target_year",
        "decision_date",
        "all_five_sites_screening_eligible",
        "uses_provisional_era5_future",
    }
    missing = required.difference(gate.columns)
    if missing:
        raise ValueError(f"crop activity gate CSV missing fields: {sorted(missing)}")
    gate["target_year"] = gate["target_year"].astype(int)
    gate["decision_date"] = pd.to_datetime(gate["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    selected = selected_cycles(contract)[["target_year", "decision_date"]]
    checked = selected.merge(
        gate[list(required)],
        on=["target_year", "decision_date"],
        how="left",
        validate="one_to_one",
    )
    if len(checked) != len(selected) or checked[
        "all_five_sites_screening_eligible"
    ].isna().any():
        raise ValueError("crop activity gate does not cover every selected cycle")
    eligible = checked["all_five_sites_screening_eligible"].astype(str).str.lower()
    provisional = checked["uses_provisional_era5_future"].astype(str).str.lower()
    if not eligible.eq("true").all():
        failed = checked.loc[eligible.ne("true"), "decision_date"].tolist()
        raise ValueError(f"selected cycles failed crop activity gate: {failed}")
    if provisional.eq("true").any():
        failed = checked.loc[provisional.eq("true"), "decision_date"].tolist()
        raise ValueError(
            "selected cycles require final crop recheck after corrected GEFS splice: "
            f"{failed}"
        )
    return checked


def load_hybrid_weather(
    path: Path, contract: dict[str, Any]
) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "target_year",
        "site_id",
        "local_date",
        "decision_date",
        "lead_day",
        "weather_source",
        *WEATHER_FIELDS,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"hybrid weather missing fields: {sorted(missing)}")
    data["target_year"] = data["target_year"].astype(int)
    data["local_date"] = pd.to_datetime(data["local_date"]).dt.strftime("%Y-%m-%d")
    if set(data["target_year"]) != {2015, 2016, 2017, 2018, 2019}:
        raise ValueError("hybrid weather year set mismatch")
    if set(data["site_id"].astype(str)) != set(contract["sites"]):
        raise ValueError("hybrid weather site set mismatch")
    if data[WEATHER_FIELDS].isna().any().any() or not np.isfinite(
        data[WEATHER_FIELDS].to_numpy(dtype=float)
    ).all():
        raise ValueError("hybrid weather contains nonfinite values")
    if (data[["solar_kj_m2_day", "actual_vapor_pressure_kpa", "wind_speed_m_s", "precipitation_mm"]] < 0).any().any():
        raise ValueError("hybrid weather contains negative physical values")
    if (data["temperature_min_c"] > data["temperature_max_c"]).any():
        raise ValueError("hybrid weather Tmin exceeds Tmax")
    cycles = selected_cycles(contract)
    future = data.loc[data["lead_day"].notna()].copy()
    if len(future) != 175:
        raise ValueError("hybrid weather must contain 175 future rows")
    if not future["weather_source"].eq(
        "GEFSv12_corrected_5member_ensemble_future"
    ).all():
        raise ValueError("future rows are not corrected GEFS")
    for row in cycles.itertuples(index=False):
        expected = pd.date_range(row.decision_date, periods=7, freq="D").strftime(
            "%Y-%m-%d"
        ).tolist()
        for site in contract["sites"]:
            group = future.loc[
                (future["target_year"] == row.target_year)
                & (future["site_id"].astype(str) == str(site))
            ]
            if group.sort_values("lead_day")["local_date"].tolist() != expected:
                raise ValueError(f"future dates incomplete for {row.target_year}/{site}")
    return data.sort_values(["target_year", "site_id", "local_date"]).reset_index(
        drop=True
    )


def write_weather_file(path: Path, weather: pd.DataFrame) -> None:
    header = [
        "*************************************************************************************************************************\n",
        "* GEFS 2015-2019 scenario-consistent pilot weather\n",
        "* ERA5 predecision state plus corrected GEFS seven-day future\n",
        "*************************************************************************************************************************\n",
        "*\n",
        "*\n",
        "*\n",
        "*************************************************************************************************************************\n",
        " Station      DD      MM    YYYY         RAD       Tmin      Tmax        HUM      WIND      RAIN     ETref\n",
        "*             nr      nr      nr       kJ/m2           C         C        kPa       m/s        mm        mm\n",
        "*************************************************************************************************************************\n",
    ]
    lines = list(header)
    for row in weather.sort_values("local_date").itertuples(index=False):
        date = pd.Timestamp(row.local_date)
        lines.append(
            f" 'Weather' {date.day:9d} {date.month:7d} {date.year:7d}"
            f" {float(row.solar_kj_m2_day):14.6f}"
            f" {float(row.temperature_min_c):12.6f}"
            f" {float(row.temperature_max_c):12.6f}"
            f" {float(row.actual_vapor_pressure_kpa):12.6f}"
            f" {float(row.wind_speed_m_s):12.6f}"
            f" {float(row.precipitation_mm):12.6f}"
            f" {float(row.etref_mm):12.6f}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def prepare_site_workspace(
    run_dir: Path,
    site: str,
    hybrid: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    source = base.source_workspace(site)
    workspace = run_dir / "workspaces" / site
    shutil.copytree(source, workspace)
    base.write_site_config(site, workspace, source)
    base.copy_generator_files(workspace)
    runner = base.RUNNER_SOURCE
    site_argument = '    parser.add_argument("--site", required=True)\n'
    if site_argument not in runner:
        raise ValueError("cannot locate site argument in generated runner template")
    runner = runner.replace(
        site_argument,
        site_argument + '    parser.add_argument("--year", type=int, required=True)\n',
        1,
    )
    swap_assignment = "    base.run_swap = run_swap\n"
    if swap_assignment not in runner:
        raise ValueError("cannot locate SWAP assignment in generated runner template")
    runner = runner.replace(
        swap_assignment,
        swap_assignment + "    base.YEAR = int(args.year)\n",
        1,
    )
    (workspace / "run_restart_smoke_one_site.py").write_text(
        runner, encoding="utf-8"
    )
    hash_rows = []
    for cycle in selected_cycles(contract).itertuples(index=False):
        weather = hybrid.loc[
            (hybrid["target_year"] == cycle.target_year)
            & (hybrid["site_id"].astype(str) == str(site))
        ].copy()
        suffix = weather_extension(cycle.target_year)
        active = workspace / f"weather{suffix}"
        original = workspace / f"WeatherOriginal{suffix}"
        write_weather_file(active, weather)
        shutil.copy2(active, original)
        weather.to_csv(
            workspace / f"hybrid_weather_{cycle.target_year}_{site}_v1.csv", index=False
        )
        hash_rows.append(
            {
                "site_id": site,
                "target_year": cycle.target_year,
                "weather_file": active.name,
                "weather_file_sha256": sha256_file(active),
                "weather_rows": len(weather),
                "gefs_future_rows": int(weather["lead_day"].notna().sum()),
            }
        )
    return {
        "site": site,
        "source_workspace": str(source),
        "run_workspace": str(workspace),
        "weather_hash_rows": hash_rows,
    }


def run_cycle_site(
    site: str,
    site_work: Path,
    year: int,
    decision: str,
    timeout: int,
    python_exe: str,
) -> dict[str, Any]:
    cmd = [
        python_exe,
        "run_restart_smoke_one_site.py",
        "--site",
        site,
        "--year",
        str(int(year)),
        "--decision-date",
        decision,
    ]
    env = base.prepend_server_runtime_library(dict(base.os.environ))
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        result = subprocess.run(
            cmd,
            cwd=site_work,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        status = "completed" if result.returncode == 0 else "returned_nonzero"
        returncode: str | int = result.returncode
        stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = ""
        stdout_tail = (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else ""
    dataset = site_work / "site_restart_generation_smoke.csv"
    best = site_work / "site_restart_generation_smoke_best_by_date.csv"
    return {
        "site": site,
        "run_workspace": str(site_work),
        "status": status,
        "returncode": returncode,
        "started_at": started_at,
        "decision_dates": decision,
        "candidate_rows": len(pd.read_csv(dataset)) if dataset.exists() else "",
        "best_rows": len(pd.read_csv(best)) if best.exists() else "",
        "dataset_csv": str(dataset) if dataset.exists() else "",
        "best_csv": str(best) if best.exists() else "",
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def add_provenance(
    frame: pd.DataFrame,
    cycle: pd.Series,
    site: str,
    corrected: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()
    weather = corrected.loc[
        (corrected["target_year"].astype(int) == int(cycle.target_year))
        & (corrected["site_id"].astype(str) == str(site))
    ]
    output["target_year"] = int(cycle.target_year)
    output["dataset_split"] = str(cycle.split)
    output["factor_fit_first_year"] = int(cycle.fit_first_year)
    output["factor_fit_last_year"] = int(cycle.fit_last_year)
    output["weather_driver_source"] = "ERA5_predecision_plus_corrected_GEFS_future"
    output["weather_label_scenario_consistent"] = True
    output["gefs_member_count"] = 5
    output["gefs_weekly_extreme_regime"] = bool(
        weather["weekly_extreme_regime"].iloc[0]
    )
    output["gefs_effective_factor"] = float(weather["effective_factor"].iloc[0])
    output["gefs_corrected_precipitation_7d_mm"] = float(
        weather["precipitation_mm_corrected_mean"].sum()
    )
    output["surrogate_training_performed"] = False
    output["tta_performed"] = False
    return output


def audit_results(
    combined: pd.DataFrame,
    corrected: pd.DataFrame,
    contract: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    rain_errors = []
    for (year, site), group in combined.groupby(["target_year", "site"], sort=True):
        expected = corrected.loc[
            (corrected["target_year"].astype(int) == int(year))
            & (corrected["site_id"].astype(str) == str(site)),
            "precipitation_mm_corrected_mean",
        ].sum()
        rain_errors.append(float((group["rain_7d_mm"].astype(float) - expected).abs().max()))
    primary = contract["labels"]["primary_outputs"]
    group_ranges = combined.groupby(["target_year", "site"])["cwdm_value"].agg(
        lambda values: float(values.max() - values.min())
    )
    best = combined.loc[combined["is_best_ir"].astype(bool)].copy()
    nonzero_years = int(
        best.groupby("target_year")["best_ir_for_date"]
        .max()
        .gt(0.0)
        .sum()
    )
    total_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    audit = {
        "candidate_rows": int(len(combined)),
        "site_cycle_count": int(
            combined[["target_year", "site"]].drop_duplicates().shape[0]
        ),
        "swap_invocations_expected": 225,
        "duplicate_sample_key_count": int(
            combined[["target_year", "site", "date_t", "ir"]].duplicated().sum()
        ),
        "primary_output_missing_value_count": int(combined[primary].isna().sum().sum()),
        "maximum_absolute_swap_rain_error_mm": float(max(rain_errors)),
        "maximum_absolute_water_balance_residual_mm": float(
            combined["water_balance_residual_0_100cm_7d_mm"].astype(float).abs().max()
        ),
        "irrigation_out_of_bounds_count": int(
            ((combined["ir"].astype(float) < 0) | (combined["ir"].astype(float) > 60)).sum()
        ),
        "future_target_year_rows_used_for_factor_fit": int(
            (combined["factor_fit_last_year"] >= combined["target_year"]).sum()
        ),
        "minimum_site_cycles_with_positive_cwdm_range": int((group_ranges > 0).sum()),
        "minimum_years_with_nonzero_best_irrigation": nonzero_years,
        "run_output_bytes": int(total_bytes),
        "run_output_gb": float(total_bytes / 1_000_000_000),
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    gate = contract["mandatory_gate"]
    mandatory_pass = (
        audit["candidate_rows"] == 200
        and audit["site_cycle_count"] == 25
        and audit["duplicate_sample_key_count"] == 0
        and audit["primary_output_missing_value_count"] == 0
        and audit["maximum_absolute_swap_rain_error_mm"]
        <= float(gate["maximum_absolute_swap_rain_error_mm"])
        and audit["maximum_absolute_water_balance_residual_mm"]
        <= float(gate["maximum_absolute_water_balance_residual_mm"])
        and audit["irrigation_out_of_bounds_count"] == 0
        and audit["future_target_year_rows_used_for_factor_fit"] == 0
        and audit["run_output_gb"]
        <= float(contract["resource_limits"]["maximum_server_output_gb"])
    )
    coverage = contract["response_coverage_gate"]
    coverage_pass = (
        audit["minimum_site_cycles_with_positive_cwdm_range"]
        >= int(coverage["minimum_site_cycles_with_positive_cwdm_range"])
        and audit["minimum_years_with_nonzero_best_irrigation"]
        >= int(coverage["minimum_years_with_nonzero_best_irrigation"])
    )
    audit["mandatory_gate_passed"] = bool(mandatory_pass)
    audit["response_coverage_gate_passed"] = bool(coverage_pass)
    if not mandatory_pass:
        audit["status"] = "mandatory_physical_gate_failed"
    elif coverage_pass:
        audit["status"] = "pilot_passed_ready_for_date_density_design_no_training_started"
    else:
        audit["status"] = "mandatory_gate_passed_response_coverage_failed_increase_date_density"
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--hybrid-weather", type=Path, default=DEFAULT_HYBRID_WEATHER)
    parser.add_argument("--corrected-daily", type=Path, default=DEFAULT_CORRECTED_DAILY)
    parser.add_argument("--crop-activity-gate-csv", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-per-cycle-site", type=int, default=1800)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    hybrid = load_hybrid_weather(args.hybrid_weather, contract)
    corrected = pd.read_csv(args.corrected_daily)
    if not args.prepare_only:
        if args.crop_activity_gate_csv is None:
            raise ValueError(
                "--crop-activity-gate-csv is required before formal SWAP generation"
            )
        validate_crop_activity_gate(args.crop_activity_gate_csv, contract)
    run_dir = args.run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    results_dir = run_dir / "cycle_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    weather_hash_rows = []
    for site in contract["sites"]:
        workspace = run_dir / "workspaces" / site
        if args.resume and workspace.is_dir():
            item = {
                "site": site,
                "source_workspace": str(base.source_workspace(site)),
                "run_workspace": str(workspace),
                "weather_hash_rows": [],
            }
        else:
            item = prepare_site_workspace(run_dir, site, hybrid, contract)
        prepared.append(item)
        weather_hash_rows.extend(item["weather_hash_rows"])
    weather_hash_path = run_dir / "prepared_hybrid_weather_files_v1.csv"
    if weather_hash_rows:
        pd.DataFrame(weather_hash_rows).to_csv(weather_hash_path, index=False)
    elif not weather_hash_path.is_file():
        raise ValueError("resume requested but prepared weather hash manifest is missing")
    pd.DataFrame(
        [{key: value for key, value in item.items() if key != "weather_hash_rows"} for item in prepared]
    ).to_csv(run_dir / "prepared_site_workspaces_v1.csv", index=False)

    if args.prepare_only:
        print(json.dumps({"run_dir": str(run_dir), "status": "prepared_only_no_SWAP_run"}, indent=2))
        return

    summary_rows = []
    candidate_frames = []
    cycles = selected_cycles(contract)
    for cycle in cycles.itertuples(index=False):
        decision = f"{cycle.date_t}:{int(cycle.decision_doy)}"
        for item in prepared:
            site = str(item["site"])
            result_copy = results_dir / f"{cycle.target_year}_{site}_candidates_v1.csv"
            if args.resume and result_copy.is_file() and len(pd.read_csv(result_copy)) == 8:
                frame = pd.read_csv(result_copy)
                candidate_frames.append(frame)
                summary_rows.append(
                    {"target_year": cycle.target_year, "site": site, "status": "resumed_completed", "candidate_rows": 8}
                )
                continue
            print(f"[PILOT] {cycle.target_year}/{site}: 1 state + 8 candidates", flush=True)
            summary = run_cycle_site(
                site=site,
                site_work=Path(item["run_workspace"]),
                year=int(cycle.target_year),
                decision=decision,
                timeout=args.timeout_per_cycle_site,
                python_exe=args.python,
            )
            summary["target_year"] = int(cycle.target_year)
            summary_rows.append(summary)
            if summary["status"] != "completed":
                pd.DataFrame(summary_rows).to_csv(run_dir / "pilot_run_summary_v1.csv", index=False)
                raise RuntimeError(f"SWAP run failed for {cycle.target_year}/{site}")
            frame = pd.read_csv(summary["dataset_csv"])
            cycle_series = pd.Series(cycle._asdict())
            frame = add_provenance(frame, cycle_series, site, corrected)
            frame.to_csv(result_copy, index=False)
            candidate_frames.append(frame)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "pilot_run_summary_v1.csv", index=False)
    combined = pd.concat(candidate_frames, ignore_index=True)
    formal = validate_smoke_dataset(
        combined,
        expected_irrigation_options_mm=contract["irrigation"]["candidate_values_mm"],
    )
    formal_summary, formal_report = write_validation_outputs(formal, run_dir)
    candidates_path = run_dir / "gefs_2015_2019_swap_candidate_labels_v1.csv"
    combined.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    audit = audit_results(combined, corrected, contract, run_dir)
    audit_path = run_dir / "gefs_2015_2019_swap_pilot_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": audit["status"],
        "contract_sha256": sha256_file(args.contract),
        "hybrid_weather_sha256": sha256_file(args.hybrid_weather),
        "corrected_daily_sha256": sha256_file(args.corrected_daily),
        "candidate_labels_sha256": sha256_file(candidates_path),
        "audit_sha256": sha256_file(audit_path),
        "formal_validation_summary": str(formal_summary),
        "formal_validation_report": str(formal_report),
        "network_download_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    manifest_path = run_dir / "gefs_2015_2019_swap_pilot_manifest_v1.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "candidates": str(candidates_path),
                "audit": str(audit_path),
                "manifest": str(manifest_path),
                "status": audit["status"],
            },
            indent=2,
        )
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError("historical pilot mandatory physical gate failed")


if __name__ == "__main__":
    main()

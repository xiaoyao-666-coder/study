#!/usr/bin/env python3
"""Run a bounded, scenario-consistent corrected-GEFS SWAP smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.simulation import run_confirmed_5site_restart_generation_smoke_v1 as base
from s2s_rtist.validation.three_output_smoke import (
    SmokeValidationError,
    validate_smoke_dataset,
    write_validation_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_corrected_swap_three_output_smoke_contract_v1.json"
)
DEFAULT_ENSEMBLE_DAILY = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "_local_gefs_corrected_surrogate_weather_smoke_v1"
    / "gefs_corrected_ensemble_daily_weather_smoke_v1.csv"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_corrected_swap_three_output_smoke_v1"
)
SWAP_WEATHER_FILES = ("weather.024", "WeatherOriginal.024")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-corrected-swap-three-output-smoke-v1":
        raise ValueError("corrected GEFS SWAP smoke contract id mismatch")
    bounds = contract["continuous_irrigation_constraint_mm"]
    if float(bounds["minimum"]) != 0.0 or float(bounds["maximum"]) != 60.0:
        raise ValueError("continuous irrigation constraint must remain 0-60 mm")
    if any(contract["scope"].values()):
        raise ValueError("corrected GEFS SWAP smoke permits a forbidden operation")
    policy = contract["result_policy"]
    if policy["weather_label_scenario_consistent_if_passed"] is not True:
        raise ValueError("passed smoke must be scenario consistent")
    if policy["training_eligible_if_passed"] is not False:
        raise ValueError("single-date smoke must remain ineligible for training")
    return contract


def load_gefs_daily(path: Path, contract: dict[str, Any]) -> pd.DataFrame:
    daily = pd.read_csv(path)
    required = {
        "site_id",
        "decision_date",
        "local_date",
        "lead_day",
        "member_count",
        "temperature_min_c_mean",
        "temperature_max_c_mean",
        "actual_vapor_pressure_kpa_mean",
        "wind_speed_m_s_mean",
        "solar_kj_m2_day_mean",
        "precipitation_mm_corrected_mean",
        "weekly_linear_scaling_factor",
        "artifact_sha256",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"corrected GEFS daily weather missing fields: {sorted(missing)}")
    daily["decision_date"] = pd.to_datetime(daily["decision_date"]).dt.strftime("%Y-%m-%d")
    daily["local_date"] = pd.to_datetime(daily["local_date"]).dt.strftime("%Y-%m-%d")
    if set(daily["site_id"].astype(str)) != set(contract["sites"]):
        raise ValueError("corrected GEFS daily site set mismatch")
    if set(daily["decision_date"]) != {contract["decision_date"]}:
        raise ValueError("corrected GEFS daily decision date mismatch")
    if len(daily) != len(contract["sites"]) * int(contract["horizon_days"]):
        raise ValueError("corrected GEFS daily row count mismatch")
    numeric = [
        "member_count",
        "temperature_min_c_mean",
        "temperature_max_c_mean",
        "actual_vapor_pressure_kpa_mean",
        "wind_speed_m_s_mean",
        "solar_kj_m2_day_mean",
        "precipitation_mm_corrected_mean",
    ]
    daily[numeric] = daily[numeric].apply(pd.to_numeric, errors="coerce")
    if daily[numeric].isna().any().any() or not np.isfinite(
        daily[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("corrected GEFS daily weather contains nonfinite values")
    if not daily["member_count"].eq(int(contract["gefs_member_count"])).all():
        raise ValueError("corrected GEFS member count mismatch")
    if (daily["temperature_min_c_mean"] > daily["temperature_max_c_mean"]).any():
        raise ValueError("corrected GEFS Tmin exceeds Tmax")
    nonnegative = [
        "actual_vapor_pressure_kpa_mean",
        "wind_speed_m_s_mean",
        "solar_kj_m2_day_mean",
        "precipitation_mm_corrected_mean",
    ]
    if (daily[nonnegative] < 0.0).any().any():
        raise ValueError("corrected GEFS daily weather contains negative physical fields")
    expected_dates = pd.date_range(
        contract["weather_replacement_policy"]["future_start"],
        contract["weather_replacement_policy"]["future_end"],
        freq="D",
    ).strftime("%Y-%m-%d").tolist()
    for site, group in daily.groupby("site_id", sort=True):
        if sorted(group["lead_day"].astype(int).tolist()) != list(
            range(1, int(contract["horizon_days"]) + 1)
        ):
            raise ValueError(f"corrected GEFS lead days incomplete for {site}")
        if sorted(group["local_date"].tolist()) != expected_dates:
            raise ValueError(f"corrected GEFS local dates incomplete for {site}")
    return daily.sort_values(["site_id", "lead_day"]).reset_index(drop=True)


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


def patch_swap_weather_file(path: Path, site_daily: pd.DataFrame) -> pd.DataFrame:
    replacements = site_daily.set_index("local_date")
    expected_dates = set(replacements.index.astype(str))
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines(
        keepends=True
    )
    audit_rows = []
    output = []
    for line in lines:
        old = parse_swap_weather_record(line)
        if old is None or old["date"] not in expected_dates:
            output.append(line)
            continue
        row = replacements.loc[old["date"]]
        output.append(format_swap_weather_record(old["date"], row, old["etref"]))
        audit_rows.append(
            {
                "patched_file": Path(path).name,
                "local_date": old["date"],
                "old_solar_kj_m2_day": old["solar_kj_m2_day"],
                "new_solar_kj_m2_day": float(row.solar_kj_m2_day_mean),
                "old_temperature_min_c": old["temperature_min_c"],
                "new_temperature_min_c": float(row.temperature_min_c_mean),
                "old_temperature_max_c": old["temperature_max_c"],
                "new_temperature_max_c": float(row.temperature_max_c_mean),
                "old_actual_vapor_pressure_kpa": old["actual_vapor_pressure_kpa"],
                "new_actual_vapor_pressure_kpa": float(
                    row.actual_vapor_pressure_kpa_mean
                ),
                "old_wind_speed_m_s": old["wind_speed_m_s"],
                "new_wind_speed_m_s": float(row.wind_speed_m_s_mean),
                "old_precipitation_mm": old["precipitation_mm"],
                "new_precipitation_mm": float(row.precipitation_mm_corrected_mean),
                "artifact_sha256": str(row.artifact_sha256),
            }
        )
    actual_dates = {row["local_date"] for row in audit_rows}
    if actual_dates != expected_dates or len(audit_rows) != len(expected_dates):
        raise ValueError(
            f"{path} patched dates={sorted(actual_dates)}, expected={sorted(expected_dates)}"
        )
    before_non_target = [
        line
        for line in lines
        if (record := parse_swap_weather_record(line)) is not None
        and record["date"] not in expected_dates
    ]
    after_non_target = [
        line
        for line in output
        if (record := parse_swap_weather_record(line)) is not None
        and record["date"] not in expected_dates
    ]
    if before_non_target != after_non_target:
        raise ValueError(f"{path} changed a non-target weather record")
    Path(path).write_text("".join(output), encoding="utf-8")
    return pd.DataFrame(audit_rows)


def patch_gridmet_file(path: Path, site_daily: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"gridMET file lacks Date: {path}")
    dates = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    replacements = site_daily.set_index("local_date")
    audit_rows = []
    mapping = {
        "Solar": "solar_kj_m2_day_mean",
        "T-max": "temperature_max_c_mean",
        "T-min": "temperature_min_c_mean",
        "RelHum": "actual_vapor_pressure_kpa_mean",
        "Precip": "precipitation_mm_corrected_mean",
        "WindSpeed": "wind_speed_m_s_mean",
    }
    missing = set(mapping).difference(frame.columns)
    if missing:
        raise ValueError(f"gridMET file missing columns: {sorted(missing)}")
    for local_date, replacement in replacements.iterrows():
        indices = frame.index[dates == local_date].tolist()
        if len(indices) != 1:
            raise ValueError(f"gridMET date {local_date} occurs {len(indices)} times")
        index = indices[0]
        audit = {
            "patched_file": Path(path).name,
            "local_date": local_date,
            "artifact_sha256": str(replacement.artifact_sha256),
        }
        for target, source in mapping.items():
            audit[f"old_{target}"] = float(frame.at[index, target])
            audit[f"new_{target}"] = float(replacement[source])
            frame.at[index, target] = float(replacement[source])
        audit_rows.append(audit)
    frame.to_csv(path, index=False)
    return pd.DataFrame(audit_rows)


def inject_site_weather(
    workspace: Path, site: str, site_daily: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, str]]:
    audit_frames = []
    hashes: dict[str, str] = {}
    for filename in SWAP_WEATHER_FILES:
        path = workspace / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing SWAP weather file: {path}")
        hashes[f"{filename}_before_sha256"] = sha256_file(path)
        audit_frames.append(patch_swap_weather_file(path, site_daily))
        hashes[f"{filename}_after_sha256"] = sha256_file(path)
    gridmet = workspace / "df_gridmet.csv"
    if not gridmet.is_file():
        raise FileNotFoundError(f"missing gridMET weather file: {gridmet}")
    hashes["df_gridmet.csv_before_sha256"] = sha256_file(gridmet)
    audit_frames.append(patch_gridmet_file(gridmet, site_daily))
    hashes["df_gridmet.csv_after_sha256"] = sha256_file(gridmet)
    audit = pd.concat(audit_frames, ignore_index=True)
    audit.insert(0, "site_id", site)
    if len(audit) != 3 * len(site_daily):
        raise ValueError(f"weather injection audit row count mismatch for {site}")
    audit.to_csv(
        workspace / "gefs_corrected_weather_injection_audit_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return audit, hashes


def add_provenance(
    frame: pd.DataFrame, site_daily: pd.DataFrame, contract: dict[str, Any]
) -> pd.DataFrame:
    output = frame.copy()
    site = str(output["site"].iloc[0])
    weather = site_daily[site_daily["site_id"].astype(str) == site]
    if len(weather) != int(contract["horizon_days"]):
        raise ValueError(f"provenance weather rows incomplete for {site}")
    output["weather_driver_source"] = contract["weather_scenario_mode"]
    output["weather_label_scenario_consistent"] = True
    output["training_eligible"] = False
    output["gefs_member_count"] = int(contract["gefs_member_count"])
    output["gefs_artifact_sha256"] = str(weather["artifact_sha256"].iloc[0])
    output["gefs_weekly_linear_scaling_factor"] = float(
        weather["weekly_linear_scaling_factor"].iloc[0]
    )
    output["gefs_corrected_precipitation_7d_mm"] = float(
        weather["precipitation_mm_corrected_mean"].sum()
    )
    return output


def audit_results(
    combined: pd.DataFrame,
    weather: pd.DataFrame,
    injection: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, Any]:
    if len(combined) != int(contract["expected_candidate_rows"]):
        raise ValueError("corrected GEFS SWAP candidate row count mismatch")
    if not combined["weather_label_scenario_consistent"].eq(True).all():
        raise ValueError("weather-label scenario consistency flag failed")
    if not combined["training_eligible"].eq(False).all():
        raise ValueError("single-date smoke must not be training eligible")
    expected_injection_rows = len(contract["sites"]) * int(contract["horizon_days"]) * 3
    if len(injection) != expected_injection_rows:
        raise ValueError("weather injection audit row count mismatch")
    rainfall_errors = []
    for site, group in combined.groupby("site", sort=True):
        expected = float(
            weather.loc[
                weather["site_id"].astype(str) == str(site),
                "precipitation_mm_corrected_mean",
            ].sum()
        )
        actual_values = group["rain_7d_mm"].astype(float)
        rainfall_errors.append(float((actual_values - expected).abs().max()))
    max_rain_error = max(rainfall_errors)
    if max_rain_error > float(contract["rainfall_sum_tolerance_mm"]):
        raise ValueError(
            f"SWAP rain differs from corrected GEFS input by {max_rain_error:.6f} mm"
        )
    maximum_residual = float(
        combined["water_balance_residual_0_100cm_7d_mm"].astype(float).abs().max()
    )
    if maximum_residual > float(
        contract["control_volume"]["maximum_absolute_water_balance_residual_mm"]
    ):
        raise ValueError("water balance residual exceeds prelocked smoke threshold")
    primary = contract["primary_outputs"]
    missing_primary = int(combined[primary].isna().sum().sum())
    if missing_primary:
        raise ValueError("scenario-consistent smoke primary outputs contain missing values")
    return {
        "contract_id": contract["contract_id"],
        "status": "corrected_gefs_swap_three_output_smoke_passed_scenario_consistent_not_training_dataset",
        "candidate_rows": int(len(combined)),
        "site_count": int(combined["site"].nunique()),
        "site_date_count": int(combined[["site", "date_t"]].drop_duplicates().shape[0]),
        "weather_injection_audit_rows": int(len(injection)),
        "swap_weather_file_patch_rows": int(
            injection["patched_file"].isin(SWAP_WEATHER_FILES).sum()
        ),
        "gridmet_file_patch_rows": int(
            injection["patched_file"].eq("df_gridmet.csv").sum()
        ),
        "predecision_weather_rows_modified": 0,
        "maximum_absolute_swap_rain_vs_gefs_error_mm": float(max_rain_error),
        "maximum_absolute_water_balance_residual_mm": maximum_residual,
        "primary_output_missing_value_count": missing_primary,
        "weather_label_scenario_consistent": True,
        "training_eligible": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--ensemble-daily", type=Path, default=DEFAULT_ENSEMBLE_DAILY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--timeout-per-site", type=int, default=1800)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    weather = load_gefs_daily(args.ensemble_daily, contract)
    run_dir = args.run_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.ensemble_daily, run_dir / "corrected_gefs_ensemble_daily_input_v1.csv")

    injection_frames = []
    prepared = []
    weather_hashes: dict[str, dict[str, str]] = {}
    for site in contract["sites"]:
        source = base.source_workspace(site)
        workspace = run_dir / site
        shutil.copytree(source, workspace)
        base.write_site_config(site, workspace, source)
        base.copy_generator_files(workspace)
        site_daily = weather[weather["site_id"].astype(str) == str(site)].copy()
        injection, hashes = inject_site_weather(workspace, site, site_daily)
        injection_frames.append(injection)
        weather_hashes[site] = hashes
        prepared.append(
            {
                "site": site,
                "source_workspace": str(source),
                "run_workspace": str(workspace),
                "patched_dates": len(site_daily),
                "artifact_sha256": str(site_daily["artifact_sha256"].iloc[0]),
            }
        )
    prepared_frame = pd.DataFrame(prepared)
    prepared_frame.to_csv(run_dir / "prepared_site_workspaces_v1.csv", index=False)
    injection_all = pd.concat(injection_frames, ignore_index=True)
    injection_path = run_dir / "gefs_corrected_weather_injection_audit_v1.csv"
    injection_all.to_csv(injection_path, index=False, encoding="utf-8-sig")

    decision = f"{contract['decision_date_swap']}:{int(contract['decision_doy'])}"
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir.resolve()),
                    "prepared": str((run_dir / "prepared_site_workspaces_v1.csv").resolve()),
                    "injection_audit": str(injection_path.resolve()),
                    "status": "prepared_only_no_SWAP_run",
                },
                indent=2,
            )
        )
        return

    summary_rows = []
    for item in prepared:
        print(f"[GEFS-SWAP] running {item['site']} (1 date x 8 candidates)", flush=True)
        summary_rows.append(
            base.run_site(
                site=item["site"],
                site_work=Path(item["run_workspace"]),
                decision_dates=[decision],
                timeout=args.timeout_per_site,
                python_exe=args.python,
                use_sampling_plan=False,
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "gefs_corrected_swap_site_run_summary_v1.csv"
    summary.to_csv(summary_path, index=False)
    incomplete = summary[summary["status"] != "completed"]
    if not incomplete.empty:
        failed = ", ".join(incomplete["site"].astype(str))
        raise RuntimeError(f"corrected GEFS SWAP site runs failed: {failed}")

    candidates = []
    for row in summary.itertuples(index=False):
        frame = pd.read_csv(row.dataset_csv)
        frame = add_provenance(frame, weather, contract)
        frame.to_csv(row.dataset_csv, index=False)
        candidates.append(frame)
    combined = pd.concat(candidates, ignore_index=True)
    formal = validate_smoke_dataset(
        combined,
        expected_irrigation_options_mm=contract["irrigation_candidates_mm"],
    )
    formal_summary, formal_report = write_validation_outputs(formal, run_dir)
    audit = audit_results(combined, weather, injection_all, contract)

    candidates_path = run_dir / "gefs_corrected_swap_three_output_candidates_v1.csv"
    audit_path = run_dir / "gefs_corrected_swap_three_output_audit_v1.json"
    manifest_path = run_dir / "gefs_corrected_swap_three_output_manifest_v1.json"
    report_path = run_dir / "gefs_corrected_swap_three_output_conclusion_v1.md"
    combined.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract),
        "corrected_gefs_ensemble_daily_input_sha256": sha256_file(args.ensemble_daily),
        "weather_file_hashes_by_site": weather_hashes,
        "candidate_file_sha256": sha256_file(candidates_path),
        "candidate_rows": int(len(combined)),
        "artifact_sha256_values": sorted(weather["artifact_sha256"].astype(str).unique()),
        "network_download_performed": False,
        "artifact_refit_performed": False,
        "hyperparameter_tuning_performed": False,
        "full_dataset_generation_performed": False,
        "surrogate_training_performed": False,
        "status": audit["status"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 订正 GEFS 驱动 SWAP 三输出 Smoke 结论

- 运行状态：通过
- 情景一致性：`weather_label_scenario_consistent=true`
- 候选样本：`{audit['candidate_rows']}` 行（5 站点 x 8 灌溉候选）
- 天气替换审计：`{audit['weather_injection_audit_rows']}` 行
- SWAP 天气文件替换：`{audit['swap_weather_file_patch_rows']}` 行
- gridMET 对照文件替换：`{audit['gridmet_file_patch_rows']}` 行
- 决策日前天气修改：`{audit['predecision_weather_rows_modified']}` 行
- SWAP 降水与订正 GEFS 7 天和最大误差：`{audit['maximum_absolute_swap_rain_vs_gefs_error_mm']:.6f} mm`
- 最大绝对水量平衡残差：`{audit['maximum_absolute_water_balance_residual_mm']:.6f} mm`
- 三输出缺失值：`{audit['primary_output_missing_value_count']}`

决策日前状态只使用原工作区历史天气；未来 7 天由冻结的订正 GEFS 31 成员日集合均值驱动。
因此本次天气输入与 SWAP 标签来自同一未来天气情景，解决了上一轮联调中的情景不一致问题。

但本次只有一个日期，`training_eligible=false`。本结果不代表已经生成训练集，也没有启动代理模型训练。

状态：`{audit['status']}`
"""
    report_path.write_text(report, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.resolve()),
                "candidates": str(candidates_path.resolve()),
                "injection_audit": str(injection_path.resolve()),
                "audit": str(audit_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "formal_validation_summary": str(formal_summary.resolve()),
                "formal_validation_report": str(formal_report.resolve()),
                "report": str(report_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

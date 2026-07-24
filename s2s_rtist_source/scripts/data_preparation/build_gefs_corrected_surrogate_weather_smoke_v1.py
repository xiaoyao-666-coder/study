#!/usr/bin/env python3
"""Build a small corrected-GEFS weather interface for surrogate integration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.fit_gefs_weekly_linear_final_artifact_v1 import artifact_hash
from scripts.diagnostics.run_gefs_weekly_linear_2024_diagnostic_v1 import apply_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_corrected_surrogate_weather_interface_contract_v1.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_final_artifact_server_v1"
    / "gefs_weekly_linear_final_artifact_2000_2019_v1.json"
)
DEFAULT_MEMBER_WEATHER = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_member_gridmet_validation_received_20260716"
    / "gefs_31member_1cycle_5site_20260716_v1"
    / "gefs_member_daily_weather.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_corrected_surrogate_weather_smoke_v1"
)
PASSTHROUGH = (
    "temperature_min_c",
    "temperature_max_c",
    "vpd_kpa",
    "wind_speed_m_s",
    "shortwave_w_m2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-corrected-surrogate-weather-interface-smoke-v1":
        raise ValueError("corrected surrogate weather interface contract id mismatch")
    if contract.get("candidate_id") != "weekly_two_stage_linear_site_factor_shrink_a075":
        raise ValueError("corrected surrogate weather candidate mismatch")
    if tuple(contract["pass_through_gefs_fields"]) != PASSTHROUGH:
        raise ValueError("corrected surrogate weather pass-through fields mismatch")
    irrigation = contract["irrigation_constraint_mm"]
    if float(irrigation["minimum"]) != 0.0 or float(irrigation["maximum"]) != 60.0:
        raise ValueError("continuous irrigation constraint must remain 0-60 mm")
    if any(contract["scope"].values()):
        raise ValueError("weather interface smoke permits a forbidden operation")
    return contract


def load_artifact(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_sha256") != artifact_hash(artifact):
        raise ValueError("weather interface artifact hash mismatch")
    if artifact.get("candidate_id") != contract["candidate_id"]:
        raise ValueError("weather interface artifact candidate mismatch")
    if artifact.get("2024_used_for_fit_or_selection") is not False:
        raise ValueError("weather interface artifact does not exclude 2024")
    return artifact


def actual_vapor_pressure_kpa(
    temperature_min_c: pd.Series,
    temperature_max_c: pd.Series,
    vpd_kpa: pd.Series,
) -> pd.Series:
    mean_temperature = 0.5 * (
        temperature_min_c.to_numpy(dtype=float)
        + temperature_max_c.to_numpy(dtype=float)
    )
    saturation = 0.6108 * np.exp(
        (17.27 * mean_temperature) / (mean_temperature + 237.3)
    )
    actual = np.maximum(0.0, saturation - vpd_kpa.to_numpy(dtype=float))
    return pd.Series(actual, index=temperature_min_c.index, dtype=float)


def to_swap_member_weather(corrected: pd.DataFrame) -> pd.DataFrame:
    output = corrected.copy()
    output["solar_kj_m2_day"] = output["shortwave_w_m2"].astype(float) * 86.4
    output["actual_vapor_pressure_kpa"] = actual_vapor_pressure_kpa(
        output["temperature_min_c"],
        output["temperature_max_c"],
        output["vpd_kpa"],
    )
    output["precipitation_mm_corrected"] = output["precipitation_mm_qm"]
    fields = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "local_date",
        "lead_day",
        "gefs_member",
        "temperature_min_c",
        "temperature_max_c",
        "vpd_kpa",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "shortwave_w_m2",
        "solar_kj_m2_day",
        "precipitation_mm_raw",
        "precipitation_mm_corrected",
        "ensemble_mean_raw_7d_mm",
        "raw_ensemble_mean_7d_q90_mm",
        "weekly_extreme_regime",
        "weekly_linear_scaling_factor",
        "factor_shrinkage_alpha",
        "artifact_sha256",
    ]
    return output[fields].sort_values(
        ["site_id", "decision_date", "gefs_member", "lead_day"]
    ).reset_index(drop=True)


def _quantile(series: pd.Series, probability: float) -> float:
    return float(np.quantile(series.to_numpy(dtype=float), probability))


def ensemble_daily(member: pd.DataFrame) -> pd.DataFrame:
    keys = ["site_id", "site_timezone", "decision_date", "local_date", "lead_day"]
    rows = []
    for key, group in member.groupby(keys, sort=True):
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "member_count": int(group["gefs_member"].nunique()),
                "temperature_min_c_mean": float(group["temperature_min_c"].mean()),
                "temperature_max_c_mean": float(group["temperature_max_c"].mean()),
                "actual_vapor_pressure_kpa_mean": float(
                    group["actual_vapor_pressure_kpa"].mean()
                ),
                "wind_speed_m_s_mean": float(group["wind_speed_m_s"].mean()),
                "solar_kj_m2_day_mean": float(group["solar_kj_m2_day"].mean()),
                "precipitation_mm_raw_mean": float(group["precipitation_mm_raw"].mean()),
                "precipitation_mm_corrected_mean": float(
                    group["precipitation_mm_corrected"].mean()
                ),
                "precipitation_mm_corrected_p10": _quantile(
                    group["precipitation_mm_corrected"], 0.1
                ),
                "precipitation_mm_corrected_p50": _quantile(
                    group["precipitation_mm_corrected"], 0.5
                ),
                "precipitation_mm_corrected_p90": _quantile(
                    group["precipitation_mm_corrected"], 0.9
                ),
                "weekly_extreme_regime": bool(group["weekly_extreme_regime"].iloc[0]),
                "weekly_linear_scaling_factor": float(
                    group["weekly_linear_scaling_factor"].iloc[0]
                ),
                "artifact_sha256": str(group["artifact_sha256"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["site_id", "decision_date", "lead_day"])


def surrogate_wide(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, decision_date), group in daily.groupby(
        ["site_id", "decision_date"], sort=True
    ):
        group = group.sort_values("lead_day")
        if list(group["lead_day"].astype(int)) != list(range(1, 8)):
            raise ValueError("surrogate wide group does not contain lead days 1-7")
        row: dict[str, Any] = {
            "site_id": site_id,
            "decision_date": pd.Timestamp(decision_date).strftime("%Y-%m-%d"),
            "horizon_days": 7,
            "gefs_member_count": int(group["member_count"].min()),
            "future_precip_raw_ensemble_mean_7d_mm": float(
                group["precipitation_mm_raw_mean"].sum()
            ),
            "future_precip_corrected_ensemble_mean_7d_mm": float(
                group["precipitation_mm_corrected_mean"].sum()
            ),
            "weekly_extreme_regime": bool(group["weekly_extreme_regime"].iloc[0]),
            "weekly_linear_scaling_factor": float(
                group["weekly_linear_scaling_factor"].iloc[0]
            ),
            "weather_source": "GEFS_31member_00UTC_frozen_precipitation_correction",
            "artifact_sha256": str(group["artifact_sha256"].iloc[0]),
        }
        sequence = []
        for day in group.itertuples(index=False):
            suffix = f"day{int(day.lead_day):02d}"
            values = {
                "solar_kj_m2_day_mean": float(day.solar_kj_m2_day_mean),
                "temperature_min_c_mean": float(day.temperature_min_c_mean),
                "temperature_max_c_mean": float(day.temperature_max_c_mean),
                "actual_vapor_pressure_kpa_mean": float(
                    day.actual_vapor_pressure_kpa_mean
                ),
                "wind_speed_m_s_mean": float(day.wind_speed_m_s_mean),
                "precipitation_mm_raw_mean": float(day.precipitation_mm_raw_mean),
                "precipitation_mm_corrected_mean": float(
                    day.precipitation_mm_corrected_mean
                ),
                "precipitation_mm_corrected_p10": float(
                    day.precipitation_mm_corrected_p10
                ),
                "precipitation_mm_corrected_p90": float(
                    day.precipitation_mm_corrected_p90
                ),
            }
            for name, value in values.items():
                row[f"future_{suffix}_{name}"] = value
            sequence.append({"lead_day": int(day.lead_day), **values})
        row["future_weather_ensemble_json"] = json.dumps(
            sequence, separators=(",", ":")
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site_id", "decision_date"])


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    artifact = load_artifact(args.artifact, contract)
    source = pd.read_csv(args.member_weather)
    required = {
        "site",
        "timezone",
        "cycle_init_utc",
        "decision_date",
        "local_date",
        "lead_day",
        "gefs_member",
        "precipitation_mm",
        *PASSTHROUGH,
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"member weather missing fields: {sorted(missing)}")
    source["decision_date"] = pd.to_datetime(source["decision_date"])
    source["local_date"] = pd.to_datetime(source["local_date"])
    source["cycle_init_utc"] = pd.to_datetime(source["cycle_init_utc"], utc=True)
    if set(source["decision_date"].dt.strftime("%Y-%m-%d")) != set(
        contract["smoke_decision_dates"]
    ):
        raise ValueError("weather interface smoke decision dates mismatch")
    if set(source["site"].astype(str)) != set(contract["expected_sites"]):
        raise ValueError("weather interface smoke site set mismatch")
    expected_rows = (
        len(contract["smoke_decision_dates"])
        * len(contract["expected_sites"])
        * int(contract["expected_member_count"])
        * int(contract["expected_horizon_days"])
    )
    if len(source) != expected_rows:
        raise ValueError(f"weather interface rows={len(source)}, expected={expected_rows}")
    if source[list(PASSTHROUGH)].isna().any().any():
        raise ValueError("weather interface pass-through fields contain missing values")
    minimal = source.rename(
        columns={
            "site": "site_id",
            "timezone": "site_timezone",
            "cycle_init_utc": "forecast_init_utc",
            "precipitation_mm": "precipitation_mm_raw",
        }
    ).copy()
    minimal["valid_date_utc"] = minimal["local_date"]
    corrected = apply_artifact(
        minimal,
        artifact,
        expected_member_count=int(contract["expected_member_count"]),
    )
    member = to_swap_member_weather(corrected)
    for column in PASSTHROUGH:
        if not np.array_equal(
            corrected[column].to_numpy(), minimal[column].to_numpy(), equal_nan=True
        ):
            raise ValueError(f"weather interface changed pass-through field {column}")
    if (member["temperature_min_c"] > member["temperature_max_c"]).any():
        raise ValueError("weather interface has minimum temperature above maximum")
    nonnegative = [
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "solar_kj_m2_day",
        "precipitation_mm_raw",
        "precipitation_mm_corrected",
    ]
    if (member[nonnegative] < 0.0).any().any():
        raise ValueError("weather interface contains a negative physical field")
    daily = ensemble_daily(member)
    wide = surrogate_wide(daily)
    if len(daily) != len(contract["expected_sites"]) * 7 or len(wide) != len(
        contract["expected_sites"]
    ):
        raise ValueError("weather interface output row count mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "member_daily": args.output_dir / "gefs_corrected_member_daily_weather_smoke_v1.csv",
        "ensemble_daily": args.output_dir / "gefs_corrected_ensemble_daily_weather_smoke_v1.csv",
        "surrogate_wide": args.output_dir / "gefs_corrected_surrogate_weather_wide_smoke_v1.csv",
        "manifest": args.output_dir / "gefs_corrected_surrogate_weather_manifest_smoke_v1.json",
        "report": args.output_dir / "gefs_corrected_surrogate_weather_conclusion_smoke_v1.md",
    }
    member.to_csv(paths["member_daily"], index=False, encoding="utf-8-sig")
    daily.to_csv(paths["ensemble_daily"], index=False, encoding="utf-8-sig")
    wide.to_csv(paths["surrogate_wide"], index=False, encoding="utf-8-sig")
    manifest = {
        "contract_id": contract["contract_id"],
        "candidate_id": artifact["candidate_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "source_file_sha256": sha256_file(args.member_weather),
        "member_daily_rows": int(len(member)),
        "ensemble_daily_rows": int(len(daily)),
        "surrogate_wide_rows": int(len(wide)),
        "pass_through_fields_unchanged": True,
        "irrigation_constraint_mm": contract["irrigation_constraint_mm"],
        "network_download_performed": False,
        "artifact_refit_performed": False,
        "surrogate_training_performed": False,
        "status": "corrected_gefs_surrogate_weather_interface_smoke_passed",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# 冻结 GEFS 订正接入代理模型天气接口 smoke",
        "",
        f"- 成员逐日行：`{len(member)}`",
        f"- 集合逐日行：`{len(daily)}`",
        f"- 代理模型宽表行：`{len(wide)}`",
        "- 非降水字段：`全部零改动`",
        "- GEFS VPD：`已转换为 SWAP 实际水汽压 HUM`",
        "- 短波单位：`W/m2 已转换为 kJ/m2/day`",
        "- 降水：`raw 与冻结订正值同时保留`",
        "- 连续灌溉约束：`0-60 mm`",
        "- 状态：`corrected_gefs_surrogate_weather_interface_smoke_passed`",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--member-weather", type=Path, default=DEFAULT_MEMBER_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

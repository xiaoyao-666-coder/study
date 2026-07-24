#!/usr/bin/env python3
"""Audit one historical GEFS cycle against the corresponding ERA5 weather."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_MEMBERS = {"c00", "p01", "p02", "p03", "p04"}
VARIABLES = (
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)
NONNEGATIVE_VARIABLES = (
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def saturation_vapor_pressure_kpa(temperature_c: pd.Series) -> pd.Series:
    values = temperature_c.to_numpy(dtype=float)
    return pd.Series(
        0.6108 * np.exp((17.27 * values) / (values + 237.3)),
        index=temperature_c.index,
        dtype=float,
    )


def normalize_era5_reference(frame: pd.DataFrame, site_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    canonical = {"local_date", *VARIABLES}
    if canonical.issubset(data.columns):
        if "site_id" in data.columns:
            data = data.loc[data["site_id"].astype(str) == site_id].copy()
        output = data[["local_date", *VARIABLES]].copy()
        metadata = {
            "era5_input_schema": "canonical_swap_weather",
            "era5_solar_conversion": "none_already_kj_m2_day",
            "era5_vapor_pressure_conversion": "none_already_actual_kpa",
        }
    else:
        required = {"Date", "Solar", "T-max", "T-min", "RelHum", "WindSpeed"}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"unsupported ERA5 schema; missing legacy columns: {missing}")
        relhum = pd.to_numeric(data["RelHum"], errors="coerce")
        if relhum.isna().any() or (relhum < 0.0).any():
            raise ValueError("legacy df_era RelHum must be finite and nonnegative")
        temperature_min = pd.to_numeric(data["T-min"], errors="coerce")
        temperature_max = pd.to_numeric(data["T-max"], errors="coerce")
        temperature_mean = (temperature_min + temperature_max) / 2.0
        if "ETref" in data.columns:
            actual_vapor_pressure = relhum
            era5_schema = "current_df_era_swap_weather"
            humidity_scale = "actual_vapor_pressure_kpa"
            vapor_pressure_conversion = "none_relhum_column_is_actual_vapor_pressure_kpa"
        elif "ET" in data.columns and (relhum <= 100.0).all():
            actual_vapor_pressure = (
                saturation_vapor_pressure_kpa(temperature_mean) * relhum / 100.0
            )
            era5_schema = "original_legacy_df_era_csv"
            humidity_scale = "relative_humidity_percent_0_to_100"
            vapor_pressure_conversion = (
                "saturation_vapor_pressure_at_daily_mean_temperature_times_relative_humidity"
            )
        else:
            raise ValueError(
                "cannot determine legacy df_era RelHum semantics from ETref/ET columns"
            )
        output = pd.DataFrame(
            {
                "local_date": pd.to_datetime(data["Date"], errors="raise"),
                "temperature_min_c": temperature_min,
                "temperature_max_c": temperature_max,
                "actual_vapor_pressure_kpa": actual_vapor_pressure,
                "wind_speed_m_s": pd.to_numeric(
                    data["WindSpeed"], errors="coerce"
                ),
                # The legacy extractor already writes daily radiation as kJ m-2.
                "solar_kj_m2_day": pd.to_numeric(data["Solar"], errors="coerce")
            }
        )
        metadata = {
            "era5_input_schema": era5_schema,
            "era5_solar_conversion": "none_already_kj_m2_day",
            "era5_relative_humidity_scale": humidity_scale,
            "era5_vapor_pressure_conversion": vapor_pressure_conversion,
        }

    output["local_date"] = pd.to_datetime(output["local_date"]).dt.strftime("%Y-%m-%d")
    output.insert(0, "site_id", site_id)
    if output["local_date"].duplicated().any():
        raise ValueError("ERA5 reference contains duplicate local dates")
    values = output[list(VARIABLES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ERA5 reference contains nonfinite canonical values")
    if (output["temperature_min_c"] > output["temperature_max_c"]).any():
        raise ValueError("ERA5 reference has Tmin greater than Tmax")
    if (output[list(NONNEGATIVE_VARIABLES)] < 0.0).any().any():
        raise ValueError("ERA5 reference has negative positive-only weather values")
    return output.sort_values("local_date").reset_index(drop=True), metadata


def build_gefs_ensemble_daily(
    member_weather: pd.DataFrame, site_id: str, decision_date: str
) -> pd.DataFrame:
    data = member_weather.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"]).dt.strftime("%Y-%m-%d")
    data["local_date"] = pd.to_datetime(data["local_date"]).dt.strftime("%Y-%m-%d")
    selected = data.loc[
        (data["site_id"].astype(str) == site_id)
        & (data["decision_date"] == decision_date)
    ].copy()
    member_keys = ["site_id", "decision_date", "gefs_member", "lead_day"]
    if len(selected) != 35 or selected[member_keys].duplicated().any():
        raise ValueError("GEFS smoke input must contain 35 unique member-day rows")
    if set(selected["gefs_member"].astype(str)) != EXPECTED_MEMBERS:
        raise ValueError("GEFS smoke input must contain the frozen five-member set")
    if sorted(selected["lead_day"].astype(int).unique()) != list(range(1, 8)):
        raise ValueError("GEFS smoke input must contain lead days 1-7")
    values = selected[list(VARIABLES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("GEFS member weather contains nonfinite canonical values")
    daily = (
        selected.groupby(
            ["site_id", "decision_date", "local_date", "lead_day"], as_index=False
        )
        .agg(
            **{variable: (variable, "mean") for variable in VARIABLES},
            member_count=("gefs_member", "nunique"),
        )
        .sort_values("lead_day")
        .reset_index(drop=True)
    )
    if len(daily) != 7 or not (daily["member_count"] == 5).all():
        raise ValueError("GEFS ensemble mean must contain seven five-member days")
    return daily


def pair_gefs_era5(
    gefs_daily: pd.DataFrame, era5_reference: pd.DataFrame
) -> pd.DataFrame:
    paired = gefs_daily.merge(
        era5_reference,
        on=["site_id", "local_date"],
        how="left",
        suffixes=("_gefs", "_era5"),
        validate="one_to_one",
    )
    expected_dates = pd.date_range(
        pd.Timestamp(paired["decision_date"].iloc[0]), periods=7, freq="D"
    ).strftime("%Y-%m-%d")
    if paired["local_date"].tolist() != expected_dates.tolist():
        raise ValueError("GEFS local dates do not match the seven-day decision horizon")
    era5_columns = [f"{variable}_era5" for variable in VARIABLES]
    if paired[era5_columns].isna().any().any():
        raise ValueError("ERA5 reference does not cover the complete GEFS horizon")
    for variable in VARIABLES:
        paired[f"{variable}_error_gefs_minus_era5"] = (
            paired[f"{variable}_gefs"] - paired[f"{variable}_era5"]
        )
    return paired


def build_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in VARIABLES:
        error = paired[f"{variable}_error_gefs_minus_era5"].to_numpy(dtype=float)
        rows.append(
            {
                "variable": variable,
                "paired_day_count": len(error),
                "raw_bias_gefs_minus_era5": float(np.mean(error)),
                "raw_mae": float(np.mean(np.abs(error))),
                "raw_rmse": float(np.sqrt(np.mean(np.square(error)))),
                "era5_mean": float(paired[f"{variable}_era5"].mean()),
                "gefs_ensemble_mean": float(paired[f"{variable}_gefs"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_audit(
    paired: pd.DataFrame, metrics: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, Any]:
    positive_scale_ratio = {}
    for variable in NONNEGATIVE_VARIABLES:
        era5_mean = float(paired[f"{variable}_era5"].mean())
        gefs_mean = float(paired[f"{variable}_gefs"].mean())
        positive_scale_ratio[variable] = (
            float(gefs_mean / era5_mean) if era5_mean > 0.0 else None
        )
    scale_ratio_outlier_count = sum(
        ratio is None or ratio < 0.05 or ratio > 20.0
        for ratio in positive_scale_ratio.values()
    )
    audit = {
        "status": "pending_gate_evaluation",
        "reference_dataset": "ERA5_corresponding_year",
        "site_id": str(paired["site_id"].iloc[0]),
        "decision_date": str(paired["decision_date"].iloc[0]),
        "paired_day_rows": int(len(paired)),
        "variable_pair_rows": int(len(paired) * len(VARIABLES)),
        "member_count_minimum": int(paired["member_count"].min()),
        "member_count_maximum": int(paired["member_count"].max()),
        "missing_value_count": int(paired.isna().sum().sum()),
        "contains_2024": bool(
            pd.to_datetime(paired["decision_date"]).dt.year.eq(2024).any()
        ),
        "gefs_tmin_greater_than_tmax_count": int(
            (paired["temperature_min_c_gefs"] > paired["temperature_max_c_gefs"]).sum()
        ),
        "era5_tmin_greater_than_tmax_count": int(
            (paired["temperature_min_c_era5"] > paired["temperature_max_c_era5"]).sum()
        ),
        "negative_positive_only_value_count": int(
            sum(
                (paired[f"{variable}_{source}"] < 0.0).sum()
                for variable in NONNEGATIVE_VARIABLES
                for source in ("gefs", "era5")
            )
        ),
        "positive_variable_gefs_to_era5_mean_ratio": positive_scale_ratio,
        "positive_variable_scale_ratio_outlier_count": int(
            scale_ratio_outlier_count
        ),
        "metric_variable_count": int(len(metrics)),
        "correction_fit_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        **metadata,
    }
    failures = []
    if len(paired) != 7:
        failures.append("paired_day_rows_not_7")
    if audit["variable_pair_rows"] != 35:
        failures.append("variable_pair_rows_not_35")
    for field in (
        "missing_value_count",
        "gefs_tmin_greater_than_tmax_count",
        "era5_tmin_greater_than_tmax_count",
        "negative_positive_only_value_count",
        "positive_variable_scale_ratio_outlier_count",
    ):
        if audit[field] != 0:
            failures.append(field)
    audit["gate_failures"] = failures
    audit["mandatory_gate_passed"] = not failures
    audit["status"] = (
        "gefs_era5_nonprecip_pair_smoke_passed"
        if not failures
        else "gefs_era5_nonprecip_pair_smoke_failed"
    )
    return audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    decision_date = pd.Timestamp(args.decision_date).strftime("%Y-%m-%d")
    gefs = build_gefs_ensemble_daily(
        pd.read_csv(args.gefs_member_weather), args.site_id, decision_date
    )
    era5, metadata = normalize_era5_reference(
        pd.read_csv(args.era5_weather), args.site_id
    )
    paired = pair_gefs_era5(gefs, era5)
    metrics = build_metrics(paired)
    audit = build_audit(paired, metrics, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "paired": args.output_dir / "gefs_era5_nonprecip_paired_daily_v1.csv",
        "metrics": args.output_dir / "gefs_era5_nonprecip_raw_metrics_v1.csv",
        "audit": args.output_dir / "gefs_era5_nonprecip_pair_audit_v1.json",
        "manifest": args.output_dir / "gefs_era5_nonprecip_pair_manifest_v1.json",
    }
    paired.to_csv(outputs["paired"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "gefs_member_weather": str(args.gefs_member_weather),
        "gefs_member_weather_sha256": sha256_file(args.gefs_member_weather),
        "era5_weather": str(args.era5_weather),
        "era5_weather_sha256": sha256_file(args.era5_weather),
        "outputs": {
            name: {"path": path.name, "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(
            "GEFS-ERA5 nonprecipitation pair smoke gate failed; "
            f"see {outputs['audit']}"
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gefs-member-weather", type=Path, required=True)
    parser.add_argument("--era5-weather", type=Path, required=True)
    parser.add_argument("--site-id", default="P1")
    parser.add_argument("--decision-date", default="2015-07-06")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

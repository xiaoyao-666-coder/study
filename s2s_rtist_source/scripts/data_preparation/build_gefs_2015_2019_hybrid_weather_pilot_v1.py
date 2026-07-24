#!/usr/bin/env python3
"""Build ERA5-predecision plus corrected-GEFS future weather for the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"
DEFAULT_CORRECTED = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_full_weather_pilot_local_v1"
) / "corrected_weather_v1" / "gefs_2015_2019_corrected_ensemble_daily_v1.csv"
DEFAULT_OUTPUT_DIR = Path(
    r"F:\s2s_rtist_source_data\gefs_2015_2019_full_weather_pilot_local_v1"
) / "hybrid_weather_v1"

SITE_COORDINATES = {
    "P1": (-98.224144, 42.015928),
    "P2": (-88.415, 40.595),
    "P3": (-96.877, 46.321),
    "P4": (-94.6686, 42.6816),
    "P15": (-112.265, 41.735),
}
VARIABLES = {
    "temperature_mean_k": "temperature_2m",
    "temperature_min_k": "temperature_2m_min",
    "temperature_max_k": "temperature_2m_max",
    "dewpoint_k": "dewpoint_temperature_2m",
    "solar_j_m2_day": "surface_solar_radiation_downwards_sum",
    "precipitation_m": "total_precipitation_sum",
    "potential_evaporation_m": "potential_evaporation_sum",
    "wind_u_m_s": "u_component_of_wind_10m",
    "wind_v_m_s": "v_component_of_wind_10m",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _day_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.tif$", path.name)
    if not match:
        raise ValueError(f"cannot parse zero-based day index from {path.name}")
    return int(match.group(1))


def extract_era5_variable(
    directory: Path, year: int, variable: str
) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform

    paths = sorted(directory.glob("*.tif"), key=_day_index)
    expected_days = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
    indices = [_day_index(path) for path in paths]
    # The uploaded ERA5 archive intentionally stops at Dec 30 in every year.
    # Pilot simulations end in July/August, so require a contiguous Jan 1-Dec 30
    # sequence and record the absent final calendar day rather than imputing it.
    accepted_indices = [list(range(expected_days)), list(range(expected_days - 1))]
    if indices not in accepted_indices:
        raise ValueError(
            f"{directory} has incomplete day indices: {len(indices)} vs {expected_days}"
        )
    site_ids = list(SITE_COORDINATES)
    lons = [SITE_COORDINATES[site][0] for site in site_ids]
    lats = [SITE_COORDINATES[site][1] for site in site_ids]
    rows = []
    transformed_cache: dict[str, tuple[list[float], list[float]]] = {}
    for path in paths:
        day_index = _day_index(path)
        with rasterio.open(path) as src:
            crs_key = str(src.crs)
            if crs_key not in transformed_cache:
                if src.crs is None or crs_key.upper() in {"EPSG:4326", "OGC:CRS84"}:
                    xs, ys = lons, lats
                else:
                    xs, ys = transform("EPSG:4326", src.crs, lons, lats)
                transformed_cache[crs_key] = (list(xs), list(ys))
            xs, ys = transformed_cache[crs_key]
            values = [float(value[0]) for value in src.sample(zip(xs, ys))]
        local_date = (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_index)).strftime(
            "%Y-%m-%d"
        )
        for site_id, value in zip(site_ids, values, strict=True):
            rows.append(
                {
                    "target_year": year,
                    "site_id": site_id,
                    "local_date": local_date,
                    variable: value,
                }
            )
    return pd.DataFrame(rows)


def extract_era5_year(era5_root: Path, year: int) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = ["target_year", "site_id", "local_date"]
    for variable, directory_name in VARIABLES.items():
        directory = era5_root / f"era5_{year}" / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"missing ERA5 directory: {directory}")
        frame = extract_era5_variable(directory, year, variable)
        merged = frame if merged is None else merged.merge(
            frame, on=keys, how="inner", validate="one_to_one"
        )
    if merged is None:
        raise RuntimeError("no ERA5 variables extracted")
    return convert_era5_to_swap(merged)


def saturation_vapor_pressure_kpa(temperature_c: pd.Series) -> pd.Series:
    values = temperature_c.to_numpy(dtype=float)
    return pd.Series(
        0.6108 * np.exp((17.27 * values) / (values + 237.3)),
        index=temperature_c.index,
    )


def convert_era5_to_swap(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["temperature_min_c"] = data["temperature_min_k"] - 273.15
    data["temperature_max_c"] = data["temperature_max_k"] - 273.15
    data["actual_vapor_pressure_kpa"] = saturation_vapor_pressure_kpa(
        data["dewpoint_k"] - 273.15
    )
    data["wind_speed_m_s"] = np.hypot(data["wind_u_m_s"], data["wind_v_m_s"])
    data["solar_kj_m2_day"] = data["solar_j_m2_day"] / 1000.0
    data["precipitation_mm"] = np.maximum(0.0, data["precipitation_m"] * 1000.0)
    data["etref_mm"] = np.maximum(
        0.0, -data["potential_evaporation_m"] * 1000.0
    )
    data["weather_source"] = "ERA5_Land_corresponding_year_predecision"
    output = [
        "target_year",
        "site_id",
        "local_date",
        "solar_kj_m2_day",
        "temperature_min_c",
        "temperature_max_c",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "precipitation_mm",
        "etref_mm",
        "weather_source",
    ]
    result = data[output].copy()
    numeric = output[3:-1]
    if result[numeric].isna().any().any() or not np.isfinite(
        result[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("ERA5 conversion contains nonfinite weather values")
    if (result["temperature_min_c"] > result["temperature_max_c"]).any():
        raise ValueError("ERA5 Tmin exceeds Tmax")
    return result


def splice_hybrid_weather(
    era5: pd.DataFrame, corrected_daily: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    future = corrected_daily.copy()
    future["local_date"] = pd.to_datetime(future["local_date"]).dt.strftime("%Y-%m-%d")
    future["decision_date"] = pd.to_datetime(future["decision_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    replacements = future.rename(
        columns={
            "solar_kj_m2_day_mean": "solar_kj_m2_day",
            "temperature_min_c_mean": "temperature_min_c",
            "temperature_max_c_mean": "temperature_max_c",
            "actual_vapor_pressure_kpa_mean": "actual_vapor_pressure_kpa",
            "wind_speed_m_s_mean": "wind_speed_m_s",
            "precipitation_mm_corrected_mean": "precipitation_mm",
        }
    )
    replacement_fields = [
        "solar_kj_m2_day",
        "temperature_min_c",
        "temperature_max_c",
        "actual_vapor_pressure_kpa",
        "wind_speed_m_s",
        "precipitation_mm",
    ]
    keys = ["target_year", "site_id", "local_date"]
    replacements = replacements[keys + ["decision_date", "lead_day", *replacement_fields]]
    hybrid = era5.merge(
        replacements,
        on=keys,
        how="left",
        suffixes=("_era5", "_gefs"),
        validate="one_to_one",
    )
    is_future = hybrid["lead_day"].notna()
    for field in replacement_fields:
        hybrid[field] = np.where(
            is_future, hybrid[f"{field}_gefs"], hybrid[f"{field}_era5"]
        )
    hybrid["weather_source"] = np.where(
        is_future,
        "GEFSv12_corrected_5member_ensemble_future",
        hybrid["weather_source"],
    )
    hybrid["lead_day"] = hybrid["lead_day"].astype("Int64")
    output_fields = [
        *keys,
        "decision_date",
        "lead_day",
        *replacement_fields,
        "etref_mm",
        "weather_source",
    ]
    output = hybrid[output_fields].sort_values(keys).reset_index(drop=True)

    future_output = output.loc[output["lead_day"].notna()].copy()
    joined = future_output.merge(
        replacements,
        on=keys,
        suffixes=("_output", "_expected"),
        validate="one_to_one",
    )
    errors = [
        np.max(
            np.abs(
                joined[f"{field}_output"].to_numpy(dtype=float)
                - joined[f"{field}_expected"].to_numpy(dtype=float)
            )
        )
        for field in replacement_fields
    ]
    predecision_gefs = output.loc[
        output["weather_source"].str.startswith("GEFS")
        & (pd.to_datetime(output["local_date"]) < pd.to_datetime(output["decision_date"]))
    ]
    audit = {
        "status": "era5_predecision_gefs_future_hybrid_passed",
        "total_rows": int(len(output)),
        "future_gefs_rows": int(is_future.sum()),
        "predecision_gefs_rows": int(len(predecision_gefs)),
        "maximum_absolute_future_splice_error": float(max(errors)),
        "target_year_2024_rows": int((output["target_year"] == 2024).sum()),
        "nonfinite_value_count": int(
            (~np.isfinite(output[replacement_fields].to_numpy(dtype=float))).sum()
        ),
        "future_site_cycle_count": int(
            future_output[["target_year", "site_id"]].drop_duplicates().shape[0]
        ),
        "era5_missing_final_calendar_day_by_design": True,
        "surrogate_training_started": False,
        "tta_started": False,
    }
    if audit["future_gefs_rows"] != 175 or audit["predecision_gefs_rows"] != 0:
        raise ValueError("hybrid future/predecision row gate failed")
    if audit["maximum_absolute_future_splice_error"] > 1e-12:
        raise ValueError("hybrid GEFS splice differs from corrected input")
    return output, audit


def run(args: argparse.Namespace) -> dict[str, Path]:
    frames = []
    for year in [2015, 2016, 2017, 2018, 2019]:
        print(f"[ERA5] extracting {year} for five sites", flush=True)
        frames.append(extract_era5_year(args.era5_root, year))
    era5 = pd.concat(frames, ignore_index=True)
    corrected = pd.read_csv(args.corrected_daily)
    hybrid, audit = splice_hybrid_weather(era5, corrected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "hybrid": args.output_dir / "gefs_2015_2019_hybrid_weather_daily_v1.csv",
        "audit": args.output_dir / "gefs_2015_2019_hybrid_weather_audit_v1.json",
        "manifest": args.output_dir
        / "gefs_2015_2019_hybrid_weather_manifest_v1.json",
    }
    hybrid.to_csv(outputs["hybrid"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "era5_root": str(args.era5_root),
        "corrected_daily_sha256": sha256_file(args.corrected_daily),
        "hybrid_weather_sha256": sha256_file(outputs["hybrid"]),
        "audit_sha256": sha256_file(outputs["audit"]),
        "network_download_performed": False,
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--corrected-daily", type=Path, default=DEFAULT_CORRECTED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

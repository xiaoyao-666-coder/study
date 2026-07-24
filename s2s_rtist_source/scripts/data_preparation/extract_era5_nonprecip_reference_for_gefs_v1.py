#!/usr/bin/env python3
"""Extract ERA5 nonprecipitation reference rows matching causal GEFS samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.data_preparation.build_gefs_2015_2019_hybrid_weather_pilot_v1 import (
    SITE_COORDINATES,
    saturation_vapor_pressure_kpa,
)


VARIABLE_DIRECTORIES = {
    "temperature_min_k": "temperature_2m_min",
    "temperature_max_k": "temperature_2m_max",
    "dewpoint_k": "dewpoint_temperature_2m",
    "solar_j_m2_day": "surface_solar_radiation_downwards_sum",
    "wind_u_m_s": "u_component_of_wind_10m",
    "wind_v_m_s": "v_component_of_wind_10m",
}
CANONICAL_FIELDS = (
    "temperature_min_c",
    "temperature_max_c",
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


def day_index(path: Path) -> int:
    match = re.search(r"_(\d+)\.tif$", path.name)
    if not match:
        raise ValueError(f"cannot parse ERA5 day index from {path.name}")
    return int(match.group(1))


def convert_era5_reference(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["temperature_min_c"] = data["temperature_min_k"] - 273.15
    data["temperature_max_c"] = data["temperature_max_k"] - 273.15
    data["actual_vapor_pressure_kpa"] = saturation_vapor_pressure_kpa(
        data["dewpoint_k"] - 273.15
    )
    data["wind_speed_m_s"] = np.hypot(data["wind_u_m_s"], data["wind_v_m_s"])
    data["solar_kj_m2_day"] = data["solar_j_m2_day"] / 1000.0
    output = data[["target_year", "site_id", "local_date", *CANONICAL_FIELDS]].copy()
    values = output[list(CANONICAL_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ERA5 reference conversion contains nonfinite values")
    if (output["temperature_min_c"] > output["temperature_max_c"]).any():
        raise ValueError("ERA5 reference Tmin exceeds Tmax")
    if (
        output[["actual_vapor_pressure_kpa", "wind_speed_m_s", "solar_kj_m2_day"]]
        < 0.0
    ).any().any():
        raise ValueError("ERA5 reference contains negative positive-only values")
    return output


def validate_reference_coverage(
    gefs_keys: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = ["target_year", "site_id", "local_date"]
    required = gefs_keys[
        ["decision_date", "lead_day", *keys]
    ].copy()
    if required[keys].duplicated().any():
        raise ValueError("GEFS causal sample has duplicate site-date reference keys")
    joined = required.merge(
        reference,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    missing = int(joined[list(CANONICAL_FIELDS)].isna().sum().sum())
    audit = {
        "status": "era5_nonprecip_causal_reference_extraction_passed"
        if missing == 0 and len(joined) == len(required)
        else "era5_nonprecip_causal_reference_extraction_failed",
        "mandatory_gate_passed": missing == 0 and len(joined) == len(required),
        "required_reference_rows": int(len(required)),
        "output_rows": int(len(joined)),
        "missing_canonical_value_count": missing,
        "year_count": int(joined["target_year"].nunique()),
        "site_count": int(joined["site_id"].nunique()),
        "contains_2024": bool((joined["target_year"] == 2024).any()),
        "reference_dataset": "ERA5_Land_corresponding_year",
        "gefs_values_used_for_reference_fit": False,
        "correction_fit_performed": False,
        "swap_simulation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
    }
    return joined, audit


def extract_variable(
    *,
    directory: Path,
    year: int,
    variable: str,
    required_dates: Sequence[str],
) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform

    paths = {day_index(path): path for path in directory.glob("*.tif")}
    if not paths:
        raise FileNotFoundError(f"no ERA5 TIFF files in {directory}")
    site_ids = list(SITE_COORDINATES)
    lons = [SITE_COORDINATES[site][0] for site in site_ids]
    lats = [SITE_COORDINATES[site][1] for site in site_ids]
    rows = []
    coordinate_cache: dict[str, tuple[list[float], list[float]]] = {}
    for local_date in required_dates:
        timestamp = pd.Timestamp(local_date)
        index = int(timestamp.dayofyear - 1)
        path = paths.get(index)
        if path is None:
            raise FileNotFoundError(f"missing ERA5 day {local_date}: {directory}")
        with rasterio.open(path) as source:
            crs_key = str(source.crs)
            if crs_key not in coordinate_cache:
                if source.crs is None or crs_key.upper() in {"EPSG:4326", "OGC:CRS84"}:
                    xs, ys = lons, lats
                else:
                    xs, ys = transform("EPSG:4326", source.crs, lons, lats)
                coordinate_cache[crs_key] = (list(xs), list(ys))
            xs, ys = coordinate_cache[crs_key]
            values = [float(value[0]) for value in source.sample(zip(xs, ys))]
        for site_id, value in zip(site_ids, values, strict=True):
            rows.append(
                {
                    "target_year": year,
                    "site_id": site_id,
                    "local_date": timestamp.strftime("%Y-%m-%d"),
                    variable: value,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    gefs = pd.read_csv(args.gefs_weather)
    gefs["decision_date"] = pd.to_datetime(gefs["decision_date"]).dt.strftime("%Y-%m-%d")
    gefs["local_date"] = pd.to_datetime(gefs["local_date"]).dt.strftime("%Y-%m-%d")
    gefs["target_year"] = pd.to_datetime(gefs["decision_date"]).dt.year
    key_columns = ["decision_date", "lead_day", "target_year", "site_id", "local_date"]
    gefs_keys = gefs[key_columns].drop_duplicates().copy()
    if len(gefs_keys) != len(gefs):
        raise ValueError("GEFS reference keys are not one-to-one")

    frames = []
    for year in sorted(gefs_keys["target_year"].unique()):
        required_dates = sorted(
            gefs_keys.loc[gefs_keys["target_year"] == year, "local_date"].unique()
        )
        merged = None
        keys = ["target_year", "site_id", "local_date"]
        for variable, directory_name in VARIABLE_DIRECTORIES.items():
            directory = args.era5_root / f"era5_{year}" / directory_name
            if not directory.is_dir():
                raise FileNotFoundError(f"missing ERA5 directory: {directory}")
            frame = extract_variable(
                directory=directory,
                year=int(year),
                variable=variable,
                required_dates=required_dates,
            )
            merged = frame if merged is None else merged.merge(
                frame, on=keys, how="inner", validate="one_to_one"
            )
        if merged is None:
            raise RuntimeError(f"no ERA5 variables extracted for {year}")
        frames.append(convert_era5_reference(merged))
        print(f"[ERA5] {year} ready ({len(required_dates)} dates)", flush=True)

    reference = pd.concat(frames, ignore_index=True)
    output, audit = validate_reference_coverage(gefs_keys, reference)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "reference": args.output_dir / "era5_nonprecip_causal_reference_daily_v1.csv",
        "audit": args.output_dir / "era5_nonprecip_causal_reference_audit_v1.json",
        "manifest": args.output_dir / "era5_nonprecip_causal_reference_manifest_v1.json",
    }
    output.to_csv(outputs["reference"], index=False)
    outputs["audit"].write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "status": audit["status"],
        "gefs_weather_sha256": sha256_file(args.gefs_weather),
        "era5_root": str(args.era5_root),
        "reference_sha256": sha256_file(outputs["reference"]),
        "audit_sha256": sha256_file(outputs["audit"]),
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not audit["mandatory_gate_passed"]:
        raise RuntimeError(f"ERA5 reference gate failed; see {outputs['audit']}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gefs-weather", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    generated = run(parse_args())
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2))

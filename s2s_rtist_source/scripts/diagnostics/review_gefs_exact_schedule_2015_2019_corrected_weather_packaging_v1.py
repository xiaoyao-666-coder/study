#!/usr/bin/env python3
"""Independently review frozen 2015-2019 GEFS weather before packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


YEARS = tuple(range(2015, 2020))
EXPECTED = {
    2015: {"rows": 2555, "cycles": 46, "site_cycles": 73},
    2016: {"rows": 2240, "cycles": 52, "site_cycles": 64},
    2017: {"rows": 2345, "cycles": 43, "site_cycles": 67},
    2018: {"rows": 2275, "cycles": 54, "site_cycles": 65},
    2019: {"rows": 2415, "cycles": 44, "site_cycles": 69},
}
SAMPLE_KEYS = ["decision_date", "site_id", "gefs_member", "local_date", "lead_day"]
NONPRECIP_FIELDS = [
    "temperature_min_c",
    "temperature_max_c",
    "actual_vapor_pressure_kpa",
    "wind_speed_m_s",
    "solar_kj_m2_day",
]
FINAL_FIELDS = ["precipitation_mm", *NONPRECIP_FIELDS]
EXPECTED_SITES = {"P1", "P2", "P3", "P4", "P15"}
EXPECTED_MEMBERS = {"c00", "p01", "p02", "p03", "p04"}
EXPECTED_WEATHER_SOURCE = "GEFSv12_5member_frozen_causal_all_variable_correction_v1"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "decision_date",
        "local_date",
        "target_date",
        "fit_first_decision_date",
        "fit_last_decision_date",
    ):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result


def max_abs_error(left: pd.Series, right: pd.Series) -> float:
    return float(
        np.max(
            np.abs(
                left.to_numpy(dtype=float)
                - right.to_numpy(dtype=float)
            )
        )
    )


def pairwise_inversions(
    frame: pd.DataFrame,
    raw_column: str,
    corrected_column: str,
) -> int:
    count = 0
    group_keys = ["decision_date", "site_id", "local_date", "lead_day"]
    for _, group in frame.groupby(group_keys, sort=False):
        ordered = group.sort_values("gefs_member")
        raw = ordered[raw_column].to_numpy(dtype=float)
        corrected = ordered[corrected_column].to_numpy(dtype=float)
        for first in range(len(raw)):
            for second in range(first + 1, len(raw)):
                if (raw[first] - raw[second]) * (
                    corrected[first] - corrected[second]
                ) < -1e-12:
                    count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for_year(
    raw_root: Path,
    frozen_parent: Path,
    year: int,
) -> dict[str, Path]:
    root = frozen_parent / f"gefs_exact_schedule_{year}_frozen_weather_v1"
    causal_name = (
        "gefs_exact_schedule_2015_causal_fit_c00_audit_v1.json"
        if year == 2015
        else f"gefs_exact_schedule_{year}_prior_year_fit_c00_audit_v1.json"
    )
    return {
        "root": root,
        "raw": raw_root / f"Y{year}" / f"gefs_exact_schedule_{year}_raw_full_weather_v1.csv",
        "final": root / "04_frozen_all_variable_weather" / "gefs_2015_2019_frozen_all_variable_member_weather_v1.csv",
        "nonprecip": root / "03_frozen_nonprecipitation" / f"gefs_exact_schedule_{year}_frozen_nonprecip_branches_v1.csv",
        "factors": root / "03_frozen_nonprecipitation" / f"gefs_exact_schedule_{year}_frozen_nonprecip_factors_v1.csv",
        "precip_cycles": root / "04_frozen_all_variable_weather" / "gefs_2015_2019_frozen_precipitation_site_cycles_v1.csv",
        "pipeline_audit": root / f"gefs_exact_schedule_{year}_frozen_weather_pipeline_audit_v1.json",
        "all_weather_audit": root / "04_frozen_all_variable_weather" / "gefs_2015_2019_frozen_all_variable_weather_audit_v1.json",
        "nonprecip_audit": root / "03_frozen_nonprecipitation" / f"gefs_exact_schedule_{year}_frozen_nonprecip_audit_v1.json",
        "causal_audit": root / "01_causal_fit_history" / causal_name,
        "era5_audit": root / "02_era5_reference" / "era5_nonprecip_causal_reference_audit_v1.json",
    }


def review_year(paths: dict[str, Path], year: int) -> tuple[dict[str, object], pd.DataFrame]:
    missing_files = [name for name, path in paths.items() if name != "root" and not path.is_file()]
    if missing_files:
        return (
            {
                "status": f"exact_schedule_{year}_packaging_review_failed",
                "mandatory_packaging_review_gate_passed": False,
                "gate_failures": [f"missing_{name}" for name in missing_files],
            },
            pd.DataFrame(),
        )

    raw = normalize_dates(pd.read_csv(paths["raw"]))
    final = normalize_dates(pd.read_csv(paths["final"]))
    nonprecip = normalize_dates(pd.read_csv(paths["nonprecip"]))
    factors = normalize_dates(pd.read_csv(paths["factors"]))
    precip_cycles = normalize_dates(pd.read_csv(paths["precip_cycles"]))
    pipeline = load_json(paths["pipeline_audit"])
    all_weather = load_json(paths["all_weather_audit"])
    nonprecip_audit = load_json(paths["nonprecip_audit"])
    causal_audit = load_json(paths["causal_audit"])
    era5_audit = load_json(paths["era5_audit"])

    raw_keys = raw[SAMPLE_KEYS].sort_values(SAMPLE_KEYS).reset_index(drop=True)
    final_keys = final[SAMPLE_KEYS].sort_values(SAMPLE_KEYS).reset_index(drop=True)
    exact_key_coverage = raw_keys.equals(final_keys)
    duplicate_key_count = int(final[SAMPLE_KEYS].duplicated().sum())
    raw_index = raw.set_index(SAMPLE_KEYS).sort_index()
    final_index = final.set_index(SAMPLE_KEYS).sort_index()
    nonprecip_index = nonprecip.set_index(SAMPLE_KEYS).sort_index()

    raw_copy_errors = {
        "precipitation_mm_raw": max_abs_error(
            final_index["precipitation_mm_raw"], raw_index["precipitation_mm_raw"]
        )
    }
    for field in NONPRECIP_FIELDS:
        raw_copy_errors[field] = max_abs_error(
            final_index[f"{field}_raw"], raw_index[field]
        )
    metadata_copy_mismatches = 0
    for field in (
        "site_timezone",
        "forecast_init_utc",
        "source_product_keys",
        "source_product_etags",
    ):
        metadata_copy_mismatches += int(
            (~final_index[field].astype(str).eq(raw_index[field].astype(str))).sum()
        )
    nonprecip_copy_error = max(
        max_abs_error(final_index[field], nonprecip_index[field])
        for field in NONPRECIP_FIELDS
    )

    numeric = final[FINAL_FIELDS].to_numpy(dtype=float)
    missing_values = int(final[FINAL_FIELDS].isna().sum().sum())
    nonfinite_values = int((~np.isfinite(numeric)).sum())
    temperature_order_errors = int(
        (final["temperature_min_c"] > final["temperature_max_c"]).sum()
    )
    negative_positive_values = int(
        (
            final[
                [
                    "precipitation_mm",
                    "actual_vapor_pressure_kpa",
                    "wind_speed_m_s",
                    "solar_kj_m2_day",
                ]
            ]
            < 0.0
        )
        .sum()
        .sum()
    )

    horizon_errors = 0
    group_size_errors = 0
    for _, group in final.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=False
    ):
        ordered = group.sort_values("lead_day")
        if ordered["lead_day"].tolist() != list(range(1, 8)):
            group_size_errors += 1
            continue
        expected_dates = pd.to_datetime(ordered["decision_date"]) + pd.to_timedelta(
            ordered["lead_day"] - 1, unit="D"
        )
        horizon_errors += int(
            (~pd.to_datetime(ordered["local_date"]).eq(expected_dates)).sum()
        )
    member_count_errors = int(
        (
            final.groupby(["decision_date", "site_id"])["gefs_member"].nunique()
            != 5
        ).sum()
    )

    factor_last = pd.to_datetime(factors["fit_last_decision_date"])
    factor_target = pd.to_datetime(factors["target_date"])
    causal_fit_leakage_rows = int(
        ((factor_last + pd.Timedelta(days=6)) >= factor_target).sum()
    )
    insufficient_fit_sample_rows = int((factors["fit_sample_count"] < 8).sum())
    expected_factor_keys = final[["decision_date", "site_id", "lead_day"]].drop_duplicates()
    actual_factor_keys = factors[["target_date", "site_id", "lead_day"]].rename(
        columns={"target_date": "decision_date"}
    )
    exact_factor_keys = expected_factor_keys.sort_values(
        ["decision_date", "site_id", "lead_day"]
    ).reset_index(drop=True).equals(
        actual_factor_keys.sort_values(
            ["decision_date", "site_id", "lead_day"]
        ).reset_index(drop=True)
    )

    cycle_keys = ["decision_date", "site_id"]
    expected_cycle_keys = final[cycle_keys].drop_duplicates().sort_values(cycle_keys)
    exact_precip_cycle_keys = expected_cycle_keys.reset_index(drop=True).equals(
        precip_cycles[cycle_keys].sort_values(cycle_keys).reset_index(drop=True)
    )
    precipitation_fit_leakage_rows = int(
        (precip_cycles["fit_last_year"] >= precip_cycles["target_year"]).sum()
    )
    raw_member_week = final.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=True
    )["precipitation_mm_raw"].sum()
    corrected_member_week = final.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=True
    )["precipitation_mm"].sum()
    factor_for_member = raw_member_week.reset_index()[cycle_keys].merge(
        precip_cycles[[*cycle_keys, "effective_factor"]],
        on=cycle_keys,
        how="left",
        validate="many_to_one",
    )["effective_factor"].to_numpy(dtype=float)
    weekly_precipitation_error = float(
        np.max(
            np.abs(
                corrected_member_week.to_numpy(dtype=float)
                - raw_member_week.to_numpy(dtype=float) * factor_for_member
            )
        )
    )
    cycle_index = precip_cycles.set_index(cycle_keys).sort_index()
    raw_ensemble_mean = raw_member_week.groupby(level=[0, 1]).mean().sort_index()
    raw_ensemble_mean_error = max_abs_error(
        raw_ensemble_mean,
        cycle_index.loc[raw_ensemble_mean.index, "raw_ensemble_mean_7d_mm"],
    )

    inversion_frame = final.copy()
    inversion_frame["temperature_center_raw"] = (
        inversion_frame["temperature_min_c_raw"]
        + inversion_frame["temperature_max_c_raw"]
    ) / 2.0
    inversion_frame["temperature_center"] = (
        inversion_frame["temperature_min_c"]
        + inversion_frame["temperature_max_c"]
    ) / 2.0
    inversion_frame["temperature_range_raw"] = (
        inversion_frame["temperature_max_c_raw"]
        - inversion_frame["temperature_min_c_raw"]
    )
    inversion_frame["temperature_range"] = (
        inversion_frame["temperature_max_c"]
        - inversion_frame["temperature_min_c"]
    )
    temperature_inversions = pairwise_inversions(
        inversion_frame, "temperature_center_raw", "temperature_center"
    ) + pairwise_inversions(
        inversion_frame, "temperature_range_raw", "temperature_range"
    )
    positive_inversions = sum(
        pairwise_inversions(inversion_frame, f"{field}_raw", field)
        for field in (
            "actual_vapor_pressure_kpa",
            "wind_speed_m_s",
            "solar_kj_m2_day",
        )
    )
    precipitation_inversions = pairwise_inversions(
        inversion_frame, "precipitation_mm_raw", "precipitation_mm"
    )
    source_key_errors = int(
        final["source_product_keys"].astype(str).str.split(";").map(len).ne(7).sum()
    )
    source_etag_errors = int(
        final["source_product_etags"].astype(str).str.split(";").map(len).ne(7).sum()
    )
    retained_gribs = len(list(paths["root"].rglob("*.grib2")))

    expected = EXPECTED[year]
    dependency_gate = all(
        (
            pipeline.get("mandatory_pipeline_gate_passed"),
            all_weather.get("mandatory_structural_gate_passed"),
            nonprecip_audit.get("mandatory_gate_passed"),
            causal_audit.get("mandatory_gate_passed"),
            era5_audit.get("mandatory_gate_passed"),
        )
    )
    strict_history = (
        not bool(pipeline.get("target_or_future_era5_used_for_fit"))
        and (
            year == 2015
            or (
                pipeline.get("strict_prior_year_history") is True
                and int(pipeline.get("history_last_year", -1)) == year - 1
            )
        )
    )
    checks = {
        "dependency_gate": dependency_gate,
        "strict_history": strict_history,
        "row_count": len(final) == expected["rows"],
        "cycle_count": final["decision_date"].nunique() == expected["cycles"],
        "site_cycle_count": len(final[cycle_keys].drop_duplicates())
        == expected["site_cycles"],
        "site_set": set(final["site_id"]) == EXPECTED_SITES,
        "member_set": set(final["gefs_member"]) == EXPECTED_MEMBERS,
        "exact_sample_keys": exact_key_coverage,
        "duplicate_sample_keys": duplicate_key_count == 0,
        "member_counts": member_count_errors == 0,
        "seven_day_horizons": group_size_errors == 0 and horizon_errors == 0,
        "missing_values": missing_values == 0,
        "nonfinite_values": nonfinite_values == 0,
        "temperature_order": temperature_order_errors == 0,
        "positive_physical_ranges": negative_positive_values == 0,
        "raw_weather_copy": max(raw_copy_errors.values()) <= 1e-10,
        "metadata_copy": metadata_copy_mismatches == 0,
        "nonprecipitation_copy": nonprecip_copy_error <= 1e-10,
        "causal_factor_dates": causal_fit_leakage_rows == 0,
        "minimum_fit_samples": insufficient_fit_sample_rows == 0,
        "factor_key_coverage": exact_factor_keys,
        "precipitation_cycle_coverage": exact_precip_cycle_keys,
        "precipitation_fit_year": precipitation_fit_leakage_rows == 0,
        "weekly_precipitation": weekly_precipitation_error <= 1e-9,
        "raw_ensemble_mean": raw_ensemble_mean_error <= 1e-9,
        "temperature_member_structure": temperature_inversions == 0,
        "positive_member_structure": positive_inversions == 0,
        "precipitation_member_structure": precipitation_inversions == 0,
        "source_product_key_count": source_key_errors == 0,
        "source_product_etag_count": source_etag_errors == 0,
        "retained_grib_files": retained_gribs == 0,
        "target_year": final["target_year"].eq(year).all(),
        "decision_year": pd.to_datetime(final["decision_date"]).dt.year.eq(year).all(),
        "weather_source": final["weather_source"].eq(EXPECTED_WEATHER_SOURCE).all(),
    }
    failures = [name for name, passed in checks.items() if not bool(passed)]
    review = {
        "status": (
            f"exact_schedule_{year}_corrected_weather_packaging_review_passed"
            if not failures
            else f"exact_schedule_{year}_corrected_weather_packaging_review_failed"
        ),
        "mandatory_packaging_review_gate_passed": not failures,
        "gate_failures": failures,
        "target_year": year,
        "final_member_rows": int(len(final)),
        "decision_date_count": int(final["decision_date"].nunique()),
        "site_cycle_count": int(len(final[cycle_keys].drop_duplicates())),
        "site_count": int(final["site_id"].nunique()),
        "member_count": int(final["gefs_member"].nunique()),
        "lead_day_count": int(final["lead_day"].nunique()),
        "exact_sample_key_coverage": exact_key_coverage,
        "duplicate_sample_key_count": duplicate_key_count,
        "missing_value_count": missing_values,
        "nonfinite_value_count": nonfinite_values,
        "temperature_order_error_count": temperature_order_errors,
        "negative_positive_variable_count": negative_positive_values,
        "maximum_absolute_raw_weather_copy_error": max(raw_copy_errors.values()),
        "metadata_copy_mismatch_count": metadata_copy_mismatches,
        "maximum_absolute_nonprecipitation_copy_error": nonprecip_copy_error,
        "causal_fit_leakage_rows": causal_fit_leakage_rows,
        "minimum_fit_sample_count": int(factors["fit_sample_count"].min()),
        "insufficient_fit_sample_rows": insufficient_fit_sample_rows,
        "precipitation_fit_leakage_rows": precipitation_fit_leakage_rows,
        "maximum_absolute_weekly_precipitation_error_mm": weekly_precipitation_error,
        "maximum_absolute_raw_ensemble_mean_error_mm": raw_ensemble_mean_error,
        "temperature_structure_member_order_inversion_count": temperature_inversions,
        "positive_variable_member_order_inversion_count": positive_inversions,
        "precipitation_member_order_inversion_count": precipitation_inversions,
        "source_product_key_count_error_rows": source_key_errors,
        "source_product_etag_count_error_rows": source_etag_errors,
        "retained_grib_file_count": retained_gribs,
        "pipeline_status": pipeline.get("status"),
        "history_last_year": pipeline.get("history_last_year"),
    }
    return review, final


def build_manifest(frozen_parent: Path, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for year in YEARS:
        root = frozen_parent / f"gefs_exact_schedule_{year}_frozen_weather_v1"
        for path in sorted(root.rglob("*")):
            if not path.is_file() or output_dir in path.parents:
                continue
            relative = path.relative_to(root).as_posix()
            if relative.startswith("05_packaging_review/"):
                continue
            records.append(
                {
                    "target_year": year,
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "artifact_stage": relative.split("/", 1)[0],
                }
            )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Path]:
    raw_root = args.raw_root.resolve()
    frozen_parent = args.frozen_parent.resolve()
    output_dir = args.output_dir.resolve()
    yearly_reviews: dict[str, dict[str, object]] = {}
    final_frames: list[pd.DataFrame] = []
    for year in YEARS:
        review, final = review_year(paths_for_year(raw_root, frozen_parent, year), year)
        yearly_reviews[str(year)] = review
        if not final.empty:
            final_frames.append(final)

    yearly_gates = all(
        bool(review.get("mandatory_packaging_review_gate_passed"))
        for review in yearly_reviews.values()
    )
    combined = pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame()
    aggregate_key_columns = ["target_year", *SAMPLE_KEYS]
    aggregate_duplicate_keys = (
        int(combined[aggregate_key_columns].duplicated().sum())
        if not combined.empty
        else -1
    )
    expected_rows = sum(item["rows"] for item in EXPECTED.values())
    expected_cycles = sum(item["cycles"] for item in EXPECTED.values())
    expected_site_cycles = sum(item["site_cycles"] for item in EXPECTED.values())
    aggregate_checks = {
        "all_yearly_gates": yearly_gates,
        "target_year_set": set(combined["target_year"].astype(int)) == set(YEARS)
        if not combined.empty
        else False,
        "total_rows": len(combined) == expected_rows,
        "total_cycles": sum(
            int(review.get("decision_date_count", -1)) for review in yearly_reviews.values()
        )
        == expected_cycles,
        "total_site_cycles": sum(
            int(review.get("site_cycle_count", -1)) for review in yearly_reviews.values()
        )
        == expected_site_cycles,
        "aggregate_duplicate_sample_keys": aggregate_duplicate_keys == 0,
    }
    aggregate_failures = [
        name for name, passed in aggregate_checks.items() if not bool(passed)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(frozen_parent, output_dir)
    manifest_path = output_dir / "gefs_exact_schedule_2015_2019_packaging_source_sha256_v1.csv"
    manifest.to_csv(manifest_path, index=False)
    audit = {
        "status": (
            "exact_schedule_2015_2019_corrected_weather_packaging_review_passed"
            if not aggregate_failures
            else "exact_schedule_2015_2019_corrected_weather_packaging_review_failed"
        ),
        "mandatory_packaging_review_gate_passed": not aggregate_failures,
        "gate_failures": aggregate_failures,
        "target_years": list(YEARS),
        "yearly_reviews": yearly_reviews,
        "total_member_rows": int(len(combined)),
        "total_decision_dates": expected_cycles,
        "total_site_cycles": expected_site_cycles,
        "aggregate_duplicate_sample_key_count": aggregate_duplicate_keys,
        "source_artifact_file_count": int(len(manifest)),
        "source_artifact_total_bytes": int(manifest["size_bytes"].sum()),
        "packaging_performed": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "ready_to_package_2015_2019_frozen_weather_for_server"
            if not aggregate_failures
            else "repair_packaging_review_failures"
        ),
    }
    audit_path = output_dir / "gefs_exact_schedule_2015_2019_packaging_review_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if aggregate_failures:
        raise RuntimeError(f"Packaging review failed; see {audit_path}")
    return {"audit": audit_path, "sha256_manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--frozen-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run(parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))

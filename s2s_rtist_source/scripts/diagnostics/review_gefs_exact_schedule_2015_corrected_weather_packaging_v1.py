#!/usr/bin/env python3
"""Independently review frozen 2015 GEFS weather before server packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("decision_date", "local_date", "target_date"):
        if column in result.columns:
            result[column] = pd.to_datetime(result[column]).dt.strftime("%Y-%m-%d")
    return result


def max_abs_error(left: pd.Series, right: pd.Series) -> float:
    return float(np.max(np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float))))


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


def build_source_manifest(frozen_root: Path, output_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(frozen_root.rglob("*")):
        if not path.is_file() or output_dir in path.parents:
            continue
        relative = path.relative_to(frozen_root).as_posix()
        records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_stage": relative.split("/", 1)[0],
            }
        )
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Path]:
    frozen_root = args.frozen_root.resolve()
    output_dir = (args.output_dir or frozen_root / "05_packaging_review").resolve()
    raw = normalize_dates(pd.read_csv(args.raw_weather))
    final_path = (
        frozen_root
        / "04_frozen_all_variable_weather"
        / "gefs_2015_2019_frozen_all_variable_member_weather_v1.csv"
    )
    nonprecip_path = (
        frozen_root
        / "03_frozen_nonprecipitation"
        / "gefs_exact_schedule_2015_frozen_nonprecip_branches_v1.csv"
    )
    factor_path = (
        frozen_root
        / "03_frozen_nonprecipitation"
        / "gefs_exact_schedule_2015_frozen_nonprecip_factors_v1.csv"
    )
    precipitation_cycle_path = (
        frozen_root
        / "04_frozen_all_variable_weather"
        / "gefs_2015_2019_frozen_precipitation_site_cycles_v1.csv"
    )
    pipeline_audit_path = (
        frozen_root / "gefs_exact_schedule_2015_frozen_weather_pipeline_audit_v1.json"
    )
    all_weather_audit_path = (
        frozen_root
        / "04_frozen_all_variable_weather"
        / "gefs_2015_2019_frozen_all_variable_weather_audit_v1.json"
    )
    nonprecip_audit_path = (
        frozen_root
        / "03_frozen_nonprecipitation"
        / "gefs_exact_schedule_2015_frozen_nonprecip_audit_v1.json"
    )
    causal_audit_path = (
        frozen_root
        / "01_causal_fit_history"
        / "gefs_exact_schedule_2015_causal_fit_c00_audit_v1.json"
    )

    final = normalize_dates(pd.read_csv(final_path))
    nonprecip = normalize_dates(pd.read_csv(nonprecip_path))
    factors = normalize_dates(pd.read_csv(factor_path))
    precipitation_cycles = normalize_dates(pd.read_csv(precipitation_cycle_path))
    pipeline_audit = load_json(pipeline_audit_path)
    all_weather_audit = load_json(all_weather_audit_path)
    nonprecip_audit = load_json(nonprecip_audit_path)
    causal_audit = load_json(causal_audit_path)
    history_audit = load_json(args.history_audit)

    raw_keys = raw[SAMPLE_KEYS].sort_values(SAMPLE_KEYS).reset_index(drop=True)
    final_keys = final[SAMPLE_KEYS].sort_values(SAMPLE_KEYS).reset_index(drop=True)
    exact_key_coverage = raw_keys.equals(final_keys)
    duplicate_key_count = int(final[SAMPLE_KEYS].duplicated().sum())
    final_index = final.set_index(SAMPLE_KEYS).sort_index()
    raw_index = raw.set_index(SAMPLE_KEYS).sort_index()
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
    metadata_copy_mismatch_count = 0
    for field in (
        "site_timezone",
        "forecast_init_utc",
        "source_product_keys",
        "source_product_etags",
    ):
        metadata_copy_mismatch_count += int(
            (~final_index[field].astype(str).eq(raw_index[field].astype(str))).sum()
        )
    nonprecip_copy_errors = {
        field: max_abs_error(final_index[field], nonprecip_index[field])
        for field in NONPRECIP_FIELDS
    }

    numeric = final[FINAL_FIELDS].to_numpy(dtype=float)
    nonfinite_value_count = int((~np.isfinite(numeric)).sum())
    missing_value_count = int(final[FINAL_FIELDS].isna().sum().sum())
    temperature_order_error_count = int(
        (final["temperature_min_c"] > final["temperature_max_c"]).sum()
    )
    negative_positive_variable_count = int(
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
    member_counts = final.groupby(["decision_date", "site_id"])[
        "gefs_member"
    ].nunique()
    member_count_error_count = int((member_counts != 5).sum())

    factor_last = pd.to_datetime(factors["fit_last_decision_date"])
    factor_target = pd.to_datetime(factors["target_date"])
    causal_fit_leakage_rows = int(((factor_last + pd.Timedelta(days=6)) >= factor_target).sum())
    insufficient_fit_sample_rows = int((factors["fit_sample_count"] < 8).sum())
    expected_factor_keys = final[["decision_date", "site_id", "lead_day"]].drop_duplicates()
    actual_factor_keys = factors[["target_date", "site_id", "lead_day"]].rename(
        columns={"target_date": "decision_date"}
    )
    exact_factor_key_coverage = expected_factor_keys.sort_values(
        ["decision_date", "site_id", "lead_day"]
    ).reset_index(drop=True).equals(
        actual_factor_keys.sort_values(
            ["decision_date", "site_id", "lead_day"]
        ).reset_index(drop=True)
    )

    cycle_keys = ["decision_date", "site_id"]
    cycle_index = precipitation_cycles.set_index(cycle_keys).sort_index()
    expected_cycle_keys = final[cycle_keys].drop_duplicates().sort_values(cycle_keys)
    exact_precipitation_cycle_coverage = expected_cycle_keys.reset_index(drop=True).equals(
        precipitation_cycles[cycle_keys].sort_values(cycle_keys).reset_index(drop=True)
    )
    precipitation_fit_leakage_rows = int(
        (precipitation_cycles["fit_last_year"] >= precipitation_cycles["target_year"]).sum()
    )
    raw_member_week = final.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=True
    )["precipitation_mm_raw"].sum()
    corrected_member_week = final.groupby(
        ["decision_date", "site_id", "gefs_member"], sort=True
    )["precipitation_mm"].sum()
    factor_for_member = raw_member_week.reset_index()[cycle_keys].merge(
        precipitation_cycles[[*cycle_keys, "effective_factor"]],
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
    raw_ensemble_mean = (
        raw_member_week.groupby(level=[0, 1]).mean().sort_index()
    )
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
        inversion_frame["temperature_min_c"] + inversion_frame["temperature_max_c"]
    ) / 2.0
    inversion_frame["temperature_range_raw"] = (
        inversion_frame["temperature_max_c_raw"]
        - inversion_frame["temperature_min_c_raw"]
    )
    inversion_frame["temperature_range"] = (
        inversion_frame["temperature_max_c"] - inversion_frame["temperature_min_c"]
    )
    temperature_structure_inversions = pairwise_inversions(
        inversion_frame, "temperature_center_raw", "temperature_center"
    ) + pairwise_inversions(
        inversion_frame, "temperature_range_raw", "temperature_range"
    )
    positive_variable_inversions = sum(
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

    source_product_key_count_errors = int(
        final["source_product_keys"].astype(str).str.split(";").map(len).ne(7).sum()
    )
    source_product_etag_count_errors = int(
        final["source_product_etags"].astype(str).str.split(";").map(len).ne(7).sum()
    )
    retained_grib_file_count = len(list(frozen_root.rglob("*.grib2")))

    dependency_statuses = {
        "history_extraction": history_audit.get("status"),
        "causal_history": causal_audit.get("status"),
        "frozen_nonprecipitation": nonprecip_audit.get("status"),
        "all_variable_weather": all_weather_audit.get("status"),
        "pipeline": pipeline_audit.get("status"),
    }
    dependency_gate_passed = all(
        (
            history_audit.get("mandatory_gate_passed"),
            causal_audit.get("mandatory_gate_passed"),
            nonprecip_audit.get("mandatory_gate_passed"),
            all_weather_audit.get("mandatory_structural_gate_passed"),
            pipeline_audit.get("mandatory_pipeline_gate_passed"),
        )
    )

    gate_failures = []
    checks = {
        "dependency_gate_failed": dependency_gate_passed,
        "row_count_mismatch": len(final) == 2555,
        "exact_sample_key_coverage_failed": exact_key_coverage,
        "duplicate_sample_keys": duplicate_key_count == 0,
        "site_set_mismatch": set(final["site_id"]) == EXPECTED_SITES,
        "member_set_mismatch": set(final["gefs_member"]) == EXPECTED_MEMBERS,
        "cycle_count_mismatch": final["decision_date"].nunique() == 46,
        "site_cycle_count_mismatch": len(final[cycle_keys].drop_duplicates()) == 73,
        "member_count_mismatch": member_count_error_count == 0,
        "seven_day_horizon_mismatch": group_size_errors == 0 and horizon_errors == 0,
        "missing_values": missing_value_count == 0,
        "nonfinite_values": nonfinite_value_count == 0,
        "temperature_order_errors": temperature_order_error_count == 0,
        "negative_physical_values": negative_positive_variable_count == 0,
        "raw_weather_copy_error": max(raw_copy_errors.values()) <= 1e-10,
        "metadata_copy_error": metadata_copy_mismatch_count == 0,
        "nonprecipitation_copy_error": max(nonprecip_copy_errors.values()) <= 1e-10,
        "causal_fit_leakage": causal_fit_leakage_rows == 0,
        "insufficient_fit_samples": insufficient_fit_sample_rows == 0,
        "factor_key_coverage_failed": exact_factor_key_coverage,
        "precipitation_cycle_coverage_failed": exact_precipitation_cycle_coverage,
        "precipitation_fit_leakage": precipitation_fit_leakage_rows == 0,
        "weekly_precipitation_error": weekly_precipitation_error <= 1e-9,
        "raw_ensemble_mean_error": raw_ensemble_mean_error <= 1e-9,
        "temperature_member_order_inversion": temperature_structure_inversions == 0,
        "positive_member_order_inversion": positive_variable_inversions == 0,
        "precipitation_member_order_inversion": precipitation_inversions == 0,
        "source_product_key_count_error": source_product_key_count_errors == 0,
        "source_product_etag_count_error": source_product_etag_count_errors == 0,
        "retained_grib_files": retained_grib_file_count == 0,
        "wrong_target_year": final["target_year"].eq(2015).all(),
        "wrong_weather_source": final["weather_source"].eq(
            "GEFSv12_5member_frozen_causal_all_variable_correction_v1"
        ).all(),
    }
    for name, passed in checks.items():
        if not bool(passed):
            gate_failures.append(name)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest(frozen_root, output_dir)
    manifest_path = output_dir / "gefs_exact_schedule_2015_packaging_source_sha256_v1.csv"
    manifest.to_csv(manifest_path, index=False)
    audit = {
        "status": (
            "exact_schedule_2015_corrected_weather_packaging_review_passed"
            if not gate_failures
            else "exact_schedule_2015_corrected_weather_packaging_review_failed"
        ),
        "mandatory_packaging_review_gate_passed": not gate_failures,
        "gate_failures": gate_failures,
        "dependency_statuses": dependency_statuses,
        "source_artifact_file_count": int(len(manifest)),
        "source_artifact_total_bytes": int(manifest["size_bytes"].sum()),
        "final_member_rows": int(len(final)),
        "decision_date_count": int(final["decision_date"].nunique()),
        "site_count": int(final["site_id"].nunique()),
        "site_cycle_count": int(len(final[cycle_keys].drop_duplicates())),
        "member_count": int(final["gefs_member"].nunique()),
        "lead_day_count": int(final["lead_day"].nunique()),
        "exact_sample_key_coverage": exact_key_coverage,
        "duplicate_sample_key_count": duplicate_key_count,
        "missing_value_count": missing_value_count,
        "nonfinite_value_count": nonfinite_value_count,
        "temperature_order_error_count": temperature_order_error_count,
        "negative_positive_variable_count": negative_positive_variable_count,
        "maximum_absolute_raw_weather_copy_error": max(raw_copy_errors.values()),
        "metadata_copy_mismatch_count": metadata_copy_mismatch_count,
        "maximum_absolute_nonprecipitation_copy_error": max(
            nonprecip_copy_errors.values()
        ),
        "causal_fit_leakage_rows": causal_fit_leakage_rows,
        "minimum_fit_sample_count": int(factors["fit_sample_count"].min()),
        "insufficient_fit_sample_rows": insufficient_fit_sample_rows,
        "exact_factor_key_coverage": exact_factor_key_coverage,
        "precipitation_fit_leakage_rows": precipitation_fit_leakage_rows,
        "maximum_absolute_weekly_precipitation_error_mm": weekly_precipitation_error,
        "maximum_absolute_raw_ensemble_mean_error_mm": raw_ensemble_mean_error,
        "temperature_structure_member_order_inversion_count": temperature_structure_inversions,
        "positive_variable_member_order_inversion_count": positive_variable_inversions,
        "precipitation_member_order_inversion_count": precipitation_inversions,
        "source_product_key_count_error_rows": source_product_key_count_errors,
        "source_product_etag_count_error_rows": source_product_etag_count_errors,
        "retained_grib_file_count": retained_grib_file_count,
        "weather_correction_applied": True,
        "packaging_performed": False,
        "swap_simulation_performed": False,
        "label_generation_performed": False,
        "surrogate_training_performed": False,
        "tta_performed": False,
        "next_gate": (
            "ready_to_package_2015_frozen_weather_for_server"
            if not gate_failures
            else "repair_packaging_review_failures"
        ),
    }
    audit_path = output_dir / "gefs_exact_schedule_2015_packaging_review_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if gate_failures:
        raise RuntimeError(f"2015 packaging review failed; see {audit_path}")
    return {"audit": audit_path, "sha256_manifest": manifest_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-weather", type=Path, required=True)
    parser.add_argument("--history-audit", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    outputs = run(parse_args())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))

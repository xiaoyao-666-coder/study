#!/usr/bin/env python3
"""Validate formal NPrintDay=24 three-output SWAP smoke datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


FORMAL_NPRINTDAY = 24
FORMAL_IRRIGATION_OPTIONS_MM = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0)
FORMAL_FLUX_INTEGRATION_METHOD = "trapezoid_actual_subdaily_interval"
FORMAL_INCREMENT_GROUPING_METHOD = "Dcum_1_to_horizon_days"
FORMAL_CONTROL_VOLUME_TYPE = "fixed_0_100cm"
FORMAL_CONTROL_DEPTH_CM = 100.0
FORMAL_DATA_PROCESSING_SPEC_VERSION = (
    "three_output_surrogate_data_processing_spec_v1_fixed_0_100cm"
)
NUMERIC_TOLERANCE_MM = 1.0e-6


class SmokeValidationError(ValueError):
    """Raised when a smoke dataset violates the locked experiment contract."""


@dataclass(frozen=True)
class SmokeValidationResult:
    passed: bool
    row_count: int
    site_count: int
    site_date_count: int
    max_abs_water_balance_residual_mm: float
    max_abs_aet_component_error_mm: float
    max_abs_balance_reconstruction_error_mm: float
    site_summary: pd.DataFrame


def _daily_columns(prefix: str) -> list[str]:
    return [f"{prefix}_day{day:02d}" for day in range(1, 8)]


def _required_columns() -> list[str]:
    required = [
        "site",
        "date_t",
        "ir",
        "net_gain_7d",
        "horizon_days_actual",
        "nprintday",
        "flux_integration_method",
        "increment_grouping_method",
        "swap_version",
        "water_depth_unit",
        "flux_rate_source_unit",
        "root_depth_unit",
        "soil_vwc_0_100cm_unit",
        "control_volume_type",
        "control_depth_cm",
        "data_processing_spec_version",
        "raw_audit_preserved",
        "raw_audit_dir",
        "rain_7d_mm",
        "snow_7d_mm",
        "irrigation_7d_mm",
        "runon_7d_mm",
        "aet_7d_mm",
        "runoff_7d_mm",
        "soil_drainage_0_100cm_7d_mm",
        "soil_boundary_waterflux_100cm_signed_7d_mm",
        "soil_boundary_outflow_100cm_7d_mm",
        "residual_flux_7d_mm",
        "predecision_soil_storage_0_100cm_mm",
        "final_soil_storage_0_100cm_mm",
        "delta_soil_storage_0_100cm_7d_mm",
        "water_balance_residual_0_100cm_7d_mm",
        "max_abs_soil_boundary_depth_error_cm",
    ]
    for prefix, suffix in (
        ("tact", "_mm"),
        ("eact", "_mm"),
        ("interc", "_mm"),
        ("aet", "_mm"),
        ("root_depth", "_cm"),
        ("soil_vwc_0_100cm", ""),
        ("soil_storage_0_100cm", "_mm"),
        ("soil_drainage_0_100cm", "_mm"),
        ("soil_boundary_waterflux_100cm_signed", "_mm"),
        ("soil_boundary_outflow_100cm", "_mm"),
        ("soil_boundary_depth", "_cm"),
    ):
        required.extend(f"{name}{suffix}" for name in _daily_columns(prefix))
    return required


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce")


def _raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise SmokeValidationError("; ".join(errors))


def validate_smoke_dataset(
    frame: pd.DataFrame,
    *,
    expected_nprintday: int = FORMAL_NPRINTDAY,
    expected_irrigation_options_mm: Sequence[float] = FORMAL_IRRIGATION_OPTIONS_MM,
) -> SmokeValidationResult:
    if frame.empty:
        raise SmokeValidationError("smoke dataset is empty")

    errors: list[str] = []
    missing = [column for column in _required_columns() if column not in frame.columns]
    if missing:
        errors.append("missing required columns: " + ", ".join(missing))
        _raise_if_errors(errors)

    work = frame.copy()
    numeric_columns = [
        column
        for column in _required_columns()
        if column
        not in {
            "site",
            "date_t",
            "flux_integration_method",
            "increment_grouping_method",
            "swap_version",
            "water_depth_unit",
            "flux_rate_source_unit",
            "root_depth_unit",
            "soil_vwc_0_100cm_unit",
            "control_volume_type",
            "data_processing_spec_version",
            "raw_audit_dir",
        }
    ]
    numeric = _numeric(work, numeric_columns)
    invalid_numeric = numeric.columns[numeric.isna().any()].tolist()
    if invalid_numeric:
        errors.append("non-finite numeric values: " + ", ".join(invalid_numeric))
    work[numeric_columns] = numeric

    if not work["nprintday"].eq(int(expected_nprintday)).all():
        values = sorted(work["nprintday"].dropna().unique().tolist())
        errors.append(f"nprintday must be {expected_nprintday}, got {values}")
    if not work["horizon_days_actual"].eq(7).all():
        errors.append("horizon_days_actual must equal 7 for every candidate")
    if not work["flux_integration_method"].eq(FORMAL_FLUX_INTEGRATION_METHOD).all():
        errors.append(
            "flux_integration_method must be " + FORMAL_FLUX_INTEGRATION_METHOD
        )
    if not work["increment_grouping_method"].eq(FORMAL_INCREMENT_GROUPING_METHOD).all():
        errors.append(
            "increment_grouping_method must be " + FORMAL_INCREMENT_GROUPING_METHOD
        )
    expected_metadata = {
        "swap_version": "4.0.1",
        "water_depth_unit": "mm",
        "flux_rate_source_unit": "cm/day",
        "root_depth_unit": "cm",
        "soil_vwc_0_100cm_unit": "cm3/cm3",
        "control_volume_type": FORMAL_CONTROL_VOLUME_TYPE,
        "data_processing_spec_version": FORMAL_DATA_PROCESSING_SPEC_VERSION,
    }
    for column, expected in expected_metadata.items():
        if not work[column].astype(str).eq(expected).all():
            actual = sorted(work[column].astype(str).unique().tolist())
            errors.append(f"{column} must be {expected!r}, got {actual}")
    if not work["control_depth_cm"].eq(FORMAL_CONTROL_DEPTH_CM).all():
        values = sorted(work["control_depth_cm"].dropna().unique().tolist())
        errors.append(
            f"control_depth_cm must be {FORMAL_CONTROL_DEPTH_CM:g}, got {values}"
        )
    legacy_moving_columns = [
        column for column in work.columns if "moving_root_boundary" in column
    ]
    if legacy_moving_columns:
        errors.append(
            "fixed control volume must not contain moving-boundary fields: "
            + ", ".join(legacy_moving_columns)
        )
    if not work["ir"].between(0.0, 60.0).all():
        errors.append("irrigation must stay within [0, 60] mm")
    if work.duplicated(["site", "date_t", "ir"]).any():
        errors.append("duplicate site/date/irrigation candidate rows")

    expected_options = sorted(float(value) for value in expected_irrigation_options_mm)
    for (site, date_t), group in work.groupby(["site", "date_t"], sort=False):
        actual_options = sorted(group["ir"].astype(float).tolist())
        if actual_options != expected_options:
            errors.append(
                f"{site}/{date_t} irrigation candidates must be {expected_options}, "
                f"got {actual_options}"
            )
        expected_raw_audit = {0.0, max(expected_options)}
        preserved = group.loc[
            group["raw_audit_preserved"].eq(1), "ir"
        ].astype(float)
        preserved_options = set(preserved.tolist())
        if preserved_options != expected_raw_audit:
            errors.append(
                f"{site}/{date_t} raw audit must preserve irrigation endpoints "
                f"{sorted(expected_raw_audit)}, got {sorted(preserved_options)}"
            )
        preserved_dirs = group.loc[
            group["raw_audit_preserved"].eq(1), "raw_audit_dir"
        ]
        if preserved_dirs.isna().any() or preserved_dirs.astype(str).str.strip().eq("").any():
            errors.append(f"{site}/{date_t} raw audit directory is missing")

    aet_component_errors = []
    for day in range(1, 8):
        suffix = f"day{day:02d}_mm"
        component_sum = (
            work[f"tact_{suffix}"]
            + work[f"eact_{suffix}"]
            + work[f"interc_{suffix}"]
        )
        aet_component_errors.append((work[f"aet_{suffix}"] - component_sum).abs())
    max_aet_component_error = float(pd.concat(aet_component_errors).max())
    if max_aet_component_error > NUMERIC_TOLERANCE_MM:
        errors.append(
            "AET component identity failed: "
            f"max error={max_aet_component_error:.9f} mm"
        )

    aet_daily_columns = [f"aet_day{day:02d}_mm" for day in range(1, 8)]
    aet_sum_error = (work[aet_daily_columns].sum(axis=1) - work["aet_7d_mm"]).abs()
    if float(aet_sum_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("daily AET sum does not equal aet_7d_mm")

    drainage_daily_columns = [
        f"soil_drainage_0_100cm_day{day:02d}_mm" for day in range(1, 8)
    ]
    drainage_sum_error = (
        work[drainage_daily_columns].sum(axis=1)
        - work["soil_drainage_0_100cm_7d_mm"]
    ).abs()
    if float(drainage_sum_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("daily fixed-layer drainage does not equal the 7-day value")

    signed_flux_daily_columns = [
        f"soil_boundary_waterflux_100cm_signed_day{day:02d}_mm"
        for day in range(1, 8)
    ]
    signed_flux_sum_error = (
        work[signed_flux_daily_columns].sum(axis=1)
        - work["soil_boundary_waterflux_100cm_signed_7d_mm"]
    ).abs()
    if float(signed_flux_sum_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("daily signed 100 cm flux does not equal the 7-day value")

    boundary_outflow_daily_columns = [
        f"soil_boundary_outflow_100cm_day{day:02d}_mm" for day in range(1, 8)
    ]
    boundary_outflow_sum_error = (
        work[boundary_outflow_daily_columns].sum(axis=1)
        - work["soil_boundary_outflow_100cm_7d_mm"]
    ).abs()
    if float(boundary_outflow_sum_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("daily 100 cm outflow does not equal the 7-day value")
    flux_sign_error = (
        work["soil_boundary_waterflux_100cm_signed_7d_mm"]
        + work["soil_boundary_outflow_100cm_7d_mm"]
    ).abs()
    if float(flux_sign_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("100 cm signed flux and downward outflow signs are inconsistent")

    direct_outflow_error = (
        work["runoff_7d_mm"]
        + work["soil_drainage_0_100cm_7d_mm"]
        + work["soil_boundary_outflow_100cm_7d_mm"]
        - work["residual_flux_7d_mm"]
    ).abs()
    if float(direct_outflow_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("direct physical outflow components do not equal residual_flux_7d_mm")

    input_water = (
        work["rain_7d_mm"]
        + work["snow_7d_mm"]
        + work["irrigation_7d_mm"]
        + work["runon_7d_mm"]
    )
    reconstructed_residual = (
        input_water
        - work["aet_7d_mm"]
        - work["delta_soil_storage_0_100cm_7d_mm"]
        - work["residual_flux_7d_mm"]
    )
    balance_error = (
        reconstructed_residual - work["water_balance_residual_0_100cm_7d_mm"]
    ).abs()
    max_balance_error = float(balance_error.max())
    if max_balance_error > NUMERIC_TOLERANCE_MM:
        errors.append(
            "water-balance residual reconstruction failed: "
            f"max error={max_balance_error:.9f} mm"
        )

    vwc_columns = [f"soil_vwc_0_100cm_day{day:02d}" for day in range(1, 8)]
    if not work[vwc_columns].apply(lambda series: series.between(0.0, 1.0)).all().all():
        errors.append("fixed 0-100 cm VWC must stay within [0, 1]")

    storage_identity_errors = []
    boundary_depth_errors = []
    for day in range(1, 8):
        storage_identity_errors.append(
            (
                work[f"soil_storage_0_100cm_day{day:02d}_mm"]
                - 1000.0 * work[f"soil_vwc_0_100cm_day{day:02d}"]
            ).abs()
        )
        boundary_depth_errors.append(
            (work[f"soil_boundary_depth_day{day:02d}_cm"] - 100.0).abs()
        )
    if float(pd.concat(storage_identity_errors).max()) > NUMERIC_TOLERANCE_MM:
        errors.append("fixed storage identity failed: storage_mm != VWC * 1000 mm")
    if float(pd.concat(boundary_depth_errors).max()) > NUMERIC_TOLERANCE_MM:
        errors.append("soil boundary depth must equal 100 cm for every day")

    storage_delta_error = (
        work["final_soil_storage_0_100cm_mm"]
        - work["predecision_soil_storage_0_100cm_mm"]
        - work["delta_soil_storage_0_100cm_7d_mm"]
    ).abs()
    final_daily_storage_error = (
        work["soil_storage_0_100cm_day07_mm"]
        - work["final_soil_storage_0_100cm_mm"]
    ).abs()
    if float(storage_delta_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("fixed 0-100 cm endpoint storage change is inconsistent")
    if float(final_daily_storage_error.max()) > NUMERIC_TOLERANCE_MM:
        errors.append("day07 fixed storage does not equal final fixed storage")

    day8_columns = [column for column in work.columns if "_day08" in column]
    if day8_columns:
        errors.append("unexpected eighth-day fields: " + ", ".join(day8_columns))

    _raise_if_errors(errors)

    site_summary = (
        work.assign(
            abs_water_balance_residual_mm=work[
                "water_balance_residual_0_100cm_7d_mm"
            ].abs(),
        )
        .groupby("site", as_index=False)
        .agg(
            candidate_rows=("ir", "size"),
            decision_dates=("date_t", "nunique"),
            raw_audit_rows=("raw_audit_preserved", "sum"),
            max_abs_water_balance_residual_mm=(
                "abs_water_balance_residual_mm",
                "max",
            ),
            median_abs_water_balance_residual_mm=(
                "abs_water_balance_residual_mm",
                "median",
            ),
        )
    )

    return SmokeValidationResult(
        passed=True,
        row_count=int(len(work)),
        site_count=int(work["site"].nunique()),
        site_date_count=int(work[["site", "date_t"]].drop_duplicates().shape[0]),
        max_abs_water_balance_residual_mm=float(
            work["water_balance_residual_0_100cm_7d_mm"].abs().max()
        ),
        max_abs_aet_component_error_mm=max_aet_component_error,
        max_abs_balance_reconstruction_error_mm=max_balance_error,
        site_summary=site_summary,
    )


def write_validation_outputs(
    result: SmokeValidationResult,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "three_output_smoke_validation_summary_v1.csv"
    report_md = output_dir / "three_output_smoke_validation_v1.md"
    result.site_summary.to_csv(summary_csv, index=False)

    lines = [
        "# Three-Output NPrintDay=24 Smoke Validation",
        "",
        f"- Passed: `{result.passed}`",
        f"- Candidate rows: `{result.row_count}`",
        f"- Sites: `{result.site_count}`",
        f"- Site/date groups: `{result.site_date_count}`",
        "- Formal output frequency: `NPrintDay=24`",
        "- Formal control volume: fixed `0-100 cm`",
        "- Flux integration: actual-Time trapezoidal integration",
        "- Increment grouping: `Dcum=1..7`",
        f"- Maximum absolute water-balance residual: `{result.max_abs_water_balance_residual_mm:.6f} mm`",
        "",
        "The water-balance residual magnitude is reported for teacher review; this validator does not impose an unapproved residual cutoff.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_csv, report_md

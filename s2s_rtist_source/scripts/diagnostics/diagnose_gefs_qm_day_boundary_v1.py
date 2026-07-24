#!/usr/bin/env python3
"""Diagnose UTC versus site-local day boundaries for the GEFS QM pilot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from run_gefs_quantile_mapping_validation_v1 import (
    FIT_YEARS,
    SITE_IDS,
    VALIDATION_YEAR,
    _deterministic_metrics,
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
    _training_summary,
    _write_csv,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    SITE_METADATA,
    aggregate_reforecast_member_daily,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    pair_member_and_reference,
    write_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v1"
    / "gefs_qm_2015_2019_pilot_v1"
)
OUTPUT_DIR = PILOT_DIR / "alignment_diagnostic_utc_day_v1"


def run_diagnostic(
    *, pilot_dir: Path = PILOT_DIR, output_dir: Path = OUTPUT_DIR
) -> dict[str, Path]:
    point_paths = sorted((pilot_dir / "cache" / "point_records").glob("*.csv"))
    if len(point_paths) != 150:
        raise ValueError(f"expected 150 cached point files, found {len(point_paths)}")
    points = pd.concat(
        [pd.read_csv(path, parse_dates=["cycle_init_utc"]) for path in point_paths],
        ignore_index=True,
    )
    points["timezone"] = "UTC"
    manifest = pd.read_csv(pilot_dir / "gefs_reforecast_download_manifest.csv")
    reference = pd.read_csv(
        pilot_dir / "era5_reference_daily_precipitation.csv",
        parse_dates=["local_date"],
    )
    member_daily = aggregate_reforecast_member_daily(points, manifest=manifest)
    member_daily["aggregation_day_boundary"] = "UTC_00_to_24"
    member_daily["site_timezone"] = member_daily["site_id"].map(
        lambda site_id: SITE_METADATA[site_id][2]
    )
    if len(member_daily) != 5250:
        raise ValueError(f"UTC aggregation rows={len(member_daily)}, expected=5250")
    paired = pair_member_and_reference(member_daily, reference)
    years = pd.to_datetime(paired["decision_date"]).dt.year
    fit = paired.loc[years.isin(FIT_YEARS)].copy()
    validation = paired.loc[years.eq(VALIDATION_YEAR)].copy()
    if len(fit) != 4200 or len(validation) != 1050:
        raise ValueError("UTC fit/validation counts do not match the contract sample")

    artifact = fit_empirical_precipitation_qm(
        fit,
        fit_years=FIT_YEARS,
        expected_members=GEFS_REFORECAST_MEMBERS,
    )
    corrected = apply_empirical_precipitation_qm(
        validation, artifact, split="validation_2019_utc_day_diagnostic"
    )
    observations, probabilistic, probabilities = _probabilistic_metrics(
        corrected, members=GEFS_REFORECAST_MEMBERS
    )
    deterministic = _deterministic_metrics(observations)
    seven_day = _seven_day_metrics(corrected)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=corrected,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "gefs_precipitation_qm_artifact_utc_day.json"
    paired_path = output_dir / "paired_raw_and_qm_members_2019_utc_day.csv"
    deterministic_path = output_dir / "deterministic_metrics_utc_day.csv"
    probabilistic_path = output_dir / "probabilistic_metrics_utc_day.csv"
    probability_path = output_dir / "precipitation_probability_utc_day.csv"
    seven_day_path = output_dir / "seven_day_precipitation_utc_day.csv"
    training_path = output_dir / "qm_training_summary_utc_day.csv"
    tail_path = output_dir / "upper_tail_events_utc_day.csv"
    gate_path = output_dir / "promotion_gate_utc_day.json"
    report_path = output_dir / "day_boundary_diagnostic.md"
    write_quantile_mapping_artifact(artifact_path, artifact)
    _write_csv(corrected, paired_path)
    _write_csv(deterministic, deterministic_path)
    _write_csv(probabilistic, probabilistic_path)
    _write_csv(probabilities, probability_path)
    _write_csv(seven_day, seven_day_path)
    _write_csv(_training_summary(artifact), training_path)
    _write_csv(corrected.loc[corrected["qm_extrapolated_upper"]], tail_path)
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    local_gate = json.loads(
        (pilot_dir / "promotion_gate_2019.json").read_text(encoding="utf-8")
    )
    report_path.write_text(
        "\n".join(
            [
                "# GEFS QM day-boundary diagnostic",
                "",
                "This branch aggregates GEFS precipitation by UTC day to match the UTC-day ERA5-Land GeoTIFF source. It does not overwrite the contract's site-local-day result.",
                "",
                "| Metric | Site-local day | UTC day diagnostic |",
                "|---|---:|---:|",
                f"| Raw 7-day MAE (mm) | {local_gate['raw_seven_day_mae_mm']:.6f} | {gate['raw_seven_day_mae_mm']:.6f} |",
                f"| QM 7-day MAE (mm) | {local_gate['qm_seven_day_mae_mm']:.6f} | {gate['qm_seven_day_mae_mm']:.6f} |",
                f"| Raw mean CRPS (mm) | {local_gate['raw_mean_crps_mm']:.6f} | {gate['raw_mean_crps_mm']:.6f} |",
                f"| QM mean CRPS (mm) | {local_gate['qm_mean_crps_mm']:.6f} | {gate['qm_mean_crps_mm']:.6f} |",
                f"| Raw mean Brier | {local_gate['raw_mean_brier_score']:.6f} | {gate['raw_mean_brier_score']:.6f} |",
                f"| QM mean Brier | {local_gate['qm_mean_brier_score']:.6f} | {gate['qm_mean_brier_score']:.6f} |",
                f"| Upper-tail rows | {local_gate['upper_tail_extrapolation_count']} | {gate['upper_tail_extrapolation_count']} |",
                "",
                f"UTC diagnostic promotion status: `{gate['promotion_status']}`.",
                "",
                "The comparison tests temporal alignment sensitivity only. A final correction module must use a forecast/reference day convention consistent with the weather data that generated SWAP labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "artifact": artifact_path,
        "paired": paired_path,
        "gate": gate_path,
        "report": report_path,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {key: str(value) for key, value in run_diagnostic().items()}, indent=2
        )
    )

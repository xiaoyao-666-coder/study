#!/usr/bin/env python3
"""Reproduce the GEFS precipitation QM v2 pilot from verified offline caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gefs_quantile_mapping_validation_v1 import (
    FIT_YEARS,
    SITE_IDS,
    VALIDATION_YEAR,
    _deterministic_metrics,
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
    _write_csv,
    pilot_cycle_dates,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    CONTRACT_VERSION_V2,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    aggregate_reforecast_member_daily_utc,
    apply_empirical_precipitation_qm,
    cycle_valid_dates,
    extract_era5_reference_precipitation_utc,
    fit_empirical_precipitation_qm,
    pair_member_and_reference,
    reforecast_site_frame,
    validate_member_daily_precipitation,
    validate_reference_daily_precipitation,
    write_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_data_contract_v2.json"
)
DEFAULT_SOURCE_PILOT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v1"
    / "gefs_qm_2015_2019_pilot_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
)
DEFAULT_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_point_cache(
    source_pilot_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_paths = sorted((source_pilot_dir / "cache" / "point_records").glob("*.csv"))
    metadata_paths = sorted((source_pilot_dir / "cache" / "metadata").glob("*.json"))
    index_paths = sorted((source_pilot_dir / "cache" / "indices").glob("*.idx"))
    counts = {
        "point records": len(point_paths),
        "metadata records": len(metadata_paths),
        "index records": len(index_paths),
    }
    if any(count != 150 for count in counts.values()):
        raise ValueError(f"verified cache requires 150 files per type: {counts}")
    points = pd.concat(
        [pd.read_csv(path, parse_dates=["cycle_init_utc"]) for path in point_paths],
        ignore_index=True,
    )
    manifest_path = source_pilot_dir / "gefs_reforecast_download_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 150:
        raise ValueError(f"source manifest rows={len(manifest)}, expected=150")
    if manifest["source_etag"].astype(str).str.strip().eq("").any():
        raise ValueError("source manifest contains an empty ETag")
    return points, manifest


def _training_summary_v2(artifact: dict[str, object]) -> pd.DataFrame:
    rows = []
    for key, group in artifact["groups"].items():
        rows.append(
            {
                "group_id": key,
                "site_id": group["site_id"],
                "lead_day": group["lead_day"],
                "sample_count": group["sample_count"],
                "reference_wet_sample_count": group[
                    "reference_wet_sample_count"
                ],
                "forecast_positive_sample_count": group[
                    "forecast_positive_sample_count"
                ],
                "forecast_wet_threshold_mm": group["forecast_wet_threshold_mm"],
                "effective_quantile_node_count": group[
                    "effective_quantile_node_count"
                ],
                "training_forecast_maximum_mm": group[
                    "training_forecast_maximum_mm"
                ],
                "training_reference_maximum_mm": group[
                    "training_reference_maximum_mm"
                ],
                "upper_tail_additive_offset_mm": group[
                    "upper_tail_additive_offset_mm"
                ],
            }
        )
    return pd.DataFrame(rows).sort_values(["site_id", "lead_day"])


def _upper_tail_audit(
    corrected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    tail = corrected.loc[corrected["qm_extrapolated_upper"]].copy()
    tail["raw_absolute_error_mm"] = (
        tail["precipitation_mm_raw"] - tail["precipitation_mm_reference"]
    ).abs()
    tail["qm_absolute_error_mm"] = (
        tail["precipitation_mm_qm"] - tail["precipitation_mm_reference"]
    ).abs()
    tail["absolute_error_change_mm"] = (
        tail["qm_absolute_error_mm"] - tail["raw_absolute_error_mm"]
    )
    corrected_values = tail["precipitation_mm_qm"].to_numpy(dtype=float)
    finite = np.isfinite(corrected_values)
    summary = {
        "event_count": int(len(tail)),
        "nonfinite_corrected_count": int((~finite).sum()),
        "negative_corrected_count": int((corrected_values < 0.0).sum()),
        "maximum_corrected_mm_day": (
            float(corrected_values.max()) if len(corrected_values) else None
        ),
        "improved_count": int(tail["absolute_error_change_mm"].lt(0.0).sum()),
        "worsened_count": int(tail["absolute_error_change_mm"].gt(0.0).sum()),
        "mean_absolute_error_change_mm_day": (
            float(tail["absolute_error_change_mm"].mean())
            if len(tail)
            else None
        ),
    }
    summary["numeric_audit_passed"] = bool(
        summary["event_count"] > 0
        and summary["nonfinite_corrected_count"] == 0
        and summary["negative_corrected_count"] == 0
    )
    summary["audit_status"] = (
        "passed_with_residual_extreme_error_caveat"
        if summary["numeric_audit_passed"]
        else "failed_numeric_audit"
    )
    return tail, summary


def _rename_metric_date(frame: pd.DataFrame) -> pd.DataFrame:
    if "local_date" not in frame.columns:
        return frame
    return frame.rename(columns={"local_date": "valid_date_utc"})


def run_validation_v2(
    *,
    source_pilot_dir: Path,
    output_dir: Path,
    era5_root: Path,
) -> dict[str, Path]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["contract_id"] != CONTRACT_ID_V2:
        raise ValueError("v2 contract id does not match the implementation")
    if contract["contract_version"] != CONTRACT_VERSION_V2:
        raise ValueError("v2 contract version does not match the implementation")

    output_dir.mkdir(parents=True, exist_ok=True)
    points, manifest = load_verified_point_cache(source_pilot_dir)
    cycle_dates = pilot_cycle_dates()
    sites = reforecast_site_frame(SITE_IDS)
    members = GEFS_REFORECAST_MEMBERS
    member_daily = aggregate_reforecast_member_daily_utc(
        points, manifest=manifest
    )
    all_valid_dates = sorted(
        {valid_date for cycle in cycle_dates for valid_date in cycle_valid_dates(cycle)}
    )
    reference = extract_era5_reference_precipitation_utc(
        era5_root=era5_root,
        sites=sites,
        valid_dates=all_valid_dates,
    )
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=SITE_IDS,
        expected_members=members,
        expected_cycles=cycle_dates,
        date_column="valid_date_utc",
    )
    validate_reference_daily_precipitation(
        reference,
        expected_sites=SITE_IDS,
        expected_dates=all_valid_dates,
        date_column="valid_date_utc",
    )
    expected = contract["expected_counts"]
    total_member_rows = expected["fit_member_rows"] + expected["validation_member_rows"]
    total_reference_rows = (
        expected["fit_unique_reference_observations"]
        + expected["validation_unique_reference_observations"]
    )
    if len(member_daily) != total_member_rows or len(reference) != total_reference_rows:
        raise ValueError("v2 fixed-sample extraction does not match contract counts")

    member_path = output_dir / "gefs_reforecast_member_daily_precipitation_utc_v2.csv"
    manifest_path = output_dir / "gefs_reforecast_download_manifest.csv"
    reference_path = output_dir / "era5_reference_daily_precipitation_utc_v2.csv"
    _write_csv(member_daily, member_path)
    _write_csv(manifest, manifest_path)
    _write_csv(reference, reference_path)

    paired = pair_member_and_reference(
        member_daily, reference, date_column="valid_date_utc"
    )
    years = pd.to_datetime(paired["decision_date"]).dt.year
    fit = paired.loc[years.isin(FIT_YEARS)].copy()
    validation = paired.loc[years.eq(VALIDATION_YEAR)].copy()
    if len(fit) != expected["fit_member_rows"]:
        raise ValueError("v2 fit row count does not match the contract")
    if len(validation) != expected["validation_member_rows"]:
        raise ValueError("v2 validation row count does not match the contract")

    artifact = fit_empirical_precipitation_qm(
        fit,
        fit_years=FIT_YEARS,
        expected_members=members,
        contract_id=CONTRACT_ID_V2,
        contract_version=CONTRACT_VERSION_V2,
        aggregation_day_boundary=UTC_DAY_BOUNDARY,
        canonical_valid_date_column="valid_date_utc",
        upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
    )
    artifact_path = output_dir / "gefs_precipitation_qm_artifact_v2.json"
    write_quantile_mapping_artifact(artifact_path, artifact)
    training_summary = _training_summary_v2(artifact)
    training_summary_path = (
        output_dir / "gefs_precipitation_qm_training_summary_v2.csv"
    )
    _write_csv(training_summary, training_summary_path)

    training_manifest = {
        "contract_id": contract["contract_id"],
        "contract_version": contract["contract_version"],
        "aggregation_day_boundary": UTC_DAY_BOUNDARY,
        "canonical_valid_date_column": "valid_date_utc",
        "fit_years": list(FIT_YEARS),
        "fit_member_rows": int(len(fit)),
        "fit_unique_reference_observations": int(
            fit[["site_id", "decision_date", "valid_date_utc"]]
            .drop_duplicates()
            .shape[0]
        ),
        "site_ids": list(SITE_IDS),
        "members": list(members),
        "artifact_sha256": artifact["artifact_sha256"],
        "training_input_sha256": artifact["training_input_sha256"],
        "member_daily_file_sha256": _sha256_file(member_path),
        "reference_daily_file_sha256": _sha256_file(reference_path),
        "source_manifest_file_sha256": _sha256_file(manifest_path),
        "validation_or_test_years_used_for_fit": [],
        "network_download_performed": False,
        "source_point_cache_directory": str(
            source_pilot_dir / "cache" / "point_records"
        ),
    }
    training_manifest_path = (
        output_dir / "gefs_precipitation_qm_training_manifest_v2.json"
    )
    training_manifest_path.write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    corrected = apply_empirical_precipitation_qm(
        validation, artifact, split="validation_2019_v2"
    )
    paired_path = output_dir / "paired_raw_and_qm_members_2019_v2.csv"
    _write_csv(corrected, paired_path)

    metric_input = corrected.rename(columns={"valid_date_utc": "local_date"})
    observations, probabilistic, probabilities = _probabilistic_metrics(
        metric_input, members=members
    )
    deterministic = _deterministic_metrics(observations)
    seven_day = _seven_day_metrics(metric_input)
    automatic_gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=metric_input,
    )
    tail_events, tail_audit = _upper_tail_audit(corrected)
    gate = dict(automatic_gate)
    gate["contract_id"] = CONTRACT_ID_V2
    gate["contract_version"] = CONTRACT_VERSION_V2
    gate["aggregation_day_boundary"] = UTC_DAY_BOUNDARY
    gate["upper_tail_policy"] = UPPER_TAIL_CONSTANT_ADDITIVE
    gate["upper_tail_numeric_audit"] = tail_audit
    gate["upper_tail_review_status"] = tail_audit["audit_status"]
    gate["promotion_status"] = (
        "passed_for_fuller_2019_validation"
        if gate["automatic_requirements_passed"]
        and tail_audit["numeric_audit_passed"]
        else "failed_v2_promotion_requirements"
    )

    deterministic_path = (
        output_dir / "deterministic_metrics_raw_vs_qm_2019_v2.csv"
    )
    probabilistic_path = (
        output_dir / "probabilistic_metrics_raw_vs_qm_2019_v2.csv"
    )
    probability_path = (
        output_dir / "precipitation_probability_raw_vs_qm_2019_v2.csv"
    )
    seven_day_path = output_dir / "seven_day_precipitation_raw_vs_qm_2019_v2.csv"
    tail_path = output_dir / "upper_tail_events_2019_v2.csv"
    tail_summary_path = output_dir / "upper_tail_audit_summary_2019_v2.json"
    gate_path = output_dir / "promotion_gate_2019_v2.json"
    report_path = output_dir / "validation_scope_and_conclusion_v2.md"
    _write_csv(deterministic, deterministic_path)
    _write_csv(_rename_metric_date(probabilistic), probabilistic_path)
    _write_csv(_rename_metric_date(probabilities), probability_path)
    _write_csv(seven_day, seven_day_path)
    _write_csv(tail_events, tail_path)
    tail_summary_path.write_text(
        json.dumps(tail_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        "\n".join(
            [
                "# GEFS precipitation QM v2 fixed-sample validation",
                "",
                "v2 uses UTC 00:00-24:00 aggregation and constant-additive upper-tail extrapolation.",
                "It fits only 2015-2018 and validates on the six fixed 2019 cycles.",
                "",
                f"- Fit member rows: `{len(fit)}`.",
                f"- Validation member rows: `{len(validation)}`.",
                f"- Frozen artifact SHA-256: `{artifact['artifact_sha256']}`.",
                f"- Raw 7-day MAE: `{gate['raw_seven_day_mae_mm']:.6f} mm`.",
                f"- QM 7-day MAE: `{gate['qm_seven_day_mae_mm']:.6f} mm`.",
                f"- Raw mean CRPS: `{gate['raw_mean_crps_mm']:.6f} mm`.",
                f"- QM mean CRPS: `{gate['qm_mean_crps_mm']:.6f} mm`.",
                f"- Raw mean Brier: `{gate['raw_mean_brier_score']:.6f}`.",
                f"- QM mean Brier: `{gate['qm_mean_brier_score']:.6f}`.",
                f"- Upper-tail events: `{tail_audit['event_count']}`.",
                f"- Promotion status: `{gate['promotion_status']}`.",
                "",
                "Passing this pilot permits only fuller 2019 validation. It does not permit model training, 2024 tuning, or production use.",
                "Residual extreme errors remain visible in the event table and must not be described as fully corrected.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "member_daily": member_path,
        "manifest": manifest_path,
        "reference": reference_path,
        "artifact": artifact_path,
        "training_summary": training_summary_path,
        "training_manifest": training_manifest_path,
        "paired_validation": paired_path,
        "deterministic_metrics": deterministic_path,
        "probabilistic_metrics": probabilistic_path,
        "probability_metrics": probability_path,
        "seven_day_metrics": seven_day_path,
        "tail_events": tail_path,
        "tail_audit": tail_summary_path,
        "promotion_gate": gate_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pilot-dir", type=Path, default=DEFAULT_SOURCE_PILOT_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_validation_v2(
        source_pilot_dir=args.source_pilot_dir,
        output_dir=args.output_dir,
        era5_root=args.era5_root,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

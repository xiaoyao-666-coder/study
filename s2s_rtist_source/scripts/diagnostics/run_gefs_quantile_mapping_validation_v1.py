#!/usr/bin/env python3
"""Run the contract-scale 2015-2018 fit and 2019 GEFS precipitation QM pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from s2s_rtist.weather.gefs_ensemble_validation import (
    aggregate_probabilistic_metrics,
    compute_precipitation_probability_metrics,
    summarize_ensemble_observations,
)
from s2s_rtist.weather.gefs_gridmet_bias import add_reference_condition
from s2s_rtist.weather.gefs_quantile_mapping import (
    GEFS_REFORECAST_MEMBERS,
    aggregate_reforecast_member_daily,
    apply_empirical_precipitation_qm,
    cycle_valid_dates,
    download_reforecast_member_points,
    extract_era5_reference_precipitation,
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
    / "gefs_quantile_mapping_data_contract_v1.json"
)
DEFAULT_ERA5_ROOT = PROJECT_ROOT / "model3_opt_sto_upload" / "data"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v1"
    / "gefs_qm_2015_2019_pilot_v1"
)
MONTH_DAYS = ("06-01", "06-15", "07-01", "07-15", "08-01", "08-15")
FIT_YEARS = (2015, 2016, 2017, 2018)
VALIDATION_YEAR = 2019
SITE_IDS = ("P1", "P2", "P3", "P4", "P15")


def pilot_cycle_dates() -> tuple[str, ...]:
    return tuple(
        f"{year}-{month_day}"
        for year in (*FIT_YEARS, VALIDATION_YEAR)
        for month_day in MONTH_DAYS
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def extract_contract_sample(
    *,
    cycle_dates: Sequence[str],
    sites: pd.DataFrame,
    members: Sequence[str],
    cache_dir: Path,
    partial_manifest_path: Path,
    timeout: int,
    retries: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(cycle_date, member) for cycle_date in cycle_dates for member in members]
    point_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    def run_task(task: tuple[str, str]):
        cycle_date, member = task
        points, metadata = download_reforecast_member_points(
            cycle_date=cycle_date,
            member=member,
            sites=sites,
            cache_dir=cache_dir,
            timeout=timeout,
            retries=retries,
            keep_grib=False,
        )
        return task, points, metadata

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_task, task): task for task in tasks}
        errors: list[tuple[tuple[str, str], Exception]] = []
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                (cycle_date, member), points, metadata = future.result()
            except Exception as exc:
                task = futures[future]
                errors.append((task, exc))
                print(
                    f"[pilot] {task[0]} {task[1]} failed: {exc}", flush=True
                )
                continue
            point_parts.append(points)
            manifest_rows.append(metadata)
            if completed % 5 == 0 or completed == len(tasks):
                partial = pd.DataFrame(manifest_rows).sort_values(
                    ["cycle_date", "gefs_member"]
                )
                _write_csv(partial, partial_manifest_path)
            print(
                f"[pilot] {cycle_date} {member} ready ({completed}/{len(tasks)})",
                flush=True,
            )
    if errors:
        examples = "; ".join(
            f"{cycle} {member}: {error}"
            for (cycle, member), error in errors[:5]
        )
        raise RuntimeError(
            f"{len(errors)} GEFS reforecast tasks failed after retries: {examples}"
        )
    points = pd.concat(point_parts, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["cycle_date", "gefs_member"]
    ).reset_index(drop=True)
    return points, manifest


def _validation_long(
    paired: pd.DataFrame, *, method: str, forecast_column: str
) -> pd.DataFrame:
    output = paired[
        [
            "site_id",
            "decision_date",
            "local_date",
            "lead_day",
            "gefs_member",
            forecast_column,
            "precipitation_mm_reference",
        ]
    ].copy()
    output = output.rename(
        columns={
            "site_id": "site",
            forecast_column: "forecast_value",
            "precipitation_mm_reference": "reference_value",
        }
    )
    output["variable"] = "precipitation_mm"
    output["method"] = method
    return output


def _probabilistic_metrics(
    paired: pd.DataFrame, *, members: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observation_parts = []
    probability_parts = []
    for method, column in (
        ("GEFS_raw", "precipitation_mm_raw"),
        ("GEFS_QM", "precipitation_mm_qm"),
    ):
        long = _validation_long(paired, method=method, forecast_column=column)
        observations = summarize_ensemble_observations(
            long, expected_members=members
        )
        observations["method"] = method
        observation_parts.append(add_reference_condition(observations))
        overall_probability = compute_precipitation_probability_metrics(
            long, expected_members=members
        )
        overall_probability["scope"] = "overall"
        overall_probability["scope_value"] = "all"
        by_lead = compute_precipitation_probability_metrics(
            long, expected_members=members, group_columns=("lead_day",)
        )
        by_lead["scope"] = "lead_day"
        by_lead["scope_value"] = by_lead["lead_day"].astype(str)
        by_site = compute_precipitation_probability_metrics(
            long, expected_members=members, group_columns=("site",)
        )
        by_site["scope"] = "site"
        by_site["scope_value"] = by_site["site"].astype(str)
        probability = pd.concat(
            [overall_probability, by_lead, by_site], ignore_index=True
        )
        probability["method"] = method
        probability_parts.append(probability)

    observations = pd.concat(observation_parts, ignore_index=True)
    metric_parts = []
    for scope, groups in (
        ("overall", ("method",)),
        ("site", ("method", "site")),
        ("lead_day", ("method", "lead_day")),
        ("condition", ("method", "reference_condition")),
    ):
        metrics = aggregate_probabilistic_metrics(
            observations, group_columns=groups
        )
        metrics["scope"] = scope
        if scope == "overall":
            metrics["scope_value"] = "all"
        else:
            metrics["scope_value"] = metrics[groups[-1]].astype(str)
        metric_parts.append(metrics)
    metrics = pd.concat(metric_parts, ignore_index=True)
    probabilities = pd.concat(probability_parts, ignore_index=True)
    return observations, metrics, probabilities


def _deterministic_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = (
        ("overall", ()),
        ("site", ("site",)),
        ("lead_day", ("lead_day",)),
        ("condition", ("reference_condition",)),
    )
    for method, method_rows in observations.groupby("method", sort=True):
        for scope, group_columns in scopes:
            if group_columns:
                grouped = method_rows.groupby(
                    list(group_columns), sort=True, dropna=False
                )
            else:
                grouped = [("all", method_rows)]
            for keys, group in grouped:
                if not isinstance(keys, tuple):
                    keys = (keys,)
                error = (
                    group["ensemble_mean"].to_numpy(dtype=float)
                    - group["reference_value"].to_numpy(dtype=float)
                )
                forecast_wet = group["ensemble_mean"].ge(0.1)
                reference_wet = group["reference_value"].ge(0.1)
                rows.append(
                    {
                        "method": method,
                        "scope": scope,
                        "scope_value": "all" if not group_columns else str(keys[-1]),
                        "n_observations": int(len(group)),
                        "bias": float(error.mean()),
                        "mae": float(np.abs(error).mean()),
                        "rmse": float(np.sqrt(np.mean(error * error))),
                        "forecast_wet_frequency": float(forecast_wet.mean()),
                        "reference_wet_frequency": float(reference_wet.mean()),
                        "wet_day_frequency_error": float(
                            forecast_wet.mean() - reference_wet.mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _seven_day_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    member_totals = paired.groupby(
        ["site_id", "decision_date", "gefs_member"], as_index=False
    ).agg(
        precipitation_mm_raw=("precipitation_mm_raw", "sum"),
        precipitation_mm_qm=("precipitation_mm_qm", "sum"),
        precipitation_mm_reference=("precipitation_mm_reference", "sum"),
    )
    rows = []
    for (site_id, decision_date), group in member_totals.groupby(
        ["site_id", "decision_date"], sort=True
    ):
        reference = float(group["precipitation_mm_reference"].iloc[0])
        for method, column in (
            ("GEFS_raw", "precipitation_mm_raw"),
            ("GEFS_QM", "precipitation_mm_qm"),
        ):
            values = group[column].to_numpy(dtype=float)
            mean_value = float(values.mean())
            rows.append(
                {
                    "site_id": site_id,
                    "decision_date": decision_date,
                    "method": method,
                    "member_count": int(len(values)),
                    "ensemble_mean_7d_mm": mean_value,
                    "ensemble_min_7d_mm": float(values.min()),
                    "ensemble_p10_7d_mm": float(np.quantile(values, 0.1)),
                    "ensemble_p90_7d_mm": float(np.quantile(values, 0.9)),
                    "ensemble_max_7d_mm": float(values.max()),
                    "reference_7d_mm": reference,
                    "error_7d_mm": mean_value - reference,
                    "absolute_error_7d_mm": abs(mean_value - reference),
                }
            )
    return pd.DataFrame(rows)


def _training_summary(artifact: dict[str, object]) -> pd.DataFrame:
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
                "forecast_wet_threshold_mm": group[
                    "forecast_wet_threshold_mm"
                ],
                "effective_quantile_node_count": group[
                    "effective_quantile_node_count"
                ],
                "training_forecast_maximum_mm": group[
                    "training_forecast_maximum_mm"
                ],
                "training_reference_maximum_mm": group[
                    "training_reference_maximum_mm"
                ],
                "upper_tail_multiplicative_ratio": group[
                    "upper_tail_multiplicative_ratio"
                ],
            }
        )
    return pd.DataFrame(rows).sort_values(["site_id", "lead_day"])


def _promotion_gate(
    *,
    observations: pd.DataFrame,
    probabilities: pd.DataFrame,
    seven_day: pd.DataFrame,
    paired: pd.DataFrame,
) -> dict[str, object]:
    seven_day_mae = seven_day.groupby("method")["absolute_error_7d_mm"].mean()
    mean_crps = observations.groupby("method")["crps"].mean()
    probability_overall = probabilities.loc[probabilities["scope"].eq("overall")]
    mean_brier = probability_overall.groupby("method")["brier_score"].mean()
    heavy = observations.loc[observations["reference_condition"].eq("heavy")]
    if heavy.empty:
        heavy_coverage_passed = True
        heavy_coverage_reason = "no heavy events in the validation sample"
    else:
        coverage = heavy.groupby("method").agg(
            p10_p90=("covered_by_p10_p90", "mean"),
            min_max=("covered_by_min_max", "mean"),
        )
        heavy_coverage_passed = not (
            coverage.loc["GEFS_QM", "p10_p90"]
            < coverage.loc["GEFS_raw", "p10_p90"]
            and coverage.loc["GEFS_QM", "min_max"]
            < coverage.loc["GEFS_raw", "min_max"]
        )
        heavy_coverage_reason = coverage.to_dict()
    extrapolation_count = int(paired["qm_extrapolated_upper"].sum())
    automatic = {
        "seven_day_mae_not_worse": bool(
            seven_day_mae["GEFS_QM"] <= seven_day_mae["GEFS_raw"]
        ),
        "crps_not_worse": bool(mean_crps["GEFS_QM"] <= mean_crps["GEFS_raw"]),
        "mean_brier_not_worse": bool(
            mean_brier["GEFS_QM"] <= mean_brier["GEFS_raw"]
        ),
        "heavy_coverage_not_both_worse": bool(heavy_coverage_passed),
    }
    automatic_passed = all(automatic.values())
    return {
        "automatic_requirements": automatic,
        "automatic_requirements_passed": automatic_passed,
        "raw_seven_day_mae_mm": float(seven_day_mae["GEFS_raw"]),
        "qm_seven_day_mae_mm": float(seven_day_mae["GEFS_QM"]),
        "raw_mean_crps_mm": float(mean_crps["GEFS_raw"]),
        "qm_mean_crps_mm": float(mean_crps["GEFS_QM"]),
        "raw_mean_brier_score": float(mean_brier["GEFS_raw"]),
        "qm_mean_brier_score": float(mean_brier["GEFS_QM"]),
        "heavy_coverage_evidence": heavy_coverage_reason,
        "upper_tail_extrapolation_count": extrapolation_count,
        "upper_tail_review_status": (
            "pending_manual_numeric_audit" if extrapolation_count else "passed_no_events"
        ),
        "promotion_status": (
            "failed_automatic_requirements"
            if not automatic_passed
            else (
                "pending_upper_tail_review"
                if extrapolation_count
                else "passed_for_fuller_2019_validation"
            )
        ),
    }


def run_validation(
    *,
    output_dir: Path,
    era5_root: Path,
    workers: int,
    timeout: int,
    retries: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    cycle_dates = pilot_cycle_dates()
    sites = reforecast_site_frame(SITE_IDS)
    members = GEFS_REFORECAST_MEMBERS
    points, manifest = extract_contract_sample(
        cycle_dates=cycle_dates,
        sites=sites,
        members=members,
        cache_dir=output_dir / "cache",
        partial_manifest_path=output_dir
        / "gefs_reforecast_download_manifest.partial.csv",
        timeout=timeout,
        retries=retries,
        workers=workers,
    )
    member_daily = aggregate_reforecast_member_daily(points, manifest=manifest)
    all_valid_dates = sorted(
        {valid_date for cycle in cycle_dates for valid_date in cycle_valid_dates(cycle)}
    )
    reference = extract_era5_reference_precipitation(
        era5_root=era5_root, sites=sites, valid_dates=all_valid_dates
    )
    validate_member_daily_precipitation(
        member_daily,
        expected_sites=SITE_IDS,
        expected_members=members,
        expected_cycles=cycle_dates,
    )
    validate_reference_daily_precipitation(
        reference, expected_sites=SITE_IDS, expected_dates=all_valid_dates
    )
    if len(member_daily) != 5250 or len(reference) != 1050:
        raise ValueError("full pilot extraction does not match the contract counts")

    member_path = output_dir / "gefs_reforecast_member_daily_precipitation.csv"
    manifest_path = output_dir / "gefs_reforecast_download_manifest.csv"
    reference_path = output_dir / "era5_reference_daily_precipitation.csv"
    _write_csv(member_daily, member_path)
    _write_csv(manifest, manifest_path)
    _write_csv(reference, reference_path)
    paired = pair_member_and_reference(member_daily, reference)
    years = pd.to_datetime(paired["decision_date"]).dt.year
    fit = paired.loc[years.isin(FIT_YEARS)].copy()
    validation = paired.loc[years.eq(VALIDATION_YEAR)].copy()
    if len(fit) != 4200 or len(validation) != 1050:
        raise ValueError("fit/validation row counts do not match the contract")

    artifact = fit_empirical_precipitation_qm(fit, fit_years=FIT_YEARS)
    artifact_path = output_dir / "gefs_precipitation_qm_artifact.json"
    write_quantile_mapping_artifact(artifact_path, artifact)
    training_summary = _training_summary(artifact)
    training_summary_path = output_dir / "gefs_precipitation_qm_training_summary.csv"
    _write_csv(training_summary, training_summary_path)
    training_manifest = {
        "contract_id": contract["contract_id"],
        "contract_version": contract["contract_version"],
        "fit_years": list(FIT_YEARS),
        "fit_member_rows": int(len(fit)),
        "fit_unique_reference_observations": int(
            fit[["site_id", "decision_date", "local_date"]].drop_duplicates().shape[0]
        ),
        "site_ids": list(SITE_IDS),
        "members": list(members),
        "artifact_sha256": artifact["artifact_sha256"],
        "training_input_sha256": artifact["training_input_sha256"],
        "member_daily_file_sha256": _sha256_file(member_path),
        "reference_daily_file_sha256": _sha256_file(reference_path),
        "source_manifest_file_sha256": _sha256_file(manifest_path),
        "validation_or_test_years_used_for_fit": [],
    }
    training_manifest_path = (
        output_dir / "gefs_precipitation_qm_training_manifest.json"
    )
    training_manifest_path.write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    corrected = apply_empirical_precipitation_qm(
        validation, artifact, split="validation_2019"
    )
    paired_path = output_dir / "paired_raw_and_qm_members_2019.csv"
    _write_csv(corrected, paired_path)
    observations, probabilistic, probabilities = _probabilistic_metrics(
        corrected, members=members
    )
    deterministic = _deterministic_metrics(observations)
    seven_day = _seven_day_metrics(corrected)
    probabilistic_path = output_dir / "probabilistic_metrics_raw_vs_qm_2019.csv"
    probability_path = output_dir / "precipitation_probability_raw_vs_qm_2019.csv"
    deterministic_path = output_dir / "deterministic_metrics_raw_vs_qm_2019.csv"
    seven_day_path = output_dir / "seven_day_precipitation_raw_vs_qm_2019.csv"
    _write_csv(probabilistic, probabilistic_path)
    _write_csv(probabilities, probability_path)
    _write_csv(deterministic, deterministic_path)
    _write_csv(seven_day, seven_day_path)
    tail_events = corrected.loc[corrected["qm_extrapolated_upper"]].copy()
    tail_path = output_dir / "upper_tail_extrapolation_events_2019.csv"
    _write_csv(tail_events, tail_path)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=corrected,
    )
    gate_path = output_dir / "promotion_gate_2019.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "validation_scope_and_conclusion.md"
    report_path.write_text(
        "\n".join(
            [
                "# GEFS precipitation QM 2019 pilot validation",
                "",
                "## Scope",
                "",
                "- QM fit: 2015-2018 only.",
                "- Validation: 2019 only.",
                "- Sites: P1, P2, P3, P4, P15.",
                "- Fixed cycles per year: 06-01, 06-15, 07-01, 07-15, 08-01, 08-15.",
                "- No 2024 values were used and no surrogate model was trained.",
                "",
                "## Structural evidence",
                "",
                f"- Fit member rows: `{len(fit)}`.",
                f"- Validation member rows: `{len(validation)}`.",
                f"- Reference observations: `{len(reference)}` across 2015-2019.",
                f"- Frozen artifact SHA-256: `{artifact['artifact_sha256']}`.",
                "",
                "## Automatic promotion gate",
                "",
                f"- Raw 7-day MAE: `{gate['raw_seven_day_mae_mm']:.6f} mm`.",
                f"- QM 7-day MAE: `{gate['qm_seven_day_mae_mm']:.6f} mm`.",
                f"- Raw mean CRPS: `{gate['raw_mean_crps_mm']:.6f} mm`.",
                f"- QM mean CRPS: `{gate['qm_mean_crps_mm']:.6f} mm`.",
                f"- Raw mean Brier score: `{gate['raw_mean_brier_score']:.6f}`.",
                f"- QM mean Brier score: `{gate['qm_mean_brier_score']:.6f}`.",
                f"- Upper-tail extrapolation rows: `{gate['upper_tail_extrapolation_count']}`.",
                f"- Status before numeric tail audit: `{gate['promotion_status']}`.",
                "",
                "This pilot can only decide whether to expand 2019 validation. It cannot authorize model training or establish a final 2024 correction.",
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
        "probabilistic_metrics": probabilistic_path,
        "probability_metrics": probability_path,
        "deterministic_metrics": deterministic_path,
        "seven_day_metrics": seven_day_path,
        "tail_events": tail_path,
        "promotion_gate": gate_path,
        "report": report_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    outputs = run_validation(
        output_dir=args.output_dir,
        era5_root=args.era5_root,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

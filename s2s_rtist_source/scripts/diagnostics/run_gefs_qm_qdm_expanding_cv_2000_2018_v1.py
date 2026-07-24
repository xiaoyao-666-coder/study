#!/usr/bin/env python3
"""Run prelocked expanding-window QM/QDM CV with selected GHCN stations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_qdm_2019_station_reference_v1 import (
    complete_site_cycle_rows,
)
from scripts.diagnostics.run_gefs_qm_training_cv_v1 import (
    _metric_bundle,
    _metric_row,
)
from s2s_rtist.weather.gefs_quantile_delta_mapping import (
    apply_offline_precipitation_qdm,
    fit_offline_precipitation_qdm,
    verify_qdm_artifact,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    CONTRACT_VERSION_V2,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    write_quantile_mapping_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRED_2000_2002 = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_reforecast_2000_2002_smoke_v1"
    / "gefs_ghcnd_paired_member_daily_2000_2002_v1.csv"
)
DEFAULT_PAIRED_2003_2014 = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_reforecast_2003_2014_expansion_v1"
    / "gefs_ghcnd_paired_member_daily_2003_2014_v1.csv"
)
DEFAULT_MEMBER_2015_2019 = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
    / "gefs_reforecast_member_daily_precipitation_utc_v2.csv"
)
DEFAULT_REFERENCE_2000_2019 = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "selected_station_reference"
    / "ghcnd_selected_station_daily_precipitation_2000_2019_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qm_qdm_expanding_cv_2000_2018_v1"
)
VALIDATION_YEARS = (2015, 2016, 2017, 2018)
CANDIDATES = {
    "qm_global": ("qm", ()),
    "qm_site_only": ("qm", ("site_id",)),
    "qdm_global": ("qdm", ()),
    "qdm_site_only": ("qdm", ("site_id",)),
}


def expanding_folds() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "fold_id": f"F{validation_year}",
            "fit_years": tuple(range(2000, validation_year)),
            "validation_year": validation_year,
        }
        for validation_year in VALIDATION_YEARS
    )


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )


def load_inputs(args: argparse.Namespace) -> pd.DataFrame:
    parts = []
    for path in (args.paired_2000_2002, args.paired_2003_2014):
        frame = pd.read_csv(path)
        frame["reference_valid_unflagged"] = as_bool(
            frame["reference_valid_unflagged"]
        )
        parts.append(frame)

    member = pd.read_csv(args.member_2015_2019)
    member["valid_date_utc"] = pd.to_datetime(member["valid_date_utc"])
    reference = pd.read_csv(args.reference_2000_2019)
    reference["valid_date_utc"] = pd.to_datetime(reference["station_record_date"])
    reference["reference_valid_unflagged"] = as_bool(
        reference["reference_valid_unflagged"]
    )
    recent = member.merge(
        reference[
            [
                "site_id",
                "valid_date_utc",
                "ghcnd_station_id",
                "precipitation_mm_reference",
                "reference_valid_unflagged",
                "q_flag",
            ]
        ],
        on=["site_id", "valid_date_utc"],
        how="left",
        validate="many_to_one",
    )
    recent["date_offset_days_applied"] = 0
    parts.append(recent)

    columns = [
        "site_id",
        "site_timezone",
        "forecast_init_utc",
        "decision_date",
        "gefs_member",
        "valid_date_utc",
        "lead_day",
        "precipitation_mm_raw",
        "source_key",
        "source_etag",
        "ghcnd_station_id",
        "precipitation_mm_reference",
        "reference_valid_unflagged",
        "date_offset_days_applied",
    ]
    data = pd.concat([part[columns] for part in parts], ignore_index=True)
    data["forecast_init_utc"] = pd.to_datetime(data["forecast_init_utc"], utc=True)
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["valid_date_utc"] = pd.to_datetime(data["valid_date_utc"])
    data = data.loc[data["decision_date"].dt.year.between(2000, 2018)].copy()
    key = ["site_id", "decision_date", "valid_date_utc", "gefs_member"]
    if data.duplicated(key).any():
        raise ValueError("duplicate combined GEFS member key")
    expected_rows = 19 * 6 * 5 * 5 * 7
    if len(data) != expected_rows:
        raise ValueError(f"combined rows={len(data)}, expected={expected_rows}")
    if data["source_etag"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("combined input contains an empty GEFS source ETag")
    return data.sort_values(key).reset_index(drop=True)


def fit_apply_candidate(
    candidate_id: str,
    fit: pd.DataFrame,
    target: pd.DataFrame,
    *,
    fit_years: tuple[int, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    method, group_keys = CANDIDATES[candidate_id]
    if method == "qm":
        artifact = fit_empirical_precipitation_qm(
            fit,
            fit_years=fit_years,
            expected_members=GEFS_REFORECAST_MEMBERS,
            contract_id=CONTRACT_ID_V2,
            contract_version=CONTRACT_VERSION_V2,
            aggregation_day_boundary=UTC_DAY_BOUNDARY,
            canonical_valid_date_column="valid_date_utc",
            upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
            group_keys=group_keys,
        )
        corrected = apply_empirical_precipitation_qm(
            target, artifact, split="expanding_window_oof"
        )
    else:
        artifact = fit_offline_precipitation_qdm(
            fit,
            fit_years=fit_years,
            group_keys=group_keys,
        )
        corrected = apply_offline_precipitation_qdm(
            target, artifact, split="expanding_window_oof"
        ).rename(columns={"precipitation_mm_qdm": "precipitation_mm_qm"})
        corrected["qm_extrapolated_upper"] = False
    corrected["candidate_id"] = candidate_id
    return corrected, artifact


def occurrence_row(candidate_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    unique_reference = frame.drop_duplicates(
        ["site_id", "decision_date", "valid_date_utc"]
    )
    observed_dry = float(
        (unique_reference["precipitation_mm_reference"] < 0.05).mean()
    )
    raw_dry = float((frame["precipitation_mm_raw"] < 0.05).mean())
    corrected_dry = float((frame["precipitation_mm_qm"] < 0.05).mean())
    raw_error = abs(raw_dry - observed_dry)
    corrected_error = abs(corrected_dry - observed_dry)
    return {
        "candidate_id": candidate_id,
        "observed_dry_fraction": observed_dry,
        "raw_member_dry_fraction": raw_dry,
        "corrected_member_dry_fraction": corrected_dry,
        "raw_absolute_dry_fraction_error": raw_error,
        "corrected_absolute_dry_fraction_error": corrected_error,
        "occurrence_not_worse": bool(corrected_error <= raw_error),
    }


def save_artifact(path: Path, artifact: dict[str, Any], method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if method == "qm":
        write_quantile_mapping_artifact(path, artifact)
    else:
        verify_qdm_artifact(artifact)
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> dict[str, Path]:
    data = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    oof_parts = []
    artifact_rows = []
    numeric_rows = []

    for fold in expanding_folds():
        fit_years = tuple(fold["fit_years"])
        validation_year = int(fold["validation_year"])
        fit = data.loc[
            data["decision_date"].dt.year.isin(fit_years)
            & data["reference_valid_unflagged"]
            & data["precipitation_mm_reference"].notna()
        ].copy()
        target = data.loc[data["decision_date"].dt.year.eq(validation_year)].copy()
        if len(target) != 1050:
            raise ValueError(f"fold {validation_year} target rows must be 1050")
        print(
            f"[cv] {fold['fold_id']} fit={min(fit_years)}-{max(fit_years)} "
            f"fit_rows={len(fit)} target_rows={len(target)}",
            flush=True,
        )
        for candidate_id, (method, _) in CANDIDATES.items():
            corrected, artifact = fit_apply_candidate(
                candidate_id,
                fit,
                target,
                fit_years=fit_years,
            )
            evaluation = complete_site_cycle_rows(corrected)
            if evaluation.empty:
                raise ValueError(f"fold {validation_year} has no complete evaluation cycles")
            evaluation["fold_id"] = fold["fold_id"]
            evaluation["validation_year"] = validation_year
            oof_parts.append(evaluation)
            row = _metric_row(
                candidate_id,
                validation_year,
                _metric_bundle(evaluation),
            )
            row["fold_id"] = fold["fold_id"]
            row["fit_start_year"] = min(fit_years)
            row["fit_end_year"] = max(fit_years)
            row["fit_member_rows"] = int(len(fit))
            row["evaluation_member_rows"] = int(len(evaluation))
            row["complete_site_cycle_count"] = int(
                evaluation[["site_id", "decision_date"]].drop_duplicates().shape[0]
            )
            metric_rows.append(row)

            artifact_path = (
                args.output_dir
                / "fold_artifacts"
                / str(fold["fold_id"])
                / f"{candidate_id}.json"
            )
            save_artifact(artifact_path, artifact, method)
            artifact_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "validation_year": validation_year,
                    "candidate_id": candidate_id,
                    "method": method,
                    "fit_start_year": min(fit_years),
                    "fit_end_year": max(fit_years),
                    "fit_member_rows": int(len(fit)),
                    "artifact_sha256": artifact["artifact_sha256"],
                    "artifact_file": str(artifact_path),
                }
            )
            values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
            numeric = {
                "fold_id": fold["fold_id"],
                "candidate_id": candidate_id,
                "negative_count": int((values < 0.0).sum()),
                "nonfinite_count": int((~np.isfinite(values)).sum()),
                "maximum_corrected_mm_day": float(values.max()),
            }
            if method == "qdm":
                delta = corrected["qdm_relative_quantile_change"].to_numpy(dtype=float)
                numeric["maximum_relative_quantile_change"] = float(delta.max())
                numeric["p99_relative_quantile_change"] = float(
                    np.quantile(delta, 0.99)
                )
            numeric_rows.append(numeric)
            print(
                f"[cv] {fold['fold_id']} {candidate_id} ready "
                f"evaluation_rows={len(evaluation)}",
                flush=True,
            )

    oof = pd.concat(oof_parts, ignore_index=True)
    year_metrics = pd.DataFrame(metric_rows).sort_values(
        ["validation_year", "candidate_id"]
    )
    pooled_rows = []
    site_rows = []
    occurrence_rows = []
    for candidate_id, candidate in oof.groupby("candidate_id", sort=True):
        pooled = _metric_row(candidate_id, None, _metric_bundle(candidate))
        pooled["scope"] = "pooled"
        pooled["complete_site_cycle_count"] = int(
            candidate[["site_id", "decision_date"]].drop_duplicates().shape[0]
        )
        pooled_rows.append(pooled)
        occurrence_rows.append(occurrence_row(candidate_id, candidate))
        for site_id, site in candidate.groupby("site_id", sort=True):
            row = _metric_row(candidate_id, None, _metric_bundle(site))
            row["scope"] = site_id
            row["complete_site_cycle_count"] = int(
                site[["site_id", "decision_date"]].drop_duplicates().shape[0]
            )
            site_rows.append(row)
    pooled_metrics = pd.DataFrame(pooled_rows).sort_values("candidate_id")
    site_metrics = pd.DataFrame(site_rows).sort_values(["scope", "candidate_id"])
    occurrence = pd.DataFrame(occurrence_rows).sort_values("candidate_id")
    numeric = pd.DataFrame(numeric_rows).sort_values(["fold_id", "candidate_id"])

    gate_rows = []
    for pooled in pooled_metrics.itertuples(index=False):
        candidate_id = pooled.candidate_id
        years = year_metrics.loc[year_metrics["candidate_id"].eq(candidate_id)]
        occurrence_passed = bool(
            occurrence.loc[
                occurrence["candidate_id"].eq(candidate_id), "occurrence_not_worse"
            ].iloc[0]
        )
        numeric_subset = numeric.loc[numeric["candidate_id"].eq(candidate_id)]
        mae_years = int(years["seven_day_mae_not_worse"].sum())
        crps_years = int(years["crps_not_worse"].sum())
        brier_years = int(years["mean_brier_not_worse"].sum())
        numeric_passed = bool(
            numeric_subset["negative_count"].sum() == 0
            and numeric_subset["nonfinite_count"].sum() == 0
        )
        eligible = bool(
            pooled.seven_day_mae_not_worse
            and pooled.crps_not_worse
            and pooled.mean_brier_not_worse
            and pooled.heavy_coverage_not_both_worse
            and occurrence_passed
            and mae_years >= 3
            and crps_years >= 3
            and brier_years >= 3
            and numeric_passed
        )
        gate_rows.append(
            {
                "candidate_id": candidate_id,
                "pooled_seven_day_mae_not_worse": bool(
                    pooled.seven_day_mae_not_worse
                ),
                "pooled_crps_not_worse": bool(pooled.crps_not_worse),
                "pooled_mean_brier_not_worse": bool(
                    pooled.mean_brier_not_worse
                ),
                "pooled_heavy_coverage_not_both_worse": bool(
                    pooled.heavy_coverage_not_both_worse
                ),
                "pooled_occurrence_not_worse": occurrence_passed,
                "mae_years_not_worse": mae_years,
                "crps_years_not_worse": crps_years,
                "brier_years_not_worse": brier_years,
                "required_years_not_worse": 3,
                "numeric_audit_passed": numeric_passed,
                "eligible_for_2019_application": eligible,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values("candidate_id")
    eligible = gate.loc[gate["eligible_for_2019_application"], "candidate_id"].tolist()

    paths = {
        "fold_assignment": args.output_dir / "expanding_cv_fold_assignment_v1.json",
        "oof_predictions": args.output_dir / "expanding_cv_oof_member_predictions_v1.csv",
        "year_metrics": args.output_dir / "expanding_cv_year_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "expanding_cv_pooled_metrics_v1.csv",
        "site_metrics": args.output_dir / "expanding_cv_site_metrics_v1.csv",
        "occurrence": args.output_dir / "expanding_cv_occurrence_audit_v1.csv",
        "numeric": args.output_dir / "expanding_cv_numeric_audit_v1.csv",
        "artifacts": args.output_dir / "expanding_cv_artifact_manifest_v1.csv",
        "gate": args.output_dir / "expanding_cv_candidate_gate_v1.json",
        "report": args.output_dir / "expanding_cv_conclusion_v1.md",
    }
    paths["fold_assignment"].write_text(
        json.dumps(expanding_folds(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    oof.to_csv(paths["oof_predictions"], index=False, encoding="utf-8-sig")
    year_metrics.to_csv(paths["year_metrics"], index=False, encoding="utf-8-sig")
    pooled_metrics.to_csv(paths["pooled_metrics"], index=False, encoding="utf-8-sig")
    site_metrics.to_csv(paths["site_metrics"], index=False, encoding="utf-8-sig")
    occurrence.to_csv(paths["occurrence"], index=False, encoding="utf-8-sig")
    numeric.to_csv(paths["numeric"], index=False, encoding="utf-8-sig")
    pd.DataFrame(artifact_rows).to_csv(
        paths["artifacts"], index=False, encoding="utf-8-sig"
    )
    gate_payload = {
        "contract_id": "gefs-qm-qdm-expanding-window-cv-2000-2018-v1",
        "candidate_set_prelocked": list(CANDIDATES),
        "validation_years": list(VALIDATION_YEARS),
        "2019_used_for_fit_or_selection": False,
        "2024_used_for_fit_or_selection": False,
        "evaluation_policy": "complete_7day_site_cycles_only",
        "qdm_target_cdf_mode": "offline_complete_withheld_gefs_year_batch",
        "realtime_deployment_supported": False,
        "eligible_candidates": eligible,
        "promotion_status": (
            "no_candidate_eligible_retain_raw"
            if not eligible
            else "eligible_candidates_ready_for_exploratory_2019_application"
        ),
        "candidate_gate": gate.to_dict(orient="records"),
    }
    paths["gate"].write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS QM/QDM 2000-2018 扩展窗口交叉验证",
        "",
        "四个候选固定为全局/按站点 QM 与 QDM；2019 和 2024 均未用于拟合或选择。",
        "评价只使用 GHCN-D 记录完整的 7 天站点-周期。",
        "",
        "| 候选 | 7天MAE差值 | CRPS差值 | Brier差值 | 发生频率 | MAE年数 | CRPS年数 | Brier年数 | 晋级 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in pooled_metrics.itertuples(index=False):
        gate_row = gate.loc[gate["candidate_id"].eq(row.candidate_id)].iloc[0]
        report.append(
            f"| `{row.candidate_id}` | "
            f"{row.seven_day_mae_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.crps_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{gate_row.pooled_occurrence_not_worse} | "
            f"{gate_row.mae_years_not_worse}/4 | "
            f"{gate_row.crps_years_not_worse}/4 | "
            f"{gate_row.brier_years_not_worse}/4 | "
            f"{gate_row.eligible_for_2019_application} |"
        )
    report.extend(
        [
            "",
            f"晋级候选：{', '.join(eligible) if eligible else '无，保留 raw GEFS'}。",
        ]
    )
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-2000-2002", type=Path, default=DEFAULT_PAIRED_2000_2002)
    parser.add_argument("--paired-2003-2014", type=Path, default=DEFAULT_PAIRED_2003_2014)
    parser.add_argument("--member-2015-2019", type=Path, default=DEFAULT_MEMBER_2015_2019)
    parser.add_argument("--reference-2000-2019", type=Path, default=DEFAULT_REFERENCE_2000_2019)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

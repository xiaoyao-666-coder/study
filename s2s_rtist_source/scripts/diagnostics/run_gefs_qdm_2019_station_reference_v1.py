#!/usr/bin/env python3
"""Compare global/site-only QM and offline QDM using selected GHCN stations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_gefs_ghcnd_day_alignment_v1 import load_selected_station_reference
from run_gefs_qdm_2019_integration_smoke_v1 import qdm_candidate, qm_candidate
from run_gefs_qm_training_cv_v1 import _metric_bundle, _metric_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBER_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
    / "gefs_reforecast_member_daily_precipitation_utc_v2.csv"
)
DEFAULT_SELECTION_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "station_quality_audit_final"
    / "ghcnd_primary_station_selection_v1.json"
)
DEFAULT_STATION_FILES_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "candidate_station_files"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_2019_station_reference_v1"
)
FIT_YEARS = (2015, 2016, 2017, 2018)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-file", type=Path, default=DEFAULT_MEMBER_FILE)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    parser.add_argument(
        "--station-files-dir", type=Path, default=DEFAULT_STATION_FILES_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def paired_input(
    member_file: Path,
    selection_file: Path,
    station_files_dir: Path,
) -> pd.DataFrame:
    member = pd.read_csv(member_file)
    member["forecast_init_utc"] = pd.to_datetime(member["forecast_init_utc"], utc=True)
    member["decision_date"] = pd.to_datetime(member["decision_date"])
    member["valid_date_utc"] = pd.to_datetime(member["valid_date_utc"])
    reference = load_selected_station_reference(selection_file, station_files_dir)
    reference = reference.loc[reference["reference_valid_unflagged"]].copy()
    reference = reference.rename(columns={"station_record_date": "valid_date_utc"})
    paired = member.merge(
        reference[
            [
                "site_id",
                "valid_date_utc",
                "ghcnd_station_id",
                "precipitation_mm_reference",
            ]
        ],
        on=["site_id", "valid_date_utc"],
        how="left",
        validate="many_to_one",
    )
    paired["station_date_offset_from_gefs_valid_date_days"] = 0
    paired["reference_dataset"] = "NOAA_NCEI_GHCN_Daily_selected_fixed_station"
    return paired


def complete_site_cycle_rows(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame["precipitation_mm_reference"].notna()].copy()
    unique = valid.drop_duplicates(
        ["site_id", "decision_date", "valid_date_utc"]
    )
    counts = unique.groupby(
        ["site_id", "decision_date"], as_index=False
    )["valid_date_utc"].nunique()
    complete = counts.loc[counts["valid_date_utc"].eq(7), ["site_id", "decision_date"]]
    return valid.merge(
        complete,
        on=["site_id", "decision_date"],
        how="inner",
        validate="many_to_one",
    )


def run(
    member_file: Path,
    selection_file: Path,
    station_files_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    paired = paired_input(member_file, selection_file, station_files_dir)
    fit = paired.loc[
        paired["decision_date"].dt.year.isin(FIT_YEARS)
        & paired["precipitation_mm_reference"].notna()
    ].copy()
    target = paired.loc[paired["decision_date"].dt.year.eq(2019)].copy()
    if len(target) != 1050:
        raise ValueError("2019 target GEFS member row count must be 1050")

    candidates = {
        "qm_global": (qm_candidate, ()),
        "qm_site_only": (qm_candidate, ("site_id",)),
        "qdm_global": (qdm_candidate, ()),
        "qdm_site_only": (qdm_candidate, ("site_id",)),
    }
    metric_rows = []
    audit_rows = []
    artifacts = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id, (runner, group_keys) in candidates.items():
        corrected, artifact = runner(fit, target, group_keys=group_keys)
        evaluation = complete_site_cycle_rows(corrected)
        scopes = [("pooled", evaluation)] + [
            (site_id, group)
            for site_id, group in evaluation.groupby("site_id", sort=True)
        ]
        for scope, group in scopes:
            row = _metric_row(candidate_id, 2019, _metric_bundle(group))
            row["scope"] = scope
            row["evaluation_member_rows"] = int(len(group))
            row["complete_site_cycle_count"] = int(
                group[["site_id", "decision_date"]].drop_duplicates().shape[0]
            )
            metric_rows.append(row)
        values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
        audit = {
            "candidate_id": candidate_id,
            "fit_member_rows": int(len(fit)),
            "target_member_rows": int(len(target)),
            "evaluation_member_rows_complete_cycles": int(len(evaluation)),
            "negative_count": int((values < 0.0).sum()),
            "nonfinite_count": int((~np.isfinite(values)).sum()),
            "maximum_corrected_mm_day": float(values.max()),
        }
        if candidate_id.startswith("qdm_"):
            relative = corrected["qdm_relative_quantile_change"].to_numpy(dtype=float)
            audit.update(
                {
                    "maximum_relative_quantile_change": float(relative.max()),
                    "p99_relative_quantile_change": float(np.quantile(relative, 0.99)),
                    "trace_censored_input_count": int(
                        corrected["qdm_trace_censored_input"].sum()
                    ),
                    "trace_censored_output_count": int(
                        corrected["qdm_trace_censored_output"].sum()
                    ),
                }
            )
        audit_rows.append(audit)
        artifacts[candidate_id] = {
            "artifact_sha256": artifact["artifact_sha256"],
            "group_keys": artifact["group_keys"],
            "fit_years": artifact["fit_years"],
        }

    metrics = pd.DataFrame(metric_rows).sort_values(["scope", "candidate_id"])
    audits = pd.DataFrame(audit_rows).sort_values("candidate_id")
    pooled = metrics.loc[metrics["scope"].eq("pooled")].copy()
    metrics_path = output_dir / "gefs_qm_qdm_station_reference_metrics_2019_v1.csv"
    audit_path = output_dir / "gefs_qm_qdm_station_reference_numeric_audit_2019_v1.csv"
    manifest_path = output_dir / "gefs_qm_qdm_station_reference_manifest_2019_v1.json"
    report_path = output_dir / "gefs_qm_qdm_station_reference_conclusion_2019_v1.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    audits.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = {
        "contract_id": "gefs-qm-qdm-selected-ghcnd-station-reference-2019-v1",
        "fit_years": list(FIT_YEARS),
        "validation_year": 2019,
        "station_date_offset_from_gefs_valid_date_days": 0,
        "fit_member_rows_with_valid_reference": int(len(fit)),
        "target_member_rows_used_for_qdm_cdf": int(len(target)),
        "evaluation_policy": "complete_7day_site_cycles_only",
        "validation_date_status": "previously_used_qm_pilot_dates_exploratory_only",
        "independent_holdout_claim_allowed": False,
        "offline_complete_target_gefs_batch_used_by_qdm": True,
        "realtime_deployment_supported": False,
        "artifacts": artifacts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS QM/QDM 地面站参考 2019 探索性比较",
        "",
        "参考数据为已冻结的五个 GHCN-D 主站，日期偏移锁定为 0。",
        "评价只使用地面站记录完整的 7 天站点-周期；QDM 的目标 CDF 使用完整 2019 GEFS 批次。",
        "这些起报日已用于旧 QM pilot，因此本结果用于决定下一步扩展年份实验，不作为独立晋级证据。",
        "",
        "| 候选 | 7天MAE差值 | CRPS差值 | Brier差值 | 重事件覆盖 gate | 完整站点-周期 |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in pooled.itertuples(index=False):
        report.append(
            f"| `{row.candidate_id}` | "
            f"{row.seven_day_mae_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.crps_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.heavy_coverage_not_both_worse} | "
            f"{row.complete_site_cycle_count} |"
        )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    return {
        "metrics": metrics_path,
        "numeric_audit": audit_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def main() -> None:
    args = parse_args()
    outputs = run(
        args.member_file,
        args.selection_file,
        args.station_files_dir,
        args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

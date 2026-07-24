#!/usr/bin/env python3
"""Compare global/site-only QM and offline QDM on the existing 2019 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gefs_qm_training_cv_v1 import _metric_bundle, _metric_row
from s2s_rtist.weather.gefs_quantile_delta_mapping import (
    apply_offline_precipitation_qdm,
    fit_offline_precipitation_qdm,
)
from s2s_rtist.weather.gefs_quantile_mapping import (
    CONTRACT_ID_V2,
    CONTRACT_VERSION_V2,
    GEFS_REFORECAST_MEMBERS,
    UPPER_TAIL_CONSTANT_ADDITIVE,
    UTC_DAY_BOUNDARY,
    apply_empirical_precipitation_qm,
    fit_empirical_precipitation_qm,
    pair_member_and_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_quantile_mapping_v2"
    / "gefs_qm_2015_2019_pilot_v2"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "qdm_2019_integration_smoke_era5_reference"
)
FIT_YEARS = (2015, 2016, 2017, 2018)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_paired(input_dir: Path) -> pd.DataFrame:
    member = pd.read_csv(
        input_dir / "gefs_reforecast_member_daily_precipitation_utc_v2.csv"
    )
    reference = pd.read_csv(
        input_dir / "era5_reference_daily_precipitation_utc_v2.csv"
    )
    paired = pair_member_and_reference(
        member, reference, date_column="valid_date_utc"
    )
    paired["decision_date"] = pd.to_datetime(paired["decision_date"])
    paired["valid_date_utc"] = pd.to_datetime(paired["valid_date_utc"])
    return paired


def qm_candidate(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    group_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    artifact = fit_empirical_precipitation_qm(
        fit,
        fit_years=FIT_YEARS,
        expected_members=GEFS_REFORECAST_MEMBERS,
        contract_id=CONTRACT_ID_V2,
        contract_version=CONTRACT_VERSION_V2,
        aggregation_day_boundary=UTC_DAY_BOUNDARY,
        canonical_valid_date_column="valid_date_utc",
        upper_tail_policy=UPPER_TAIL_CONSTANT_ADDITIVE,
        group_keys=group_keys,
    )
    corrected = apply_empirical_precipitation_qm(
        validation, artifact, split="2019_integration_smoke"
    )
    return corrected, artifact


def qdm_candidate(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    group_keys: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    artifact = fit_offline_precipitation_qdm(
        fit,
        fit_years=FIT_YEARS,
        group_keys=group_keys,
    )
    corrected = apply_offline_precipitation_qdm(
        validation, artifact, split="2019_integration_smoke"
    ).rename(columns={"precipitation_mm_qdm": "precipitation_mm_qm"})
    corrected["qm_extrapolated_upper"] = False
    return corrected, artifact


def run(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    paired = load_paired(input_dir)
    fit = paired.loc[paired["decision_date"].dt.year.isin(FIT_YEARS)].copy()
    validation = paired.loc[paired["decision_date"].dt.year.eq(2019)].copy()
    if len(fit) != 4200 or len(validation) != 1050:
        raise ValueError("unexpected existing pilot fit/validation row counts")

    candidates = {
        "qm_global": (qm_candidate, ()),
        "qm_site_only": (qm_candidate, ("site_id",)),
        "qdm_global": (qdm_candidate, ()),
        "qdm_site_only": (qdm_candidate, ("site_id",)),
    }
    metric_rows = []
    audit_rows = []
    artifact_summaries = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate_id, (runner, group_keys) in candidates.items():
        corrected, artifact = runner(fit, validation, group_keys=group_keys)
        bundle = _metric_bundle(corrected)
        metric_rows.append(_metric_row(candidate_id, 2019, bundle))
        values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
        audit = {
            "candidate_id": candidate_id,
            "row_count": int(len(corrected)),
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
        artifact_summaries[candidate_id] = {
            "artifact_sha256": artifact["artifact_sha256"],
            "group_keys": artifact["group_keys"],
            "fit_years": artifact["fit_years"],
        }

    metrics = pd.DataFrame(metric_rows).sort_values("candidate_id")
    audits = pd.DataFrame(audit_rows).sort_values("candidate_id")
    metrics_path = output_dir / "qdm_integration_smoke_metrics_2019_v1.csv"
    audit_path = output_dir / "qdm_integration_smoke_numeric_audit_2019_v1.csv"
    manifest_path = output_dir / "qdm_integration_smoke_manifest_2019_v1.json"
    report_path = output_dir / "qdm_integration_smoke_conclusion_2019_v1.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    audits.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = {
        "contract_id": "gefs-qdm-2019-integration-smoke-era5-reference-v1",
        "scope": "method_integration_only_not_station_reference_conclusion",
        "reference_dataset": "ERA5_Land_existing_pilot",
        "fit_years": list(FIT_YEARS),
        "validation_year": 2019,
        "validation_date_status": "previously_used_strategy_selection_pilot_dates",
        "independent_holdout_claim_allowed": False,
        "fit_member_rows": int(len(fit)),
        "validation_member_rows": int(len(validation)),
        "offline_complete_target_gefs_batch_used_by_qdm": True,
        "realtime_deployment_supported": False,
        "artifacts": artifact_summaries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# QDM 2019 集成 smoke（ERA5 参考，非地面站正式结论）",
        "",
        "本结果只验证 QDM 在现有 GEFS 集合数据上的集成与方向，不替代扩展 GHCN-D 地面站实验。",
        "QDM 使用完整 2019 GEFS 批次估计目标 CDF，因此属于离线方法验证，不能直接用于 2024 实时部署。",
        "六个 2019 起报日已在旧 QM pilot 中使用，因此本结果不是独立留出验证，也不能据此晋级。",
        "",
        "| 候选 | 7天MAE差值 | CRPS差值 | Brier差值 | 重事件覆盖 gate |",
        "|---|---:|---:|---:|---|",
    ]
    for row in metrics.itertuples(index=False):
        report_lines.append(
            f"| `{row.candidate_id}` | "
            f"{row.seven_day_mae_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.crps_difference_candidate_minus_raw_mm:+.4f} | "
            f"{row.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.heavy_coverage_not_both_worse} |"
        )
    report_lines.extend(
        [
            "",
            "数值审计重点：`qdm_site_only` 的最大相对分位数变化为 "
            f"{float(audits.loc[audits['candidate_id'].eq('qdm_site_only'), 'maximum_relative_quantile_change'].iloc[0]):.4f} 倍。",
            "该值未产生负值或非有限值，但必须在扩展地面站 OOF 中继续审计低分位和极端事件。",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8-sig")
    return {
        "metrics": metrics_path,
        "numeric_audit": audit_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def main() -> None:
    args = parse_args()
    outputs = run(args.input_dir, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

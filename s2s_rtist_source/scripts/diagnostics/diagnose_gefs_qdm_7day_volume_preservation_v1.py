#!/usr/bin/env python3
"""Diagnose QDM with exact preservation of each raw member's 7-day total."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import (
    occurrence_row,
)
from scripts.diagnostics.run_gefs_qm_training_cv_v1 import (
    _metric_bundle,
    _metric_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_7day_volume_preservation_contract_v1.json"
)
DEFAULT_OOF = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qm_qdm_expanding_cv_2000_2018_v1"
    / "expanding_cv_oof_member_predictions_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_7day_volume_preservation_v1"
)
BASE_TO_CANDIDATE = {
    "qdm_global": "qdm_global_7d_volume_preserving",
    "qdm_site_only": "qdm_site_only_7d_volume_preserving",
}


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != (
        "gefs-qdm-seven-day-volume-preservation-diagnostic-v1"
    ):
        raise ValueError("volume-preservation contract id mismatch")
    if contract.get("contract_version") != 1:
        raise ValueError("volume-preservation contract version mismatch")
    if contract["scope"]["use_2019_allowed"] or contract["scope"]["use_2024_allowed"]:
        raise ValueError("volume-preservation contract must prohibit 2019 and 2024")
    return contract


def volume_preserve_group(
    group: pd.DataFrame,
    *,
    tolerance_mm: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if group["valid_date_utc"].nunique() != 7 or len(group) != 7:
        raise ValueError("volume-preservation group must contain exactly seven days")
    raw = group["precipitation_mm_raw"].to_numpy(dtype=float)
    qdm = group["precipitation_mm_qm"].to_numpy(dtype=float)
    raw_total = float(raw.sum())
    qdm_total = float(qdm.sum())
    fallback = False
    if raw_total <= tolerance_mm:
        corrected = np.zeros_like(qdm)
        scale = 0.0
    elif qdm_total <= tolerance_mm:
        corrected = raw.copy()
        scale = np.nan
        fallback = True
    else:
        scale = raw_total / qdm_total
        corrected = qdm * scale
        residual = raw_total - float(corrected.sum())
        index = int(np.argmax(corrected))
        corrected[index] += residual
    if np.any(~np.isfinite(corrected)) or np.any(corrected < -tolerance_mm):
        raise ValueError("volume preservation produced invalid precipitation")
    corrected = np.maximum(corrected, 0.0)
    total_error = float(corrected.sum() - raw_total)
    if abs(total_error) > tolerance_mm:
        raise ValueError("volume preservation failed the member-total tolerance")
    return corrected, {
        "raw_total_mm": raw_total,
        "qdm_total_mm": qdm_total,
        "scale_factor": scale,
        "fallback_to_raw": fallback,
        "member_total_error_mm": total_error,
    }


def transform_candidate(
    frame: pd.DataFrame,
    *,
    output_candidate_id: str,
    tolerance_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy().sort_values(
        ["site_id", "decision_date", "gefs_member", "valid_date_utc"]
    )
    corrected_parts = []
    audit_rows = []
    keys = ["site_id", "decision_date", "gefs_member"]
    for key, group in data.groupby(keys, sort=True):
        group = group.copy()
        corrected, audit = volume_preserve_group(
            group,
            tolerance_mm=tolerance_mm,
        )
        group["precipitation_mm_qdm_base"] = group["precipitation_mm_qm"]
        group["precipitation_mm_qm"] = corrected
        group["candidate_id"] = output_candidate_id
        corrected_parts.append(group)
        audit_rows.append(
            {
                "candidate_id": output_candidate_id,
                "site_id": key[0],
                "decision_date": key[1],
                "gefs_member": key[2],
                **audit,
            }
        )
    return pd.concat(corrected_parts, ignore_index=True), pd.DataFrame(audit_rows)


def metric_tables(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_rows = []
    pooled_rows = []
    site_rows = []
    for candidate_id, candidate in oof.groupby("candidate_id", sort=True):
        for year, group in candidate.groupby("validation_year", sort=True):
            row = _metric_row(candidate_id, int(year), _metric_bundle(group))
            row["complete_site_cycle_count"] = int(
                group[["site_id", "decision_date"]].drop_duplicates().shape[0]
            )
            year_rows.append(row)
        pooled = _metric_row(candidate_id, None, _metric_bundle(candidate))
        pooled["scope"] = "pooled"
        pooled["complete_site_cycle_count"] = int(
            candidate[["site_id", "decision_date"]].drop_duplicates().shape[0]
        )
        pooled_rows.append(pooled)
        for site_id, site in candidate.groupby("site_id", sort=True):
            row = _metric_row(candidate_id, None, _metric_bundle(site))
            row["scope"] = site_id
            row["complete_site_cycle_count"] = int(
                site[["site_id", "decision_date"]].drop_duplicates().shape[0]
            )
            site_rows.append(row)
    return (
        pd.DataFrame(year_rows).sort_values(["validation_year", "candidate_id"]),
        pd.DataFrame(pooled_rows).sort_values("candidate_id"),
        pd.DataFrame(site_rows).sort_values(["scope", "candidate_id"]),
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    tolerance = float(contract["member_total_tolerance_mm"])
    source = pd.read_csv(args.oof_predictions)
    source["decision_date"] = pd.to_datetime(source["decision_date"])
    source["valid_date_utc"] = pd.to_datetime(source["valid_date_utc"])
    if set(source["validation_year"].astype(int)) != {2015, 2016, 2017, 2018}:
        raise ValueError("volume-preservation input must contain only 2015-2018 OOF")

    transformed_parts = []
    audit_parts = []
    for base_id, output_id in BASE_TO_CANDIDATE.items():
        base = source.loc[source["candidate_id"].eq(base_id)].copy()
        if base.empty:
            raise ValueError(f"missing base OOF candidate: {base_id}")
        transformed, audit = transform_candidate(
            base,
            output_candidate_id=output_id,
            tolerance_mm=tolerance,
        )
        transformed_parts.append(transformed)
        audit_parts.append(audit)
        print(
            f"[volume] {output_id} ready rows={len(transformed)} groups={len(audit)}",
            flush=True,
        )
    oof = pd.concat(transformed_parts, ignore_index=True)
    member_audit = pd.concat(audit_parts, ignore_index=True)
    year_metrics, pooled_metrics, site_metrics = metric_tables(oof)
    occurrence = pd.DataFrame(
        [
            occurrence_row(candidate_id, candidate)
            for candidate_id, candidate in oof.groupby("candidate_id", sort=True)
        ]
    )

    numeric_rows = []
    for candidate_id, candidate in oof.groupby("candidate_id", sort=True):
        values = candidate["precipitation_mm_qm"].to_numpy(dtype=float)
        audit = member_audit.loc[member_audit["candidate_id"].eq(candidate_id)]
        finite_scales = audit["scale_factor"].dropna().to_numpy(dtype=float)
        numeric_rows.append(
            {
                "candidate_id": candidate_id,
                "negative_count": int((values < 0.0).sum()),
                "nonfinite_count": int((~np.isfinite(values)).sum()),
                "maximum_corrected_mm_day": float(values.max()),
                "maximum_absolute_member_total_error_mm": float(
                    audit["member_total_error_mm"].abs().max()
                ),
                "fallback_to_raw_group_count": int(audit["fallback_to_raw"].sum()),
                "maximum_scale_factor": (
                    float(finite_scales.max()) if len(finite_scales) else None
                ),
                "p99_scale_factor": (
                    float(np.quantile(finite_scales, 0.99))
                    if len(finite_scales)
                    else None
                ),
            }
        )
    numeric = pd.DataFrame(numeric_rows).sort_values("candidate_id")

    gate_rows = []
    required_years = int(
        contract["candidate_gate"]["minimum_years_not_worse_per_primary_metric"]
    )
    for pooled in pooled_metrics.itertuples(index=False):
        candidate_id = pooled.candidate_id
        years = year_metrics.loc[year_metrics["candidate_id"].eq(candidate_id)]
        occurrence_passed = bool(
            occurrence.loc[
                occurrence["candidate_id"].eq(candidate_id), "occurrence_not_worse"
            ].iloc[0]
        )
        audit = numeric.loc[numeric["candidate_id"].eq(candidate_id)].iloc[0]
        volume_passed = bool(
            audit.maximum_absolute_member_total_error_mm <= tolerance
        )
        numeric_passed = bool(
            audit.negative_count == 0 and audit.nonfinite_count == 0
        )
        mae_years = int(
            (
                years["seven_day_mae_difference_candidate_minus_raw_mm"]
                <= tolerance
            ).sum()
        )
        crps_years = int(years["crps_not_worse"].sum())
        brier_years = int(years["mean_brier_not_worse"].sum())
        pooled_mae_passed = bool(
            pooled.seven_day_mae_difference_candidate_minus_raw_mm <= tolerance
        )
        eligible = bool(
            pooled_mae_passed
            and pooled.crps_not_worse
            and pooled.mean_brier_not_worse
            and pooled.heavy_coverage_not_both_worse
            and occurrence_passed
            and mae_years >= required_years
            and crps_years >= required_years
            and brier_years >= required_years
            and numeric_passed
            and volume_passed
        )
        gate_rows.append(
            {
                "candidate_id": candidate_id,
                "pooled_seven_day_mae_not_worse": pooled_mae_passed,
                "pooled_crps_not_worse": bool(pooled.crps_not_worse),
                "pooled_mean_brier_not_worse": bool(pooled.mean_brier_not_worse),
                "pooled_heavy_coverage_not_both_worse": bool(
                    pooled.heavy_coverage_not_both_worse
                ),
                "pooled_occurrence_not_worse": occurrence_passed,
                "mae_years_not_worse": mae_years,
                "crps_years_not_worse": crps_years,
                "brier_years_not_worse": brier_years,
                "required_years_not_worse": required_years,
                "numeric_audit_passed": numeric_passed,
                "member_total_constraint_passed": volume_passed,
                "eligible_for_exploratory_2019_application": eligible,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values("candidate_id")
    eligible = gate.loc[
        gate["eligible_for_exploratory_2019_application"], "candidate_id"
    ].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof": args.output_dir / "volume_preserving_oof_predictions_v1.csv",
        "member_audit": args.output_dir / "volume_preserving_member_total_audit_v1.csv",
        "year_metrics": args.output_dir / "volume_preserving_year_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "volume_preserving_pooled_metrics_v1.csv",
        "site_metrics": args.output_dir / "volume_preserving_site_metrics_v1.csv",
        "occurrence": args.output_dir / "volume_preserving_occurrence_audit_v1.csv",
        "numeric": args.output_dir / "volume_preserving_numeric_audit_v1.csv",
        "gate": args.output_dir / "volume_preserving_candidate_gate_v1.json",
        "report": args.output_dir / "volume_preserving_conclusion_v1.md",
    }
    oof.to_csv(paths["oof"], index=False, encoding="utf-8-sig")
    member_audit.to_csv(paths["member_audit"], index=False, encoding="utf-8-sig")
    year_metrics.to_csv(paths["year_metrics"], index=False, encoding="utf-8-sig")
    pooled_metrics.to_csv(paths["pooled_metrics"], index=False, encoding="utf-8-sig")
    site_metrics.to_csv(paths["site_metrics"], index=False, encoding="utf-8-sig")
    occurrence.to_csv(paths["occurrence"], index=False, encoding="utf-8-sig")
    numeric.to_csv(paths["numeric"], index=False, encoding="utf-8-sig")
    gate_payload = {
        "contract_id": contract["contract_id"],
        "2019_used": False,
        "2024_used": False,
        "eligible_candidates": eligible,
        "promotion_status": (
            "no_volume_preserving_candidate_eligible"
            if not eligible
            else "eligible_for_exploratory_2019_application"
        ),
        "candidate_gate": gate.to_dict(orient="records"),
    }
    paths["gate"].write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS QDM 7 天水量保持诊断",
        "",
        "每个站点-周期-成员的 7 天订正总量被约束为 raw GEFS 总量；2019 和 2024 未使用。",
        "",
        "| 候选 | 7天MAE差值 | CRPS差值 | Brier差值 | 发生频率 | CRPS年数 | Brier年数 | 水量约束 | 晋级 |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for pooled in pooled_metrics.itertuples(index=False):
        row = gate.loc[gate["candidate_id"].eq(pooled.candidate_id)].iloc[0]
        report.append(
            f"| `{pooled.candidate_id}` | "
            f"{pooled.seven_day_mae_difference_candidate_minus_raw_mm:+.8f} | "
            f"{pooled.crps_difference_candidate_minus_raw_mm:+.4f} | "
            f"{pooled.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.pooled_occurrence_not_worse} | "
            f"{row.crps_years_not_worse}/4 | "
            f"{row.brier_years_not_worse}/4 | "
            f"{row.member_total_constraint_passed} | "
            f"{row.eligible_for_exploratory_2019_application} |"
        )
    report.extend(
        [
            "",
            f"晋级候选：{', '.join(eligible) if eligible else '无'}。",
        ]
    )
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run(args)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

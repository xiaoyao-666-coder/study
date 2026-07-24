#!/usr/bin/env python3
"""Run causal current-cycle global QDM plus 7-day volume-preserving OOF CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.diagnose_gefs_qdm_7day_volume_preservation_v1 import (
    metric_tables,
    transform_candidate,
)
from scripts.diagnostics.run_gefs_qdm_2019_station_reference_v1 import (
    complete_site_cycle_rows,
)
from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import (
    DEFAULT_MEMBER_2015_2019,
    DEFAULT_PAIRED_2000_2002,
    DEFAULT_PAIRED_2003_2014,
    DEFAULT_REFERENCE_2000_2019,
    expanding_folds,
    load_inputs,
    occurrence_row,
)
from s2s_rtist.weather.gefs_quantile_delta_mapping import (
    apply_current_cycle_precipitation_qdm,
    fit_offline_precipitation_qdm,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_causal_current_cycle_cv_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_causal_current_cycle_cv_v1"
)
CANDIDATE_ID = "qdm_global_current_cycle_7d_volume_preserving"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-qdm-causal-current-cycle-cv-v1":
        raise ValueError("causal QDM contract id mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("causal QDM candidate mismatch")
    if contract.get("target_cdf_mode") != "causal_current_cycle_global_batch":
        raise ValueError("causal QDM target CDF mode mismatch")
    if contract.get("scale_cap") is not None:
        raise ValueError("post-hoc scale cap is prohibited")
    if contract["scope"]["use_2019_allowed"] or contract["scope"]["use_2024_allowed"]:
        raise ValueError("2019 and 2024 must be prohibited")
    expected = (
        int(contract["expected_sites"])
        * int(contract["expected_members"])
        * int(contract["expected_lead_days"])
    )
    if expected != int(contract["expected_target_cdf_rows_per_cycle"]):
        raise ValueError("causal QDM cycle sample count mismatch")
    return contract


def apply_fold(
    data: pd.DataFrame,
    fold: dict[str, Any],
    *,
    expected_rows_per_cycle: int,
    tolerance_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    fit_years = tuple(int(value) for value in fold["fit_years"])
    validation_year = int(fold["validation_year"])
    fit = data.loc[
        data["decision_date"].dt.year.isin(fit_years)
        & data["reference_valid_unflagged"]
        & data["precipitation_mm_reference"].notna()
    ].copy()
    target = data.loc[data["decision_date"].dt.year.eq(validation_year)].copy()
    artifact = fit_offline_precipitation_qdm(
        fit,
        fit_years=fit_years,
        group_keys=(),
    )
    qdm = apply_current_cycle_precipitation_qdm(
        target,
        artifact,
        split=f"causal_current_cycle_oof_{validation_year}",
        expected_rows_per_cycle=expected_rows_per_cycle,
    ).rename(columns={"precipitation_mm_qdm": "precipitation_mm_qm"})
    qdm["qm_extrapolated_upper"] = False
    qdm["candidate_id"] = "qdm_global_current_cycle"
    corrected, member_audit = transform_candidate(
        qdm,
        output_candidate_id=CANDIDATE_ID,
        tolerance_mm=tolerance_mm,
    )
    corrected["validation_year"] = validation_year
    member_audit["validation_year"] = validation_year
    evaluation = complete_site_cycle_rows(corrected)
    if evaluation.empty:
        raise ValueError(f"no complete cycles in validation year {validation_year}")
    cycle_manifest = (
        corrected.groupby("decision_date", as_index=False)
        .agg(
            target_cdf_rows=("qdm_target_cdf_sample_count", "first"),
            corrected_rows=("precipitation_mm_qm", "size"),
            target_cdf_cycle=("qdm_target_cdf_cycle", "first"),
        )
        .assign(validation_year=validation_year)
    )
    cycle_manifest["future_cycle_rows_used"] = 0
    return evaluation, member_audit, artifact, cycle_manifest


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    tolerance = float(contract["member_total_tolerance_mm"])
    expected_rows = int(contract["expected_target_cdf_rows_per_cycle"])
    data = load_inputs(args)
    if set(data["decision_date"].dt.year).intersection({2019, 2024}):
        raise ValueError("forbidden year entered causal QDM CV")

    evaluation_parts = []
    audit_parts = []
    cycle_parts = []
    artifact_rows = []
    for fold in expanding_folds():
        evaluation, audit, artifact, cycles = apply_fold(
            data,
            fold,
            expected_rows_per_cycle=expected_rows,
            tolerance_mm=tolerance,
        )
        evaluation_parts.append(evaluation)
        audit_parts.append(audit)
        cycle_parts.append(cycles)
        artifact_rows.append(
            {
                "fold_id": fold["fold_id"],
                "validation_year": fold["validation_year"],
                "fit_first_year": min(fold["fit_years"]),
                "fit_last_year": max(fold["fit_years"]),
                "validation_rows_used_for_fit": 0,
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
        print(
            f"[causal-qdm] {fold['fold_id']} rows={len(evaluation)} ready",
            flush=True,
        )

    oof = pd.concat(evaluation_parts, ignore_index=True)
    member_audit = pd.concat(audit_parts, ignore_index=True)
    cycle_manifest = pd.concat(cycle_parts, ignore_index=True)
    artifacts = pd.DataFrame(artifact_rows)
    year_metrics, pooled_metrics, site_metrics = metric_tables(oof)
    pooled = pooled_metrics.iloc[0]
    occurrence = occurrence_row(CANDIDATE_ID, oof)

    values = oof["precipitation_mm_qm"].to_numpy(dtype=float)
    scales = member_audit["scale_factor"].dropna().to_numpy(dtype=float)
    relative = oof["qdm_relative_quantile_change"].to_numpy(dtype=float)
    numeric = {
        "candidate_id": CANDIDATE_ID,
        "negative_count": int((values < 0.0).sum()),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "maximum_corrected_mm_day": float(values.max()),
        "maximum_absolute_member_total_error_mm": float(
            member_audit["member_total_error_mm"].abs().max()
        ),
        "fallback_to_raw_group_count": int(member_audit["fallback_to_raw"].sum()),
        "maximum_scale_factor": float(scales.max()),
        "p99_scale_factor": float(np.quantile(scales, 0.99)),
        "maximum_qdm_relative_quantile_change": float(relative.max()),
        "p99_qdm_relative_quantile_change": float(np.quantile(relative, 0.99)),
    }
    required_years = int(
        contract["candidate_gate"]["minimum_years_not_worse_per_primary_metric"]
    )
    mae_years = int(
        (
            year_metrics["seven_day_mae_difference_candidate_minus_raw_mm"]
            <= tolerance
        ).sum()
    )
    crps_years = int(year_metrics["crps_not_worse"].sum())
    brier_years = int(year_metrics["mean_brier_not_worse"].sum())
    numeric_passed = bool(
        numeric["negative_count"] == 0 and numeric["nonfinite_count"] == 0
    )
    volume_passed = bool(
        numeric["maximum_absolute_member_total_error_mm"] <= tolerance
    )
    cycle_causality_passed = bool(
        cycle_manifest["target_cdf_rows"].eq(expected_rows).all()
        and cycle_manifest["corrected_rows"].eq(expected_rows).all()
        and cycle_manifest["future_cycle_rows_used"].eq(0).all()
        and pd.to_datetime(cycle_manifest["target_cdf_cycle"]).eq(
            pd.to_datetime(cycle_manifest["decision_date"])
        ).all()
    )
    eligible = bool(
        pooled["seven_day_mae_difference_candidate_minus_raw_mm"] <= tolerance
        and pooled["crps_not_worse"]
        and pooled["mean_brier_not_worse"]
        and pooled["heavy_coverage_not_both_worse"]
        and occurrence["occurrence_not_worse"]
        and mae_years >= required_years
        and crps_years >= required_years
        and brier_years >= required_years
        and numeric_passed
        and volume_passed
        and cycle_causality_passed
    )
    gate = {
        "contract_id": contract["contract_id"],
        "candidate_id": CANDIDATE_ID,
        "2019_used": False,
        "2024_used": False,
        "target_cdf_mode": contract["target_cdf_mode"],
        "cycle_causality_passed": cycle_causality_passed,
        "pooled_seven_day_mae_not_worse": bool(
            pooled["seven_day_mae_difference_candidate_minus_raw_mm"] <= tolerance
        ),
        "pooled_crps_not_worse": bool(pooled["crps_not_worse"]),
        "pooled_mean_brier_not_worse": bool(pooled["mean_brier_not_worse"]),
        "pooled_heavy_coverage_not_both_worse": bool(
            pooled["heavy_coverage_not_both_worse"]
        ),
        "pooled_occurrence_not_worse": bool(occurrence["occurrence_not_worse"]),
        "mae_years_not_worse": mae_years,
        "crps_years_not_worse": crps_years,
        "brier_years_not_worse": brier_years,
        "required_years_not_worse": required_years,
        "numeric_audit_passed": numeric_passed,
        "member_total_constraint_passed": volume_passed,
        "eligible_for_causal_2019_application": eligible,
        "status": (
            "eligible_for_causal_2019_application"
            if eligible
            else "failed_causal_training_oof_retain_raw"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof": args.output_dir / "causal_current_cycle_oof_predictions_v1.csv",
        "member_audit": args.output_dir / "causal_current_cycle_member_audit_v1.csv",
        "cycle_manifest": args.output_dir / "causal_current_cycle_target_cdf_manifest_v1.csv",
        "artifacts": args.output_dir / "causal_current_cycle_fold_artifacts_v1.csv",
        "year_metrics": args.output_dir / "causal_current_cycle_year_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "causal_current_cycle_pooled_metrics_v1.csv",
        "site_metrics": args.output_dir / "causal_current_cycle_site_metrics_v1.csv",
        "occurrence": args.output_dir / "causal_current_cycle_occurrence_v1.json",
        "numeric": args.output_dir / "causal_current_cycle_numeric_audit_v1.json",
        "gate": args.output_dir / "causal_current_cycle_candidate_gate_v1.json",
        "report": args.output_dir / "causal_current_cycle_conclusion_v1.md",
    }
    for frame, key in (
        (oof, "oof"),
        (member_audit, "member_audit"),
        (cycle_manifest, "cycle_manifest"),
        (artifacts, "artifacts"),
        (year_metrics, "year_metrics"),
        (pooled_metrics, "pooled_metrics"),
        (site_metrics, "site_metrics"),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    for payload, key in (
        (occurrence, "occurrence"),
        (numeric, "numeric"),
        (gate, "gate"),
    ):
        paths[key].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = [
        "# GEFS QDM 因果当前周期目标 CDF 训练期 OOF",
        "",
        "每个目标 CDF 只使用当前起报周期可见的 175 个 GEFS 值；2019 与 2024 未使用。",
        "",
        f"- 7 天 MAE 差值：`{pooled['seven_day_mae_difference_candidate_minus_raw_mm']:+.8f} mm`",
        f"- CRPS 差值：`{pooled['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier 差值：`{pooled['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- CRPS 不劣年份：`{crps_years}/4`",
        f"- Brier 不劣年份：`{brier_years}/4`",
        f"- 最大缩放系数：`{numeric['maximum_scale_factor']:.6f}`",
        f"- P99 缩放系数：`{numeric['p99_scale_factor']:.6f}`",
        f"- Gate：`{gate['status']}`",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--paired-2000-2002", type=Path, default=DEFAULT_PAIRED_2000_2002)
    parser.add_argument("--paired-2003-2014", type=Path, default=DEFAULT_PAIRED_2003_2014)
    parser.add_argument("--member-2015-2019", type=Path, default=DEFAULT_MEMBER_2015_2019)
    parser.add_argument("--reference-2000-2019", type=Path, default=DEFAULT_REFERENCE_2000_2019)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

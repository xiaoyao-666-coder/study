#!/usr/bin/env python3
"""Validate frozen global QDM plus exact 7-day member-volume preservation in 2019."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.diagnose_gefs_qdm_7day_volume_preservation_v1 import (
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
    load_inputs,
    occurrence_row,
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_volume_preserving_2019_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_volume_preserving_2019_validation_v1"
)
CANDIDATE_ID = "qdm_global_7d_volume_preserving"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != (
        "gefs-qdm-global-seven-day-volume-preserving-2019-validation-v1"
    ):
        raise ValueError("2019 volume-preserving contract id mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("2019 volume-preserving candidate mismatch")
    if contract["scope"]["use_2019_reference_for_fit"]:
        raise ValueError("2019 reference cannot be used for fit")
    if contract["scope"]["use_2024_allowed"]:
        raise ValueError("2024 must be prohibited")
    if contract.get("scale_cap") is not None:
        raise ValueError("post-hoc scale cap is prohibited by the frozen contract")
    if contract.get("target_cdf_mode") != "offline_complete_2019_gefs_batch":
        raise ValueError("2019 target CDF mode mismatch")
    return contract


def load_2019_target(args: argparse.Namespace) -> pd.DataFrame:
    member = pd.read_csv(args.member_2015_2019)
    member["decision_date"] = pd.to_datetime(member["decision_date"])
    member["valid_date_utc"] = pd.to_datetime(member["valid_date_utc"])
    member["forecast_init_utc"] = pd.to_datetime(member["forecast_init_utc"], utc=True)
    member = member.loc[member["decision_date"].dt.year.eq(2019)].copy()
    reference = pd.read_csv(args.reference_2000_2019)
    reference["valid_date_utc"] = pd.to_datetime(reference["station_record_date"])
    reference["reference_valid_unflagged"] = reference[
        "reference_valid_unflagged"
    ].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )
    reference = reference.loc[reference["reference_valid_unflagged"]].copy()
    target = member.merge(
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
    if len(target) != 1050:
        raise ValueError(f"2019 target rows={len(target)}, expected=1050")
    return target


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    tolerance = float(contract["member_total_tolerance_mm"])
    history = load_inputs(args)
    fit = history.loc[
        history["reference_valid_unflagged"]
        & history["precipitation_mm_reference"].notna()
    ].copy()
    if set(fit["decision_date"].dt.year.astype(int)).difference(range(2000, 2019)):
        raise ValueError("fit contains a year outside 2000-2018")
    target = load_2019_target(args)
    print(
        f"[2019] fit_rows={len(fit)} target_rows={len(target)}",
        flush=True,
    )

    artifact = fit_offline_precipitation_qdm(
        fit,
        fit_years=tuple(range(2000, 2019)),
        group_keys=(),
    )
    qdm = apply_offline_precipitation_qdm(
        target,
        artifact,
        split="validation_2019_exploratory",
    ).rename(columns={"precipitation_mm_qdm": "precipitation_mm_qm"})
    qdm["qm_extrapolated_upper"] = False
    qdm["candidate_id"] = "qdm_global"
    corrected, member_audit = transform_candidate(
        qdm,
        output_candidate_id=CANDIDATE_ID,
        tolerance_mm=tolerance,
    )
    evaluation = complete_site_cycle_rows(corrected)
    if evaluation.empty:
        raise ValueError("2019 has no complete station-cycle evaluation rows")
    metric = _metric_row(CANDIDATE_ID, 2019, _metric_bundle(evaluation))
    metric["complete_site_cycle_count"] = int(
        evaluation[["site_id", "decision_date"]].drop_duplicates().shape[0]
    )
    occurrence = occurrence_row(CANDIDATE_ID, evaluation)

    values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    delta = qdm["qdm_relative_quantile_change"].to_numpy(dtype=float)
    finite_scales = member_audit["scale_factor"].dropna().to_numpy(dtype=float)
    numeric = {
        "candidate_id": CANDIDATE_ID,
        "negative_count": int((values < 0.0).sum()),
        "nonfinite_count": int((~np.isfinite(values)).sum()),
        "maximum_corrected_mm_day": float(values.max()),
        "maximum_qdm_relative_quantile_change": float(delta.max()),
        "p99_qdm_relative_quantile_change": float(np.quantile(delta, 0.99)),
        "maximum_absolute_member_total_error_mm": float(
            member_audit["member_total_error_mm"].abs().max()
        ),
        "fallback_to_raw_group_count": int(member_audit["fallback_to_raw"].sum()),
        "maximum_scale_factor": float(finite_scales.max()),
        "p99_scale_factor": float(np.quantile(finite_scales, 0.99)),
    }
    pooled_mae_passed = bool(
        metric["seven_day_mae_difference_candidate_minus_raw_mm"] <= tolerance
    )
    numeric_passed = bool(
        numeric["negative_count"] == 0 and numeric["nonfinite_count"] == 0
    )
    volume_passed = bool(
        numeric["maximum_absolute_member_total_error_mm"] <= tolerance
    )
    passed = bool(
        pooled_mae_passed
        and metric["crps_not_worse"]
        and metric["mean_brier_not_worse"]
        and metric["heavy_coverage_not_both_worse"]
        and occurrence["occurrence_not_worse"]
        and numeric_passed
        and volume_passed
    )
    gate = {
        "contract_id": contract["contract_id"],
        "candidate_id": CANDIDATE_ID,
        "fit_years": list(range(2000, 2019)),
        "validation_year": 2019,
        "2019_reference_used_for_fit": False,
        "2024_used": False,
        "validation_date_status": "previously_used_exploratory_dates_not_independent",
        "seven_day_mae_not_worse": pooled_mae_passed,
        "crps_not_worse": bool(metric["crps_not_worse"]),
        "mean_brier_not_worse": bool(metric["mean_brier_not_worse"]),
        "heavy_coverage_not_both_worse": bool(
            metric["heavy_coverage_not_both_worse"]
        ),
        "occurrence_not_worse": bool(occurrence["occurrence_not_worse"]),
        "numeric_audit_passed": numeric_passed,
        "member_total_constraint_passed": volume_passed,
        "all_requirements_passed": passed,
        "status": (
            "passed_exploratory_2019_requires_causal_target_cdf_design"
            if passed
            else "failed_exploratory_2019_retain_raw"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "artifact": args.output_dir / "qdm_global_fit_2000_2018_artifact_v1.json",
        "predictions": args.output_dir / "qdm_volume_preserving_predictions_2019_v1.csv",
        "evaluation": args.output_dir / "qdm_volume_preserving_complete_cycle_evaluation_2019_v1.csv",
        "member_audit": args.output_dir / "qdm_volume_preserving_member_total_audit_2019_v1.csv",
        "metrics": args.output_dir / "qdm_volume_preserving_metrics_2019_v1.json",
        "occurrence": args.output_dir / "qdm_volume_preserving_occurrence_2019_v1.json",
        "numeric": args.output_dir / "qdm_volume_preserving_numeric_audit_2019_v1.json",
        "gate": args.output_dir / "qdm_volume_preserving_gate_2019_v1.json",
        "report": args.output_dir / "qdm_volume_preserving_conclusion_2019_v1.md",
    }
    verify_qdm_artifact(artifact)
    paths["artifact"].write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corrected.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    evaluation.to_csv(paths["evaluation"], index=False, encoding="utf-8-sig")
    member_audit.to_csv(paths["member_audit"], index=False, encoding="utf-8-sig")
    for path, payload in (
        (paths["metrics"], metric),
        (paths["occurrence"], occurrence),
        (paths["numeric"], numeric),
        (paths["gate"], gate),
    ):
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = [
        "# GEFS 全局 QDM 7 天水量保持 2019 验证",
        "",
        "候选在 2015-2018 OOF 后冻结；2000-2018 用于拟合，2019 GHCN-D 仅用于评分。",
        "2019 日期并非完全独立，且 QDM 使用完整 2019 GEFS 批次估计目标 CDF。",
        "",
        f"- 7 天 MAE 差值：`{metric['seven_day_mae_difference_candidate_minus_raw_mm']:+.8f} mm`",
        f"- CRPS 差值：`{metric['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier 差值：`{metric['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- 完整站点-周期：`{metric['complete_site_cycle_count']}`",
        f"- 最大缩放系数：`{numeric['maximum_scale_factor']:.6f}`",
        f"- P99 缩放系数：`{numeric['p99_scale_factor']:.6f}`",
        f"- raw 回退组：`{numeric['fallback_to_raw_group_count']}`",
        f"- Gate：`{gate['status']}`",
    ]
    paths["report"].write_text(
        "\n".join(report) + "\n", encoding="utf-8-sig"
    )
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

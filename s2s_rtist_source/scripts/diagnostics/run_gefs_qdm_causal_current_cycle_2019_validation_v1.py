#!/usr/bin/env python3
"""Validate frozen causal current-cycle global QDM in 2019."""

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
from scripts.diagnostics.run_gefs_qdm_volume_preserving_2019_validation_v1 import (
    load_2019_target,
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
    apply_current_cycle_precipitation_qdm,
    fit_offline_precipitation_qdm,
    verify_qdm_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_causal_current_cycle_2019_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_causal_current_cycle_2019_validation_v1"
)
CANDIDATE_ID = "qdm_global_current_cycle_7d_volume_preserving"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != (
        "gefs-qdm-causal-current-cycle-2019-validation-v1"
    ):
        raise ValueError("causal 2019 contract id mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("causal 2019 candidate mismatch")
    if contract.get("target_cdf_mode") != "causal_current_cycle_global_batch":
        raise ValueError("causal 2019 target CDF mode mismatch")
    if contract.get("scale_cap") is not None:
        raise ValueError("post-hoc scale cap is prohibited")
    scope = contract["scope"]
    if scope["use_2019_reference_for_fit"]:
        raise ValueError("2019 reference cannot be used for fit")
    if scope["use_future_2019_cycles_for_target_cdf"]:
        raise ValueError("future 2019 cycles cannot enter target CDF")
    if scope["use_2024_allowed"]:
        raise ValueError("2024 must be prohibited")
    return contract


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    tolerance = float(contract["member_total_tolerance_mm"])
    expected_rows = int(contract["expected_target_cdf_rows_per_cycle"])
    expected_cycles = int(contract["expected_cycles"])
    history = load_inputs(args)
    fit = history.loc[
        history["reference_valid_unflagged"]
        & history["precipitation_mm_reference"].notna()
    ].copy()
    fit_years = tuple(range(2000, 2019))
    if set(fit["decision_date"].dt.year.astype(int)).difference(fit_years):
        raise ValueError("fit contains a year outside 2000-2018")
    target = load_2019_target(args)
    artifact = fit_offline_precipitation_qdm(
        fit,
        fit_years=fit_years,
        group_keys=(),
    )
    qdm = apply_current_cycle_precipitation_qdm(
        target,
        artifact,
        split="causal_current_cycle_2019_exploratory",
        expected_rows_per_cycle=expected_rows,
    ).rename(columns={"precipitation_mm_qdm": "precipitation_mm_qm"})
    qdm["qm_extrapolated_upper"] = False
    qdm["candidate_id"] = "qdm_global_current_cycle"
    corrected, member_audit = transform_candidate(
        qdm,
        output_candidate_id=CANDIDATE_ID,
        tolerance_mm=tolerance,
    )
    evaluation = complete_site_cycle_rows(corrected)
    if evaluation.empty:
        raise ValueError("2019 has no complete station-cycle evaluation rows")

    cycle_manifest = (
        corrected.groupby("decision_date", as_index=False)
        .agg(
            target_cdf_rows=("qdm_target_cdf_sample_count", "first"),
            corrected_rows=("precipitation_mm_qm", "size"),
            target_cdf_cycle=("qdm_target_cdf_cycle", "first"),
        )
    )
    cycle_manifest["future_cycle_rows_used"] = 0
    cycle_causality_passed = bool(
        len(cycle_manifest) == expected_cycles
        and cycle_manifest["target_cdf_rows"].eq(expected_rows).all()
        and cycle_manifest["corrected_rows"].eq(expected_rows).all()
        and cycle_manifest["future_cycle_rows_used"].eq(0).all()
        and pd.to_datetime(cycle_manifest["target_cdf_cycle"]).eq(
            pd.to_datetime(cycle_manifest["decision_date"])
        ).all()
    )

    metric = _metric_row(CANDIDATE_ID, 2019, _metric_bundle(evaluation))
    metric["complete_site_cycle_count"] = int(
        evaluation[["site_id", "decision_date"]].drop_duplicates().shape[0]
    )
    occurrence = occurrence_row(CANDIDATE_ID, evaluation)
    values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    relative = corrected["qdm_relative_quantile_change"].to_numpy(dtype=float)
    scales = member_audit["scale_factor"].dropna().to_numpy(dtype=float)
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
    numeric_passed = bool(
        numeric["negative_count"] == 0 and numeric["nonfinite_count"] == 0
    )
    volume_passed = bool(
        numeric["maximum_absolute_member_total_error_mm"] <= tolerance
    )
    mae_passed = bool(
        metric["seven_day_mae_difference_candidate_minus_raw_mm"] <= tolerance
    )
    passed = bool(
        mae_passed
        and metric["crps_not_worse"]
        and metric["mean_brier_not_worse"]
        and metric["heavy_coverage_not_both_worse"]
        and occurrence["occurrence_not_worse"]
        and cycle_causality_passed
        and numeric_passed
        and volume_passed
    )
    gate = {
        "contract_id": contract["contract_id"],
        "candidate_id": CANDIDATE_ID,
        "fit_years": list(fit_years),
        "validation_year": 2019,
        "2019_reference_used_for_fit": False,
        "future_2019_cycles_used_for_target_cdf": False,
        "2024_used": False,
        "validation_date_status": "previously_used_exploratory_dates_not_independent",
        "target_cdf_mode": contract["target_cdf_mode"],
        "cycle_causality_passed": cycle_causality_passed,
        "seven_day_mae_not_worse": mae_passed,
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
            "passed_causal_2019_exploratory_with_scale_stability_caveat"
            if passed
            else "failed_causal_2019_retain_raw"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "artifact": args.output_dir / "causal_qdm_global_fit_2000_2018_artifact_v1.json",
        "predictions": args.output_dir / "causal_qdm_predictions_2019_v1.csv",
        "evaluation": args.output_dir / "causal_qdm_complete_cycle_evaluation_2019_v1.csv",
        "member_audit": args.output_dir / "causal_qdm_member_audit_2019_v1.csv",
        "cycle_manifest": args.output_dir / "causal_qdm_target_cdf_manifest_2019_v1.csv",
        "metrics": args.output_dir / "causal_qdm_metrics_2019_v1.json",
        "occurrence": args.output_dir / "causal_qdm_occurrence_2019_v1.json",
        "numeric": args.output_dir / "causal_qdm_numeric_audit_2019_v1.json",
        "gate": args.output_dir / "causal_qdm_candidate_gate_2019_v1.json",
        "report": args.output_dir / "causal_qdm_conclusion_2019_v1.md",
    }
    verify_qdm_artifact(artifact)
    paths["artifact"].write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for frame, key in (
        (corrected, "predictions"),
        (evaluation, "evaluation"),
        (member_audit, "member_audit"),
        (cycle_manifest, "cycle_manifest"),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    for payload, key in (
        (metric, "metrics"),
        (occurrence, "occurrence"),
        (numeric, "numeric"),
        (gate, "gate"),
    ):
        paths[key].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    report = [
        "# GEFS QDM 因果当前周期目标 CDF 2019 验证",
        "",
        "2000-2018 用于拟合；每个 2019 目标 CDF 只使用当前周期 175 个 GEFS 值。",
        "2019 GHCN-D 仅用于评分，验证日期不是全新独立留出集。",
        "",
        f"- 7 天 MAE 差值：`{metric['seven_day_mae_difference_candidate_minus_raw_mm']:+.8f} mm`",
        f"- CRPS 差值：`{metric['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier 差值：`{metric['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- 完整站点-周期：`{metric['complete_site_cycle_count']}`",
        f"- 最大 QDM 相对变化：`{numeric['maximum_qdm_relative_quantile_change']:.6f}`",
        f"- 最大水量缩放系数：`{numeric['maximum_scale_factor']:.6f}`",
        f"- P99 水量缩放系数：`{numeric['p99_scale_factor']:.6f}`",
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

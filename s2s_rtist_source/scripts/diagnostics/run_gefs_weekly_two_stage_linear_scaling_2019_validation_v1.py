#!/usr/bin/env python3
"""Validate frozen site-only weekly two-stage GEFS linear scaling in 2019."""

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
from scripts.diagnostics.run_gefs_weekly_two_stage_linear_scaling_cv_v1 import (
    apply_two_stage_factors,
    fit_two_stage_factors,
    weekly_cycle_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_two_stage_linear_scaling_2019_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_two_stage_linear_scaling_2019_validation_v1"
)
CANDIDATE_ID = "weekly_two_stage_linear_site_only"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != (
        "gefs-weekly-two-stage-linear-scaling-2019-validation-v1"
    ):
        raise ValueError("weekly linear 2019 contract id mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("weekly linear 2019 candidate mismatch")
    if contract.get("group_keys") != ["site_id"]:
        raise ValueError("weekly linear 2019 group mismatch")
    if float(contract.get("extreme_quantile")) != 0.9:
        raise ValueError("weekly linear 2019 quantile mismatch")
    scope = contract["scope"]
    if scope["use_2019_reference_for_fit"]:
        raise ValueError("2019 reference cannot be used for factor fit")
    if scope["use_future_2019_cycles_for_factor_fit"]:
        raise ValueError("future 2019 cycles cannot be used for factor fit")
    if scope["use_2024_allowed"]:
        raise ValueError("2024 must be prohibited")
    return contract


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    history = load_inputs(args)
    fit_years = tuple(range(2000, 2019))
    if set(history["decision_date"].dt.year.astype(int)).difference(fit_years):
        raise ValueError("factor fit contains a year outside 2000-2018")
    fit_cycles = weekly_cycle_table(history, require_reference=True)
    factors = fit_two_stage_factors(
        fit_cycles,
        group_keys=("site_id",),
        extreme_quantile=float(contract["extreme_quantile"]),
    )
    if set(factors["site_id"].astype(str)) != set(contract["expected_sites"]):
        raise ValueError("weekly linear 2019 factor sites mismatch")
    factors["candidate_id"] = CANDIDATE_ID
    factors["fit_first_year"] = min(fit_years)
    factors["fit_last_year"] = max(fit_years)
    factors["validation_year"] = 2019
    factors["validation_rows_used_for_fit"] = 0

    target = load_2019_target(args)
    corrected = apply_two_stage_factors(
        target,
        factors,
        candidate=CANDIDATE_ID,
        group_keys=("site_id",),
    )
    evaluation = complete_site_cycle_rows(corrected)
    if evaluation.empty:
        raise ValueError("2019 has no complete weekly linear evaluation cycles")
    metric = _metric_row(CANDIDATE_ID, 2019, _metric_bundle(evaluation))
    metric["complete_site_cycle_count"] = int(
        evaluation[["site_id", "decision_date"]].drop_duplicates().shape[0]
    )
    occurrence = occurrence_row(CANDIDATE_ID, evaluation)

    factor_values = corrected["weekly_linear_scaling_factor"].to_numpy(dtype=float)
    corrected_values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    numeric = {
        "candidate_id": CANDIDATE_ID,
        "negative_count": int((corrected_values < 0.0).sum()),
        "nonfinite_count": int((~np.isfinite(corrected_values)).sum()),
        "minimum_applied_factor": float(factor_values.min()),
        "maximum_applied_factor": float(factor_values.max()),
        "extreme_target_site_cycle_count": int(
            corrected.loc[
                corrected["weekly_extreme_regime"],
                ["site_id", "decision_date"],
            ].drop_duplicates().shape[0]
        ),
        "maximum_corrected_mm_day": float(corrected_values.max()),
    }
    numeric_passed = bool(
        numeric["negative_count"] == 0
        and numeric["nonfinite_count"] == 0
        and np.isfinite(factor_values).all()
        and np.all(factor_values >= 0.0)
    )
    seven_day_passed = bool(metric["seven_day_mae_not_worse"])
    daily_mae_passed = bool(
        metric["candidate_ensemble_mean_mae"] <= metric["raw_ensemble_mean_mae"]
    )
    daily_rmse_passed = bool(
        metric["candidate_ensemble_mean_rmse"] <= metric["raw_ensemble_mean_rmse"]
    )
    passed = bool(
        seven_day_passed
        and daily_mae_passed
        and daily_rmse_passed
        and metric["crps_not_worse"]
        and metric["mean_brier_not_worse"]
        and metric["heavy_coverage_not_both_worse"]
        and occurrence["occurrence_not_worse"]
        and numeric_passed
    )
    gate = {
        "contract_id": contract["contract_id"],
        "candidate_id": CANDIDATE_ID,
        "fit_years": list(fit_years),
        "validation_year": 2019,
        "2019_reference_used_for_fit": False,
        "future_2019_cycles_used_for_factor_fit": False,
        "2024_used": False,
        "validation_date_status": "validation_set_previously_explored_not_independent",
        "seven_day_mae_not_worse": seven_day_passed,
        "daily_ensemble_mean_mae_not_worse": daily_mae_passed,
        "daily_ensemble_mean_rmse_not_worse": daily_rmse_passed,
        "crps_not_worse": bool(metric["crps_not_worse"]),
        "mean_brier_not_worse": bool(metric["mean_brier_not_worse"]),
        "heavy_coverage_not_both_worse": bool(
            metric["heavy_coverage_not_both_worse"]
        ),
        "occurrence_not_worse": bool(occurrence["occurrence_not_worse"]),
        "numeric_audit_passed": numeric_passed,
        "all_requirements_passed": passed,
        "status": (
            "passed_2019_validation_candidate_for_raw_gefs_postprocessing"
            if passed
            else "failed_2019_validation_continue_to_gamma_qm"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "factors": args.output_dir / "weekly_two_stage_linear_site_factors_2000_2018_v1.csv",
        "predictions": args.output_dir / "weekly_two_stage_linear_predictions_2019_v1.csv",
        "evaluation": args.output_dir / "weekly_two_stage_linear_complete_cycle_evaluation_2019_v1.csv",
        "metrics": args.output_dir / "weekly_two_stage_linear_metrics_2019_v1.json",
        "occurrence": args.output_dir / "weekly_two_stage_linear_occurrence_2019_v1.json",
        "numeric": args.output_dir / "weekly_two_stage_linear_numeric_audit_2019_v1.json",
        "gate": args.output_dir / "weekly_two_stage_linear_candidate_gate_2019_v1.json",
        "report": args.output_dir / "weekly_two_stage_linear_conclusion_2019_v1.md",
    }
    factors.to_csv(paths["factors"], index=False, encoding="utf-8-sig")
    corrected.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    evaluation.to_csv(paths["evaluation"], index=False, encoding="utf-8-sig")
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
        "# GEFS 按站点七日累计两阶段线性缩放 2019 验证",
        "",
        "候选由2015-2018 OOF冻结；2000-2018拟合，2019观测仅用于评分，2024未使用。",
        "",
        f"- 7天MAE差值：`{metric['seven_day_mae_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- 日MAE差值：`{metric['candidate_ensemble_mean_mae'] - metric['raw_ensemble_mean_mae']:+.6f} mm`",
        f"- 日RMSE差值：`{metric['candidate_ensemble_mean_rmse'] - metric['raw_ensemble_mean_rmse']:+.6f} mm`",
        f"- CRPS差值：`{metric['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier差值：`{metric['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- 因子范围：`{numeric['minimum_applied_factor']:.6f}–{numeric['maximum_applied_factor']:.6f}`",
        f"- 完整站点-周期：`{metric['complete_site_cycle_count']}`",
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

#!/usr/bin/env python3
"""Select shrinkage for frozen site-only weekly linear scaling using OOF and 2019."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.diagnose_gefs_qdm_7day_volume_preservation_v1 import (
    metric_tables,
)
from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import (
    occurrence_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_factor_shrinkage_selection_contract_v1.json"
)
OOF_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_two_stage_linear_scaling_cv_server_v1"
)
VALIDATION_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_two_stage_linear_scaling_2019_validation_server_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_factor_shrinkage_selection_v1"
)
BASE_CANDIDATE_ID = "weekly_two_stage_linear_site_only"


def candidate_id(alpha: float) -> str:
    return f"weekly_two_stage_linear_site_factor_shrink_a{int(round(alpha * 100)):03d}"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-factor-shrinkage-selection-v1":
        raise ValueError("weekly factor shrinkage contract id mismatch")
    if contract.get("base_candidate_id") != BASE_CANDIDATE_ID:
        raise ValueError("weekly factor shrinkage base candidate mismatch")
    alphas = [float(value) for value in contract["candidate_alphas"]]
    if alphas != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("weekly factor shrinkage candidate set mismatch")
    scope = contract["scope"]
    if scope["refit_base_factors_allowed"]:
        raise ValueError("weekly factor shrinkage cannot refit base factors")
    if not scope["use_2019_for_hyperparameter_selection"]:
        raise ValueError("2019 must be explicitly designated as validation")
    if scope["use_2019_reference_for_base_factor_fit"]:
        raise ValueError("2019 reference cannot fit base factors")
    if scope["use_2024_allowed"]:
        raise ValueError("2024 must be prohibited")
    return contract


def shrink_factors(frame: pd.DataFrame, alpha: float, *, split: str) -> pd.DataFrame:
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("factor shrinkage alpha must be in (0, 1]")
    required = {
        "precipitation_mm_raw",
        "weekly_linear_scaling_factor",
        "candidate_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing factor shrinkage columns: {sorted(missing)}")
    data = frame.loc[frame["candidate_id"].eq(BASE_CANDIDATE_ID)].copy()
    if data.empty:
        raise ValueError(f"{split} has no frozen site-only base candidate")
    data["base_weekly_linear_scaling_factor"] = data["weekly_linear_scaling_factor"]
    data["factor_shrinkage_alpha"] = float(alpha)
    data["weekly_linear_scaling_factor"] = 1.0 + float(alpha) * (
        data["base_weekly_linear_scaling_factor"] - 1.0
    )
    data["precipitation_mm_qm"] = (
        data["precipitation_mm_raw"] * data["weekly_linear_scaling_factor"]
    )
    data["candidate_id"] = candidate_id(alpha)
    data["factor_shrinkage_split"] = split
    factors = data["weekly_linear_scaling_factor"].to_numpy(dtype=float)
    corrected = data["precipitation_mm_qm"].to_numpy(dtype=float)
    if (
        np.any(~np.isfinite(factors))
        or np.any(factors < 0.0)
        or np.any(~np.isfinite(corrected))
        or np.any(corrected < 0.0)
    ):
        raise ValueError("factor shrinkage produced invalid precipitation")
    return data


def split_gates(
    predictions: pd.DataFrame,
    *,
    split: str,
    required_years: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_metrics, pooled_metrics, _ = metric_tables(predictions)
    occurrence = pd.DataFrame(
        [
            occurrence_row(candidate, group)
            for candidate, group in predictions.groupby("candidate_id", sort=True)
        ]
    )
    rows = []
    for pooled in pooled_metrics.itertuples(index=False):
        candidate = pooled.candidate_id
        group = predictions.loc[predictions["candidate_id"].eq(candidate)]
        occurrence_passed = bool(
            occurrence.loc[
                occurrence["candidate_id"].eq(candidate), "occurrence_not_worse"
            ].iloc[0]
        )
        daily_mae_passed = bool(
            pooled.candidate_ensemble_mean_mae <= pooled.raw_ensemble_mean_mae
        )
        daily_rmse_passed = bool(
            pooled.candidate_ensemble_mean_rmse <= pooled.raw_ensemble_mean_rmse
        )
        values = group["precipitation_mm_qm"].to_numpy(dtype=float)
        factors = group["weekly_linear_scaling_factor"].to_numpy(dtype=float)
        numeric_passed = bool(
            np.isfinite(values).all()
            and np.all(values >= 0.0)
            and np.isfinite(factors).all()
            and np.all(factors >= 0.0)
        )
        years = year_metrics.loc[year_metrics["candidate_id"].eq(candidate)]
        mae_years = int(years["seven_day_mae_not_worse"].sum())
        crps_years = int(years["crps_not_worse"].sum())
        brier_years = int(years["mean_brier_not_worse"].sum())
        years_passed = bool(
            required_years is None
            or (
                mae_years >= required_years
                and crps_years >= required_years
                and brier_years >= required_years
            )
        )
        passed = bool(
            daily_mae_passed
            and daily_rmse_passed
            and pooled.seven_day_mae_not_worse
            and pooled.crps_not_worse
            and pooled.mean_brier_not_worse
            and pooled.heavy_coverage_not_both_worse
            and occurrence_passed
            and numeric_passed
            and years_passed
        )
        alpha = float(group["factor_shrinkage_alpha"].iloc[0])
        rows.append(
            {
                "split": split,
                "candidate_id": candidate,
                "factor_shrinkage_alpha": alpha,
                "daily_ensemble_mean_mae_not_worse": daily_mae_passed,
                "daily_ensemble_mean_rmse_not_worse": daily_rmse_passed,
                "seven_day_mae_not_worse": bool(pooled.seven_day_mae_not_worse),
                "crps_not_worse": bool(pooled.crps_not_worse),
                "mean_brier_not_worse": bool(pooled.mean_brier_not_worse),
                "heavy_coverage_not_both_worse": bool(
                    pooled.heavy_coverage_not_both_worse
                ),
                "occurrence_not_worse": occurrence_passed,
                "mae_years_not_worse": mae_years,
                "crps_years_not_worse": crps_years,
                "brier_years_not_worse": brier_years,
                "required_years_not_worse": required_years,
                "numeric_audit_passed": numeric_passed,
                "minimum_effective_factor": float(factors.min()),
                "maximum_effective_factor": float(factors.max()),
                "all_requirements_passed": passed,
            }
        )
    return pd.DataFrame(rows), year_metrics, pooled_metrics, occurrence


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    oof_source = pd.read_csv(args.oof_predictions)
    validation_source = pd.read_csv(args.validation_predictions)
    for frame in (oof_source, validation_source):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"])
        frame["valid_date_utc"] = pd.to_datetime(frame["valid_date_utc"])
    if set(oof_source["validation_year"].astype(int)) != {2015, 2016, 2017, 2018}:
        raise ValueError("factor shrinkage OOF input years mismatch")
    validation_source["validation_year"] = 2019

    oof = pd.concat(
        [
            shrink_factors(oof_source, float(alpha), split="training_oof_2015_2018")
            for alpha in contract["candidate_alphas"]
        ],
        ignore_index=True,
    )
    validation = pd.concat(
        [
            shrink_factors(validation_source, float(alpha), split="validation_2019")
            for alpha in contract["candidate_alphas"]
        ],
        ignore_index=True,
    )
    required_years = int(contract["minimum_oof_years_not_worse_per_primary_metric"])
    oof_gate, oof_year, oof_pooled, oof_occurrence = split_gates(
        oof,
        split="training_oof_2015_2018",
        required_years=required_years,
    )
    validation_gate, validation_year, validation_pooled, validation_occurrence = split_gates(
        validation,
        split="validation_2019",
        required_years=None,
    )
    combined_gate = oof_gate.merge(
        validation_gate,
        on=["candidate_id", "factor_shrinkage_alpha"],
        suffixes=("_oof", "_2019"),
        validate="one_to_one",
    )
    combined_gate["eligible_for_2024_frozen_test"] = (
        combined_gate["all_requirements_passed_oof"]
        & combined_gate["all_requirements_passed_2019"]
    )
    eligible = combined_gate.loc[
        combined_gate["eligible_for_2024_frozen_test"], "candidate_id"
    ].tolist()
    selected = None
    if eligible:
        selection = validation_pooled.loc[
            validation_pooled["candidate_id"].isin(eligible)
        ].merge(
            combined_gate[["candidate_id", "factor_shrinkage_alpha"]],
            on="candidate_id",
            validate="one_to_one",
        )
        selected = str(
            selection.sort_values(
                [
                    "candidate_seven_day_mae_mm",
                    "candidate_ensemble_mean_rmse",
                    "candidate_mean_crps_mm",
                    "candidate_mean_brier_score",
                    "factor_shrinkage_alpha",
                ]
            ).iloc[0]["candidate_id"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof_predictions": args.output_dir / "weekly_factor_shrinkage_oof_predictions_v1.csv",
        "validation_predictions": args.output_dir / "weekly_factor_shrinkage_2019_predictions_v1.csv",
        "oof_year_metrics": args.output_dir / "weekly_factor_shrinkage_oof_year_metrics_v1.csv",
        "oof_pooled_metrics": args.output_dir / "weekly_factor_shrinkage_oof_pooled_metrics_v1.csv",
        "validation_metrics": args.output_dir / "weekly_factor_shrinkage_2019_metrics_v1.csv",
        "oof_occurrence": args.output_dir / "weekly_factor_shrinkage_oof_occurrence_v1.csv",
        "validation_occurrence": args.output_dir / "weekly_factor_shrinkage_2019_occurrence_v1.csv",
        "gate": args.output_dir / "weekly_factor_shrinkage_selection_gate_v1.json",
        "report": args.output_dir / "weekly_factor_shrinkage_selection_conclusion_v1.md",
    }
    for frame, key in (
        (oof, "oof_predictions"),
        (validation, "validation_predictions"),
        (oof_year, "oof_year_metrics"),
        (oof_pooled, "oof_pooled_metrics"),
        (validation_pooled, "validation_metrics"),
        (oof_occurrence, "oof_occurrence"),
        (validation_occurrence, "validation_occurrence"),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    payload = {
        "contract_id": contract["contract_id"],
        "2019_role": "previously_explored_validation_hyperparameter_selection",
        "2019_reference_used_for_base_factor_fit": False,
        "2024_used": False,
        "candidate_set": contract["candidate_alphas"],
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "selected_alpha": (
            float(
                combined_gate.loc[
                    combined_gate["candidate_id"].eq(selected),
                    "factor_shrinkage_alpha",
                ].iloc[0]
            )
            if selected is not None
            else None
        ),
        "selection_order": contract["selection_order"],
        "status": (
            "factor_shrinkage_selected_freeze_before_2024"
            if selected is not None
            else "no_factor_shrinkage_candidate_eligible_continue_to_gamma_qm"
        ),
        "candidate_gate": combined_gate.to_dict(orient="records"),
    }
    paths["gate"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    oof_index = oof_pooled.set_index("candidate_id")
    val_index = validation_pooled.set_index("candidate_id")
    report = [
        "# GEFS 周线性缩放因子收缩选择",
        "",
        "2015-2018 OOF用于资格筛选，2019作为已探索验证集选择alpha；2024未使用。",
        "",
        "| alpha | OOF 7天MAE差值 | OOF日RMSE差值 | OOF Brier差值 | 2019 7天MAE差值 | 2019日RMSE差值 | 2019 Brier差值 | 晋级 |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in combined_gate.sort_values("factor_shrinkage_alpha").itertuples(index=False):
        oof_metric = oof_index.loc[row.candidate_id]
        val_metric = val_index.loc[row.candidate_id]
        report.append(
            f"| {row.factor_shrinkage_alpha:.2f} | "
            f"{oof_metric.seven_day_mae_difference_candidate_minus_raw_mm:+.6f} | "
            f"{oof_metric.candidate_ensemble_mean_rmse - oof_metric.raw_ensemble_mean_rmse:+.6f} | "
            f"{oof_metric.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{val_metric.seven_day_mae_difference_candidate_minus_raw_mm:+.6f} | "
            f"{val_metric.candidate_ensemble_mean_rmse - val_metric.raw_ensemble_mean_rmse:+.6f} | "
            f"{val_metric.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.eligible_for_2024_frozen_test} |"
        )
    report.extend(["", f"锁定候选：`{selected}`。" if selected else "无候选晋级，继续 Piani Gamma QM。"])
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=OOF_ROOT / "weekly_two_stage_linear_oof_predictions_v1.csv",
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=VALIDATION_ROOT / "weekly_two_stage_linear_complete_cycle_evaluation_2019_v1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

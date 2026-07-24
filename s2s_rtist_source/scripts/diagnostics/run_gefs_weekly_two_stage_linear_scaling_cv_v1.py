#!/usr/bin/env python3
"""Run expanding-window OOF CV for GEFS weekly two-stage linear scaling."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_two_stage_linear_scaling_cv_contract_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_two_stage_linear_scaling_cv_v1"
)
CANDIDATES = {
    "weekly_two_stage_linear_global": (),
    "weekly_two_stage_linear_site_only": ("site_id",),
}


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-two-stage-linear-scaling-cv-v1":
        raise ValueError("weekly linear scaling contract id mismatch")
    if contract.get("candidate_set") != list(CANDIDATES):
        raise ValueError("weekly linear scaling candidate set mismatch")
    if float(contract.get("extreme_quantile")) != 0.9:
        raise ValueError("weekly linear scaling extreme quantile mismatch")
    scope = contract["scope"]
    if scope["use_2019_allowed"] or scope["use_2024_allowed"]:
        raise ValueError("2019 and 2024 must be prohibited")
    return contract


def weekly_cycle_table(frame: pd.DataFrame, *, require_reference: bool) -> pd.DataFrame:
    data = frame.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["valid_date_utc"] = pd.to_datetime(data["valid_date_utc"])
    keys = ["site_id", "decision_date"]
    member_keys = keys + ["gefs_member"]
    member_counts = data.groupby(member_keys)["valid_date_utc"].nunique()
    complete_members = member_counts.loc[member_counts.eq(7)].reset_index()[member_keys]
    data = data.merge(
        complete_members,
        on=member_keys,
        how="inner",
        validate="many_to_one",
    )
    member_totals = (
        data.groupby(member_keys, as_index=False)["precipitation_mm_raw"]
        .sum()
        .rename(columns={"precipitation_mm_raw": "member_raw_7d_mm"})
    )
    member_number = member_totals.groupby(keys)["gefs_member"].nunique()
    complete_cycles = member_number.loc[member_number.eq(5)].reset_index()[keys]
    member_totals = member_totals.merge(
        complete_cycles,
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    cycles = (
        member_totals.groupby(keys, as_index=False)["member_raw_7d_mm"]
        .mean()
        .rename(columns={"member_raw_7d_mm": "ensemble_mean_raw_7d_mm"})
    )
    if require_reference:
        reference = data.loc[
            data["reference_valid_unflagged"]
            & data["precipitation_mm_reference"].notna()
        ].drop_duplicates(keys + ["valid_date_utc"])
        reference_counts = reference.groupby(keys)["valid_date_utc"].nunique()
        complete_reference = reference_counts.loc[reference_counts.eq(7)].reset_index()[keys]
        reference_totals = (
            reference.merge(
                complete_reference,
                on=keys,
                how="inner",
                validate="many_to_one",
            )
            .groupby(keys, as_index=False)["precipitation_mm_reference"]
            .sum()
            .rename(columns={"precipitation_mm_reference": "reference_7d_mm"})
        )
        cycles = cycles.merge(
            reference_totals,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    return cycles.sort_values(keys).reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float, *, label: str) -> float:
    if denominator <= 0.0:
        if numerator == 0.0:
            return 1.0
        raise ValueError(f"positive {label} numerator with zero denominator")
    result = numerator / denominator
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"invalid {label} scaling factor")
    return float(result)


def fit_two_stage_factors(
    cycles: pd.DataFrame,
    *,
    group_keys: tuple[str, ...],
    extreme_quantile: float,
) -> pd.DataFrame:
    required = {"site_id", "ensemble_mean_raw_7d_mm", "reference_7d_mm"}
    missing = required.difference(cycles.columns)
    if missing:
        raise ValueError(f"missing weekly factor columns: {sorted(missing)}")
    grouped = [((), cycles)] if not group_keys else cycles.groupby(list(group_keys), sort=True)
    rows = []
    for raw_key, group in grouped:
        values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        raw = group["ensemble_mean_raw_7d_mm"].to_numpy(dtype=float)
        obs = group["reference_7d_mm"].to_numpy(dtype=float)
        if len(group) < 10:
            raise ValueError("weekly factor group has fewer than 10 complete cycles")
        q90 = float(np.quantile(raw, extreme_quantile))
        extreme = raw > q90
        if int(extreme.sum()) < 2:
            raise ValueError("weekly factor group has fewer than two extreme cycles")
        extreme_factor = _safe_ratio(
            float(obs[extreme].sum()),
            float(raw[extreme].sum()),
            label="extreme",
        )
        stage1 = raw.copy()
        stage1[extreme] *= extreme_factor
        overall_factor = _safe_ratio(
            float(obs.sum()),
            float(stage1.sum()),
            label="overall",
        )
        row = {
            "fit_complete_cycle_count": int(len(group)),
            "fit_extreme_cycle_count": int(extreme.sum()),
            "extreme_quantile": float(extreme_quantile),
            "raw_ensemble_mean_7d_q90_mm": q90,
            "extreme_factor": extreme_factor,
            "overall_factor": overall_factor,
            "final_extreme_factor": extreme_factor * overall_factor,
        }
        for column, value in zip(group_keys, values, strict=True):
            row[column] = value
        rows.append(row)
    return pd.DataFrame(rows)


def apply_two_stage_factors(
    frame: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    candidate: str,
    group_keys: tuple[str, ...],
) -> pd.DataFrame:
    data = frame.copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["valid_date_utc"] = pd.to_datetime(data["valid_date_utc"])
    cycles = weekly_cycle_table(data, require_reference=False)
    merge_keys = list(group_keys)
    if merge_keys:
        cycles = cycles.merge(factors, on=merge_keys, how="left", validate="many_to_one")
    else:
        if len(factors) != 1:
            raise ValueError("global weekly factor artifact must have one row")
        for column in factors.columns:
            cycles[column] = factors.iloc[0][column]
    if cycles["overall_factor"].isna().any():
        raise ValueError("weekly scaling target has a missing factor")
    cycles["weekly_extreme_regime"] = cycles["ensemble_mean_raw_7d_mm"].gt(
        cycles["raw_ensemble_mean_7d_q90_mm"]
    )
    cycles["weekly_linear_scaling_factor"] = np.where(
        cycles["weekly_extreme_regime"],
        cycles["final_extreme_factor"],
        cycles["overall_factor"],
    )
    factor_values = cycles["weekly_linear_scaling_factor"].to_numpy(dtype=float)
    if np.any(~np.isfinite(factor_values)) or np.any(factor_values < 0.0):
        raise ValueError("weekly scaling produced an invalid target factor")
    cycle_columns = [
        "site_id",
        "decision_date",
        "ensemble_mean_raw_7d_mm",
        "raw_ensemble_mean_7d_q90_mm",
        "extreme_factor",
        "overall_factor",
        "final_extreme_factor",
        "weekly_extreme_regime",
        "weekly_linear_scaling_factor",
    ]
    data = data.merge(
        cycles[cycle_columns],
        on=["site_id", "decision_date"],
        how="left",
        validate="many_to_one",
    )
    if data["weekly_linear_scaling_factor"].isna().any():
        raise ValueError("weekly scaling left target rows without a factor")
    data["precipitation_mm_qm"] = (
        data["precipitation_mm_raw"] * data["weekly_linear_scaling_factor"]
    )
    data["qm_extrapolated_upper"] = False
    data["candidate_id"] = candidate
    values = data["precipitation_mm_qm"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weekly scaling produced invalid precipitation")
    return data


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    data = load_inputs(args)
    if set(data["decision_date"].dt.year).intersection({2019, 2024}):
        raise ValueError("forbidden year entered weekly linear scaling CV")
    extreme_quantile = float(contract["extreme_quantile"])
    evaluation_parts = []
    artifact_parts = []
    for fold in expanding_folds():
        fit_years = tuple(int(value) for value in fold["fit_years"])
        validation_year = int(fold["validation_year"])
        fit = data.loc[data["decision_date"].dt.year.isin(fit_years)].copy()
        target = data.loc[data["decision_date"].dt.year.eq(validation_year)].copy()
        fit_cycles = weekly_cycle_table(fit, require_reference=True)
        for candidate, group_keys in CANDIDATES.items():
            factors = fit_two_stage_factors(
                fit_cycles,
                group_keys=group_keys,
                extreme_quantile=extreme_quantile,
            )
            factors["candidate_id"] = candidate
            factors["fold_id"] = fold["fold_id"]
            factors["validation_year"] = validation_year
            factors["fit_first_year"] = min(fit_years)
            factors["fit_last_year"] = max(fit_years)
            factors["validation_rows_used_for_fit"] = 0
            artifact_parts.append(factors)
            corrected = apply_two_stage_factors(
                target,
                factors,
                candidate=candidate,
                group_keys=group_keys,
            )
            corrected.loc[
                ~corrected["reference_valid_unflagged"],
                "precipitation_mm_reference",
            ] = np.nan
            evaluation = complete_site_cycle_rows(corrected)
            if evaluation.empty:
                raise ValueError(f"no complete evaluation cycles for {candidate} {validation_year}")
            evaluation["validation_year"] = validation_year
            evaluation_parts.append(evaluation)
            print(
                f"[weekly-linear] {fold['fold_id']} {candidate} rows={len(evaluation)}",
                flush=True,
            )

    oof = pd.concat(evaluation_parts, ignore_index=True)
    artifacts = pd.concat(artifact_parts, ignore_index=True)
    year_metrics, pooled_metrics, site_metrics = metric_tables(oof)
    occurrence = pd.DataFrame(
        [
            occurrence_row(candidate, group)
            for candidate, group in oof.groupby("candidate_id", sort=True)
        ]
    )
    required_years = int(
        contract["candidate_gate"]["minimum_years_not_worse_per_primary_metric"]
    )
    gate_rows = []
    for pooled in pooled_metrics.itertuples(index=False):
        candidate = pooled.candidate_id
        years = year_metrics.loc[year_metrics["candidate_id"].eq(candidate)]
        candidate_rows = oof.loc[oof["candidate_id"].eq(candidate)]
        factor_values = candidate_rows["weekly_linear_scaling_factor"].to_numpy(dtype=float)
        corrected_values = candidate_rows["precipitation_mm_qm"].to_numpy(dtype=float)
        numeric_passed = bool(
            np.isfinite(factor_values).all()
            and np.all(factor_values >= 0.0)
            and np.isfinite(corrected_values).all()
            and np.all(corrected_values >= 0.0)
        )
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
        mae_years = int(years["seven_day_mae_not_worse"].sum())
        crps_years = int(years["crps_not_worse"].sum())
        brier_years = int(years["mean_brier_not_worse"].sum())
        eligible = bool(
            daily_mae_passed
            and daily_rmse_passed
            and pooled.seven_day_mae_not_worse
            and pooled.crps_not_worse
            and pooled.mean_brier_not_worse
            and pooled.heavy_coverage_not_both_worse
            and occurrence_passed
            and mae_years >= required_years
            and crps_years >= required_years
            and brier_years >= required_years
            and numeric_passed
        )
        gate_rows.append(
            {
                "candidate_id": candidate,
                "pooled_daily_ensemble_mean_mae_not_worse": daily_mae_passed,
                "pooled_daily_ensemble_mean_rmse_not_worse": daily_rmse_passed,
                "pooled_seven_day_mae_not_worse": bool(pooled.seven_day_mae_not_worse),
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
                "minimum_applied_factor": float(factor_values.min()),
                "maximum_applied_factor": float(factor_values.max()),
                "eligible_for_2019_validation": eligible,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values("candidate_id")
    eligible_ids = gate.loc[gate["eligible_for_2019_validation"], "candidate_id"].tolist()
    selected = None
    if eligible_ids:
        selection = pooled_metrics.loc[pooled_metrics["candidate_id"].isin(eligible_ids)]
        selected = str(
            selection.sort_values(
                [
                    "candidate_seven_day_mae_mm",
                    "candidate_ensemble_mean_rmse",
                    "candidate_mean_crps_mm",
                    "candidate_mean_brier_score",
                    "candidate_id",
                ]
            ).iloc[0]["candidate_id"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": args.output_dir / "weekly_two_stage_linear_oof_predictions_v1.csv",
        "artifacts": args.output_dir / "weekly_two_stage_linear_fold_factors_v1.csv",
        "year_metrics": args.output_dir / "weekly_two_stage_linear_year_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "weekly_two_stage_linear_pooled_metrics_v1.csv",
        "site_metrics": args.output_dir / "weekly_two_stage_linear_site_metrics_v1.csv",
        "occurrence": args.output_dir / "weekly_two_stage_linear_occurrence_v1.csv",
        "gate": args.output_dir / "weekly_two_stage_linear_candidate_gate_v1.json",
        "report": args.output_dir / "weekly_two_stage_linear_conclusion_v1.md",
    }
    for frame, key in (
        (oof, "predictions"),
        (artifacts, "artifacts"),
        (year_metrics, "year_metrics"),
        (pooled_metrics, "pooled_metrics"),
        (site_metrics, "site_metrics"),
        (occurrence, "occurrence"),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    payload = {
        "contract_id": contract["contract_id"],
        "2019_used": False,
        "2024_used": False,
        "candidate_set_prelocked": list(CANDIDATES),
        "eligible_candidates": eligible_ids,
        "selected_candidate": selected,
        "selection_order": contract["selection_order"],
        "status": (
            "candidate_selected_for_2019_validation"
            if selected is not None
            else "no_weekly_linear_candidate_eligible_retain_raw"
        ),
        "candidate_gate": gate.to_dict(orient="records"),
    }
    paths["gate"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_index = pooled_metrics.set_index("candidate_id")
    report = [
        "# GEFS 七日累计两阶段线性缩放训练期 OOF",
        "",
        "按 Shah-Mishra GEFS 论文构造；2019 和 2024 未使用。",
        "",
        "| 候选 | 7天MAE差值 | 日MAE差值 | 日RMSE差值 | CRPS差值 | Brier差值 | 晋级 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate.itertuples(index=False):
        metric = metrics_index.loc[row.candidate_id]
        report.append(
            f"| `{row.candidate_id}` | "
            f"{metric.seven_day_mae_difference_candidate_minus_raw_mm:+.6f} | "
            f"{metric.candidate_ensemble_mean_mae - metric.raw_ensemble_mean_mae:+.6f} | "
            f"{metric.candidate_ensemble_mean_rmse - metric.raw_ensemble_mean_rmse:+.6f} | "
            f"{metric.crps_difference_candidate_minus_raw_mm:+.6f} | "
            f"{metric.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.eligible_for_2019_validation} |"
        )
    report.extend(["", f"锁定候选：`{selected}`。" if selected else "无候选晋级，保留 raw GEFS。"])
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

#!/usr/bin/env python3
"""Evaluate prelocked raw/QDM daily-allocation shrinkage on causal OOF rows."""

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
    / "gefs_qdm_raw_allocation_shrinkage_cv_contract_v1.json"
)
DEFAULT_OOF = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_causal_current_cycle_cv_server_v1"
    / "causal_current_cycle_oof_predictions_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_raw_allocation_shrinkage_cv_v1"
)
BASE_CANDIDATE_ID = "qdm_global_current_cycle_7d_volume_preserving"


def candidate_id(alpha: float) -> str:
    return f"qdm_global_current_cycle_vp_raw_shrink_a{int(round(alpha * 100)):03d}"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-qdm-raw-allocation-shrinkage-cv-v1":
        raise ValueError("raw-allocation shrinkage contract id mismatch")
    if contract.get("base_candidate_id") != BASE_CANDIDATE_ID:
        raise ValueError("raw-allocation shrinkage base candidate mismatch")
    alphas = [float(value) for value in contract["candidate_alphas"]]
    if alphas != [0.25, 0.5, 0.75, 1.0]:
        raise ValueError("raw-allocation shrinkage candidate set mismatch")
    if any(value <= 0.0 or value > 1.0 for value in alphas):
        raise ValueError("shrinkage alpha must be in (0, 1]")
    scope = contract["scope"]
    if scope["refit_qdm_allowed"] or scope["use_2019_allowed"] or scope["use_2024_allowed"]:
        raise ValueError("shrinkage CV cannot refit QDM or use 2019/2024")
    return contract


def shrink_prediction(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("shrinkage alpha must be in (0, 1]")
    required = {
        "site_id",
        "decision_date",
        "valid_date_utc",
        "gefs_member",
        "validation_year",
        "precipitation_mm_raw",
        "precipitation_mm_qm",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing shrinkage columns: {sorted(missing)}")
    data = frame.copy()
    data["precipitation_mm_qdm_vp_alpha_1"] = data["precipitation_mm_qm"]
    data["precipitation_mm_qm"] = data["precipitation_mm_raw"] + float(alpha) * (
        data["precipitation_mm_qdm_vp_alpha_1"] - data["precipitation_mm_raw"]
    )
    data["shrinkage_alpha"] = float(alpha)
    data["candidate_id"] = candidate_id(alpha)
    values = data["precipitation_mm_qm"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("shrinkage produced invalid precipitation")
    return data


def member_total_audit(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_id", "site_id", "decision_date", "gefs_member"]
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        if len(group) != 7:
            raise ValueError(f"shrinkage member group {key} has {len(group)} rows")
        raw_total = float(group["precipitation_mm_raw"].sum())
        corrected_total = float(group["precipitation_mm_qm"].sum())
        rows.append(
            {
                "candidate_id": key[0],
                "site_id": key[1],
                "decision_date": key[2],
                "gefs_member": key[3],
                "shrinkage_alpha": float(group["shrinkage_alpha"].iloc[0]),
                "raw_total_mm": raw_total,
                "corrected_total_mm": corrected_total,
                "member_total_error_mm": corrected_total - raw_total,
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    tolerance = float(contract["member_total_tolerance_mm"])
    source = pd.read_csv(args.oof_predictions)
    source["decision_date"] = pd.to_datetime(source["decision_date"])
    source["valid_date_utc"] = pd.to_datetime(source["valid_date_utc"])
    if set(source["validation_year"].astype(int)) != {2015, 2016, 2017, 2018}:
        raise ValueError("shrinkage input must contain only 2015-2018 OOF")
    if set(source["candidate_id"].astype(str)) != {BASE_CANDIDATE_ID}:
        raise ValueError("shrinkage input candidate mismatch")

    candidates = pd.concat(
        [
            shrink_prediction(source, float(alpha))
            for alpha in contract["candidate_alphas"]
        ],
        ignore_index=True,
    )
    audit = member_total_audit(candidates)
    year_metrics, pooled_metrics, site_metrics = metric_tables(candidates)
    occurrence = pd.DataFrame(
        [
            occurrence_row(candidate, group)
            for candidate, group in candidates.groupby("candidate_id", sort=True)
        ]
    )

    required_years = int(
        contract["candidate_gate"]["minimum_years_not_worse_per_primary_metric"]
    )
    gate_rows = []
    for pooled in pooled_metrics.itertuples(index=False):
        candidate = pooled.candidate_id
        years = year_metrics.loc[year_metrics["candidate_id"].eq(candidate)]
        candidate_audit = audit.loc[audit["candidate_id"].eq(candidate)]
        occurrence_passed = bool(
            occurrence.loc[
                occurrence["candidate_id"].eq(candidate), "occurrence_not_worse"
            ].iloc[0]
        )
        alpha = float(
            candidates.loc[
                candidates["candidate_id"].eq(candidate), "shrinkage_alpha"
            ].iloc[0]
        )
        mae_years = int(
            (years["seven_day_mae_difference_candidate_minus_raw_mm"] <= tolerance).sum()
        )
        crps_years = int(years["crps_not_worse"].sum())
        brier_years = int(years["mean_brier_not_worse"].sum())
        daily_mae_passed = bool(
            pooled.candidate_ensemble_mean_mae <= pooled.raw_ensemble_mean_mae + tolerance
        )
        daily_rmse_passed = bool(
            pooled.candidate_ensemble_mean_rmse <= pooled.raw_ensemble_mean_rmse + tolerance
        )
        volume_passed = bool(
            candidate_audit["member_total_error_mm"].abs().max() <= tolerance
        )
        values = candidates.loc[
            candidates["candidate_id"].eq(candidate), "precipitation_mm_qm"
        ].to_numpy(dtype=float)
        numeric_passed = bool(np.isfinite(values).all() and np.all(values >= 0.0))
        eligible = bool(
            daily_mae_passed
            and daily_rmse_passed
            and pooled.seven_day_mae_difference_candidate_minus_raw_mm <= tolerance
            and pooled.crps_not_worse
            and pooled.mean_brier_not_worse
            and pooled.heavy_coverage_not_both_worse
            and occurrence_passed
            and mae_years >= required_years
            and crps_years >= required_years
            and brier_years >= required_years
            and volume_passed
            and numeric_passed
        )
        gate_rows.append(
            {
                "candidate_id": candidate,
                "shrinkage_alpha": alpha,
                "pooled_daily_ensemble_mean_mae_not_worse": daily_mae_passed,
                "pooled_daily_ensemble_mean_rmse_not_worse": daily_rmse_passed,
                "pooled_seven_day_mae_not_worse": bool(
                    pooled.seven_day_mae_difference_candidate_minus_raw_mm <= tolerance
                ),
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
                "eligible_for_2019_validation": eligible,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values("shrinkage_alpha")
    eligible_ids = gate.loc[gate["eligible_for_2019_validation"], "candidate_id"]
    selected = None
    if len(eligible_ids):
        eligible_metrics = pooled_metrics.loc[
            pooled_metrics["candidate_id"].isin(eligible_ids)
        ].merge(
            gate[["candidate_id", "shrinkage_alpha"]],
            on="candidate_id",
            validate="one_to_one",
        )
        selected = str(
            eligible_metrics.sort_values(
                [
                    "candidate_mean_crps_mm",
                    "candidate_mean_brier_score",
                    "candidate_ensemble_mean_mae",
                    "shrinkage_alpha",
                ]
            ).iloc[0]["candidate_id"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": args.output_dir / "raw_allocation_shrinkage_oof_predictions_v1.csv",
        "member_audit": args.output_dir / "raw_allocation_shrinkage_member_total_audit_v1.csv",
        "year_metrics": args.output_dir / "raw_allocation_shrinkage_year_metrics_v1.csv",
        "pooled_metrics": args.output_dir / "raw_allocation_shrinkage_pooled_metrics_v1.csv",
        "site_metrics": args.output_dir / "raw_allocation_shrinkage_site_metrics_v1.csv",
        "occurrence": args.output_dir / "raw_allocation_shrinkage_occurrence_v1.csv",
        "gate": args.output_dir / "raw_allocation_shrinkage_candidate_gate_v1.json",
        "report": args.output_dir / "raw_allocation_shrinkage_conclusion_v1.md",
    }
    for frame, key in (
        (candidates, "predictions"),
        (audit, "member_audit"),
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
        "candidate_set_prelocked": contract["candidate_alphas"],
        "eligible_candidates": eligible_ids.tolist(),
        "selected_candidate": selected,
        "selection_order": contract["selection_order"],
        "status": (
            "candidate_selected_for_2019_validation"
            if selected is not None
            else "no_shrinkage_candidate_eligible_retain_raw"
        ),
        "candidate_gate": gate.to_dict(orient="records"),
    }
    paths["gate"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS QDM 日分配向 raw 收缩训练期 OOF",
        "",
        "候选只使用 2015-2018 因果 OOF；2019 与 2024 未使用。",
        "",
        "| alpha | 日MAE差值 | 日RMSE差值 | CRPS差值 | Brier差值 | 发生频率 | 晋级 |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    metrics_index = pooled_metrics.set_index("candidate_id")
    for row in gate.itertuples(index=False):
        metrics = metrics_index.loc[row.candidate_id]
        report.append(
            f"| {row.shrinkage_alpha:.2f} | "
            f"{metrics.candidate_ensemble_mean_mae - metrics.raw_ensemble_mean_mae:+.6f} | "
            f"{metrics.candidate_ensemble_mean_rmse - metrics.raw_ensemble_mean_rmse:+.6f} | "
            f"{metrics.crps_difference_candidate_minus_raw_mm:+.6f} | "
            f"{metrics.mean_brier_difference_candidate_minus_raw:+.6f} | "
            f"{row.pooled_occurrence_not_worse} | "
            f"{row.eligible_for_2019_validation} |"
        )
    report.extend(["", f"锁定候选：`{selected}`。" if selected else "无候选晋级，保留 raw GEFS。"])
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

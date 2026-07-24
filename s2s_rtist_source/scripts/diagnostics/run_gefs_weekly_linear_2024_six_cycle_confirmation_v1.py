#!/usr/bin/env python3
"""Score the frozen correction on six prelocked 2024 GEFS cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.diagnostics.audit_gefs_weekly_linear_2024_ghcnd_reference_v1 import (
    bootstrap_cycle_metrics,
    metrics_for_scope,
    station_reference,
    target_dates,
)
from scripts.diagnostics.fit_gefs_weekly_linear_final_artifact_v1 import artifact_hash
from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import occurrence_row
from scripts.diagnostics.run_gefs_quantile_mapping_validation_v1 import (
    _deterministic_metrics,
)
from scripts.diagnostics.run_gefs_weekly_linear_2024_diagnostic_v1 import (
    apply_artifact,
    load_member_data,
    metric_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_2024_six_cycle_confirmation_contract_v1.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_final_artifact_server_v1"
    / "gefs_weekly_linear_final_artifact_2000_2019_v1.json"
)
DEFAULT_MEMBER_FILE = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_2024_six_cycle_extraction_server_v1"
    / "gefs_member_daily_precipitation_2024_six_cycle_v1.csv"
)
DEFAULT_STATION_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "candidate_station_files"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_2024_six_cycle_confirmation_v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-2024-six-cycle-confirmation-v1":
        raise ValueError("six-cycle confirmation contract id mismatch")
    if contract.get("candidate_id") != "weekly_two_stage_linear_site_factor_shrink_a075":
        raise ValueError("six-cycle confirmation candidate mismatch")
    if float(contract.get("factor_shrinkage_alpha")) != 0.75:
        raise ValueError("six-cycle confirmation alpha mismatch")
    if set(contract["decision_dates"]).intersection(
        contract["previously_scored_decision_dates"]
    ):
        raise ValueError("six-cycle decision dates overlap previous scored cycles")
    scope = contract["scope"]
    if any(
        scope[key]
        for key in (
            "artifact_refit_allowed",
            "candidate_reselection_allowed",
            "hyperparameter_tuning_allowed",
            "station_reselection_allowed",
            "reference_used_for_application_factor",
            "surrogate_training_allowed",
        )
    ):
        raise ValueError("six-cycle confirmation permits a forbidden operation")
    return contract


def load_artifact(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_sha256") != artifact_hash(artifact):
        raise ValueError("final artifact hash mismatch")
    if artifact.get("candidate_id") != contract["candidate_id"]:
        raise ValueError("final artifact candidate mismatch")
    if float(artifact.get("factor_shrinkage_alpha")) != float(
        contract["factor_shrinkage_alpha"]
    ):
        raise ValueError("final artifact alpha mismatch")
    if artifact.get("fit_years") != list(range(2000, 2020)):
        raise ValueError("final artifact fit years mismatch")
    if artifact.get("2024_used_for_fit_or_selection") is not False:
        raise ValueError("final artifact does not exclude 2024")
    return artifact


def reference_assignment_table(
    contract: dict[str, Any], reference: pd.DataFrame, station_dir: Path
) -> pd.DataFrame:
    rows = []
    for site_id, station_id in contract["ghcnd_station_assignments"].items():
        group = reference.loc[reference["site_id"].eq(site_id)]
        rows.append(
            {
                "site_id": site_id,
                "ghcnd_station_id": station_id,
                "valid_target_dates": int(group["reference_valid_unflagged"].sum()),
                "required_target_dates": int(len(group)),
                "station_file": str((station_dir / f"{station_id}.csv.gz").resolve()),
                "station_assignment_policy": contract["station_assignment_policy"],
                "station_selection_used_forecast_scores": False,
            }
        )
    return pd.DataFrame(rows).sort_values("site_id")


def promotion_gate(
    metric: dict[str, Any], occurrence: dict[str, Any], numeric: dict[str, Any]
) -> dict[str, bool]:
    requirements = {
        "daily_ensemble_mean_mae_not_worse": bool(
            metric["candidate_ensemble_mean_mae"] <= metric["raw_ensemble_mean_mae"]
        ),
        "daily_ensemble_mean_rmse_not_worse": bool(
            metric["candidate_ensemble_mean_rmse"] <= metric["raw_ensemble_mean_rmse"]
        ),
        "seven_day_mae_not_worse": bool(metric["seven_day_mae_not_worse"]),
        "crps_not_worse": bool(metric["crps_not_worse"]),
        "mean_brier_not_worse": bool(metric["mean_brier_not_worse"]),
        "heavy_coverage_not_both_worse": bool(
            metric["heavy_coverage_not_both_worse"]
        ),
        "occurrence_not_worse": bool(occurrence["occurrence_not_worse"]),
        "numeric_audit_passed": bool(
            numeric["negative_count"] == 0 and numeric["nonfinite_count"] == 0
        ),
    }
    return requirements


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    artifact = load_artifact(args.artifact, contract)
    member_contract = {
        "expected_decision_dates": contract["decision_dates"],
        "expected_sites": contract["expected_sites"],
        "expected_lead_days": contract["expected_lead_days"],
        "expected_member_count": contract["expected_member_count"],
    }
    raw = load_member_data([args.member_file], member_contract)
    corrected = apply_artifact(
        raw,
        artifact,
        expected_member_count=int(contract["expected_member_count"]),
    )
    dates = target_dates(
        {
            "expected_decision_dates": contract["decision_dates"],
        }
    )
    reference = station_reference(
        contract["ghcnd_station_assignments"],
        station_dir=args.station_dir,
        dates=dates,
    )
    if not reference["reference_valid_unflagged"].all():
        missing = reference.loc[
            ~reference["reference_valid_unflagged"], ["site_id", "local_date"]
        ]
        raise ValueError(f"six-cycle GHCN-D reference is incomplete: {missing.to_dict('records')}")
    corrected = corrected.merge(
        reference,
        on=["site_id", "local_date"],
        how="left",
        validate="many_to_one",
    )
    if corrected["precipitation_mm_reference"].isna().any():
        raise ValueError("six-cycle corrected rows have missing GHCN-D reference")
    members = tuple(sorted(corrected["gefs_member"].astype(str).unique()))
    metric, cycle_metrics = metrics_for_scope(
        corrected,
        scope="frozen_six_cycle_operational_ghcnd",
        members=members,
    )
    bootstrap = bootstrap_cycle_metrics(
        cycle_metrics,
        replicates=int(contract["bootstrap_replicates"]),
        seed=int(contract["bootstrap_seed"]),
    )
    bundle = metric_bundle(corrected, members)
    deterministic = _deterministic_metrics(bundle["observations"])
    occurrence = occurrence_row(contract["candidate_id"], corrected)
    cycle_factors = corrected.drop_duplicates(["site_id", "decision_date"])
    corrected_values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    factor_values = cycle_factors["weekly_linear_scaling_factor"].to_numpy(dtype=float)
    numeric = {
        "candidate_id": contract["candidate_id"],
        "negative_count": int((corrected_values < 0.0).sum()),
        "nonfinite_count": int((~np.isfinite(corrected_values)).sum()),
        "minimum_applied_factor": float(factor_values.min()),
        "maximum_applied_factor": float(factor_values.max()),
        "extreme_site_cycle_count": int(cycle_factors["weekly_extreme_regime"].sum()),
        "maximum_corrected_mm_day": float(corrected_values.max()),
    }
    requirements = promotion_gate(metric, occurrence, numeric)
    if list(requirements) != contract["promotion_gate"]:
        raise ValueError("implemented promotion gate differs from prelocked contract")
    passed = bool(all(requirements.values()))
    gate = {
        "contract_id": contract["contract_id"],
        "candidate_id": contract["candidate_id"],
        "decision_dates": contract["decision_dates"],
        "previously_scored_decision_dates": contract["previously_scored_decision_dates"],
        "decision_cycle_initializations_disjoint": True,
        "all_valid_dates_disjoint": False,
        "artifact_sha256": artifact["artifact_sha256"],
        "artifact_refit_performed": False,
        "candidate_reselection_performed": False,
        "hyperparameter_tuning_performed": False,
        "station_reselection_performed": False,
        "requirements": requirements,
        "all_requirements_passed": passed,
        "status": (
            "passed_frozen_2024_six_cycle_confirmation"
            if passed
            else "failed_frozen_2024_six_cycle_confirmation_retain_raw"
        ),
    }
    assignments = reference_assignment_table(contract, reference, args.station_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": args.output_dir / "frozen_weekly_linear_predictions_2024_six_cycle_v1.csv",
        "assignments": args.output_dir / "ghcnd_station_assignments_2024_six_cycle_v1.csv",
        "metrics": args.output_dir / "frozen_weekly_linear_metrics_2024_six_cycle_v1.json",
        "cycle_metrics": args.output_dir / "frozen_weekly_linear_cycle_metrics_2024_six_cycle_v1.csv",
        "bootstrap": args.output_dir / "frozen_weekly_linear_cycle_bootstrap_2024_six_cycle_v1.csv",
        "deterministic": args.output_dir / "frozen_weekly_linear_deterministic_2024_six_cycle_v1.csv",
        "probabilistic": args.output_dir / "frozen_weekly_linear_probabilistic_2024_six_cycle_v1.csv",
        "probability": args.output_dir / "frozen_weekly_linear_probability_2024_six_cycle_v1.csv",
        "seven_day": args.output_dir / "frozen_weekly_linear_seven_day_2024_six_cycle_v1.csv",
        "occurrence": args.output_dir / "frozen_weekly_linear_occurrence_2024_six_cycle_v1.json",
        "numeric": args.output_dir / "frozen_weekly_linear_numeric_2024_six_cycle_v1.json",
        "gate": args.output_dir / "frozen_weekly_linear_gate_2024_six_cycle_v1.json",
        "report": args.output_dir / "frozen_weekly_linear_conclusion_2024_six_cycle_v1.md",
    }
    corrected.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    assignments.to_csv(paths["assignments"], index=False, encoding="utf-8-sig")
    cycle_metrics.to_csv(paths["cycle_metrics"], index=False, encoding="utf-8-sig")
    bootstrap.to_csv(paths["bootstrap"], index=False, encoding="utf-8-sig")
    deterministic.to_csv(paths["deterministic"], index=False, encoding="utf-8-sig")
    bundle["probabilistic"].to_csv(paths["probabilistic"], index=False, encoding="utf-8-sig")
    bundle["probabilities"].to_csv(paths["probability"], index=False, encoding="utf-8-sig")
    bundle["seven_day"].to_csv(paths["seven_day"], index=False, encoding="utf-8-sig")
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
        "# GEFS 冻结周尺度订正 2024 六周期确认",
        "",
        "六个起报周期未进入此前订正评分；artifact、alpha 和 GHCN-D 站点均在提取前冻结。",
        "部分有效日期与旧周期重叠，因此这是新起报周期确认，不声称所有有效日期完全独立。",
        "",
        f"- 7 天 MAE 差值：`{metric['seven_day_mae_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- 日 MAE 差值：`{metric['candidate_ensemble_mean_mae'] - metric['raw_ensemble_mean_mae']:+.6f} mm`",
        f"- 日 RMSE 差值：`{metric['candidate_ensemble_mean_rmse'] - metric['raw_ensemble_mean_rmse']:+.6f} mm`",
        f"- CRPS 差值：`{metric['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier 差值：`{metric['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- 完整站点-周期：`{metric['complete_site_cycle_count']}`",
        f"- 强事件站点-周期：`{numeric['extreme_site_cycle_count']}`",
        f"- Gate：`{gate['status']}`",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--member-file", type=Path, default=DEFAULT_MEMBER_FILE)
    parser.add_argument("--station-dir", type=Path, default=DEFAULT_STATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

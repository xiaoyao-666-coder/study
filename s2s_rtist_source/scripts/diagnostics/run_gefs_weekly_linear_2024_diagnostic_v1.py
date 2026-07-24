#!/usr/bin/env python3
"""Apply the frozen weekly correction to the preselected 2024 five-cycle sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.diagnostics.fit_gefs_weekly_linear_final_artifact_v1 import artifact_hash
from scripts.diagnostics.run_gefs_qm_qdm_expanding_cv_2000_2018_v1 import occurrence_row
from scripts.diagnostics.run_gefs_qm_training_cv_v1 import _metric_row
from scripts.diagnostics.run_gefs_quantile_mapping_validation_v1 import (
    _deterministic_metrics,
    _probabilistic_metrics,
    _promotion_gate,
    _seven_day_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_2024_diagnostic_contract_v1.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_final_artifact_server_v1"
    / "gefs_weekly_linear_final_artifact_2000_2019_v1.json"
)
DEFAULT_ONE_CYCLE_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_member_gridmet_validation_received_20260716"
    / "gefs_31member_1cycle_5site_20260716_v1"
)
DEFAULT_FOUR_CYCLE_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_member_gridmet_validation_v1"
    / "gefs_31member_4cycle_5site_precip_20260716_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_2024_five_cycle_diagnostic_v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-2024-frozen-diagnostic-v1":
        raise ValueError("2024 frozen diagnostic contract id mismatch")
    scope = contract["scope"]
    forbidden = (
        scope["artifact_refit_allowed"],
        scope["candidate_reselection_allowed"],
        scope["hyperparameter_tuning_allowed"],
        scope["reference_used_for_application_factor"],
    )
    if any(forbidden):
        raise ValueError("2024 diagnostic contract permits a forbidden operation")
    return contract


def load_artifact(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("artifact_sha256") != artifact_hash(artifact):
        raise ValueError("frozen artifact hash mismatch")
    if artifact.get("artifact_contract_id") != contract["required_artifact_contract_id"]:
        raise ValueError("frozen artifact contract mismatch")
    if artifact.get("candidate_id") != contract["candidate_id"]:
        raise ValueError("frozen artifact candidate mismatch")
    if artifact.get("fit_years") != contract["required_artifact_fit_years"]:
        raise ValueError("frozen artifact fit years mismatch")
    if float(artifact.get("factor_shrinkage_alpha")) != float(
        contract["required_factor_shrinkage_alpha"]
    ):
        raise ValueError("frozen artifact alpha mismatch")
    if artifact.get("2024_used_for_fit_or_selection") is not False:
        raise ValueError("artifact does not prove 2024 exclusion")
    return artifact


def load_member_data(paths: Sequence[Path], contract: dict[str, Any]) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"site", "decision_date", "local_date", "lead_day", "gefs_member", "precipitation_mm"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"2024 member file missing columns: {sorted(missing)}")
        parts.append(
            frame[list(required)].rename(
                columns={"site": "site_id", "precipitation_mm": "precipitation_mm_raw"}
            )
        )
    data = pd.concat(parts, ignore_index=True)
    data["decision_date"] = pd.to_datetime(data["decision_date"])
    data["local_date"] = pd.to_datetime(data["local_date"])
    data["valid_date_utc"] = data["local_date"]
    expected_dates = set(pd.to_datetime(contract["expected_decision_dates"]))
    if set(data["decision_date"].unique()) != expected_dates:
        raise ValueError("2024 decision date set mismatch")
    if set(data["site_id"].astype(str)) != set(contract["expected_sites"]):
        raise ValueError("2024 site set mismatch")
    if set(data["lead_day"].astype(int)) != set(contract["expected_lead_days"]):
        raise ValueError("2024 lead-day set mismatch")
    members = sorted(data["gefs_member"].astype(str).unique())
    if len(members) != int(contract["expected_member_count"]):
        raise ValueError("2024 member count mismatch")
    key = ["site_id", "decision_date", "local_date", "gefs_member"]
    if data.duplicated(key).any():
        raise ValueError("duplicate 2024 member key")
    expected_rows = len(expected_dates) * len(contract["expected_sites"]) * len(members) * 7
    if len(data) != expected_rows:
        raise ValueError(f"2024 member rows={len(data)}, expected={expected_rows}")
    values = data["precipitation_mm_raw"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("2024 raw precipitation contains invalid values")
    return data.sort_values(key).reset_index(drop=True)


def load_reference_data(paths: Sequence[Path], contract: dict[str, Any]) -> pd.DataFrame:
    parts = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame.loc[frame["variable"].eq("precipitation_mm")].copy()
        parts.append(
            frame[["site", "local_date", "reference_value"]].rename(
                columns={
                    "site": "site_id",
                    "reference_value": "precipitation_mm_reference",
                }
            )
        )
    reference = pd.concat(parts, ignore_index=True)
    reference["local_date"] = pd.to_datetime(reference["local_date"])
    key = ["site_id", "local_date"]
    if reference.duplicated(key).any():
        raise ValueError("duplicate 2024 reference key")
    expected_rows = len(contract["expected_decision_dates"]) * len(contract["expected_sites"]) * 7
    if len(reference) != expected_rows:
        raise ValueError(f"2024 reference rows={len(reference)}, expected={expected_rows}")
    return reference


def cycle_ensemble_mean_totals(
    frame: pd.DataFrame, *, expected_member_count: int
) -> pd.DataFrame:
    keys = ["site_id", "decision_date"]
    member_keys = keys + ["gefs_member"]
    counts = frame.groupby(member_keys)["local_date"].nunique()
    if not counts.eq(7).all():
        raise ValueError("2024 cycle contains an incomplete member")
    member_totals = (
        frame.groupby(member_keys, as_index=False)["precipitation_mm_raw"]
        .sum()
        .rename(columns={"precipitation_mm_raw": "member_raw_7d_mm"})
    )
    member_counts = member_totals.groupby(keys)["gefs_member"].nunique()
    if not member_counts.eq(expected_member_count).all():
        raise ValueError("2024 cycle does not contain the expected ensemble")
    return (
        member_totals.groupby(keys, as_index=False)["member_raw_7d_mm"]
        .mean()
        .rename(columns={"member_raw_7d_mm": "ensemble_mean_raw_7d_mm"})
    )


def apply_artifact(
    frame: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    expected_member_count: int,
) -> pd.DataFrame:
    cycles = cycle_ensemble_mean_totals(
        frame, expected_member_count=expected_member_count
    )
    groups = pd.DataFrame(artifact["groups"])
    cycles = cycles.merge(groups, on="site_id", how="left", validate="many_to_one")
    if cycles["raw_ensemble_mean_7d_q90_mm"].isna().any():
        raise ValueError("2024 site has no frozen factor group")
    cycles["weekly_extreme_regime"] = cycles["ensemble_mean_raw_7d_mm"].gt(
        cycles["raw_ensemble_mean_7d_q90_mm"]
    )
    cycles["weekly_linear_scaling_factor"] = np.where(
        cycles["weekly_extreme_regime"],
        cycles["effective_extreme_factor"],
        cycles["effective_overall_factor"],
    )
    corrected = frame.merge(
        cycles[
            [
                "site_id",
                "decision_date",
                "ensemble_mean_raw_7d_mm",
                "raw_ensemble_mean_7d_q90_mm",
                "weekly_extreme_regime",
                "weekly_linear_scaling_factor",
            ]
        ],
        on=["site_id", "decision_date"],
        how="left",
        validate="many_to_one",
    )
    corrected["precipitation_mm_qm"] = (
        corrected["precipitation_mm_raw"]
        * corrected["weekly_linear_scaling_factor"]
    )
    corrected["candidate_id"] = artifact["candidate_id"]
    corrected["factor_shrinkage_alpha"] = artifact["factor_shrinkage_alpha"]
    corrected["artifact_sha256"] = artifact["artifact_sha256"]
    corrected["qm_extrapolated_upper"] = False
    values = corrected["precipitation_mm_qm"].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("frozen correction produced invalid precipitation")
    return corrected


def metric_bundle(corrected: pd.DataFrame, members: Sequence[str]) -> dict[str, Any]:
    observations, probabilistic, probabilities = _probabilistic_metrics(
        corrected, members=members
    )
    seven_day = _seven_day_metrics(corrected)
    gate = _promotion_gate(
        observations=observations,
        probabilities=probabilities,
        seven_day=seven_day,
        paired=corrected,
    )
    return {
        "observations": observations,
        "probabilistic": probabilistic,
        "probabilities": probabilities,
        "seven_day": seven_day,
        "gate": gate,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    artifact = load_artifact(args.artifact, contract)
    member_paths = [args.one_cycle_dir / "gefs_member_daily_weather.csv", args.four_cycle_dir / "gefs_member_daily_weather.csv"]
    reference_paths = [args.one_cycle_dir / "gridmet_reference_daily_long.csv", args.four_cycle_dir / "gridmet_reference_daily_long.csv"]
    raw = load_member_data(member_paths, contract)
    reference = load_reference_data(reference_paths, contract)
    corrected = apply_artifact(
        raw,
        artifact,
        expected_member_count=int(contract["expected_member_count"]),
    ).merge(reference, on=["site_id", "local_date"], how="left", validate="many_to_one")
    if corrected["precipitation_mm_reference"].isna().any():
        raise ValueError("2024 diagnostic has missing gridMET references")
    members = tuple(sorted(corrected["gefs_member"].astype(str).unique()))
    bundle = metric_bundle(corrected, members)
    metric = _metric_row(artifact["candidate_id"], 2024, bundle)
    metric["complete_site_cycle_count"] = int(
        corrected[["site_id", "decision_date"]].drop_duplicates().shape[0]
    )
    occurrence = occurrence_row(artifact["candidate_id"], corrected)
    deterministic = _deterministic_metrics(bundle["observations"])
    factor_cycles = corrected.drop_duplicates(["site_id", "decision_date"])
    factors = factor_cycles["weekly_linear_scaling_factor"].to_numpy(dtype=float)
    numeric = {
        "candidate_id": artifact["candidate_id"],
        "negative_count": int((corrected["precipitation_mm_qm"] < 0.0).sum()),
        "nonfinite_count": int((~np.isfinite(corrected["precipitation_mm_qm"])).sum()),
        "minimum_applied_factor": float(factors.min()),
        "maximum_applied_factor": float(factors.max()),
        "extreme_site_cycle_count": int(factor_cycles["weekly_extreme_regime"].sum()),
        "maximum_corrected_mm_day": float(corrected["precipitation_mm_qm"].max()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": args.output_dir / "weekly_linear_frozen_predictions_2024_five_cycle_v1.csv",
        "metrics": args.output_dir / "weekly_linear_frozen_metrics_2024_five_cycle_v1.json",
        "probabilistic": args.output_dir / "weekly_linear_frozen_probabilistic_metrics_2024_v1.csv",
        "probability": args.output_dir / "weekly_linear_frozen_probability_metrics_2024_v1.csv",
        "deterministic": args.output_dir / "weekly_linear_frozen_deterministic_metrics_2024_v1.csv",
        "seven_day": args.output_dir / "weekly_linear_frozen_seven_day_metrics_2024_v1.csv",
        "occurrence": args.output_dir / "weekly_linear_frozen_occurrence_2024_v1.json",
        "numeric": args.output_dir / "weekly_linear_frozen_numeric_audit_2024_v1.json",
        "manifest": args.output_dir / "weekly_linear_frozen_diagnostic_manifest_2024_v1.json",
        "report": args.output_dir / "weekly_linear_frozen_conclusion_2024_five_cycle_v1.md",
    }
    corrected.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    bundle["probabilistic"].to_csv(paths["probabilistic"], index=False, encoding="utf-8-sig")
    bundle["probabilities"].to_csv(paths["probability"], index=False, encoding="utf-8-sig")
    deterministic.to_csv(paths["deterministic"], index=False, encoding="utf-8-sig")
    bundle["seven_day"].to_csv(paths["seven_day"], index=False, encoding="utf-8-sig")
    for payload, key in ((metric, "metrics"), (occurrence, "occurrence"), (numeric, "numeric")):
        paths[key].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "contract_id": contract["contract_id"],
        "candidate_id": artifact["candidate_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "artifact_fit_years": artifact["fit_years"],
        "artifact_refit_performed": False,
        "candidate_reselection_performed": False,
        "hyperparameter_tuning_performed": False,
        "2024_reference_used_for_application_factor": False,
        "decision_dates": contract["expected_decision_dates"],
        "member_count": len(members),
        "reference_product": contract["reference_product"],
        "evidence_role": contract["evidence_role"],
        "independent_final_test_claim_allowed": False,
        "retuning_after_result_allowed": False,
        "status": "post_freeze_diagnostic_completed_no_retuning_allowed",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS 冻结周尺度线性订正 2024 五周期诊断",
        "",
        "该候选在读取 2024 数据前已经冻结；本次没有重拟合、重选候选或调 alpha。",
        "现有五周期曾用于偏差诊断，参考为 gridMET 而非训练使用的 GHCN-D，",
        "所以这里只报告外部诊断结果，不声称是完全独立最终测试，也不得根据结果继续调参。",
        "",
        f"- 7 天 MAE 差值：`{metric['seven_day_mae_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- 日 MAE 差值：`{metric['candidate_ensemble_mean_mae'] - metric['raw_ensemble_mean_mae']:+.6f} mm`",
        f"- 日 RMSE 差值：`{metric['candidate_ensemble_mean_rmse'] - metric['raw_ensemble_mean_rmse']:+.6f} mm`",
        f"- CRPS 差值：`{metric['crps_difference_candidate_minus_raw_mm']:+.6f} mm`",
        f"- Brier 差值：`{metric['mean_brier_difference_candidate_minus_raw']:+.6f}`",
        f"- P10-P90 覆盖率：raw `{metric['raw_p10_p90_coverage']:.6f}`，订正 `{metric['candidate_p10_p90_coverage']:.6f}`",
        f"- min-max 覆盖率：raw `{metric['raw_min_max_coverage']:.6f}`，订正 `{metric['candidate_min_max_coverage']:.6f}`",
        f"- 站点-周期数：`{metric['complete_site_cycle_count']}`",
        f"- 有效因子范围：`{numeric['minimum_applied_factor']:.6f}–{numeric['maximum_applied_factor']:.6f}`",
        f"- 强事件站点-周期数：`{numeric['extreme_site_cycle_count']}`",
        "- 结论状态：`post_freeze_diagnostic_completed_no_retuning_allowed`",
    ]
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--one-cycle-dir", type=Path, default=DEFAULT_ONE_CYCLE_DIR)
    parser.add_argument("--four-cycle-dir", type=Path, default=DEFAULT_FOUR_CYCLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

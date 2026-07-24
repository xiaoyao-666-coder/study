#!/usr/bin/env python3
"""Audit the frozen 2024 correction against GHCN-D station references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.diagnostics.audit_ghcnd_precipitation_candidates_v1 import (
    read_station_prcp,
    sha256_file,
)
from scripts.diagnostics.run_gefs_qdm_2019_station_reference_v1 import (
    complete_site_cycle_rows,
)
from scripts.diagnostics.run_gefs_qm_training_cv_v1 import _metric_row
from scripts.diagnostics.run_gefs_weekly_linear_2024_diagnostic_v1 import (
    metric_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_weekly_linear_2024_ghcnd_reference_audit_contract_v1.json"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_weekly_linear_2024_five_cycle_diagnostic_server_v1"
    / "weekly_linear_frozen_predictions_2024_five_cycle_v1.csv"
)
DEFAULT_SELECTION = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "station_quality_audit_final"
    / "ghcnd_primary_station_selection_v1.json"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "station_inventory"
    / "ghcnd_precipitation_station_candidates_v1.csv"
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
    / "gefs_weekly_linear_2024_ghcnd_reference_audit_v1"
)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-weekly-linear-2024-ghcnd-reference-audit-v1":
        raise ValueError("2024 GHCN-D reference audit contract id mismatch")
    if contract.get("candidate_id") != "weekly_two_stage_linear_site_factor_shrink_a075":
        raise ValueError("2024 GHCN-D audit candidate mismatch")
    if int(contract.get("station_record_date_offset_days")) != 0:
        raise ValueError("2024 GHCN-D audit date offset must remain zero")
    if set(contract["primary_station_assignments"]) != set(contract["expected_sites"]):
        raise ValueError("primary station assignment sites mismatch")
    if set(contract["complete_fallback_station_assignments"]) != set(
        contract["expected_sites"]
    ):
        raise ValueError("fallback station assignment sites mismatch")
    if any(contract["scope"].values()):
        raise ValueError("2024 GHCN-D audit contract permits a forbidden operation")
    return contract


def target_dates(contract: dict[str, Any]) -> pd.DatetimeIndex:
    dates = []
    for cycle in pd.to_datetime(contract["expected_decision_dates"]):
        dates.extend(pd.date_range(cycle, periods=7))
    result = pd.DatetimeIndex(sorted(set(dates)))
    expected = len(contract["expected_decision_dates"]) * 7
    if len(result) != expected:
        raise ValueError("2024 audit cycle dates overlap unexpectedly")
    return result


def load_predictions(path: Path, contract: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("decision_date", "local_date", "valid_date_utc"):
        frame[column] = pd.to_datetime(frame[column])
    if set(frame["candidate_id"].astype(str)) != {contract["candidate_id"]}:
        raise ValueError("frozen prediction candidate mismatch")
    if frame["artifact_sha256"].astype(str).nunique() != 1:
        raise ValueError("frozen predictions contain multiple artifact hashes")
    if set(frame["site_id"].astype(str)) != set(contract["expected_sites"]):
        raise ValueError("frozen prediction site set mismatch")
    if set(frame["decision_date"].dt.strftime("%Y-%m-%d")) != set(
        contract["expected_decision_dates"]
    ):
        raise ValueError("frozen prediction decision dates mismatch")
    members = frame["gefs_member"].astype(str).unique()
    if len(members) != int(contract["expected_member_count"]):
        raise ValueError("frozen prediction member count mismatch")
    key = ["site_id", "decision_date", "local_date", "gefs_member"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate frozen prediction key")
    expected_rows = (
        len(contract["expected_sites"])
        * len(contract["expected_decision_dates"])
        * int(contract["expected_member_count"])
        * 7
    )
    if len(frame) != expected_rows:
        raise ValueError(f"frozen prediction rows={len(frame)}, expected={expected_rows}")
    return frame


def validate_primary_assignments(
    selection_path: Path, contract: dict[str, Any]
) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    observed = {
        site: str(details["station_id"])
        for site, details in selection["sites"].items()
    }
    if observed != contract["primary_station_assignments"]:
        raise ValueError("contract primary stations differ from frozen selection")


def station_reference(
    assignments: Mapping[str, str],
    *,
    station_dir: Path,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    parts = []
    for site_id, station_id in sorted(assignments.items()):
        path = station_dir / f"{station_id}.csv.gz"
        station = read_station_prcp(path).set_index("date").reindex(dates)
        valid = (
            station["precipitation_mm"].notna()
            & station["q_flag"].fillna("").str.strip().eq("")
        )
        part = pd.DataFrame(
            {
                "site_id": site_id,
                "local_date": dates,
                "ghcnd_station_id": station_id,
                "precipitation_mm_reference": station["precipitation_mm"].where(valid).to_numpy(),
                "reference_valid_unflagged": valid.to_numpy(dtype=bool),
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def validate_fallback_coverage(
    reference: pd.DataFrame, contract: dict[str, Any]
) -> None:
    counts = reference.groupby("site_id")["reference_valid_unflagged"].sum()
    expected = len(target_dates(contract))
    if not counts.eq(expected).all():
        raise ValueError(f"fallback reference is not complete: {counts.to_dict()}")


def pair_complete_cycles(
    predictions: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    paired = predictions.drop(
        columns=["precipitation_mm_reference"], errors="ignore"
    ).merge(
        reference,
        on=["site_id", "local_date"],
        how="left",
        validate="many_to_one",
    )
    paired.loc[
        ~paired["reference_valid_unflagged"].fillna(False),
        "precipitation_mm_reference",
    ] = np.nan
    return complete_site_cycle_rows(paired)


def metrics_for_scope(
    frame: pd.DataFrame,
    *,
    scope: str,
    members: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    bundle = metric_bundle(frame, members)
    metric = _metric_row(str(frame["candidate_id"].iloc[0]), 2024, bundle)
    metric["reference_scope"] = scope
    metric["complete_site_cycle_count"] = int(
        frame[["site_id", "decision_date"]].drop_duplicates().shape[0]
    )
    cycle_rows = []
    for decision_date, group in frame.groupby("decision_date", sort=True):
        cycle_bundle = metric_bundle(group, members)
        row = _metric_row(str(group["candidate_id"].iloc[0]), 2024, cycle_bundle)
        cycle_rows.append(
            {
                "reference_scope": scope,
                "decision_date": pd.Timestamp(decision_date).strftime("%Y-%m-%d"),
                "complete_site_count": int(group["site_id"].nunique()),
                "seven_day_mae_difference_candidate_minus_raw_mm": row[
                    "seven_day_mae_difference_candidate_minus_raw_mm"
                ],
                "crps_difference_candidate_minus_raw_mm": row[
                    "crps_difference_candidate_minus_raw_mm"
                ],
                "mean_brier_difference_candidate_minus_raw": row[
                    "mean_brier_difference_candidate_minus_raw"
                ],
            }
        )
    return metric, pd.DataFrame(cycle_rows)


def bootstrap_cycle_metrics(
    cycle_metrics: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    metric_columns = {
        "seven_day_mae": "seven_day_mae_difference_candidate_minus_raw_mm",
        "crps": "crps_difference_candidate_minus_raw_mm",
        "mean_brier": "mean_brier_difference_candidate_minus_raw",
    }
    rows = []
    for scope, group in cycle_metrics.groupby("reference_scope", sort=True):
        group = group.sort_values("decision_date")
        for offset, (metric, column) in enumerate(metric_columns.items()):
            values = group[column].to_numpy(dtype=float)
            rng = np.random.default_rng(int(seed) + offset)
            indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
            means = values[indices].mean(axis=1)
            rows.append(
                {
                    "reference_scope": scope,
                    "metric": metric,
                    "cycle_count": int(len(values)),
                    "point_difference_candidate_minus_raw": float(values.mean()),
                    "bootstrap_ci_lower_95": float(np.quantile(means, 0.025)),
                    "bootstrap_ci_upper_95": float(np.quantile(means, 0.975)),
                    "bootstrap_replicates": int(replicates),
                    "random_seed": int(seed) + offset,
                    "fraction_bootstrap_improved": float((means <= 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def assignment_audit(
    contract: dict[str, Any],
    *,
    candidates_path: Path,
    primary_reference: pd.DataFrame,
    fallback_reference: pd.DataFrame,
) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path)
    rows = []
    for scope, assignments, reference in (
        ("primary_station_available_cycles", contract["primary_station_assignments"], primary_reference),
        ("complete_fallback_station_all_sites", contract["complete_fallback_station_assignments"], fallback_reference),
    ):
        for site_id, station_id in assignments.items():
            meta = candidates.loc[
                candidates["project_site_id"].eq(site_id)
                & candidates["station_id"].eq(station_id)
            ]
            if len(meta) != 1:
                raise ValueError(f"missing station metadata for {site_id} {station_id}")
            item = meta.iloc[0]
            site_reference = reference.loc[reference["site_id"].eq(site_id)]
            rows.append(
                {
                    "reference_scope": scope,
                    "site_id": site_id,
                    "ghcnd_station_id": station_id,
                    "station_name": item["station_name"],
                    "candidate_rank_by_distance": int(item["candidate_rank_by_distance"]),
                    "distance_km": float(item["distance_km"]),
                    "same_as_2000_2019_primary": bool(
                        station_id == contract["primary_station_assignments"][site_id]
                    ),
                    "valid_target_dates": int(site_reference["reference_valid_unflagged"].sum()),
                    "target_date_count": int(len(site_reference)),
                    "selection_used_forecast_scores": False,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    validate_primary_assignments(args.selection, contract)
    predictions = load_predictions(args.predictions, contract)
    dates = target_dates(contract)
    primary_reference = station_reference(
        contract["primary_station_assignments"],
        station_dir=args.station_dir,
        dates=dates,
    )
    fallback_reference = station_reference(
        contract["complete_fallback_station_assignments"],
        station_dir=args.station_dir,
        dates=dates,
    )
    validate_fallback_coverage(fallback_reference, contract)
    primary = pair_complete_cycles(predictions, primary_reference)
    fallback = pair_complete_cycles(predictions, fallback_reference)
    if fallback[["site_id", "decision_date"]].drop_duplicates().shape[0] != 25:
        raise ValueError("fallback audit does not contain all 25 site-cycles")
    members = tuple(sorted(predictions["gefs_member"].astype(str).unique()))
    primary_metric, primary_cycles = metrics_for_scope(
        primary, scope="primary_station_available_cycles", members=members
    )
    fallback_metric, fallback_cycles = metrics_for_scope(
        fallback, scope="complete_fallback_station_all_sites", members=members
    )
    metrics = [primary_metric, fallback_metric]
    cycle_metrics = pd.concat([primary_cycles, fallback_cycles], ignore_index=True)
    bootstrap = bootstrap_cycle_metrics(
        cycle_metrics,
        replicates=int(contract["bootstrap_replicates"]),
        seed=int(contract["bootstrap_seed"]),
    )
    assignments = assignment_audit(
        contract,
        candidates_path=args.candidates,
        primary_reference=primary_reference,
        fallback_reference=fallback_reference,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "primary_paired": args.output_dir / "ghcnd_primary_station_available_predictions_2024_v1.csv",
        "fallback_paired": args.output_dir / "ghcnd_complete_fallback_predictions_2024_v1.csv",
        "assignments": args.output_dir / "ghcnd_reference_station_assignments_2024_v1.csv",
        "cycle_metrics": args.output_dir / "ghcnd_reference_cycle_metrics_2024_v1.csv",
        "bootstrap": args.output_dir / "ghcnd_reference_cycle_block_bootstrap_2024_v1.csv",
        "metrics": args.output_dir / "ghcnd_reference_metrics_2024_v1.json",
        "manifest": args.output_dir / "ghcnd_reference_audit_manifest_2024_v1.json",
        "report": args.output_dir / "ghcnd_reference_audit_conclusion_2024_v1.md",
    }
    primary.to_csv(paths["primary_paired"], index=False, encoding="utf-8-sig")
    fallback.to_csv(paths["fallback_paired"], index=False, encoding="utf-8-sig")
    assignments.to_csv(paths["assignments"], index=False, encoding="utf-8-sig")
    cycle_metrics.to_csv(paths["cycle_metrics"], index=False, encoding="utf-8-sig")
    bootstrap.to_csv(paths["bootstrap"], index=False, encoding="utf-8-sig")
    paths["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract_id": contract["contract_id"],
        "candidate_id": contract["candidate_id"],
        "artifact_sha256": str(predictions["artifact_sha256"].iloc[0]),
        "prediction_file_sha256": sha256_file(args.predictions),
        "selection_file_sha256": sha256_file(args.selection),
        "primary_complete_site_cycle_count": int(
            primary[["site_id", "decision_date"]].drop_duplicates().shape[0]
        ),
        "fallback_complete_site_cycle_count": 25,
        "artifact_refit_performed": False,
        "candidate_reselection_performed": False,
        "hyperparameter_tuning_performed": False,
        "station_selection_used_forecast_scores": False,
        "independent_final_test_claim_allowed": False,
        "retuning_after_result_allowed": False,
        "evidence_role": contract["evidence_role"],
        "status": "post_freeze_ghcnd_reference_audit_completed_no_retuning_allowed",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_index = {item["reference_scope"]: item for item in metrics}
    report = [
        "# GEFS 冻结周尺度订正 2024 GHCN-D 参考一致性审计",
        "",
        "订正 artifact、候选和 alpha 均未改变。替代站只按记录完整性与距离选择，未查看预报误差。",
        "这些日期已经被探索，因此两种口径都不能声称是完全独立最终测试，也不得据此调参。",
        "",
        "| 参考口径 | 完整站点-周期 | 7天MAE差值 | 日RMSE差值 | CRPS差值 | Brier差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scope in ("primary_station_available_cycles", "complete_fallback_station_all_sites"):
        item = metric_index[scope]
        report.append(
            f"| `{scope}` | {item['complete_site_cycle_count']} | "
            f"{item['seven_day_mae_difference_candidate_minus_raw_mm']:+.6f} | "
            f"{item['candidate_ensemble_mean_rmse'] - item['raw_ensemble_mean_rmse']:+.6f} | "
            f"{item['crps_difference_candidate_minus_raw_mm']:+.6f} | "
            f"{item['mean_brier_difference_candidate_minus_raw']:+.6f} |"
        )
    report.extend(
        [
            "",
            "Bootstrap 结果见 `ghcnd_reference_cycle_block_bootstrap_2024_v1.csv`。",
            "结论状态：`post_freeze_ghcnd_reference_audit_completed_no_retuning_allowed`。",
        ]
    )
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--station-dir", type=Path, default=DEFAULT_STATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit high scaling factors in causal current-cycle volume-preserving QDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_scale_stability_audit_contract_v1.json"
)
OOF_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_causal_current_cycle_cv_server_v1"
)
VALIDATION_ROOT = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_causal_current_cycle_2019_validation_server_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "site_general_surrogate_eval"
    / "gefs_qdm_station_reference_v1"
    / "gefs_qdm_scale_stability_audit_v1"
)
CANDIDATE_ID = "qdm_global_current_cycle_7d_volume_preserving"


def classify_scale(scale: float, fallback: bool) -> str:
    if fallback or not np.isfinite(scale):
        return "fallback_raw"
    if scale <= 2.0:
        return "le_2"
    if scale <= 5.0:
        return "gt_2_le_5"
    if scale <= 10.0:
        return "gt_5_le_10"
    if scale <= 20.0:
        return "gt_10_le_20"
    return "gt_20"


def as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("contract_id") != "gefs-qdm-scale-stability-audit-v1":
        raise ValueError("scale stability contract id mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("scale stability candidate mismatch")
    scope = contract["scope"]
    if scope["modify_predictions_allowed"] or scope["refit_qdm_allowed"]:
        raise ValueError("scale audit cannot modify predictions or refit QDM")
    if scope["use_2024_allowed"]:
        raise ValueError("2024 must be prohibited")
    return contract


def group_diagnostics(
    predictions_path: Path,
    audit_path: Path,
    *,
    split: str,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    audit = pd.read_csv(audit_path)
    for frame in (predictions, audit):
        frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    audit["fallback_to_raw"] = audit["fallback_to_raw"].map(as_bool)
    keys = ["site_id", "decision_date", "gefs_member"]
    if predictions.duplicated(keys + ["valid_date_utc"]).any():
        raise ValueError(f"duplicate prediction rows in {split}")
    if audit.duplicated(keys).any():
        raise ValueError(f"duplicate member audit rows in {split}")
    required = {
        "precipitation_mm_raw",
        "precipitation_mm_qm",
        "precipitation_mm_qdm_base",
        "precipitation_mm_reference",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"missing prediction columns in {split}: {sorted(missing)}")
    rows = []
    audit_index = audit.set_index(keys)
    for key, group in predictions.groupby(keys, sort=True):
        if len(group) != 7:
            raise ValueError(f"{split} group {key} has {len(group)} rows, expected 7")
        if key not in audit_index.index:
            raise ValueError(f"missing member audit for {split} group {key}")
        member = audit_index.loc[key]
        raw = group["precipitation_mm_raw"].to_numpy(dtype=float)
        base = group["precipitation_mm_qdm_base"].to_numpy(dtype=float)
        corrected = group["precipitation_mm_qm"].to_numpy(dtype=float)
        reference = group["precipitation_mm_reference"].to_numpy(dtype=float)
        scale = float(member["scale_factor"])
        fallback = bool(member["fallback_to_raw"])
        raw_mae = float(np.mean(np.abs(raw - reference)))
        corrected_mae = float(np.mean(np.abs(corrected - reference)))
        rows.append(
            {
                "split": split,
                "site_id": key[0],
                "decision_date": key[1],
                "gefs_member": key[2],
                "scale_bin": classify_scale(scale, fallback),
                "scale_factor": scale if np.isfinite(scale) else None,
                "fallback_to_raw": fallback,
                "raw_total_mm": float(member["raw_total_mm"]),
                "qdm_base_total_mm": float(member["qdm_total_mm"]),
                "reference_total_mm": float(reference.sum()),
                "raw_daily_mae_mm": raw_mae,
                "corrected_daily_mae_mm": corrected_mae,
                "daily_mae_change_corrected_minus_raw_mm": corrected_mae - raw_mae,
                "daily_mae_improved": bool(corrected_mae <= raw_mae),
                "maximum_raw_mm_day": float(raw.max()),
                "maximum_qdm_base_mm_day": float(base.max()),
                "maximum_corrected_mm_day": float(corrected.max()),
                "maximum_reference_mm_day": float(reference.max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_bins(groups: pd.DataFrame) -> pd.DataFrame:
    order = [
        "fallback_raw",
        "le_2",
        "gt_2_le_5",
        "gt_5_le_10",
        "gt_10_le_20",
        "gt_20",
    ]
    rows = []
    for split, split_frame in groups.groupby("split", sort=False):
        total = len(split_frame)
        for scale_bin in order:
            frame = split_frame.loc[split_frame["scale_bin"].eq(scale_bin)]
            rows.append(
                {
                    "split": split,
                    "scale_bin": scale_bin,
                    "member_group_count": int(len(frame)),
                    "member_group_fraction": float(len(frame) / total),
                    "mean_scale_factor": (
                        float(frame["scale_factor"].mean()) if len(frame) else None
                    ),
                    "mean_daily_mae_change_corrected_minus_raw_mm": (
                        float(
                            frame[
                                "daily_mae_change_corrected_minus_raw_mm"
                            ].mean()
                        )
                        if len(frame)
                        else None
                    ),
                    "fraction_daily_mae_not_worse": (
                        float(frame["daily_mae_improved"].mean())
                        if len(frame)
                        else None
                    ),
                    "maximum_corrected_mm_day": (
                        float(frame["maximum_corrected_mm_day"].max())
                        if len(frame)
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def threshold_summary(groups: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for split, frame in groups.groupby("split", sort=False):
        nonfallback = frame.loc[~frame["fallback_to_raw"]].copy()
        for threshold in thresholds:
            high = nonfallback.loc[nonfallback["scale_factor"].gt(threshold)]
            rows.append(
                {
                    "split": split,
                    "scale_threshold": threshold,
                    "high_scale_group_count": int(len(high)),
                    "high_scale_fraction_of_all_groups": float(len(high) / len(frame)),
                    "mean_daily_mae_change_corrected_minus_raw_mm": (
                        float(
                            high["daily_mae_change_corrected_minus_raw_mm"].mean()
                        )
                        if len(high)
                        else None
                    ),
                    "fraction_daily_mae_not_worse": (
                        float(high["daily_mae_improved"].mean())
                        if len(high)
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Path]:
    contract = load_contract(args.contract)
    oof = group_diagnostics(
        args.oof_predictions,
        args.oof_member_audit,
        split="training_oof_2015_2018",
    )
    validation = group_diagnostics(
        args.validation_predictions,
        args.validation_member_audit,
        split="exploratory_2019",
    )
    groups = pd.concat([oof, validation], ignore_index=True)
    bins = summarize_bins(groups)
    thresholds = threshold_summary(
        groups,
        [float(value) for value in contract["high_scale_thresholds"]],
    )
    top_count = int(contract["top_event_count_per_split"])
    top_events = (
        groups.loc[~groups["fallback_to_raw"]]
        .sort_values(["split", "scale_factor"], ascending=[True, False])
        .groupby("split", sort=False)
        .head(top_count)
        .reset_index(drop=True)
    )
    split_rows = []
    for split, frame in groups.groupby("split", sort=False):
        split_rows.append(
            {
                "split": split,
                "member_group_count": int(len(frame)),
                "fallback_to_raw_count": int(frame["fallback_to_raw"].sum()),
                "maximum_scale_factor": float(frame["scale_factor"].max()),
                "p99_scale_factor": float(frame["scale_factor"].quantile(0.99)),
                "mean_daily_mae_change_corrected_minus_raw_mm": float(
                    frame["daily_mae_change_corrected_minus_raw_mm"].mean()
                ),
                "fraction_daily_mae_not_worse": float(
                    frame["daily_mae_improved"].mean()
                ),
                "maximum_corrected_mm_day": float(
                    frame["maximum_corrected_mm_day"].max()
                ),
            }
        )
    summary = {
        "contract_id": contract["contract_id"],
        "candidate_id": CANDIDATE_ID,
        "2019_used_for_refit_or_selection": False,
        "2024_used": False,
        "decision_policy": contract["decision_policy"],
        "split_summary": split_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "groups": args.output_dir / "scale_stability_member_group_diagnostics_v1.csv",
        "bins": args.output_dir / "scale_stability_bin_summary_v1.csv",
        "thresholds": args.output_dir / "scale_stability_threshold_summary_v1.csv",
        "top_events": args.output_dir / "scale_stability_top_events_v1.csv",
        "summary": args.output_dir / "scale_stability_summary_v1.json",
        "report": args.output_dir / "scale_stability_conclusion_v1.md",
    }
    for frame, key in (
        (groups, "groups"),
        (bins, "bins"),
        (thresholds, "thresholds"),
        (top_events, "top_events"),
    ):
        frame.to_csv(paths[key], index=False, encoding="utf-8-sig")
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# GEFS QDM 水量缩放稳定性审计",
        "",
        "本审计不修改预测、不重新拟合、不使用 2024，也不新增事后晋级 gate。",
        "",
        "| 数据 | 成员组 | raw回退 | 最大缩放 | P99缩放 | 日MAE差值 | 日MAE不劣比例 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        report.append(
            f"| `{row['split']}` | {row['member_group_count']} | "
            f"{row['fallback_to_raw_count']} | {row['maximum_scale_factor']:.4f} | "
            f"{row['p99_scale_factor']:.4f} | "
            f"{row['mean_daily_mae_change_corrected_minus_raw_mm']:+.4f} | "
            f"{row['fraction_daily_mae_not_worse']:.4f} |"
        )
    report.extend(["", "高倍缩放分组和具体事件见配套 CSV。"])
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=OOF_ROOT / "causal_current_cycle_oof_predictions_v1.csv",
    )
    parser.add_argument(
        "--oof-member-audit",
        type=Path,
        default=OOF_ROOT / "causal_current_cycle_member_audit_v1.csv",
    )
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=VALIDATION_ROOT / "causal_qdm_complete_cycle_evaluation_2019_v1.csv",
    )
    parser.add_argument(
        "--validation-member-audit",
        type=Path,
        default=VALIDATION_ROOT / "causal_qdm_member_audit_2019_v1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

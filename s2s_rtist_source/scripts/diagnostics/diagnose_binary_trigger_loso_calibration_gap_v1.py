#!/usr/bin/env python3
"""Diagnose the gap between deployable LOSO calibration and site oracle.

Run this after evaluate_binary_trigger_loso_calibration_policies_v1.py. The
default target policy is nearest_1_oracle_rate_transfer because it is currently
the best deployable learned-trigger calibration. The script compares it with
the held-out-label site oracle, then breaks the remaining regret down by site
and trigger error type.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_loso_calibration_policies_v1"
)
DEFAULT_DECISIONS = DEFAULT_ROOT / "binary_trigger_loso_calibration_policy_decisions_v1.csv"
DEFAULT_OUT = DEFAULT_ROOT / "gap_diagnostic_v1"
PAPER_FIXED_LIST_REGRET = 0.614875609


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def classify_error(row: pd.Series) -> str:
    pred = bool(row["pred_should_irrigate"])
    actual = bool(row["should_irrigate"])
    if pred and actual:
        return "true_positive"
    if pred and not actual:
        return "false_positive"
    if not pred and actual:
        return "false_negative"
    return "true_negative"


def summarize_policy(df: pd.DataFrame) -> dict:
    pred = df["pred_should_irrigate"].astype(bool)
    actual = df["should_irrigate"].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "policy": str(df["policy"].iloc[0]),
        "mean_regret": float(df["trigger_decision_regret_oracle_amount"].mean()),
        "median_regret": float(df["trigger_decision_regret_oracle_amount"].median()),
        "paper_fixed_list_gap": float(df["trigger_decision_regret_oracle_amount"].mean() - PAPER_FIXED_LIST_REGRET),
        "trigger_accuracy": float((pred == actual).mean()),
        "trigger_recall": recall,
        "trigger_specificity": specificity,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "predicted_irrigation_rate": float(pred.mean()),
        "true_irrigation_rate": float(actual.mean()),
    }


def write_csv(path: Path, df: pd.DataFrame) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def write_text(path: Path, text: str) -> bool:
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--target-policy", default="nearest_1_oracle_rate_transfer")
    parser.add_argument("--oracle-policy", default="site_oracle_threshold")
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions file: {decisions_path}")
    out_dir = Path(args.output_dir)
    can_write = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print("[warn] No space left on device; will print summary only and skip writing")
            can_write = False
        else:
            raise

    decisions = pd.read_csv(decisions_path)
    for col in ["should_irrigate", "pred_should_irrigate", "trigger_correct"]:
        if col in decisions.columns:
            decisions[col] = bool_series(decisions[col])
    decisions["policy"] = decisions["policy"].astype(str)

    target = decisions.loc[decisions["policy"] == args.target_policy].copy()
    oracle = decisions.loc[decisions["policy"] == args.oracle_policy].copy()
    if target.empty:
        raise ValueError(f"Missing target policy: {args.target_policy}")
    if oracle.empty:
        raise ValueError(f"Missing oracle policy: {args.oracle_policy}")

    target["error_type"] = target.apply(classify_error, axis=1)
    oracle["error_type"] = oracle.apply(classify_error, axis=1)
    summary = pd.DataFrame([summarize_policy(target), summarize_policy(oracle)])

    by_site_target = (
        target.groupby("site_id")
        .agg(
            target_mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            target_max_regret=("trigger_decision_regret_oracle_amount", "max"),
            target_accuracy=("trigger_correct", "mean"),
            target_assigned_threshold=("assigned_threshold", "first"),
            target_irrigation_rate=("pred_should_irrigate", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    by_site_oracle = (
        oracle.groupby("site_id")
        .agg(
            oracle_mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            oracle_max_regret=("trigger_decision_regret_oracle_amount", "max"),
            oracle_accuracy=("trigger_correct", "mean"),
            oracle_assigned_threshold=("assigned_threshold", "first"),
            oracle_irrigation_rate=("pred_should_irrigate", "mean"),
        )
        .reset_index()
    )
    by_site_gap = by_site_target.merge(by_site_oracle, on="site_id", how="inner")
    by_site_gap["target_minus_oracle_mean_regret"] = (
        by_site_gap["target_mean_regret"] - by_site_gap["oracle_mean_regret"]
    )
    by_site_gap["target_minus_paper_fixed_list"] = by_site_gap["target_mean_regret"] - PAPER_FIXED_LIST_REGRET
    by_site_gap = by_site_gap.sort_values("target_minus_oracle_mean_regret", ascending=False)

    error_breakdown = (
        target.groupby("error_type")
        .agg(
            count=("site_date_id", "count"),
            total_regret=("trigger_decision_regret_oracle_amount", "sum"),
            mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            max_regret=("trigger_decision_regret_oracle_amount", "max"),
        )
        .reset_index()
        .sort_values("total_regret", ascending=False)
    )
    error_by_site = (
        target.groupby(["site_id", "error_type"])
        .agg(
            count=("site_date_id", "count"),
            total_regret=("trigger_decision_regret_oracle_amount", "sum"),
            mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            max_regret=("trigger_decision_regret_oracle_amount", "max"),
        )
        .reset_index()
        .sort_values(["total_regret", "site_id"], ascending=[False, True])
    )

    merge_cols = [
        "site_date_id",
        "site_id",
        "trigger_decision_regret_oracle_amount",
        "pred_should_irrigate",
        "assigned_threshold",
        "chosen_ir_oracle_amount",
    ]
    gap = target[merge_cols + ["error_type"]].merge(
        oracle[merge_cols + ["error_type"]],
        on=["site_date_id", "site_id"],
        how="inner",
        suffixes=("_target", "_oracle"),
    )
    gap["target_minus_oracle_regret"] = (
        gap["trigger_decision_regret_oracle_amount_target"]
        - gap["trigger_decision_regret_oracle_amount_oracle"]
    )
    worst_gap = gap.sort_values("target_minus_oracle_regret", ascending=False).head(60)

    summary_path = out_dir / "binary_trigger_loso_calibration_gap_summary_v1.csv"
    by_site_path = out_dir / "binary_trigger_loso_calibration_gap_by_site_v1.csv"
    error_path = out_dir / "binary_trigger_loso_calibration_gap_error_breakdown_v1.csv"
    error_by_site_path = out_dir / "binary_trigger_loso_calibration_gap_error_by_site_v1.csv"
    worst_path = out_dir / "binary_trigger_loso_calibration_gap_worst_decisions_v1.csv"
    report_path = out_dir / "binary_trigger_loso_calibration_gap_v1.md"

    lines = [
        "# Binary Trigger LOSO Calibration Gap V1",
        "",
        "## Inputs",
        "",
        f"- Decisions: `{decisions_path}`",
        f"- Target policy: `{args.target_policy}`",
        f"- Oracle policy: `{args.oracle_policy}`",
        f"- Paper fixed-list mean regret: `{PAPER_FIXED_LIST_REGRET}`",
        "",
        "## Policy Summary",
        "",
        markdown_table(summary),
        "",
        "## Target vs Site Oracle By Site",
        "",
        markdown_table(by_site_gap),
        "",
        "## Target Error Breakdown",
        "",
        markdown_table(error_breakdown),
        "",
        "## Target Error By Site",
        "",
        markdown_table(error_by_site),
        "",
        "## Worst Target Minus Oracle Decisions",
        "",
        markdown_table(worst_gap),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{error_path}`",
        f"- `{error_by_site_path}`",
        f"- `{worst_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(by_site_path, by_site_gap)
        write_csv(error_path, error_breakdown)
        write_csv(error_by_site_path, error_by_site)
        write_csv(worst_path, worst_gap)
        write_text(report_path, report_text)

    print("Binary trigger LOSO calibration gap v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"error_breakdown: {error_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site_gap.to_string(index=False))
    print("")
    print(error_breakdown.to_string(index=False))


if __name__ == "__main__":
    main()

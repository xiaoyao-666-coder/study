#!/usr/bin/env python3
"""Evaluate a two-stage irrigation policy on LSTM candidate predictions.

Stage 1 decides whether to irrigate at all by comparing the best positive
candidate score with the zero-irrigation score. Stage 2 selects the best
positive candidate only when the positive margin passes a threshold.

This is a lightweight diagnostic over an existing prediction CSV; it does not
retrain the surrogate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import TARGET, bool_series, markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_PREDICTIONS = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_surrogate_lstm_loso_v1"
    / "continuous_irrigation_surrogate_lstm_v1_predictions.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_lstm_twostage_policy_v1"


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def select_candidate(group: pd.DataFrame, pred_col: str, threshold: float) -> pd.Series:
    g = group.copy()
    g["candidate_ir"] = pd.to_numeric(g["candidate_ir"], errors="coerce")
    zero = g.loc[g["candidate_ir"].abs() <= 1e-9]
    if zero.empty:
        raise ValueError(f"Missing zero-irrigation candidate for {group['site_date_id'].iloc[0]}")
    zero_row = zero.iloc[0]
    positive = g.loc[g["candidate_ir"] > 1e-9]
    if positive.empty:
        return zero_row
    positive_best = positive.loc[positive[pred_col].idxmax()]
    margin = float(positive_best[pred_col] - zero_row[pred_col])
    return positive_best if margin >= threshold else zero_row


def evaluate_threshold(df: pd.DataFrame, pred_col: str, threshold: float) -> pd.DataFrame:
    rows = []
    for site_date_id, group in df.groupby("site_date_id", sort=False):
        true_best = group.loc[group[TARGET].idxmax()]
        chosen = select_candidate(group, pred_col, threshold)
        zero = group.loc[pd.to_numeric(group["candidate_ir"], errors="coerce").abs() <= 1e-9].iloc[0]
        positive = group.loc[pd.to_numeric(group["candidate_ir"], errors="coerce") > 1e-9]
        best_positive = positive.loc[positive[pred_col].idxmax()] if len(positive) else zero
        rows.append(
            {
                "threshold": float(threshold),
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "true_best_net_gain": float(true_best[TARGET]),
                "chosen_true_net_gain": float(chosen[TARGET]),
                "zero_pred": float(zero[pred_col]),
                "best_positive_ir": float(best_positive["candidate_ir"]),
                "best_positive_pred": float(best_positive[pred_col]),
                "positive_margin_over_zero": float(best_positive[pred_col] - zero[pred_col]),
                "decision_correct": float(chosen["candidate_ir"]) == float(true_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - chosen[TARGET]),
                "chosen_is_zero": abs(float(chosen["candidate_ir"])) <= 1e-9,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--pred-col", default="pred_net_gain_7d")
    parser.add_argument(
        "--thresholds",
        default="-50,-20,-10,-5,-2,-1,0,1,2,5,10,20,30,50,75,100",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    if args.pred_col not in df.columns:
        raise ValueError(f"Missing prediction column: {args.pred_col}")
    for col in ["target_collapse", "same_date_duplicate_target_curve", "is_best_ir"]:
        if col in df.columns:
            df[col] = bool_series(df[col])

    all_decisions = [
        evaluate_threshold(df, args.pred_col, threshold)
        for threshold in parse_thresholds(args.thresholds)
    ]
    decisions = pd.concat(all_decisions, ignore_index=True)
    summary = (
        decisions.groupby("threshold")
        .agg(
            mean_decision_regret=("decision_regret", "mean"),
            median_decision_regret=("decision_regret", "median"),
            decision_accuracy=("decision_correct", "mean"),
            collapse_decision_accuracy=("decision_correct", lambda s: safe_mean(s[decisions.loc[s.index, "target_collapse"]])),
            noncollapse_decision_accuracy=("decision_correct", lambda s: safe_mean(s[~decisions.loc[s.index, "target_collapse"]])),
            zero_choice_rate=("chosen_is_zero", "mean"),
            false_positive_irrigation_rate=("chosen_is_zero", lambda s: safe_mean(~s[decisions.loc[s.index, "target_collapse"]])),
            missed_irrigation_rate=("chosen_is_zero", lambda s: safe_mean(s[~decisions.loc[s.index, "target_collapse"]])),
        )
        .reset_index()
        .sort_values("mean_decision_regret")
    )
    best_threshold = float(summary.iloc[0]["threshold"])
    best_decisions = decisions.loc[decisions["threshold"] == best_threshold].copy()
    by_site = (
        best_decisions.groupby("site_id")
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
            max_decision_regret=("decision_regret", "max"),
            zero_choice_rate=("chosen_is_zero", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_decision_regret", ascending=False)
    )
    worst = best_decisions.sort_values("decision_regret", ascending=False).head(25)

    decisions_path = out_dir / "continuous_ir_lstm_twostage_policy_decisions_v1.csv"
    summary_path = out_dir / "continuous_ir_lstm_twostage_policy_threshold_sweep_v1.csv"
    by_site_path = out_dir / "continuous_ir_lstm_twostage_policy_by_site_v1.csv"
    report_path = out_dir / "continuous_ir_lstm_twostage_policy_v1.md"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# LSTM Two-Stage Irrigation Policy V1",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_path}`",
        f"- Prediction column: `{args.pred_col}`",
        "",
        "## Threshold Sweep",
        "",
        markdown_table(summary),
        "",
        f"Best threshold by mean regret: `{best_threshold}`",
        "",
        "## Best Threshold By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decisions At Best Threshold",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("LSTM two-stage irrigation policy v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply a safe-threshold policy on top of tree surrogate predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRED = Path("Maize_shortterm_surrogate_v1/surrogate_tree_nosklearn_v1_predictions.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
POLICY_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_safe_policy_eval.csv"
METRICS_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_safe_policy_metrics.txt"

THRESHOLDS = [0.0, 1.0, 2.0, 5.0]


def select_safe_candidate(group: pd.DataFrame, threshold: float) -> pd.Series:
    g = group.sort_values("candidate_ir").reset_index(drop=True)
    g = g.copy()
    g["pred_rank"] = g["pred_net_gain_7d"].rank(ascending=False, method="first")
    best = g.loc[g["pred_net_gain_7d"].idxmax()]
    zero = g.loc[g["candidate_ir"] == 0].iloc[0]
    second = g.sort_values("pred_net_gain_7d", ascending=False).iloc[1]

    if float(best["candidate_ir"]) == 0.0:
        return best

    improvement_over_zero = float(best["pred_net_gain_7d"] - zero["pred_net_gain_7d"])
    margin_over_second = float(best["pred_net_gain_7d"] - second["pred_net_gain_7d"])

    if improvement_over_zero < threshold or margin_over_second < threshold:
        # Safe choice: prefer the smallest candidate that is near the best score.
        close = g[g["pred_net_gain_7d"] >= best["pred_net_gain_7d"] - threshold].sort_values("candidate_ir")
        return close.iloc[0]

    return best


def evaluate_policy(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for date_t, group in df.groupby("date_t", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        chosen = select_safe_candidate(group, threshold)
        rows.append(
            {
                "threshold": threshold,
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "chosen_pred_net_gain": float(chosen["pred_net_gain_7d"]),
                "chosen_true_net_gain": float(chosen["net_gain_7d"]),
                "decision_correct": float(chosen["candidate_ir"]) == float(true_best["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
                "chosen_is_zero": float(chosen["candidate_ir"]) == 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not PRED.exists():
        raise FileNotFoundError(f"Missing prediction file: {PRED}")

    df = pd.read_csv(PRED)
    if "pred_net_gain_7d" not in df.columns:
        raise ValueError("Prediction file missing pred_net_gain_7d")

    all_rows = []
    for threshold in THRESHOLDS:
        all_rows.append(evaluate_policy(df, threshold))
    policy_df = pd.concat(all_rows, ignore_index=True)
    policy_df.to_csv(POLICY_OUT, index=False)

    summary_rows = []
    for threshold, group in policy_df.groupby("threshold", sort=True):
        summary_rows.append(
            {
                "threshold": threshold,
                "decision_accuracy": float(group["decision_correct"].mean()),
                "mean_regret": float(group["decision_regret"].mean()),
                "zero_rate": float(group["chosen_is_zero"].mean()),
                "avg_chosen_true_gain": float(group["chosen_true_net_gain"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["policy_note"] = summary["threshold"].map(
        lambda x: "very conservative" if x >= 5 else ("conservative" if x >= 2 else "light guard")
    )
    summary.to_csv(OUT_DIR / "surrogate_tree_nosklearn_v1_safe_policy_summary.csv", index=False)

    lines = [
        "Tree surrogate safe-threshold policy v1",
        "",
        f"prediction file: {PRED}",
        f"policy output: {POLICY_OUT}",
        "",
        summary.to_string(index=False),
        "",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nPolicy eval:")
    print(policy_df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate site-normalized top-rate policies for the binary trigger.

Raw trigger probabilities are poorly calibrated across sites, but the
site-threshold oracle shows the ranking signal can be strong. This diagnostic
uses a single global within-site irrigation rate: for each site, irrigate the
top X percent of site-dates by predicted trigger probability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_loso_v1"
)
DEFAULT_PREDICTIONS = DEFAULT_ROOT / "continuous_irrigation_binary_trigger_lstm_v1_predictions.csv"
DEFAULT_OUT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_site_rate_policy_v1"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def parse_rates(text: str, n_steps: int) -> list[float]:
    if text.strip().lower() == "auto":
        return [i / n_steps for i in range(n_steps + 1)]
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one rate is required")
    return sorted(set(values))


def evaluate_rate(pred_df: pd.DataFrame, rate: float) -> pd.DataFrame:
    parts = []
    for _site_id, group in pred_df.groupby("site_id", sort=False):
        g = group.sort_values("pred_irrigate_prob", ascending=False).copy()
        n_select = int(round(float(rate) * len(g)))
        n_select = max(0, min(len(g), n_select))
        g["pred_should_irrigate"] = False
        if n_select > 0:
            g.iloc[:n_select, g.columns.get_loc("pred_should_irrigate")] = True
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    chosen_gain = np.where(
        out["pred_should_irrigate"].to_numpy(dtype=bool),
        out["oracle_positive_true_net_gain"].to_numpy(dtype=float),
        out["zero_true_net_gain"].to_numpy(dtype=float),
    )
    out["site_rate"] = float(rate)
    out["trigger_correct"] = out["pred_should_irrigate"].to_numpy(dtype=bool) == out["should_irrigate"].to_numpy(dtype=bool)
    out["trigger_decision_regret_oracle_amount"] = out["true_best_net_gain"].to_numpy(dtype=float) - chosen_gain
    out["chosen_ir_oracle_amount"] = np.where(
        out["pred_should_irrigate"].to_numpy(dtype=bool),
        out["oracle_positive_ir"].to_numpy(dtype=float),
        0.0,
    )
    return out


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rate, group in decisions.groupby("site_rate", sort=False):
        pred = group["pred_should_irrigate"].astype(bool)
        actual = group["should_irrigate"].astype(bool)
        correct = pred == actual
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        tn = int((~pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        rows.append(
            {
                "site_rate": float(rate),
                "trigger_accuracy": float(correct.mean()),
                "trigger_balanced_accuracy": float(np.nanmean([recall, specificity])),
                "trigger_precision": safe_div(tp, tp + fp),
                "trigger_recall": recall,
                "trigger_specificity": specificity,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "mean_decision_regret_oracle_amount": float(group["trigger_decision_regret_oracle_amount"].mean()),
                "median_decision_regret_oracle_amount": float(group["trigger_decision_regret_oracle_amount"].median()),
                "predicted_irrigation_rate": float(pred.mean()),
                "true_irrigation_rate": float(actual.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--rates", default="auto")
    parser.add_argument("--auto-steps", type=int, default=27)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(pred_path)
    for col in ["should_irrigate", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in pred_df.columns:
            pred_df[col] = bool_series(pred_df[col])
    all_decisions = [evaluate_rate(pred_df, rate) for rate in parse_rates(args.rates, args.auto_steps)]
    decisions = pd.concat(all_decisions, ignore_index=True)
    summary = summarize(decisions).sort_values(
        ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
        ascending=[True, False],
    )
    best_rate = float(summary.iloc[0]["site_rate"])
    best_decisions = decisions.loc[decisions["site_rate"] == best_rate].copy()
    by_site = (
        best_decisions.groupby("site_id")
        .agg(
            trigger_accuracy=("trigger_correct", "mean"),
            mean_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "mean"),
            max_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "max"),
            predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            true_irrigation_rate=("should_irrigate", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_decision_regret_oracle_amount", ascending=False)
    )
    worst = best_decisions.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(40)

    decisions_path = out_dir / "binary_trigger_site_rate_policy_decisions_v1.csv"
    summary_path = out_dir / "binary_trigger_site_rate_policy_summary_v1.csv"
    by_site_path = out_dir / "binary_trigger_site_rate_policy_by_site_v1.csv"
    report_path = out_dir / "binary_trigger_site_rate_policy_v1.md"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Binary Trigger Site-Rate Policy V1",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_path}`",
        "",
        "## Site-Rate Sweep",
        "",
        markdown_table(summary),
        "",
        f"Best site rate by oracle-amount mean regret: `{best_rate}`",
        "",
        "## Best Rate By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decisions At Best Rate",
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

    print("Binary trigger site-rate policy v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

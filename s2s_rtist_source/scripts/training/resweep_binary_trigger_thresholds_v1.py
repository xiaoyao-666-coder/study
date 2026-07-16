#!/usr/bin/env python3
"""Resweep binary-trigger thresholds over saved site-date predictions.

The first binary-trigger sweep started at 0.05, but failure diagnostics showed
some high-regret false negatives with probabilities far below that range. This
script reuses the saved predictions and evaluates a denser low-threshold grid
without retraining the trigger.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table
from train_continuous_irrigation_binary_trigger_lstm_v1 import evaluate_thresholds, parse_thresholds


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_loso_v1"
)
DEFAULT_PREDICTIONS = DEFAULT_ROOT / "continuous_irrigation_binary_trigger_lstm_v1_predictions.csv"
DEFAULT_OUT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_low_threshold_resweep_v1"
DEFAULT_THRESHOLDS = (
    "0,1e-10,1e-9,1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4,"
    "5e-4,0.001,0.0025,0.005,0.01,0.02,0.03,0.04,0.05,"
    "0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
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
    decisions, summary = evaluate_thresholds(pred_df, parse_thresholds(args.thresholds))
    summary = summary.sort_values(
        ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
        ascending=[True, False],
    )
    best_threshold = float(summary.iloc[0]["threshold"])
    best_decisions = decisions.loc[decisions["threshold"] == best_threshold].copy()
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

    decisions_path = out_dir / "binary_trigger_low_threshold_resweep_decisions_v1.csv"
    summary_path = out_dir / "binary_trigger_low_threshold_resweep_summary_v1.csv"
    by_site_path = out_dir / "binary_trigger_low_threshold_resweep_by_site_v1.csv"
    report_path = out_dir / "binary_trigger_low_threshold_resweep_v1.md"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Binary Trigger Low-Threshold Resweep V1",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_path}`",
        "",
        "## Threshold Sweep",
        "",
        markdown_table(summary),
        "",
        f"Best threshold by oracle-amount mean regret: `{best_threshold}`",
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

    print("Binary trigger low-threshold resweep v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

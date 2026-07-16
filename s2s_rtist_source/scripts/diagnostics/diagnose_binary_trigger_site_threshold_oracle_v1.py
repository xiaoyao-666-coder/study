#!/usr/bin/env python3
"""Diagnose the upper bound of site-specific binary-trigger thresholds.

This is a diagnostic only: it selects the best threshold per held-out site using
observed outcomes. If this oracle is much better than the best global threshold,
the remaining issue is threshold calibration/site heterogeneity. If not, the
trigger probabilities themselves are misordered for key positive dates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_low_threshold_resweep_v1"
DEFAULT_DECISIONS = DEFAULT_ROOT / "binary_trigger_low_threshold_resweep_decisions_v1.csv"
DEFAULT_SUMMARY = DEFAULT_ROOT / "binary_trigger_low_threshold_resweep_summary_v1.csv"
DEFAULT_OUT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_site_threshold_oracle_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    summary_path = Path(args.summary)
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions file: {decisions_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = pd.read_csv(decisions_path)
    summary = pd.read_csv(summary_path)
    for col in ["should_irrigate", "pred_should_irrigate", "trigger_correct"]:
        if col in decisions.columns:
            decisions[col] = bool_series(decisions[col])
    best_global = summary.sort_values(
        ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
        ascending=[True, False],
    ).iloc[0]
    by_site_threshold = (
        decisions.groupby(["site_id", "threshold"])
        .agg(
            n_site_dates=("site_date_id", "count"),
            trigger_accuracy=("trigger_correct", "mean"),
            mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            max_regret=("trigger_decision_regret_oracle_amount", "max"),
            predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            true_irrigation_rate=("should_irrigate", "mean"),
            false_negative_count=("pred_should_irrigate", lambda s: int((decisions.loc[s.index, "should_irrigate"] & ~s).sum())),
            false_positive_count=("pred_should_irrigate", lambda s: int((~decisions.loc[s.index, "should_irrigate"] & s).sum())),
        )
        .reset_index()
    )
    best_by_site = (
        by_site_threshold.sort_values(["site_id", "mean_regret", "trigger_accuracy"], ascending=[True, True, False])
        .groupby("site_id", as_index=False)
        .head(1)
        .sort_values("mean_regret", ascending=False)
        .reset_index(drop=True)
    )
    chosen = decisions.merge(
        best_by_site[["site_id", "threshold"]].rename(columns={"threshold": "site_oracle_threshold"}),
        on="site_id",
        how="inner",
    )
    chosen = chosen.loc[chosen["threshold"] == chosen["site_oracle_threshold"]].copy()
    oracle_summary = pd.DataFrame(
        [
            {
                "policy": "best_global_threshold",
                "threshold": float(best_global["threshold"]),
                "mean_regret": float(best_global["mean_decision_regret_oracle_amount"]),
                "trigger_accuracy": float(best_global["trigger_accuracy"]),
                "trigger_recall": float(best_global["trigger_recall"]),
                "trigger_specificity": float(best_global["trigger_specificity"]),
                "predicted_irrigation_rate": float(best_global["predicted_irrigation_rate"]),
            },
            {
                "policy": "site_oracle_thresholds",
                "threshold": "per_site",
                "mean_regret": float(chosen["trigger_decision_regret_oracle_amount"].mean()),
                "trigger_accuracy": float(chosen["trigger_correct"].mean()),
                "trigger_recall": float(chosen.loc[chosen["should_irrigate"], "pred_should_irrigate"].mean()),
                "trigger_specificity": float((~chosen.loc[~chosen["should_irrigate"], "pred_should_irrigate"]).mean()),
                "predicted_irrigation_rate": float(chosen["pred_should_irrigate"].mean()),
            },
        ]
    )
    worst = chosen.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(40)

    site_threshold_path = out_dir / "binary_trigger_site_threshold_oracle_by_site_v1.csv"
    chosen_path = out_dir / "binary_trigger_site_threshold_oracle_decisions_v1.csv"
    summary_out_path = out_dir / "binary_trigger_site_threshold_oracle_summary_v1.csv"
    report_path = out_dir / "binary_trigger_site_threshold_oracle_v1.md"
    best_by_site.to_csv(site_threshold_path, index=False)
    chosen.to_csv(chosen_path, index=False)
    oracle_summary.to_csv(summary_out_path, index=False)

    lines = [
        "# Binary Trigger Site-Threshold Oracle V1",
        "",
        "## Inputs",
        "",
        f"- Decisions: `{decisions_path}`",
        f"- Global summary: `{summary_path}`",
        "",
        "## Global vs Site-Oracle Thresholds",
        "",
        markdown_table(oracle_summary),
        "",
        "## Best Threshold By Site",
        "",
        markdown_table(best_by_site),
        "",
        "## Worst Site-Oracle Decisions",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{summary_out_path}`",
        f"- `{site_threshold_path}`",
        f"- `{chosen_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Binary trigger site-threshold oracle v1")
    print(f"summary: {summary_out_path}")
    print(f"by_site: {site_threshold_path}")
    print(f"report: {report_path}")
    print("")
    print(oracle_summary.to_string(index=False))
    print("")
    print(best_by_site.to_string(index=False))


if __name__ == "__main__":
    main()

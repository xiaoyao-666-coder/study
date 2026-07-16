#!/usr/bin/env python3
"""Diagnose failures of the binary irrigation trigger.

This script reads the binary-trigger outputs and focuses on the best threshold
selected by oracle-amount mean regret. It does not train a model; it identifies
which sites/dates drive the remaining regret before any positive-amount ranker
is attempted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


DEFAULT_ROOT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_loso_v1"
)
DEFAULT_DECISIONS = DEFAULT_ROOT / "continuous_irrigation_binary_trigger_lstm_v1_threshold_decisions.csv"
DEFAULT_SWEEP = DEFAULT_ROOT / "continuous_irrigation_binary_trigger_lstm_v1_threshold_sweep.csv"
DEFAULT_OUT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_failure_diagnostic_v1"


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def quantile(series: pd.Series, q: float) -> float:
    return float(pd.to_numeric(series, errors="coerce").quantile(q)) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--threshold-sweep", default=str(DEFAULT_SWEEP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    decisions_path = Path(args.decisions)
    sweep_path = Path(args.threshold_sweep)
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions file: {decisions_path}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing threshold sweep file: {sweep_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(decisions_path)
    sweep = pd.read_csv(sweep_path)
    for col in ["should_irrigate", "pred_should_irrigate", "trigger_correct", "target_collapse"]:
        if col in decisions.columns:
            decisions[col] = bool_series(decisions[col])

    threshold = args.threshold
    if threshold is None:
        threshold = float(
            sweep.sort_values(
                ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
                ascending=[True, False],
            ).iloc[0]["threshold"]
        )
    best = decisions.loc[decisions["threshold"].astype(float) == float(threshold)].copy()
    if best.empty:
        raise ValueError(f"No decision rows found for threshold={threshold}")

    false_negative = best.loc[best["should_irrigate"] & ~best["pred_should_irrigate"]].copy()
    false_positive = best.loc[~best["should_irrigate"] & best["pred_should_irrigate"]].copy()
    worst_fn = false_negative.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(args.top_n)
    worst_fp = false_positive.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(args.top_n)

    by_site = (
        best.groupby("site_id")
        .agg(
            n_site_dates=("site_date_id", "count"),
            true_irrigation_rate=("should_irrigate", "mean"),
            predicted_irrigation_rate=("pred_should_irrigate", "mean"),
            trigger_accuracy=("trigger_correct", "mean"),
            mean_regret=("trigger_decision_regret_oracle_amount", "mean"),
            max_regret=("trigger_decision_regret_oracle_amount", "max"),
            false_negative_count=("pred_should_irrigate", lambda s: int((best.loc[s.index, "should_irrigate"] & ~s).sum())),
            false_positive_count=("pred_should_irrigate", lambda s: int((~best.loc[s.index, "should_irrigate"] & s).sum())),
            mean_prob=("pred_irrigate_prob", "mean"),
            q10_prob=("pred_irrigate_prob", lambda s: quantile(s, 0.10)),
            median_prob=("pred_irrigate_prob", "median"),
            q90_prob=("pred_irrigate_prob", lambda s: quantile(s, 0.90)),
        )
        .reset_index()
        .sort_values("mean_regret", ascending=False)
    )
    by_error_type = pd.DataFrame(
        [
            {
                "error_type": "false_negative",
                "count": int(len(false_negative)),
                "mean_regret": safe_mean(false_negative["trigger_decision_regret_oracle_amount"]),
                "total_regret": float(false_negative["trigger_decision_regret_oracle_amount"].sum()),
                "mean_prob": safe_mean(false_negative["pred_irrigate_prob"]),
            },
            {
                "error_type": "false_positive",
                "count": int(len(false_positive)),
                "mean_regret": safe_mean(false_positive["trigger_decision_regret_oracle_amount"]),
                "total_regret": float(false_positive["trigger_decision_regret_oracle_amount"].sum()),
                "mean_prob": safe_mean(false_positive["pred_irrigate_prob"]),
            },
        ]
    )
    positive_by_site = (
        best.loc[best["should_irrigate"]]
        .groupby("site_id")
        .agg(
            positive_dates=("site_date_id", "count"),
            positive_recall=("pred_should_irrigate", "mean"),
            positive_mean_prob=("pred_irrigate_prob", "mean"),
            missed_positive_regret=("trigger_decision_regret_oracle_amount", "sum"),
            max_missed_positive_regret=("trigger_decision_regret_oracle_amount", "max"),
        )
        .reset_index()
        .sort_values("missed_positive_regret", ascending=False)
    )

    by_site_path = out_dir / "binary_trigger_failure_by_site_v1.csv"
    error_type_path = out_dir / "binary_trigger_failure_by_error_type_v1.csv"
    positive_site_path = out_dir / "binary_trigger_positive_recall_by_site_v1.csv"
    worst_fn_path = out_dir / "binary_trigger_worst_false_negative_v1.csv"
    worst_fp_path = out_dir / "binary_trigger_worst_false_positive_v1.csv"
    report_path = out_dir / "binary_trigger_failure_diagnostic_v1.md"
    by_site.to_csv(by_site_path, index=False)
    by_error_type.to_csv(error_type_path, index=False)
    positive_by_site.to_csv(positive_site_path, index=False)
    worst_fn.to_csv(worst_fn_path, index=False)
    worst_fp.to_csv(worst_fp_path, index=False)

    best_sweep = sweep.loc[sweep["threshold"].astype(float) == float(threshold)]
    lines = [
        "# Binary Trigger Failure Diagnostic V1",
        "",
        "## Inputs",
        "",
        f"- Decisions: `{decisions_path}`",
        f"- Threshold sweep: `{sweep_path}`",
        f"- Diagnosed threshold: `{threshold}`",
        "",
        "## Best Threshold Summary",
        "",
        markdown_table(best_sweep),
        "",
        "## Error Type Summary",
        "",
        markdown_table(by_error_type),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Positive Recall By Site",
        "",
        markdown_table(positive_by_site),
        "",
        "## Worst False Negatives",
        "",
        markdown_table(worst_fn.head(25)),
        "",
        "## Worst False Positives",
        "",
        markdown_table(worst_fp.head(25)),
        "",
        "## Outputs",
        "",
        f"- `{by_site_path}`",
        f"- `{error_type_path}`",
        f"- `{positive_site_path}`",
        f"- `{worst_fn_path}`",
        f"- `{worst_fp_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Binary trigger failure diagnostic v1")
    print(f"threshold: {threshold}")
    print(f"report: {report_path}")
    print("")
    print(by_error_type.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

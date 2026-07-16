#!/usr/bin/env python3
"""Train the first continuous-irrigation surrogate baseline.

This is the first model-training check after the sampling-plan SWAP runner and
feature builder. It consumes the continuous-irrigation feature table and runs a
ridge baseline with leave-one-site-out evaluation by default. The purpose is to
verify that the continuous irrigation amount can enter a site-general surrogate
workflow before scaling from the smoke plan to the full sampling plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    evaluate,
    markdown_table,
)


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_features_v1"
    / "confirmed_5site_true_input_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_surrogate_baseline_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--cv-group-col",
        default="site_id",
        help="Column to leave out for each CV fold. Use site_id for LOSO.",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing continuous-irrigation sample table: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    pred_df, decision_df, metrics_df, feature_cols = evaluate(
        df,
        alpha=args.alpha,
        cv_group_col=args.cv_group_col,
    )

    pred_path = out_dir / "continuous_irrigation_surrogate_baseline_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_baseline_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_baseline_v1_metrics.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_baseline_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_baseline_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    worst = decision_df.sort_values("decision_regret", ascending=False).head(15)
    by_site = (
        decision_df.groupby("site_id")
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
            max_decision_regret=("decision_regret", "max"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    by_site_path = out_dir / "continuous_irrigation_surrogate_baseline_v1_by_site.csv"
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Continuous Irrigation Surrogate Baseline V1",
        "",
        "## Scope",
        "",
        "- First continuous-irrigation model-training smoke check.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        "- Default CV is leave-one-site-out, which tests unseen-site generalization.",
        "- This is a ridge sanity baseline, not the final LSTM/Transformer surrogate.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decision Rows",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decision_path}`",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation surrogate baseline v1")
    print(f"input: {data_path}")
    print(f"predictions: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print(metrics_df.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print("")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()

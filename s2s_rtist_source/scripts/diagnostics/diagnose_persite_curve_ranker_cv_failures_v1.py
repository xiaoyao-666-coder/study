#!/usr/bin/env python3
"""Diagnose held-out-date failures for the per-site curve-aware ranker.

This is a lightweight post-run audit. It does not retrain a model. It reads the
ranker decision table and sampled-rank table, then summarizes where the held-out
date CV failures come from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_BASE = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_curve_mlp_ranker_cv_v1"
)


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def regret_bucket(value: float) -> str:
    if value <= 1e-9:
        return "zero"
    if value <= 1.0:
        return "0-1"
    if value <= 5.0:
        return "1-5"
    if value <= 20.0:
        return "5-20"
    return ">20"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "failure_diagnostic_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = base_dir / "persite_curve_mlp_ranker_decisions_v1.csv"
    sampled_path = base_dir / "persite_curve_mlp_ranker_sampled_rank_eval_v1.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions table: {decisions_path}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled rank table: {sampled_path}")

    decisions = pd.read_csv(decisions_path)
    sampled = pd.read_csv(sampled_path)
    if "eval_mode" in decisions.columns:
        decisions = decisions.loc[decisions["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    if "eval_mode" in sampled.columns:
        sampled = sampled.loc[sampled["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    if decisions.empty:
        raise ValueError("No heldout_date_cv decision rows found")

    merged = decisions.merge(
        sampled[
            [
                "site_date_id",
                "sampled_top1_correct",
                "sampled_curve_regret",
                "true_best_pred_rank",
            ]
        ],
        on="site_date_id",
        how="left",
    )
    merged["continuous_minus_paper_regret"] = (
        merged["continuous_ranker_regret_vs_dense_oracle"]
        - merged["paper_regret_vs_dense_oracle"]
    )
    merged["fixed_ranker_minus_paper_fixed_oracle"] = merged[
        "fixed_list_ranker_regret_vs_fixed_oracle"
    ]
    merged["continuous_regret_bucket"] = merged[
        "continuous_ranker_regret_vs_dense_oracle"
    ].apply(regret_bucket)
    merged["paper_regret_bucket"] = merged["paper_regret_vs_dense_oracle"].apply(regret_bucket)

    overall = pd.DataFrame(
        [
            {
                "site_dates": int(len(merged)),
                "paper_mean_regret": float(merged["paper_regret_vs_dense_oracle"].mean()),
                "fixed_list_ranker_mean_regret": float(
                    merged["fixed_list_ranker_regret_vs_fixed_oracle"].mean()
                ),
                "continuous_ranker_mean_regret": float(
                    merged["continuous_ranker_regret_vs_dense_oracle"].mean()
                ),
                "continuous_minus_paper_mean_regret": float(
                    merged["continuous_minus_paper_regret"].mean()
                ),
                "sampled_top1_accuracy": safe_mean(merged["sampled_top1_correct"]),
                "sampled_mean_curve_regret": float(merged["sampled_curve_regret"].mean()),
                "mean_true_best_pred_rank": float(merged["true_best_pred_rank"].mean()),
                "better_than_paper_rate": safe_mean(
                    merged["continuous_ranker_gain_over_paper"] > 1e-9
                ),
                "worse_than_paper_rate": safe_mean(
                    merged["continuous_ranker_gain_over_paper"] < -1e-9
                ),
                "nonfixed_ir_rate": safe_mean(merged["continuous_ranker_nonfixed_ir"]),
                "fixed_list_selection_bad_rate": safe_mean(
                    merged["fixed_list_ranker_regret_vs_fixed_oracle"] > 1e-9
                ),
                "large_continuous_failure_rate_regret_gt_5": safe_mean(
                    merged["continuous_ranker_regret_vs_dense_oracle"] > 5.0
                ),
            }
        ]
    )

    by_site = (
        merged.groupby("site_id")
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            fixed_list_ranker_mean_regret=("fixed_list_ranker_regret_vs_fixed_oracle", "mean"),
            continuous_ranker_mean_regret=("continuous_ranker_regret_vs_dense_oracle", "mean"),
            continuous_minus_paper_mean_regret=("continuous_minus_paper_regret", "mean"),
            sampled_top1_accuracy=("sampled_top1_correct", "mean"),
            sampled_mean_curve_regret=("sampled_curve_regret", "mean"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
            better_than_paper_rate=("continuous_ranker_better_than_paper", "mean"),
            worse_than_paper_rate=("continuous_ranker_worse_than_paper", "mean"),
            nonfixed_ir_rate=("continuous_ranker_nonfixed_ir", "mean"),
        )
        .reset_index()
        .sort_values("continuous_ranker_mean_regret", ascending=False)
    )

    by_bucket = (
        merged.groupby(["continuous_regret_bucket", "paper_regret_bucket"])
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            continuous_ranker_mean_regret=("continuous_ranker_regret_vs_dense_oracle", "mean"),
            sampled_top1_accuracy=("sampled_top1_correct", "mean"),
            nonfixed_ir_rate=("continuous_ranker_nonfixed_ir", "mean"),
        )
        .reset_index()
        .sort_values(["continuous_regret_bucket", "paper_regret_bucket"])
    )

    worst = merged.sort_values(
        "continuous_ranker_regret_vs_dense_oracle", ascending=False
    ).head(args.top_n)
    worst_cols = [
        "site_id",
        "site_date_id",
        "date_t",
        "dense_oracle_ir",
        "paper_fixed_list_oracle_ir",
        "continuous_ranker_ir",
        "paper_regret_vs_dense_oracle",
        "continuous_ranker_regret_vs_dense_oracle",
        "continuous_minus_paper_regret",
        "sampled_top1_correct",
        "sampled_curve_regret",
        "true_best_pred_rank",
        "continuous_ranker_nonfixed_ir",
    ]
    worst = worst[[col for col in worst_cols if col in worst.columns]].copy()

    overall_path = out_dir / "persite_curve_ranker_cv_failure_overall_v1.csv"
    by_site_path = out_dir / "persite_curve_ranker_cv_failure_by_site_v1.csv"
    by_bucket_path = out_dir / "persite_curve_ranker_cv_failure_by_bucket_v1.csv"
    worst_path = out_dir / "persite_curve_ranker_cv_failure_worst_dates_v1.csv"
    report_path = out_dir / "persite_curve_ranker_cv_failure_diagnostic_v1.md"
    overall.to_csv(overall_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    by_bucket.to_csv(by_bucket_path, index=False)
    worst.to_csv(worst_path, index=False)

    lines = [
        "# Per-Site Curve Ranker CV Failure Diagnostic V1",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Failure Buckets",
        "",
        markdown_table(by_bucket),
        "",
        "## Worst Dates",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{overall_path}`",
        f"- `{by_site_path}`",
        f"- `{by_bucket_path}`",
        f"- `{worst_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site curve ranker CV failure diagnostic v1")
    print(f"overall: {overall_path}")
    print(f"by_site: {by_site_path}")
    print(f"by_bucket: {by_bucket_path}")
    print(f"worst: {worst_path}")
    print(f"report: {report_path}")
    print("")
    print(overall.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

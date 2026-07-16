#!/usr/bin/env python3
"""Diagnose held-out-date failures for the per-site curve-top ranker.

The curve-top TinyForest passed the in-sample capacity check, but held-out-date
CV can still fail through a small number of catastrophic argmax choices. This
script does not retrain. It reads the curve-top ranker outputs and summarizes:

- whether the selected dense candidate is top1/top3/top5 on the true curve;
- how much worse the continuous decision is than the paper fixed-list oracle;
- which sites and dates dominate the mean regret;
- whether failures are fixed-list mistakes or off-list continuous mistakes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_BASE = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_curve_top_tinyforest_ranker_cv_v1"
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


def read_outputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions_path = base_dir / "persite_curve_top_tinyforest_ranker_decisions_v1.csv"
    sampled_path = base_dir / "persite_curve_top_tinyforest_ranker_sampled_rank_eval_v1.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing decisions table: {decisions_path}")
    if not sampled_path.exists():
        raise FileNotFoundError(f"Missing sampled-rank table: {sampled_path}")
    decisions = pd.read_csv(decisions_path)
    sampled = pd.read_csv(sampled_path)
    if "eval_mode" in decisions.columns:
        decisions = decisions.loc[decisions["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    if "eval_mode" in sampled.columns:
        sampled = sampled.loc[sampled["eval_mode"].astype(str) == "heldout_date_cv"].copy()
    if decisions.empty:
        raise ValueError("No heldout_date_cv rows found in decisions table")
    if sampled.empty:
        raise ValueError("No heldout_date_cv rows found in sampled-rank table")
    return decisions, sampled


def build_merged(decisions: pd.DataFrame, sampled: pd.DataFrame) -> pd.DataFrame:
    sampled_keep = [
        "site_date_id",
        "sampled_top1_correct",
        "sampled_top3_correct",
        "sampled_top5_correct",
        "sampled_curve_regret",
        "sampled_pred_best_true_rank",
        "true_best_pred_rank",
        "true_best_ir",
        "pred_best_ir",
    ]
    merged = decisions.merge(
        sampled[[col for col in sampled_keep if col in sampled.columns]],
        on="site_date_id",
        how="left",
    )
    merged["continuous_minus_paper_regret"] = (
        merged["continuous_top_ranker_regret_vs_dense_oracle"]
        - merged["paper_regret_vs_dense_oracle"]
    )
    merged["continuous_regret_bucket"] = merged[
        "continuous_top_ranker_regret_vs_dense_oracle"
    ].apply(regret_bucket)
    merged["paper_regret_bucket"] = merged["paper_regret_vs_dense_oracle"].apply(regret_bucket)
    merged["selected_rank_bucket"] = pd.cut(
        merged["continuous_selected_dense_rank"],
        bins=[0, 1, 3, 5, 10, 25, np.inf],
        labels=["1", "2-3", "4-5", "6-10", "11-25", ">25"],
        include_lowest=True,
        right=True,
    ).astype(str)
    merged["offlist_large_failure"] = (
        merged["continuous_top_ranker_nonfixed_ir"].astype(bool)
        & (merged["continuous_top_ranker_regret_vs_dense_oracle"] > 5.0)
    )
    merged["fixedgrid_large_failure"] = (
        ~merged["continuous_top_ranker_nonfixed_ir"].astype(bool)
        & (merged["continuous_top_ranker_regret_vs_dense_oracle"] > 5.0)
    )
    return merged


def overall_table(merged: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "site_dates": int(len(merged)),
                "sites": int(merged["site_id"].nunique()),
                "paper_mean_regret": float(merged["paper_regret_vs_dense_oracle"].mean()),
                "continuous_mean_regret": float(
                    merged["continuous_top_ranker_regret_vs_dense_oracle"].mean()
                ),
                "continuous_median_regret": float(
                    merged["continuous_top_ranker_regret_vs_dense_oracle"].median()
                ),
                "continuous_p90_regret": float(
                    np.quantile(merged["continuous_top_ranker_regret_vs_dense_oracle"], 0.9)
                ),
                "continuous_minus_paper_mean_regret": float(
                    merged["continuous_minus_paper_regret"].mean()
                ),
                "sampled_top1_accuracy": safe_mean(merged["sampled_top1_correct"]),
                "sampled_top3_accuracy": safe_mean(merged["sampled_top3_correct"]),
                "sampled_top5_accuracy": safe_mean(merged["sampled_top5_correct"]),
                "continuous_selected_top1_rate": safe_mean(merged["continuous_selected_top1"]),
                "continuous_selected_top3_rate": safe_mean(merged["continuous_selected_top3"]),
                "continuous_selected_top5_rate": safe_mean(merged["continuous_selected_top5"]),
                "mean_continuous_selected_dense_rank": float(
                    merged["continuous_selected_dense_rank"].mean()
                ),
                "large_regret_gt_5_rate": safe_mean(
                    merged["continuous_top_ranker_regret_vs_dense_oracle"] > 5.0
                ),
                "offlist_large_failure_rate": safe_mean(merged["offlist_large_failure"]),
                "fixedgrid_large_failure_rate": safe_mean(merged["fixedgrid_large_failure"]),
                "better_than_paper_rate": safe_mean(
                    merged["continuous_top_ranker_gain_over_paper"] > 1e-9
                ),
                "worse_than_paper_rate": safe_mean(
                    merged["continuous_top_ranker_gain_over_paper"] < -1e-9
                ),
                "nonfixed_ir_rate": safe_mean(merged["continuous_top_ranker_nonfixed_ir"]),
            }
        ]
    )


def by_site_table(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby("site_id")
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            continuous_mean_regret=("continuous_top_ranker_regret_vs_dense_oracle", "mean"),
            continuous_p90_regret=(
                "continuous_top_ranker_regret_vs_dense_oracle",
                lambda x: float(np.quantile(x, 0.9)),
            ),
            continuous_minus_paper_mean_regret=("continuous_minus_paper_regret", "mean"),
            sampled_top1_accuracy=("sampled_top1_correct", "mean"),
            sampled_top3_accuracy=("sampled_top3_correct", "mean"),
            sampled_mean_curve_regret=("sampled_curve_regret", "mean"),
            continuous_selected_top3_rate=("continuous_selected_top3", "mean"),
            mean_continuous_selected_dense_rank=("continuous_selected_dense_rank", "mean"),
            large_regret_gt_5_rate=(
                "continuous_top_ranker_regret_vs_dense_oracle",
                lambda x: float(np.mean(x > 5.0)),
            ),
            offlist_large_failure_rate=("offlist_large_failure", "mean"),
            fixedgrid_large_failure_rate=("fixedgrid_large_failure", "mean"),
            nonfixed_ir_rate=("continuous_top_ranker_nonfixed_ir", "mean"),
            better_than_paper_rate=("continuous_top_ranker_better_than_paper", "mean"),
        )
        .reset_index()
        .sort_values("continuous_mean_regret", ascending=False)
    )


def bucket_table(merged: pd.DataFrame) -> pd.DataFrame:
    return (
        merged.groupby(["continuous_regret_bucket", "selected_rank_bucket"], dropna=False)
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            continuous_mean_regret=("continuous_top_ranker_regret_vs_dense_oracle", "mean"),
            mean_selected_rank=("continuous_selected_dense_rank", "mean"),
            sampled_top3_accuracy=("sampled_top3_correct", "mean"),
            nonfixed_ir_rate=("continuous_top_ranker_nonfixed_ir", "mean"),
        )
        .reset_index()
        .sort_values(["continuous_regret_bucket", "selected_rank_bucket"])
    )


def worst_dates_table(merged: pd.DataFrame, top_n: int) -> pd.DataFrame:
    keep = [
        "site_id",
        "site_date_id",
        "date_t",
        "dense_oracle_ir",
        "paper_fixed_list_oracle_ir",
        "continuous_top_ranker_ir",
        "continuous_selected_dense_rank",
        "paper_regret_vs_dense_oracle",
        "continuous_top_ranker_regret_vs_dense_oracle",
        "continuous_minus_paper_regret",
        "continuous_top_ranker_gain_over_paper",
        "sampled_top1_correct",
        "sampled_top3_correct",
        "sampled_curve_regret",
        "sampled_pred_best_true_rank",
        "true_best_pred_rank",
        "continuous_top_ranker_nonfixed_ir",
        "offlist_large_failure",
        "fixedgrid_large_failure",
    ]
    return (
        merged.sort_values("continuous_top_ranker_regret_vs_dense_oracle", ascending=False)
        [[col for col in keep if col in merged.columns]]
        .head(top_n)
        .copy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir) if args.output_dir else base_dir / "failure_diagnostic_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions, sampled = read_outputs(base_dir)
    merged = build_merged(decisions, sampled)
    overall = overall_table(merged)
    by_site = by_site_table(merged)
    by_bucket = bucket_table(merged)
    worst = worst_dates_table(merged, args.top_n)

    merged_path = out_dir / "persite_curve_top_ranker_cv_failure_merged_v1.csv"
    overall_path = out_dir / "persite_curve_top_ranker_cv_failure_overall_v1.csv"
    by_site_path = out_dir / "persite_curve_top_ranker_cv_failure_by_site_v1.csv"
    by_bucket_path = out_dir / "persite_curve_top_ranker_cv_failure_by_bucket_v1.csv"
    worst_path = out_dir / "persite_curve_top_ranker_cv_failure_worst_dates_v1.csv"
    report_path = out_dir / "persite_curve_top_ranker_cv_failure_diagnostic_v1.md"

    merged.to_csv(merged_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    by_bucket.to_csv(by_bucket_path, index=False)
    worst.to_csv(worst_path, index=False)

    lines = [
        "# Per-Site Curve-Top Ranker CV Failure Diagnostic V1",
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
        f"- `{merged_path}`",
        f"- `{overall_path}`",
        f"- `{by_site_path}`",
        f"- `{by_bucket_path}`",
        f"- `{worst_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site curve-top ranker CV failure diagnostic v1")
    print(f"merged: {merged_path}")
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

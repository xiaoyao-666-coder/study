#!/usr/bin/env python3
"""Train per-site TinyForest upper baselines for continuous irrigation.

This diagnostic separates two failure modes:

1. LOSO site-generalization failure: a site-general model cannot extrapolate to
   an unseen site.
2. Model/feature capacity failure: even a per-site model trained on the same
   site cannot rank irrigation candidates well.

For each site, this script filters the candidate table to that site and runs
leave-one-site-date-out validation with the same TinyForest implementation used
by the site-general baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import TARGET, markdown_table
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import evaluate, score_metrics


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_features_v1"
    / "continuous_ir_12site_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_persite_tree_v1"
)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing continuous-irrigation sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "site_id" not in df.columns or "site_date_id" not in df.columns:
        raise ValueError("Input table must contain site_id and site_date_id")

    pred_parts = []
    decision_parts = []
    metric_parts = []
    feature_map = {}
    sites = sorted(df["site_id"].astype(str).unique())
    for i, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy()
        print(
            f"[persite] site {i + 1}/{len(sites)} {site_id} rows={len(site_df)} dates={site_df['site_date_id'].nunique()}",
            flush=True,
        )
        pred_df, decision_df, metrics_df, _by_site, feature_cols = evaluate(
            site_df,
            cv_group_col="site_date_id",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state + i * 100,
        )
        pred_df["model"] = "persite_tiny_forest"
        decision_df["model"] = "persite_tiny_forest"
        metrics_df.insert(0, "site_id", site_id)
        pred_parts.append(pred_df)
        decision_parts.append(decision_df)
        metric_parts.append(metrics_df)
        feature_map[site_id] = feature_cols

    pred_all = pd.concat(pred_parts, ignore_index=True)
    decision_all = pd.concat(decision_parts, ignore_index=True)
    per_site_metrics = pd.concat(metric_parts, ignore_index=True)

    overall = score_metrics(
        pred_all[TARGET].to_numpy(dtype=float),
        pred_all["pred_net_gain_7d"].to_numpy(dtype=float),
    )
    overall.update(
        {
            "model": "persite_tiny_forest",
            "cv_group_col": "site_date_id_within_site",
            "cv_folds": int(decision_all["site_date_id"].nunique()),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "decision_correct": int(decision_all["decision_correct"].sum()),
            "decision_total": int(len(decision_all)),
            "decision_accuracy": safe_mean(decision_all["decision_correct"]),
            "mean_decision_regret": float(decision_all["decision_regret"].mean()),
            "median_decision_regret": float(decision_all["decision_regret"].median()),
            "collapse_decision_accuracy": safe_mean(decision_all.loc[decision_all["target_collapse"], "decision_correct"]),
            "noncollapse_decision_accuracy": safe_mean(decision_all.loc[~decision_all["target_collapse"], "decision_correct"]),
        }
    )
    overall_df = pd.DataFrame([overall])

    by_site = (
        decision_all.groupby("site_id")
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
            max_decision_regret=("decision_regret", "max"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("mean_decision_regret", ascending=False)
    )
    worst = decision_all.sort_values("decision_regret", ascending=False).head(20)

    pred_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_metrics.csv"
    per_site_metrics_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_per_site_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_persite_tree_v1.md"

    pred_all.to_csv(pred_path, index=False)
    decision_all.to_csv(decision_path, index=False)
    overall_df.to_csv(metrics_path, index=False)
    per_site_metrics.to_csv(per_site_metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(json.dumps(feature_map, indent=2), encoding="utf-8")

    lines = [
        "# Continuous Irrigation Per-Site Tree V1",
        "",
        "## Scope",
        "",
        "- Per-site TinyForest upper baseline.",
        "- Validation is leave-one-site-date-out within each site.",
        "- Use this only as a diagnostic upper bound, not as the final method.",
        "",
        "## Overall Metrics",
        "",
        markdown_table(overall_df),
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
        f"- `{per_site_metrics_path}`",
        f"- `{by_site_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation per-site tree v1")
    print(f"predictions: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(overall_df.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

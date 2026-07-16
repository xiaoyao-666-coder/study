#!/usr/bin/env python3
"""Evaluate static-feature threshold transfer for the binary trigger.

Site-oracle thresholds beat the paper fixed list, while global and top-rate
policies do not. This diagnostic transfers threshold choices from training
sites to each held-out site using static-site feature similarity. It uses labels
only to compute thresholds on the other sites, not on the evaluated site.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import TARGET, bool_series, markdown_table


DEFAULT_PREDICTIONS = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_irrigation_trigger_loso_v1"
    / "continuous_irrigation_binary_trigger_lstm_v1_predictions.csv"
)
DEFAULT_SAMPLES = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = Path("site_general_surrogate_eval") / "continuous_ir_12site_10k_binary_trigger_threshold_transfer_v1"
DEFAULT_THRESHOLDS = (
    "0,1e-10,1e-9,1e-8,1e-7,1e-6,5e-6,1e-5,5e-5,1e-4,"
    "5e-4,0.001,0.0025,0.005,0.01,0.02,0.03,0.04,0.05,"
    "0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5"
)


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def evaluate_fixed_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["assigned_threshold"] = float(threshold)
    out["pred_should_irrigate"] = out["pred_irrigate_prob"] >= float(threshold)
    chosen_gain = np.where(
        out["pred_should_irrigate"].to_numpy(dtype=bool),
        out["oracle_positive_true_net_gain"].to_numpy(dtype=float),
        out["zero_true_net_gain"].to_numpy(dtype=float),
    )
    out["trigger_correct"] = out["pred_should_irrigate"].to_numpy(dtype=bool) == out["should_irrigate"].to_numpy(dtype=bool)
    out["trigger_decision_regret_oracle_amount"] = out["true_best_net_gain"].to_numpy(dtype=float) - chosen_gain
    out["chosen_ir_oracle_amount"] = np.where(
        out["pred_should_irrigate"].to_numpy(dtype=bool),
        out["oracle_positive_ir"].to_numpy(dtype=float),
        0.0,
    )
    return out


def summarize_decisions(decisions: pd.DataFrame, policy: str) -> dict:
    pred = decisions["pred_should_irrigate"].astype(bool)
    actual = decisions["should_irrigate"].astype(bool)
    correct = pred == actual
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "policy": policy,
        "trigger_accuracy": float(correct.mean()),
        "trigger_balanced_accuracy": float(np.nanmean([recall, specificity])),
        "trigger_precision": safe_div(tp, tp + fp),
        "trigger_recall": recall,
        "trigger_specificity": specificity,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "mean_decision_regret_oracle_amount": float(decisions["trigger_decision_regret_oracle_amount"].mean()),
        "median_decision_regret_oracle_amount": float(decisions["trigger_decision_regret_oracle_amount"].median()),
        "predicted_irrigation_rate": float(pred.mean()),
        "true_irrigation_rate": float(actual.mean()),
    }


def site_feature_columns(df: pd.DataFrame) -> list[str]:
    exact = ["longitude", "latitude", "site_ir_min", "site_ir_max"]
    prefixes = ("static_", "soil_")
    cols = [
        col
        for col in df.columns
        if col in exact or col.startswith(prefixes)
    ]
    numeric_cols = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isna().all():
            numeric_cols.append(col)
    return numeric_cols


def build_site_features(samples: pd.DataFrame) -> pd.DataFrame:
    if "site_id" not in samples.columns:
        raise ValueError("Missing site_id in samples")
    cols = site_feature_columns(samples)
    if not cols:
        raise ValueError("No usable static site feature columns found")
    feature_df = samples[["site_id", *cols]].copy()
    for col in cols:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    return feature_df.groupby("site_id", as_index=False)[cols].mean()


def nearest_site_thresholds(
    site_features: pd.DataFrame,
    site_oracle: pd.DataFrame,
    *,
    k: int,
    threshold_grid: list[float],
) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in site_features.columns if c != "site_id"]
    merged = site_features.merge(site_oracle[["site_id", "oracle_threshold"]], on="site_id", how="inner")
    if len(merged) < 2:
        raise ValueError("Need at least two sites for threshold transfer")

    grid = np.asarray(threshold_grid, dtype=float)
    eps = min([t for t in threshold_grid if t > 0] or [1e-10])
    for _, test in merged.iterrows():
        train = merged.loc[merged["site_id"] != test["site_id"]].copy()
        train_x = train[feature_cols].to_numpy(dtype=float)
        test_x = test[feature_cols].to_numpy(dtype=float)[None, :]
        med = np.nanmedian(train_x, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        train_inds = np.where(np.isnan(train_x))
        test_inds = np.where(np.isnan(test_x))
        train_x[train_inds] = np.take(med, train_inds[1])
        test_x[test_inds] = np.take(med, test_inds[1])
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        std = np.where(std <= 1e-12, 1.0, std)
        diff = ((train_x - mean) / std - (test_x - mean) / std).clip(-1e9, 1e9)
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        train = train.assign(distance=dist)
        neighbors = train.sort_values("distance").head(k)
        log_threshold = np.log10(np.maximum(neighbors["oracle_threshold"].to_numpy(dtype=float), eps))
        raw_threshold = float(10 ** np.median(log_threshold))
        assigned_threshold = float(grid[np.argmin(np.abs(np.log10(np.maximum(grid, eps)) - np.log10(max(raw_threshold, eps))))])
        rows.append(
            {
                "site_id": str(test["site_id"]),
                "assigned_threshold": assigned_threshold,
                "true_oracle_threshold": float(test["oracle_threshold"]),
                "nearest_sites": ",".join(neighbors["site_id"].astype(str).tolist()),
                "nearest_distances": ",".join(f"{v:.6g}" for v in neighbors["distance"].tolist()),
                "nearest_thresholds": ",".join(str(v) for v in neighbors["oracle_threshold"].tolist()),
                "k": int(k),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--neighbor-ks", default="1,3")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    samples_path = Path(args.samples)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(pred_path)
    for col in ["should_irrigate", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in pred_df.columns:
            pred_df[col] = bool_series(pred_df[col])
    samples = pd.read_csv(samples_path)
    site_features = build_site_features(samples)
    thresholds = parse_thresholds(args.thresholds)

    site_threshold_rows = []
    for site_id, group in pred_df.groupby("site_id", sort=False):
        summaries = []
        for threshold in thresholds:
            decisions = evaluate_fixed_threshold(group, threshold)
            summaries.append(summarize_decisions(decisions, f"threshold_{threshold}"))
        site_summary = pd.DataFrame(summaries)
        best = site_summary.sort_values(
            ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
            ascending=[True, False],
        ).iloc[0]
        site_threshold_rows.append(
            {
                "site_id": site_id,
                "oracle_threshold": float(str(best["policy"]).replace("threshold_", "")),
                "oracle_mean_regret": float(best["mean_decision_regret_oracle_amount"]),
                "oracle_trigger_accuracy": float(best["trigger_accuracy"]),
            }
        )
    site_oracle = pd.DataFrame(site_threshold_rows)

    policy_summaries = []
    all_policy_decisions = []
    all_assignments = []
    for k in [int(part.strip()) for part in args.neighbor_ks.split(",") if part.strip()]:
        assignments = nearest_site_thresholds(site_features, site_oracle, k=k, threshold_grid=thresholds)
        parts = []
        for _, row in assignments.iterrows():
            site_rows = pred_df.loc[pred_df["site_id"].astype(str) == str(row["site_id"])].copy()
            decisions = evaluate_fixed_threshold(site_rows, float(row["assigned_threshold"]))
            decisions["policy"] = f"nearest_{k}_threshold_transfer"
            decisions["true_oracle_threshold"] = float(row["true_oracle_threshold"])
            decisions["nearest_sites"] = row["nearest_sites"]
            parts.append(decisions)
        policy_decisions = pd.concat(parts, ignore_index=True)
        policy_summaries.append(summarize_decisions(policy_decisions, f"nearest_{k}_threshold_transfer"))
        all_policy_decisions.append(policy_decisions)
        assignments["policy"] = f"nearest_{k}_threshold_transfer"
        all_assignments.append(assignments)

    oracle_parts = []
    for _, row in site_oracle.iterrows():
        site_rows = pred_df.loc[pred_df["site_id"].astype(str) == str(row["site_id"])].copy()
        decisions = evaluate_fixed_threshold(site_rows, float(row["oracle_threshold"]))
        decisions["policy"] = "site_oracle_threshold"
        oracle_parts.append(decisions)
    oracle_decisions = pd.concat(oracle_parts, ignore_index=True)
    policy_summaries.append(summarize_decisions(oracle_decisions, "site_oracle_threshold"))
    all_policy_decisions.append(oracle_decisions)

    all_decisions = pd.concat(all_policy_decisions, ignore_index=True)
    summary = pd.DataFrame(policy_summaries).sort_values("mean_decision_regret_oracle_amount")
    assignments_df = pd.concat(all_assignments, ignore_index=True)
    by_site = (
        all_decisions.groupby(["policy", "site_id"])
        .agg(
            trigger_accuracy=("trigger_correct", "mean"),
            mean_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "mean"),
            max_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "max"),
            assigned_threshold=("assigned_threshold", "first"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["policy", "mean_decision_regret_oracle_amount"], ascending=[True, False])
    )
    worst = all_decisions.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(50)

    summary_path = out_dir / "binary_trigger_threshold_transfer_summary_v1.csv"
    assignments_path = out_dir / "binary_trigger_threshold_transfer_assignments_v1.csv"
    by_site_path = out_dir / "binary_trigger_threshold_transfer_by_site_v1.csv"
    decisions_path = out_dir / "binary_trigger_threshold_transfer_decisions_v1.csv"
    report_path = out_dir / "binary_trigger_threshold_transfer_v1.md"
    summary.to_csv(summary_path, index=False)
    assignments_df.to_csv(assignments_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    all_decisions.to_csv(decisions_path, index=False)

    lines = [
        "# Binary Trigger Threshold Transfer V1",
        "",
        "## Inputs",
        "",
        f"- Predictions: `{pred_path}`",
        f"- Samples: `{samples_path}`",
        "",
        "## Policy Summary",
        "",
        markdown_table(summary),
        "",
        "## Threshold Assignments",
        "",
        markdown_table(assignments_df),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decisions",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{assignments_path}`",
        f"- `{by_site_path}`",
        f"- `{decisions_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Binary trigger threshold transfer v1")
    print(f"summary: {summary_path}")
    print(f"assignments: {assignments_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(assignments_df.to_string(index=False))


if __name__ == "__main__":
    main()

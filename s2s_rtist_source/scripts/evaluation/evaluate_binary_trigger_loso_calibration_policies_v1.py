#!/usr/bin/env python3
"""Evaluate deployable LOSO calibration policies for the binary trigger.

The site-oracle threshold diagnostic showed that the trigger probabilities have
useful ranking signal, but raw global thresholds and simple static threshold
transfer did not beat the paper fixed list. This script tests calibration rules
that do not use labels from the evaluated site:

- choose one global threshold using all other sites;
- transfer the mean/median oracle irrigation rate from all other sites;
- transfer the oracle irrigation rate from nearest static-feature sites.

The rate-transfer policies use the held-out site's unlabeled trigger probability
distribution to convert a target irrigation rate into a probability threshold.
They still evaluate with oracle positive irrigation amount, so this isolates
trigger calibration quality before any positive-amount ranker.
"""

from __future__ import annotations

import argparse
import errno
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table


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
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_binary_trigger_loso_calibration_policies_v1"
)
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
    pred = out["pred_irrigate_prob"].to_numpy(dtype=float) >= float(threshold)
    actual = out["should_irrigate"].to_numpy(dtype=bool)
    chosen_gain = np.where(
        pred,
        out["oracle_positive_true_net_gain"].to_numpy(dtype=float),
        out["zero_true_net_gain"].to_numpy(dtype=float),
    )
    out["assigned_threshold"] = float(threshold)
    out["pred_should_irrigate"] = pred
    out["trigger_correct"] = pred == actual
    out["trigger_decision_regret_oracle_amount"] = out["true_best_net_gain"].to_numpy(dtype=float) - chosen_gain
    out["chosen_ir_oracle_amount"] = np.where(
        pred,
        out["oracle_positive_ir"].to_numpy(dtype=float),
        0.0,
    )
    return out


def summarize_decisions(decisions: pd.DataFrame, policy: str) -> dict:
    pred = decisions["pred_should_irrigate"].astype(bool)
    actual = decisions["should_irrigate"].astype(bool)
    tp = int((pred & actual).sum())
    fp = int((pred & ~actual).sum())
    tn = int((~pred & ~actual).sum())
    fn = int((~pred & actual).sum())
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "policy": policy,
        "trigger_accuracy": float((pred == actual).mean()),
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
        "mean_assigned_threshold": float(decisions["assigned_threshold"].mean()),
    }


def site_feature_columns(df: pd.DataFrame) -> list[str]:
    exact = ["longitude", "latitude", "site_ir_min", "site_ir_max"]
    prefixes = ("static_", "soil_")
    cols = [col for col in df.columns if col in exact or col.startswith(prefixes)]
    out = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.isna().all():
            out.append(col)
    return out


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


def static_distances(site_features: pd.DataFrame, test_site: str, train_sites: set[str]) -> pd.DataFrame:
    feature_cols = [col for col in site_features.columns if col != "site_id"]
    features = site_features.copy()
    features["site_id"] = features["site_id"].astype(str)
    train = features.loc[features["site_id"].isin(train_sites)].copy()
    test = features.loc[features["site_id"] == str(test_site)].copy()
    if train.empty:
        raise ValueError(f"No training site features available for {test_site}")
    if test.empty:
        raise ValueError(f"No test site features available for {test_site}")

    train_x = train[feature_cols].to_numpy(dtype=float)
    test_x = test[feature_cols].to_numpy(dtype=float)
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
    train["distance"] = np.sqrt(np.sum(diff * diff, axis=1))
    return train.sort_values("distance")


def choose_best_threshold(df: pd.DataFrame, thresholds: list[float]) -> tuple[float, dict]:
    summaries = []
    for threshold in thresholds:
        decisions = evaluate_fixed_threshold(df, threshold)
        summaries.append(summarize_decisions(decisions, f"threshold_{threshold}"))
    summary = pd.DataFrame(summaries)
    best = summary.sort_values(
        ["mean_decision_regret_oracle_amount", "trigger_balanced_accuracy"],
        ascending=[True, False],
    ).iloc[0]
    threshold = float(str(best["policy"]).replace("threshold_", ""))
    return threshold, best.to_dict()


def threshold_for_target_rate(site_rows: pd.DataFrame, target_rate: float, thresholds: list[float]) -> tuple[float, float]:
    probs = site_rows["pred_irrigate_prob"].to_numpy(dtype=float)
    rows = []
    for threshold in thresholds:
        rate = float((probs >= float(threshold)).mean())
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_rate": rate,
                "rate_abs_error": abs(rate - float(target_rate)),
            }
        )
    best = pd.DataFrame(rows).sort_values(["rate_abs_error", "threshold"], ascending=[True, True]).iloc[0]
    return float(best["threshold"]), float(best["predicted_rate"])


def compute_site_oracle(pred_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for site_id, group in pred_df.groupby("site_id", sort=False):
        threshold, summary = choose_best_threshold(group, thresholds)
        rows.append(
            {
                "site_id": str(site_id),
                "oracle_threshold": threshold,
                "oracle_mean_regret": float(summary["mean_decision_regret_oracle_amount"]),
                "oracle_trigger_accuracy": float(summary["trigger_accuracy"]),
                "oracle_predicted_irrigation_rate": float(summary["predicted_irrigation_rate"]),
                "true_irrigation_rate": float(summary["true_irrigation_rate"]),
            }
        )
    return pd.DataFrame(rows)


def add_policy_decisions(
    *,
    rows: list[pd.DataFrame],
    assignments: list[dict],
    site_rows: pd.DataFrame,
    policy: str,
    threshold: float,
    target_rate: float | None,
    achieved_rate: float | None,
    note: str,
    nearest_sites: str = "",
    nearest_distances: str = "",
    nearest_rates: str = "",
) -> None:
    decisions = evaluate_fixed_threshold(site_rows, threshold)
    decisions["policy"] = policy
    decisions["target_irrigation_rate"] = np.nan if target_rate is None else float(target_rate)
    decisions["achieved_irrigation_rate"] = np.nan if achieved_rate is None else float(achieved_rate)
    rows.append(decisions)
    assignments.append(
        {
            "site_id": str(site_rows["site_id"].iloc[0]),
            "policy": policy,
            "assigned_threshold": float(threshold),
            "target_irrigation_rate": np.nan if target_rate is None else float(target_rate),
            "achieved_irrigation_rate": np.nan if achieved_rate is None else float(achieved_rate),
            "nearest_sites": nearest_sites,
            "nearest_distances": nearest_distances,
            "nearest_oracle_rates": nearest_rates,
            "note": note,
        }
    )


def write_csv(path: Path, df: pd.DataFrame) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def write_text(path: Path, text: str) -> bool:
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--neighbor-ks", default="1,3,5")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    samples_path = Path(args.samples)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    out_dir = Path(args.output_dir)
    can_write = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print("[warn] No space left on device; will print summary only and skip writing")
            can_write = False
        else:
            raise

    pred_df = pd.read_csv(pred_path)
    for col in ["should_irrigate", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in pred_df.columns:
            pred_df[col] = bool_series(pred_df[col])
    pred_df["site_id"] = pred_df["site_id"].astype(str)
    samples = pd.read_csv(samples_path)
    site_features = build_site_features(samples)
    thresholds = parse_thresholds(args.thresholds)
    neighbor_ks = [int(part.strip()) for part in args.neighbor_ks.split(",") if part.strip()]

    site_oracle = compute_site_oracle(pred_df, thresholds)
    site_oracle_index = site_oracle.set_index("site_id")
    decision_parts: list[pd.DataFrame] = []
    assignment_rows: list[dict] = []

    for test_site, site_rows in pred_df.groupby("site_id", sort=False):
        train_rows = pred_df.loc[pred_df["site_id"] != str(test_site)].copy()
        train_sites = set(train_rows["site_id"].astype(str).unique().tolist())
        train_oracle = site_oracle.loc[site_oracle["site_id"].isin(train_sites)].copy()

        global_threshold, _global_summary = choose_best_threshold(train_rows, thresholds)
        add_policy_decisions(
            rows=decision_parts,
            assignments=assignment_rows,
            site_rows=site_rows,
            policy="other_sites_global_threshold",
            threshold=global_threshold,
            target_rate=None,
            achieved_rate=None,
            note="Best threshold selected on all non-held-out sites.",
        )

        for label, target_rate in [
            ("other_sites_mean_oracle_rate", float(train_oracle["oracle_predicted_irrigation_rate"].mean())),
            ("other_sites_median_oracle_rate", float(train_oracle["oracle_predicted_irrigation_rate"].median())),
        ]:
            threshold, achieved_rate = threshold_for_target_rate(site_rows, target_rate, thresholds)
            add_policy_decisions(
                rows=decision_parts,
                assignments=assignment_rows,
                site_rows=site_rows,
                policy=label,
                threshold=threshold,
                target_rate=target_rate,
                achieved_rate=achieved_rate,
                note="Target irrigation rate estimated from non-held-out site oracles.",
            )

        distances = static_distances(site_features, str(test_site), train_sites)
        train_oracle_by_site = train_oracle.set_index("site_id")
        for k in neighbor_ks:
            neighbors = distances.head(k).copy()
            neighbor_rates = [
                float(train_oracle_by_site.loc[str(site_id), "oracle_predicted_irrigation_rate"])
                for site_id in neighbors["site_id"].astype(str).tolist()
            ]
            target_rate = float(np.median(neighbor_rates))
            threshold, achieved_rate = threshold_for_target_rate(site_rows, target_rate, thresholds)
            add_policy_decisions(
                rows=decision_parts,
                assignments=assignment_rows,
                site_rows=site_rows,
                policy=f"nearest_{k}_oracle_rate_transfer",
                threshold=threshold,
                target_rate=target_rate,
                achieved_rate=achieved_rate,
                nearest_sites=",".join(neighbors["site_id"].astype(str).tolist()),
                nearest_distances=",".join(f"{v:.6g}" for v in neighbors["distance"].tolist()),
                nearest_rates=",".join(f"{v:.6g}" for v in neighbor_rates),
                note="Median oracle irrigation rate transferred from nearest non-held-out static-feature sites.",
            )

        oracle_threshold = float(site_oracle_index.loc[str(test_site), "oracle_threshold"])
        oracle_rate = float(site_oracle_index.loc[str(test_site), "oracle_predicted_irrigation_rate"])
        add_policy_decisions(
            rows=decision_parts,
            assignments=assignment_rows,
            site_rows=site_rows,
            policy="site_oracle_threshold",
            threshold=oracle_threshold,
            target_rate=oracle_rate,
            achieved_rate=oracle_rate,
            note="Diagnostic only; uses held-out site labels.",
        )

    all_decisions = pd.concat(decision_parts, ignore_index=True)
    assignments = pd.DataFrame(assignment_rows)
    summary = pd.DataFrame(
        [summarize_decisions(group, policy) for policy, group in all_decisions.groupby("policy", sort=False)]
    ).sort_values("mean_decision_regret_oracle_amount")
    by_site = (
        all_decisions.groupby(["policy", "site_id"])
        .agg(
            trigger_accuracy=("trigger_correct", "mean"),
            mean_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "mean"),
            max_decision_regret_oracle_amount=("trigger_decision_regret_oracle_amount", "max"),
            assigned_threshold=("assigned_threshold", "first"),
            target_irrigation_rate=("target_irrigation_rate", "first"),
            achieved_irrigation_rate=("achieved_irrigation_rate", "first"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["policy", "mean_decision_regret_oracle_amount"], ascending=[True, False])
    )
    worst = all_decisions.sort_values("trigger_decision_regret_oracle_amount", ascending=False).head(50)

    summary_path = out_dir / "binary_trigger_loso_calibration_policy_summary_v1.csv"
    assignments_path = out_dir / "binary_trigger_loso_calibration_policy_assignments_v1.csv"
    by_site_path = out_dir / "binary_trigger_loso_calibration_policy_by_site_v1.csv"
    decisions_path = out_dir / "binary_trigger_loso_calibration_policy_decisions_v1.csv"
    site_oracle_path = out_dir / "binary_trigger_loso_calibration_policy_site_oracle_v1.csv"
    report_path = out_dir / "binary_trigger_loso_calibration_policies_v1.md"

    lines = [
        "# Binary Trigger LOSO Calibration Policies V1",
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
        "## Site Oracle Rates",
        "",
        markdown_table(site_oracle),
        "",
        "## Assignments",
        "",
        markdown_table(assignments),
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
        f"- `{site_oracle_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(assignments_path, assignments)
        write_csv(by_site_path, by_site)
        write_csv(decisions_path, all_decisions)
        write_csv(site_oracle_path, site_oracle)
        write_text(report_path, report_text)

    print("Binary trigger LOSO calibration policies v1")
    print(f"summary: {summary_path}")
    print(f"assignments: {assignments_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(assignments.to_string(index=False))


if __name__ == "__main__":
    main()

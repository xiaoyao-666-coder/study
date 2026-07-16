#!/usr/bin/env python3
"""Calibrate the collapse threshold for two-head continuous surrogate outputs.

This script does not retrain the model. It reads the two-head prediction table,
replays candidate ranking under several collapse thresholds, and reports regret
and false-collapse/missed-collapse counts. It is a calibration check before
deciding whether to keep the two-head baseline or move to a better classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    markdown_table,
)


DEFAULT_PRED = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_surrogate_twohead_v1"
    / "continuous_irrigation_surrogate_twohead_v1_predictions.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_surrogate_twohead_v1"
)


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def evaluate_threshold(pred: pd.DataFrame, threshold: float) -> tuple[dict, pd.DataFrame]:
    work = pred.copy()
    work["target_collapse"] = bool_series(work["target_collapse"])
    work["same_date_duplicate_target_curve"] = bool_series(work["same_date_duplicate_target_curve"])
    work["candidate_ir"] = pd.to_numeric(work["candidate_ir"], errors="coerce")
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work["collapse_prob"] = pd.to_numeric(work["collapse_prob"], errors="coerce").fillna(0.0)
    work["response_pred_net_gain_7d"] = pd.to_numeric(work["response_pred_net_gain_7d"], errors="coerce")
    work["collapse_cost_pred_net_gain_7d"] = pd.to_numeric(work["collapse_cost_pred_net_gain_7d"], errors="coerce")
    work["pred_net_gain_7d"] = np.where(
        work["collapse_prob"] >= threshold,
        work["collapse_cost_pred_net_gain_7d"],
        work["response_pred_net_gain_7d"],
    )
    work["collapse_selected_by_head"] = work["collapse_prob"] >= threshold

    decisions = []
    for site_date_id, part in work.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
                "threshold": float(threshold),
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "collapse_prob": float(true_best["collapse_prob"]),
                "collapse_selected_by_head": bool(true_best["collapse_selected_by_head"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "true_best_net_gain": float(true_best[TARGET]),
                "pred_best_true_net_gain": float(pred_best[TARGET]),
                "pred_best_pred_net_gain": float(pred_best["pred_net_gain_7d"]),
                "decision_correct": float(true_best["candidate_ir"]) == float(pred_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - pred_best[TARGET]),
            }
        )
    decision_df = pd.DataFrame(decisions)
    metrics = score_metrics(
        work[TARGET].to_numpy(dtype=float),
        work["pred_net_gain_7d"].to_numpy(dtype=float),
    )
    false_collapse = decision_df[(~decision_df["target_collapse"]) & (decision_df["collapse_selected_by_head"])]
    missed_collapse = decision_df[(decision_df["target_collapse"]) & (~decision_df["collapse_selected_by_head"])]
    metrics.update(
        {
            "threshold": float(threshold),
            "decision_correct": int(decision_df["decision_correct"].sum()),
            "decision_total": int(len(decision_df)),
            "decision_accuracy": safe_mean(decision_df["decision_correct"]),
            "mean_decision_regret": float(decision_df["decision_regret"].mean()),
            "median_decision_regret": float(decision_df["decision_regret"].median()),
            "max_decision_regret": float(decision_df["decision_regret"].max()),
            "collapse_decision_accuracy": safe_mean(decision_df.loc[decision_df["target_collapse"], "decision_correct"]),
            "noncollapse_decision_accuracy": safe_mean(decision_df.loc[~decision_df["target_collapse"], "decision_correct"]),
            "false_collapse_noncollapse": int(len(false_collapse)),
            "missed_collapse": int(len(missed_collapse)),
        }
    )
    return metrics, decision_df


def parse_thresholds(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("No thresholds provided")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=str(DEFAULT_PRED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing two-head predictions: {pred_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(pred_path)
    required = [
        "collapse_prob",
        "response_pred_net_gain_7d",
        "collapse_cost_pred_net_gain_7d",
        TARGET,
        "site_date_id",
        "candidate_ir",
    ]
    missing = [col for col in required if col not in pred.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    summaries = []
    decisions = []
    for threshold in parse_thresholds(args.thresholds):
        metrics, decision_df = evaluate_threshold(pred, threshold)
        summaries.append(metrics)
        decisions.append(decision_df)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["mean_decision_regret", "false_collapse_noncollapse", "missed_collapse"]
    )
    decision_all = pd.concat(decisions, ignore_index=True)
    best_threshold = float(summary_df.iloc[0]["threshold"])
    best_decision = decision_all[decision_all["threshold"] == best_threshold].copy()
    worst = best_decision.sort_values("decision_regret", ascending=False).head(15)

    summary_path = out_dir / "continuous_irrigation_surrogate_twohead_threshold_calibration_v1_summary.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_twohead_threshold_calibration_v1_decision_eval.csv"
    report_path = out_dir / "continuous_irrigation_surrogate_twohead_threshold_calibration_v1.md"
    summary_df.to_csv(summary_path, index=False)
    decision_all.to_csv(decision_path, index=False)

    lines = [
        "# Continuous Irrigation Two-Head Threshold Calibration V1",
        "",
        "## Scope",
        "",
        "- Replays two-head predictions under several collapse thresholds.",
        "- Does not retrain the model.",
        "- Selects the lowest mean regret threshold as the smoke calibration candidate.",
        "",
        "## Summary",
        "",
        markdown_table(summary_df),
        "",
        f"Best threshold by mean regret: `{best_threshold}`",
        "",
        "## Worst Rows At Best Threshold",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{decision_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation two-head threshold calibration v1")
    print(f"predictions: {pred_path}")
    print(f"summary: {summary_path}")
    print(f"decision_eval: {decision_path}")
    print(f"report: {report_path}")
    print(summary_df.to_string(index=False))
    print("")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()

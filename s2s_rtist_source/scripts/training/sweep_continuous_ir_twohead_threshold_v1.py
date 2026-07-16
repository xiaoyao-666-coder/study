#!/usr/bin/env python3
"""Sweep the collapse threshold for two-head continuous-irrigation predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "net_gain_7d"


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> dict:
    work = df.copy()
    work["target_collapse"] = bool_series(work["target_collapse"])
    work["candidate_ir"] = pd.to_numeric(work["candidate_ir"], errors="coerce")
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work["collapse_prob"] = pd.to_numeric(work["collapse_prob"], errors="coerce")
    work["response_pred_net_gain_7d"] = pd.to_numeric(work["response_pred_net_gain_7d"], errors="coerce")
    work["collapse_cost_pred_net_gain_7d"] = pd.to_numeric(work["collapse_cost_pred_net_gain_7d"], errors="coerce")
    work["pred_net_gain_7d"] = np.where(
        work["collapse_prob"] >= threshold,
        work["collapse_cost_pred_net_gain_7d"],
        work["response_pred_net_gain_7d"],
    )

    decisions = []
    for site_date_id, part in work.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "collapse_prob": float(true_best["collapse_prob"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "true_best_net_gain": float(true_best[TARGET]),
                "pred_best_true_net_gain": float(pred_best[TARGET]),
                "decision_correct": float(true_best["candidate_ir"]) == float(pred_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - pred_best[TARGET]),
            }
        )
    decision_df = pd.DataFrame(decisions)
    return {
        "collapse_threshold": float(threshold),
        "decision_correct": int(decision_df["decision_correct"].sum()),
        "decision_total": int(len(decision_df)),
        "decision_accuracy": safe_mean(decision_df["decision_correct"]),
        "mean_decision_regret": float(decision_df["decision_regret"].mean()),
        "median_decision_regret": float(decision_df["decision_regret"].median()),
        "max_decision_regret": float(decision_df["decision_regret"].max()),
        "collapse_decision_accuracy": safe_mean(decision_df.loc[decision_df["target_collapse"], "decision_correct"]),
        "noncollapse_decision_accuracy": safe_mean(decision_df.loc[~decision_df["target_collapse"], "decision_correct"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_path)

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
    rows = [evaluate_threshold(df, float(t)) for t in thresholds]
    result = pd.DataFrame(rows).sort_values(
        ["mean_decision_regret", "decision_accuracy"],
        ascending=[True, False],
    )

    out_path = out_dir / "continuous_ir_twohead_threshold_sweep_v1.csv"
    report_path = out_dir / "continuous_ir_twohead_threshold_sweep_v1.md"
    result.to_csv(out_path, index=False)

    top = result.head(10)
    lines = [
        "# Continuous Irrigation Two-Head Threshold Sweep V1",
        "",
        f"- Predictions: `{pred_path}`",
        "- Primary selection metric: lowest mean decision regret.",
        "",
        "## Top Thresholds",
        "",
        markdown_table(top),
        "",
        "## Output",
        "",
        f"- `{out_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation two-head threshold sweep v1")
    print(f"input: {pred_path}")
    print(f"output: {out_path}")
    print(f"report: {report_path}")
    print("")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()

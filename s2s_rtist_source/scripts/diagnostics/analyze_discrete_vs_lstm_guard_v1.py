#!/usr/bin/env python3
"""Analyze when LSTM continuous optimization should override the paper list.

This diagnostic does not retrain a model. It checks whether the LSTM surrogate's
own predicted margin over the paper-style discrete candidate decision is a
usable confidence signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_COMPARISON = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_discrete_vs_lstm_continuous_v1"
    / "continuous_ir_discrete_vs_lstm_continuous_decisions_v1.csv"
)
DEFAULT_DENSE = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_lstm_continuous_optimization_v1"
    / "continuous_ir_lstm_surrogate_dense_predictions_v1.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_discrete_vs_lstm_guard_v1"


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def interp_pred(part: pd.DataFrame, ir: float) -> float:
    tmp = part[["candidate_ir", "pred_net_gain_7d"]].copy()
    tmp["candidate_ir"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    tmp["pred_net_gain_7d"] = pd.to_numeric(tmp["pred_net_gain_7d"], errors="coerce")
    tmp = tmp.dropna().sort_values("candidate_ir")
    return float(np.interp(float(ir), tmp["candidate_ir"].to_numpy(), tmp["pred_net_gain_7d"].to_numpy()))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-decisions", default=str(DEFAULT_COMPARISON))
    parser.add_argument("--dense-predictions", default=str(DEFAULT_DENSE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--thresholds", default="0,1,2,5,10,20,30,50,75,100")
    args = parser.parse_args()

    decisions = pd.read_csv(args.comparison_decisions)
    dense = pd.read_csv(args.dense_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    pred_rows = []
    dense_groups = {str(k): v.copy() for k, v in dense.groupby("site_date_id", sort=False)}
    for row in decisions.itertuples(index=False):
        site_date_id = str(row.site_date_id)
        if site_date_id not in dense_groups:
            raise ValueError(f"Missing dense predictions for {site_date_id}")
        part = dense_groups[site_date_id]
        paper_pred = interp_pred(part, float(row.paper_best_ir))
        lstm_pred = interp_pred(part, float(row.lstm_opt_ir))
        pred_rows.append(
            {
                "site_date_id": site_date_id,
                "paper_pred_gain": paper_pred,
                "lstm_pred_gain": lstm_pred,
                "lstm_pred_margin_over_paper": lstm_pred - paper_pred,
            }
        )

    pred_df = pd.DataFrame(pred_rows)
    df = decisions.merge(pred_df, on="site_date_id", how="left")
    df["paper_regret"] = df["paper_regret_vs_dense_oracle"]
    df["lstm_regret"] = df["lstm_regret_vs_dense_oracle"]
    df["oracle_gate_regret"] = np.minimum(df["paper_regret"], df["lstm_regret"])
    df["lstm_worse_than_paper"] = df["lstm_regret"] > df["paper_regret"]

    overall = pd.DataFrame(
        [
            {
                "site_dates": int(len(df)),
                "paper_mean_regret": float(df["paper_regret"].mean()),
                "lstm_mean_regret": float(df["lstm_regret"].mean()),
                "perfect_gate_mean_regret": float(df["oracle_gate_regret"].mean()),
                "perfect_gate_lstm_use_rate": safe_mean(df["lstm_regret"] < df["paper_regret"]),
                "lstm_worse_than_paper_rate": safe_mean(df["lstm_worse_than_paper"]),
                "mean_pred_margin": float(df["lstm_pred_margin_over_paper"].mean()),
                "median_pred_margin": float(df["lstm_pred_margin_over_paper"].median()),
            }
        ]
    )

    sweep_rows = []
    for threshold in thresholds:
        use_lstm = df["lstm_pred_margin_over_paper"] >= threshold
        regret = np.where(use_lstm, df["lstm_regret"], df["paper_regret"])
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "mean_regret": float(np.mean(regret)),
                "median_regret": float(np.median(regret)),
                "lstm_use_rate": safe_mean(use_lstm),
                "bad_override_rate": safe_mean(use_lstm & df["lstm_worse_than_paper"]),
                "good_override_rate": safe_mean(use_lstm & (df["lstm_regret"] < df["paper_regret"])),
            }
        )
    sweep = pd.DataFrame(sweep_rows).sort_values("mean_regret")

    by_site = (
        df.groupby("site_id")
        .agg(
            paper_mean_regret=("paper_regret", "mean"),
            lstm_mean_regret=("lstm_regret", "mean"),
            perfect_gate_mean_regret=("oracle_gate_regret", "mean"),
            mean_pred_margin=("lstm_pred_margin_over_paper", "mean"),
            lstm_worse_than_paper_rate=("lstm_worse_than_paper", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("lstm_mean_regret", ascending=False)
    )

    bad_high_margin = (
        df.loc[df["lstm_worse_than_paper"]]
        .sort_values(["lstm_pred_margin_over_paper", "lstm_regret"], ascending=False)
        .head(30)
    )
    good_overrides = (
        df.loc[df["lstm_regret"] < df["paper_regret"]]
        .sort_values("paper_regret", ascending=False)
        .head(30)
    )

    annotated_path = out_dir / "continuous_ir_discrete_vs_lstm_guard_annotated_decisions_v1.csv"
    overall_path = out_dir / "continuous_ir_discrete_vs_lstm_guard_overall_v1.csv"
    sweep_path = out_dir / "continuous_ir_discrete_vs_lstm_guard_threshold_sweep_v1.csv"
    by_site_path = out_dir / "continuous_ir_discrete_vs_lstm_guard_by_site_v1.csv"
    report_path = out_dir / "continuous_ir_discrete_vs_lstm_guard_v1.md"
    df.to_csv(annotated_path, index=False)
    overall.to_csv(overall_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Discrete vs LSTM Guard Diagnostic V1",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## Threshold Sweep",
        "",
        markdown_table(sweep),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Bad High-Margin LSTM Overrides",
        "",
        markdown_table(bad_high_margin),
        "",
        "## Good LSTM Overrides",
        "",
        markdown_table(good_overrides),
        "",
        "## Outputs",
        "",
        f"- `{annotated_path}`",
        f"- `{overall_path}`",
        f"- `{sweep_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Discrete vs LSTM guard diagnostic v1")
    print(f"overall: {overall_path}")
    print(f"sweep: {sweep_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(overall.to_string(index=False))
    print("")
    print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose LSTM ranking errors against the strong fixed-list baseline.

The previous guard diagnostic showed that the LSTM predicted margin is not a
safe override signal. This script separates two possible failure modes:

1. Continuous-search failure: dense-grid argmax moves outside a good fixed-list
   decision.
2. Surrogate-ranking failure: even within the paper fixed list, the LSTM
   surrogate ranks candidates poorly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    candidate_set_for_site,
    interp_gain,
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_SAMPLES = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
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
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_lstm_ranking_diagnostic_v1"

TARGET = "net_gain_7d"


def interp_pred(part: pd.DataFrame, ir: float) -> float:
    tmp = part[["candidate_ir", "pred_net_gain_7d"]].copy()
    tmp["candidate_ir"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    tmp["pred_net_gain_7d"] = pd.to_numeric(tmp["pred_net_gain_7d"], errors="coerce")
    tmp = tmp.dropna().sort_values("candidate_ir")
    return float(np.interp(float(ir), tmp["candidate_ir"].to_numpy(), tmp["pred_net_gain_7d"].to_numpy()))


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--comparison-decisions", default=str(DEFAULT_COMPARISON))
    parser.add_argument("--dense-predictions", default=str(DEFAULT_DENSE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    comparison = pd.read_csv(args.comparison_decisions)
    dense = pd.read_csv(args.dense_predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_candidates = parse_candidates(args.paper_candidates)

    dense_groups = {str(k): v.copy() for k, v in dense.groupby("site_date_id", sort=False)}
    rows = []
    for site_date_id, curve in samples.groupby("site_date_id", sort=False):
        site_date_id = str(site_date_id)
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        true_best = curve.loc[curve[TARGET].idxmax()]
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        dense_part = dense_groups[site_date_id]
        compare_row = comparison.loc[comparison["site_date_id"] == site_date_id]
        if compare_row.empty:
            raise ValueError(f"Missing comparison row for {site_date_id}")
        compare_row = compare_row.iloc[0]

        paper_values = candidate_set_for_site(site_ir_max, paper_candidates)
        paper_pred_scores = [(ir, interp_pred(dense_part, ir)) for ir in paper_values]
        paper_lstm_ir, paper_lstm_pred_gain = max(paper_pred_scores, key=lambda item: item[1])
        paper_lstm_true_gain = interp_gain(curve, paper_lstm_ir)

        dense_lstm_ir = float(compare_row["lstm_opt_ir"])
        dense_lstm_pred_gain = interp_pred(dense_part, dense_lstm_ir)
        dense_lstm_true_gain = float(compare_row["lstm_opt_interp_gain"])
        paper_oracle_ir = float(compare_row["paper_best_ir"])
        paper_oracle_true_gain = float(compare_row["paper_best_interp_gain"])
        paper_oracle_pred_gain = interp_pred(dense_part, paper_oracle_ir)
        dense_oracle_gain = float(compare_row["dense_oracle_gain"])

        dense_part = dense_part.copy()
        dense_part["pred_rank"] = dense_part["pred_net_gain_7d"].rank(method="min", ascending=False)
        nearest_true_ir_idx = (pd.to_numeric(dense_part["candidate_ir"], errors="coerce") - float(true_best["candidate_ir"])).abs().idxmin()
        true_best_pred_rank = float(dense_part.loc[nearest_true_ir_idx, "pred_rank"])

        rows.append(
            {
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "site_ir_max": site_ir_max,
                "dense_oracle_ir": float(compare_row["dense_oracle_ir"]),
                "dense_oracle_gain": dense_oracle_gain,
                "paper_oracle_ir": paper_oracle_ir,
                "paper_oracle_true_gain": paper_oracle_true_gain,
                "paper_oracle_pred_gain": paper_oracle_pred_gain,
                "paper_oracle_regret": dense_oracle_gain - paper_oracle_true_gain,
                "lstm_fixed_list_ir": float(paper_lstm_ir),
                "lstm_fixed_list_pred_gain": float(paper_lstm_pred_gain),
                "lstm_fixed_list_true_gain": float(paper_lstm_true_gain),
                "lstm_fixed_list_regret": dense_oracle_gain - paper_lstm_true_gain,
                "lstm_dense_ir": dense_lstm_ir,
                "lstm_dense_pred_gain": dense_lstm_pred_gain,
                "lstm_dense_true_gain": dense_lstm_true_gain,
                "lstm_dense_regret": dense_oracle_gain - dense_lstm_true_gain,
                "true_best_pred_rank_in_dense_grid": true_best_pred_rank,
                "dense_minus_fixed_lstm_regret": (dense_oracle_gain - dense_lstm_true_gain) - (dense_oracle_gain - paper_lstm_true_gain),
                "fixed_lstm_minus_paper_oracle_regret": (dense_oracle_gain - paper_lstm_true_gain) - (dense_oracle_gain - paper_oracle_true_gain),
                "paper_oracle_pred_margin_over_lstm_fixed": paper_oracle_pred_gain - paper_lstm_pred_gain,
            }
        )

    df = pd.DataFrame(rows)
    overall = pd.DataFrame(
        [
            {
                "site_dates": int(len(df)),
                "paper_oracle_mean_regret": float(df["paper_oracle_regret"].mean()),
                "lstm_fixed_list_mean_regret": float(df["lstm_fixed_list_regret"].mean()),
                "lstm_dense_mean_regret": float(df["lstm_dense_regret"].mean()),
                "dense_search_added_regret_vs_lstm_fixed": float(df["dense_minus_fixed_lstm_regret"].mean()),
                "lstm_fixed_added_regret_vs_paper_oracle": float(df["fixed_lstm_minus_paper_oracle_regret"].mean()),
                "lstm_fixed_better_than_paper_oracle_rate": safe_mean(df["lstm_fixed_list_regret"] < df["paper_oracle_regret"]),
                "lstm_dense_better_than_lstm_fixed_rate": safe_mean(df["lstm_dense_regret"] < df["lstm_fixed_list_regret"]),
                "median_true_best_pred_rank": float(df["true_best_pred_rank_in_dense_grid"].median()),
                "mean_true_best_pred_rank": float(df["true_best_pred_rank_in_dense_grid"].mean()),
                "pred_true_gain_corr_dense_choice": safe_corr(df["lstm_dense_pred_gain"], df["lstm_dense_true_gain"]),
                "pred_true_gain_corr_paper_oracle": safe_corr(df["paper_oracle_pred_gain"], df["paper_oracle_true_gain"]),
            }
        ]
    )

    by_site = (
        df.groupby("site_id")
        .agg(
            paper_oracle_mean_regret=("paper_oracle_regret", "mean"),
            lstm_fixed_list_mean_regret=("lstm_fixed_list_regret", "mean"),
            lstm_dense_mean_regret=("lstm_dense_regret", "mean"),
            mean_true_best_pred_rank=("true_best_pred_rank_in_dense_grid", "mean"),
            dense_search_added_regret_vs_lstm_fixed=("dense_minus_fixed_lstm_regret", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("lstm_dense_mean_regret", ascending=False)
    )

    worst_fixed = df.sort_values("lstm_fixed_list_regret", ascending=False).head(30)
    worst_dense_extra = df.sort_values("dense_minus_fixed_lstm_regret", ascending=False).head(30)
    collapse_false_positive = (
        df.loc[df["target_collapse"] & (df["lstm_dense_ir"] > 0)]
        .sort_values("lstm_dense_regret", ascending=False)
        .head(30)
    )

    decisions_path = out_dir / "continuous_ir_lstm_ranking_diagnostic_decisions_v1.csv"
    overall_path = out_dir / "continuous_ir_lstm_ranking_diagnostic_overall_v1.csv"
    by_site_path = out_dir / "continuous_ir_lstm_ranking_diagnostic_by_site_v1.csv"
    report_path = out_dir / "continuous_ir_lstm_ranking_diagnostic_v1.md"
    df.to_csv(decisions_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# LSTM Ranking vs Discrete Policy Diagnostic V1",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst LSTM Fixed-List Ranking Errors",
        "",
        markdown_table(worst_fixed),
        "",
        "## Worst Extra Regret From Dense Search",
        "",
        markdown_table(worst_dense_extra),
        "",
        "## Collapse False Positives",
        "",
        markdown_table(collapse_false_positive),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{overall_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("LSTM ranking vs discrete policy diagnostic v1")
    print(f"overall: {overall_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(overall.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

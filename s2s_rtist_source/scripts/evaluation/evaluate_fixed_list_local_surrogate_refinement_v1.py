#!/usr/bin/env python3
"""Evaluate surrogate-guided local refinement around the paper fixed list.

The previous oracle diagnostic showed how much regret can be removed if a true
SWAP oracle searches only a small neighborhood around the paper fixed-list
anchor. This script asks the next deployability question without retraining:
can the existing dense LSTM surrogate predictions choose a better local amount,
and can a predicted margin safely guard when to leave the fixed-list anchor?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    TARGET,
    candidate_set_for_site,
    interp_gain,
    parse_candidates,
)
from diagnose_fixed_list_local_refinement_headroom_v1 import (
    float_list,
    interpolated_grid_best,
    parse_bool,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_SAMPLES = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_DENSE = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_lstm_continuous_optimization_v1"
    / "continuous_ir_lstm_surrogate_dense_predictions_v1.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_fixed_list_local_surrogate_refinement_v1"


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def interp_pred(part: pd.DataFrame, ir: float) -> float:
    tmp = part[["candidate_ir", "pred_net_gain_7d"]].copy()
    tmp["candidate_ir"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    tmp["pred_net_gain_7d"] = pd.to_numeric(tmp["pred_net_gain_7d"], errors="coerce")
    tmp = tmp.dropna().sort_values("candidate_ir")
    return float(np.interp(float(ir), tmp["candidate_ir"].to_numpy(), tmp["pred_net_gain_7d"].to_numpy()))


def local_predicted_best(dense_part: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    part = dense_part.copy()
    part["candidate_ir"] = pd.to_numeric(part["candidate_ir"], errors="coerce")
    part["pred_net_gain_7d"] = pd.to_numeric(part["pred_net_gain_7d"], errors="coerce")
    part = part.dropna(subset=["candidate_ir", "pred_net_gain_7d"])
    local = part.loc[(part["candidate_ir"] >= lower - 1e-9) & (part["candidate_ir"] <= upper + 1e-9)]
    if local.empty:
        raise ValueError(f"No dense prediction rows inside local window [{lower}, {upper}]")
    return local.loc[local["pred_net_gain_7d"].idxmax()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--dense-predictions", default=str(DEFAULT_DENSE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--windows-mm", default="1,2.5,5,10")
    parser.add_argument("--oracle-grid-step-mm", type=float, default=0.5)
    parser.add_argument("--thresholds", default="-50,-20,-10,-5,-2,-1,0,1,2,5,10,20,50,100")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    dense_path = Path(args.dense_predictions)
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not dense_path.exists():
        raise FileNotFoundError(f"Missing dense predictions file: {dense_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(samples_path)
    dense = pd.read_csv(dense_path)
    required_samples = {"site_date_id", "site_id", "date_t", "candidate_ir", "site_ir_max", TARGET}
    required_dense = {"site_date_id", "candidate_ir", "pred_net_gain_7d"}
    missing_samples = sorted(required_samples.difference(samples.columns))
    missing_dense = sorted(required_dense.difference(dense.columns))
    if missing_samples:
        raise ValueError(f"Missing sample columns: {missing_samples}")
    if missing_dense:
        raise ValueError(f"Missing dense prediction columns: {missing_dense}")

    paper_candidates = parse_candidates(args.paper_candidates)
    windows = float_list(args.windows_mm)
    thresholds = parse_thresholds(args.thresholds)
    dense_groups = {str(k): v.copy() for k, v in dense.groupby("site_date_id", sort=False)}

    rows = []
    for site_date_id, curve in samples.groupby("site_date_id", sort=False):
        site_date_id = str(site_date_id)
        if site_date_id not in dense_groups:
            raise ValueError(f"Missing dense predictions for site_date_id={site_date_id}")
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        dense_part = dense_groups[site_date_id]

        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        dense_oracle_ir = float(dense_oracle["candidate_ir"])
        dense_oracle_gain = float(dense_oracle[TARGET])
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        paper_values = candidate_set_for_site(site_ir_max, paper_candidates)
        paper_scores = [(ir, interp_gain(curve, ir)) for ir in paper_values]
        paper_ir, paper_gain = max(paper_scores, key=lambda item: item[1])
        paper_regret = dense_oracle_gain - paper_gain
        paper_pred_gain = interp_pred(dense_part, paper_ir)

        for window in windows:
            lower = max(0.0, paper_ir - window)
            upper = min(site_ir_max, paper_ir + window)
            local_oracle_ir, local_oracle_gain = interpolated_grid_best(
                curve, lower, upper, args.oracle_grid_step_mm
            )
            local_oracle_regret = dense_oracle_gain - local_oracle_gain
            pred_best = local_predicted_best(dense_part, lower, upper)
            local_pred_ir = float(pred_best["candidate_ir"])
            local_pred_gain = interp_gain(curve, local_pred_ir)
            local_pred_regret = dense_oracle_gain - local_pred_gain
            local_pred_pred_gain = float(pred_best["pred_net_gain_7d"])
            pred_margin = local_pred_pred_gain - paper_pred_gain
            rows.append(
                {
                    "site_date_id": site_date_id,
                    "site_id": str(dense_oracle["site_id"]),
                    "date_t": str(dense_oracle["date_t"]),
                    "target_collapse": parse_bool(dense_oracle["target_collapse"])
                    if "target_collapse" in dense_oracle
                    else False,
                    "window_mm": float(window),
                    "site_ir_max": site_ir_max,
                    "paper_best_ir": float(paper_ir),
                    "paper_true_gain": float(paper_gain),
                    "paper_pred_gain": paper_pred_gain,
                    "paper_regret": float(paper_regret),
                    "local_oracle_ir": float(local_oracle_ir),
                    "local_oracle_true_gain": float(local_oracle_gain),
                    "local_oracle_regret": float(local_oracle_regret),
                    "local_pred_ir": local_pred_ir,
                    "local_pred_true_gain": float(local_pred_gain),
                    "local_pred_pred_gain": local_pred_pred_gain,
                    "local_pred_regret": float(local_pred_regret),
                    "local_pred_margin_over_paper": float(pred_margin),
                    "dense_oracle_ir": dense_oracle_ir,
                    "dense_oracle_gain": dense_oracle_gain,
                    "local_oracle_improvement_over_paper": float(paper_regret - local_oracle_regret),
                    "local_pred_improvement_over_paper": float(paper_regret - local_pred_regret),
                    "local_pred_changed_ir": abs(local_pred_ir - float(paper_ir)) > 1e-9,
                    "local_pred_better_than_paper": local_pred_regret < paper_regret - 1e-9,
                    "local_pred_worse_than_paper": local_pred_regret > paper_regret + 1e-9,
                    "local_pred_reaches_dense_oracle": abs(local_pred_regret) <= 1e-9,
                }
            )

    decisions = pd.DataFrame(rows)
    summary = (
        decisions.groupby("window_mm")
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            local_pred_mean_regret=("local_pred_regret", "mean"),
            local_oracle_mean_improvement=("local_oracle_improvement_over_paper", "mean"),
            local_pred_mean_improvement=("local_pred_improvement_over_paper", "mean"),
            local_pred_better_rate=("local_pred_better_than_paper", "mean"),
            local_pred_worse_rate=("local_pred_worse_than_paper", "mean"),
            local_pred_changed_ir_rate=("local_pred_changed_ir", "mean"),
            local_pred_reaches_dense_oracle_rate=("local_pred_reaches_dense_oracle", "mean"),
            mean_pred_margin=("local_pred_margin_over_paper", "mean"),
            median_pred_margin=("local_pred_margin_over_paper", "median"),
        )
        .reset_index()
        .sort_values("window_mm")
    )

    sweep_rows = []
    for window, part in decisions.groupby("window_mm", sort=True):
        for threshold in thresholds:
            use_local = part["local_pred_margin_over_paper"] >= threshold
            guarded_regret = np.where(use_local, part["local_pred_regret"], part["paper_regret"])
            sweep_rows.append(
                {
                    "window_mm": float(window),
                    "threshold": float(threshold),
                    "mean_regret": float(np.mean(guarded_regret)),
                    "median_regret": float(np.median(guarded_regret)),
                    "local_use_rate": safe_mean(use_local),
                    "good_override_rate": safe_mean(use_local & part["local_pred_better_than_paper"]),
                    "bad_override_rate": safe_mean(use_local & part["local_pred_worse_than_paper"]),
                    "paper_mean_regret": float(part["paper_regret"].mean()),
                    "local_pred_mean_regret": float(part["local_pred_regret"].mean()),
                    "local_oracle_mean_regret": float(part["local_oracle_regret"].mean()),
                }
            )
    sweep = pd.DataFrame(sweep_rows).sort_values(["mean_regret", "window_mm", "threshold"])
    best_guard = sweep.iloc[0]
    best_window = float(best_guard["window_mm"])
    best_threshold = float(best_guard["threshold"])
    best_decisions = decisions.loc[decisions["window_mm"] == best_window].copy()
    best_decisions["guarded_use_local"] = best_decisions["local_pred_margin_over_paper"] >= best_threshold
    best_decisions["guarded_regret"] = np.where(
        best_decisions["guarded_use_local"],
        best_decisions["local_pred_regret"],
        best_decisions["paper_regret"],
    )

    by_site = (
        best_decisions.groupby("site_id")
        .agg(
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            local_pred_mean_regret=("local_pred_regret", "mean"),
            guarded_mean_regret=("guarded_regret", "mean"),
            guarded_use_local_rate=("guarded_use_local", "mean"),
            local_pred_better_rate=("local_pred_better_than_paper", "mean"),
            local_pred_worse_rate=("local_pred_worse_than_paper", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("guarded_mean_regret", ascending=False)
    )
    bad_overrides = best_decisions.loc[
        best_decisions["guarded_use_local"] & best_decisions["local_pred_worse_than_paper"]
    ].sort_values("guarded_regret", ascending=False).head(40)
    good_overrides = best_decisions.loc[
        best_decisions["guarded_use_local"] & best_decisions["local_pred_better_than_paper"]
    ].sort_values("paper_regret", ascending=False).head(40)

    decisions_path = out_dir / "fixed_list_local_surrogate_refinement_decisions_v1.csv"
    summary_path = out_dir / "fixed_list_local_surrogate_refinement_summary_v1.csv"
    sweep_path = out_dir / "fixed_list_local_surrogate_refinement_threshold_sweep_v1.csv"
    by_site_path = out_dir / "fixed_list_local_surrogate_refinement_by_site_v1.csv"
    report_path = out_dir / "fixed_list_local_surrogate_refinement_v1.md"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Fixed-List Local Surrogate Refinement V1",
        "",
        "## Inputs",
        "",
        f"- Samples: `{samples_path}`",
        f"- Dense predictions: `{dense_path}`",
        f"- Paper candidates: `{args.paper_candidates}`",
        f"- Windows mm: `{args.windows_mm}`",
        f"- Oracle grid step mm: `{args.oracle_grid_step_mm}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Margin Guard Sweep",
        "",
        markdown_table(sweep),
        "",
        "## By Site At Best Guard",
        "",
        markdown_table(by_site),
        "",
        "## Bad Guarded Overrides At Best Guard",
        "",
        markdown_table(bad_overrides),
        "",
        "## Good Guarded Overrides At Best Guard",
        "",
        markdown_table(good_overrides),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{summary_path}`",
        f"- `{sweep_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Fixed-list local surrogate refinement v1")
    print(f"summary: {summary_path}")
    print(f"sweep: {sweep_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(sweep.head(20).to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

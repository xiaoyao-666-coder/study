#!/usr/bin/env python3
"""Evaluate plateau-aware amount policies for expanded learned-trigger runs.

The worst-date diagnostics show that several mistakes happen when the tree
prediction curve is nearly flat around the peak. This script keeps the learned
trigger decision unchanged and only changes the irrigation amount selected
inside the predicted plateau.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")


TOLERANCES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def load_inputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples_path = out_dir / "shortterm_surrogate_expanded_samples_v1.csv"
    pred_path = out_dir / "surrogate_tree_nosklearn_expanded_v1_predictions.csv"
    learned_path = out_dir / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"

    for path in [samples_path, pred_path, learned_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    samples = pd.read_csv(samples_path)
    pred = pd.read_csv(pred_path)
    learned = pd.read_csv(learned_path)
    learned = learned[learned["amount_policy"] == "raw_tree_peak"].copy()

    df = samples.merge(pred[["sample_id", "pred_net_gain_7d"]], on="sample_id", how="left")
    if df["pred_net_gain_7d"].isna().any():
        raise RuntimeError("Some samples did not match tree predictions.")
    return df, learned


def choose_plateau(group: pd.DataFrame, tolerance: float, tie_rule: str) -> pd.Series:
    max_pred = float(group["pred_net_gain_7d"].max())
    plateau = group[group["pred_net_gain_7d"] >= max_pred - tolerance].sort_values("candidate_ir").reset_index(drop=True)
    if plateau.empty:
        raise RuntimeError("Empty plateau")

    if tie_rule == "raw_peak":
        return group.loc[group["pred_net_gain_7d"].idxmax()]
    if tie_rule == "low":
        return plateau.iloc[0]
    if tie_rule == "high":
        return plateau.iloc[-1]
    if tie_rule == "middle":
        return plateau.iloc[len(plateau) // 2]
    if tie_rule == "closest_to_raw":
        raw = group.loc[group["pred_net_gain_7d"].idxmax()]
        raw_ir = float(raw["candidate_ir"])
        idx = (plateau["candidate_ir"].astype(float) - raw_ir).abs().idxmin()
        return plateau.loc[idx]
    raise ValueError(tie_rule)


def evaluate_policy(df: pd.DataFrame, learned: pd.DataFrame, policy: str, tolerance: float, tie_rule: str) -> pd.DataFrame:
    rows = []
    for _, trigger_row in learned.sort_values("decision_doy").iterrows():
        date_t = trigger_row["date_t"]
        group = df[df["date_t"] == date_t].sort_values("candidate_ir").reset_index(drop=True)
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        zero = group[group["candidate_ir"].astype(float) == 0.0].iloc[0]

        if not bool(trigger_row["triggered"]):
            chosen = zero
            choice_source = "trigger_zero"
            plateau_size = 0
        else:
            chosen = choose_plateau(group, tolerance, tie_rule)
            choice_source = tie_rule
            max_pred = float(group["pred_net_gain_7d"].max())
            plateau_size = int((group["pred_net_gain_7d"] >= max_pred - tolerance).sum())

        rows.append(
            {
                "policy": policy,
                "tolerance": tolerance,
                "tie_rule": tie_rule,
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "triggered": bool(trigger_row["triggered"]),
                "trigger_prob": float(trigger_row["trigger_prob"]),
                "trigger_threshold": float(trigger_row["trigger_threshold"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "chosen_true_net_gain": float(chosen["net_gain_7d"]),
                "chosen_pred_net_gain": float(chosen["pred_net_gain_7d"]),
                "decision_correct": float(chosen["candidate_ir"]) == float(true_best["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
                "plateau_size": plateau_size,
                "choice_source": choice_source,
            }
        )
    return pd.DataFrame(rows)


def summarize(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in decisions.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_dates": len(group),
                "top1_accuracy": float(group["decision_correct"].mean()),
                "near_correct_regret_le_1": float((group["decision_regret"] <= 1.0).mean()),
                "near_correct_regret_le_3": float((group["decision_regret"] <= 3.0).mean()),
                "near_correct_regret_le_5": float((group["decision_regret"] <= 5.0).mean()),
                "mean_regret": float(group["decision_regret"].mean()),
                "median_regret": float(group["decision_regret"].median()),
                "max_regret": float(group["decision_regret"].max()),
                "total_regret": float(group["decision_regret"].sum()),
                "avg_plateau_size": float(group["plateau_size"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_regret", "max_regret", "total_regret"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Expanded surrogate result directory.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    decision_out = out_dir / "expanded_plateau_amount_policy_v1_decision_eval.csv"
    summary_out = out_dir / "expanded_plateau_amount_policy_v1_comparison.csv"
    report_out = out_dir / "expanded_plateau_amount_policy_v1.txt"

    df, learned = load_inputs(out_dir)
    parts = []

    parts.append(evaluate_policy(df, learned, "raw_peak_baseline", 0.0, "raw_peak"))
    for tolerance in TOLERANCES:
        for tie_rule in ["low", "middle", "high", "closest_to_raw"]:
            policy = f"plateau_{tie_rule}_tol_{tolerance:g}"
            parts.append(evaluate_policy(df, learned, policy, tolerance, tie_rule))

    decisions = pd.concat(parts, ignore_index=True)
    summary = summarize(decisions)
    decisions.to_csv(decision_out, index=False)
    summary.to_csv(summary_out, index=False)

    best = summary.head(12)
    lines = [
        "Expanded plateau amount policy v1",
        "",
        "Trigger: learned trigger from learned_trigger_curve_policy_expanded_v1",
        "Amount: choose within predicted peak plateau when triggered",
        "",
        "Best policies by mean regret:",
        best.to_string(index=False),
        "",
        f"wrote: {decision_out}",
        f"wrote: {summary_out}",
    ]
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

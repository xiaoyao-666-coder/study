#!/usr/bin/env python3
"""Evaluate feature-based early-stage irrigation caps for expanded policies.

This is an exploratory diagnostic policy, not a final recommended rule. It
tests whether early crop-stage states like 18-Jul-2024 should avoid high
irrigation amounts when the tree prediction curve overestimates them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")
SAMPLES = OUT_DIR / "shortterm_surrogate_expanded_samples_v1.csv"
PRED = OUT_DIR / "surrogate_tree_nosklearn_expanded_v1_predictions.csv"
LEARNED = OUT_DIR / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"

DECISION_OUT = OUT_DIR / "expanded_stage_cap_policy_v1_decision_eval.csv"
SUMMARY_OUT = OUT_DIR / "expanded_stage_cap_policy_v1_comparison.csv"
REPORT_OUT = OUT_DIR / "expanded_stage_cap_policy_v1.txt"

DVS_THRESHOLDS = [1.18, 1.20, 1.22, 1.25, 1.30]
DOY_THRESHOLDS = [200, 202, 204, 206]
CAPS = [10.0, 15.0, 20.0]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in [SAMPLES, PRED, LEARNED]:
        if not path.exists():
            raise FileNotFoundError(path)
    samples = pd.read_csv(SAMPLES)
    pred = pd.read_csv(PRED)
    learned = pd.read_csv(LEARNED)
    learned = learned[learned["amount_policy"] == "raw_tree_peak"].copy()
    df = samples.merge(pred[["sample_id", "pred_net_gain_7d"]], on="sample_id", how="left")
    if df["pred_net_gain_7d"].isna().any():
        raise RuntimeError("Some samples did not match tree predictions.")
    return df, learned


def choose_middle_plateau(group: pd.DataFrame, tolerance: float = 1.0) -> pd.Series:
    max_pred = float(group["pred_net_gain_7d"].max())
    plateau = group[group["pred_net_gain_7d"] >= max_pred - tolerance].sort_values("candidate_ir").reset_index(drop=True)
    return plateau.iloc[len(plateau) // 2]


def choose_with_cap(group: pd.DataFrame, cap: float, tolerance: float = 1.0) -> pd.Series:
    capped = group[group["candidate_ir"].astype(float) <= cap].copy()
    if capped.empty:
        capped = group.copy()
    return choose_middle_plateau(capped, tolerance=tolerance)


def evaluate(df: pd.DataFrame, learned: pd.DataFrame, policy: str, condition: str, threshold: float | None, cap: float | None) -> pd.DataFrame:
    rows = []
    for _, trigger_row in learned.sort_values("decision_doy").iterrows():
        date_t = trigger_row["date_t"]
        group = df[df["date_t"] == date_t].sort_values("candidate_ir").reset_index(drop=True)
        first = group.iloc[0]
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        zero = group[group["candidate_ir"].astype(float) == 0.0].iloc[0]

        cap_active = False
        if not bool(trigger_row["triggered"]):
            chosen = zero
            source = "trigger_zero"
        else:
            if condition == "none":
                chosen = choose_middle_plateau(group, tolerance=1.0)
                source = "plateau_middle"
            elif condition == "dvs_le":
                cap_active = float(first["state_dvs"]) <= float(threshold)
                chosen = choose_with_cap(group, float(cap)) if cap_active else choose_middle_plateau(group)
                source = f"dvs_le_{threshold}_cap_{cap}" if cap_active else "plateau_middle"
            elif condition == "doy_le":
                cap_active = int(first["decision_doy"]) <= int(threshold)
                chosen = choose_with_cap(group, float(cap)) if cap_active else choose_middle_plateau(group)
                source = f"doy_le_{threshold}_cap_{cap}" if cap_active else "plateau_middle"
            else:
                raise ValueError(condition)

        rows.append(
            {
                "policy": policy,
                "condition": condition,
                "threshold": threshold,
                "cap": cap,
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "state_dvs": float(first["state_dvs"]),
                "cap_active": cap_active,
                "triggered": bool(trigger_row["triggered"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "chosen_true_net_gain": float(chosen["net_gain_7d"]),
                "decision_correct": float(chosen["candidate_ir"]) == float(true_best["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
                "choice_source": source,
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
                "cap_active_dates": int(group["cap_active"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_regret", "max_regret"]).reset_index(drop=True)


def main() -> None:
    df, learned = load_inputs()
    parts = [evaluate(df, learned, "plateau_middle_tol_1_baseline", "none", None, None)]

    for threshold in DVS_THRESHOLDS:
        for cap in CAPS:
            policy = f"stage_cap_dvs_le_{threshold:g}_cap_{cap:g}"
            parts.append(evaluate(df, learned, policy, "dvs_le", threshold, cap))

    for threshold in DOY_THRESHOLDS:
        for cap in CAPS:
            policy = f"stage_cap_doy_le_{threshold:g}_cap_{cap:g}"
            parts.append(evaluate(df, learned, policy, "doy_le", threshold, cap))

    decisions = pd.concat(parts, ignore_index=True)
    summary = summarize(decisions)
    decisions.to_csv(DECISION_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    lines = [
        "Expanded stage-cap policy v1",
        "",
        "Base amount policy: learned trigger + plateau_middle_tol_1",
        "Exploratory guards: cap candidate irrigation when early DVS or early DOY condition holds.",
        "",
        "Best policies by mean regret:",
        summary.head(15).to_string(index=False),
        "",
        f"wrote: {DECISION_OUT}",
        f"wrote: {SUMMARY_OUT}",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

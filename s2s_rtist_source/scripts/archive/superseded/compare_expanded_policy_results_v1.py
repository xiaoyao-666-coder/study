#!/usr/bin/env python3
"""Compare expanded surrogate decision policies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")
TREE = OUT_DIR / "surrogate_tree_nosklearn_expanded_v1_decision_eval.csv"
LEARNED = OUT_DIR / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"
SUMMARY = OUT_DIR / "expanded_policy_comparison_v1.csv"
WORST = OUT_DIR / "expanded_policy_worst_dates_v1.csv"
REPORT = OUT_DIR / "expanded_policy_comparison_v1.txt"


def load_tree() -> pd.DataFrame:
    df = pd.read_csv(TREE)
    return pd.DataFrame(
        {
            "policy": "tree_raw",
            "date_t": df["date_t"],
            "decision_doy": df["decision_doy"],
            "true_best_ir": df["true_best_ir"],
            "chosen_ir": df["pred_best_ir"],
            "true_best_net_gain": df["true_best_net_gain"],
            "chosen_true_net_gain": df["pred_best_true_net_gain"],
            "decision_correct": df["decision_correct"],
            "decision_regret": df["decision_regret"],
        }
    )


def load_learned() -> pd.DataFrame:
    df = pd.read_csv(LEARNED)
    df = df[df["amount_policy"] == "raw_tree_peak"].copy()
    return pd.DataFrame(
        {
            "policy": "learned_trigger_raw_tree",
            "date_t": df["date_t"],
            "decision_doy": df["decision_doy"],
            "true_best_ir": df["true_best_ir"],
            "chosen_ir": df["chosen_ir"],
            "true_best_net_gain": df["true_best_net_gain"],
            "chosen_true_net_gain": df["chosen_true_net_gain"],
            "decision_correct": df["decision_correct"],
            "decision_regret": df["decision_regret"],
        }
    )


def main() -> None:
    if not TREE.exists():
        raise FileNotFoundError(TREE)
    if not LEARNED.exists():
        raise FileNotFoundError(LEARNED)

    all_decisions = pd.concat([load_tree(), load_learned()], ignore_index=True)
    rows = []
    for policy, group in all_decisions.groupby("policy", sort=False):
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
                "avg_chosen_gain": float(group["chosen_true_net_gain"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)

    worst = all_decisions.sort_values(["policy", "decision_regret"], ascending=[True, False])
    worst.to_csv(WORST, index=False)

    lines = [
        "Expanded policy comparison v1",
        "",
        summary.to_string(index=False),
        "",
        "Worst dates by policy:",
        worst.groupby("policy").head(6).to_string(index=False),
        "",
        f"wrote: {SUMMARY}",
        f"wrote: {WORST}",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

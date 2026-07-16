#!/usr/bin/env python3
"""Compare expanded policies with teacher-facing irrigation metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")
TREE = OUT_DIR / "surrogate_tree_nosklearn_expanded_v1_decision_eval.csv"
LEARNED = OUT_DIR / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"
PLATEAU = OUT_DIR / "expanded_plateau_amount_policy_v1_decision_eval.csv"
STAGE_CAP = OUT_DIR / "expanded_stage_cap_policy_v1_decision_eval.csv"

SUMMARY = OUT_DIR / "expanded_policy_comparison_v3.csv"
WORST = OUT_DIR / "expanded_policy_worst_dates_v3.csv"
REPORT = OUT_DIR / "expanded_policy_comparison_v3.txt"


def from_common(
    df: pd.DataFrame,
    *,
    policy: str,
    chosen_col: str,
    chosen_gain_col: str,
    correct_col: str,
    regret_col: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy": policy,
            "date_t": df["date_t"],
            "decision_doy": df["decision_doy"],
            "true_best_ir": df["true_best_ir"],
            "chosen_ir": df[chosen_col],
            "true_best_net_gain": df["true_best_net_gain"],
            "chosen_true_net_gain": df[chosen_gain_col],
            "decision_correct": df[correct_col],
            "decision_regret": df[regret_col],
        }
    )


def load_tree() -> pd.DataFrame:
    df = pd.read_csv(TREE)
    return from_common(
        df,
        policy="tree_raw",
        chosen_col="pred_best_ir",
        chosen_gain_col="pred_best_true_net_gain",
        correct_col="decision_correct",
        regret_col="decision_regret",
    )


def load_learned() -> pd.DataFrame:
    df = pd.read_csv(LEARNED)
    df = df[df["amount_policy"] == "raw_tree_peak"].copy()
    return from_common(
        df,
        policy="learned_trigger_raw_tree",
        chosen_col="chosen_ir",
        chosen_gain_col="chosen_true_net_gain",
        correct_col="decision_correct",
        regret_col="decision_regret",
    )


def load_plateau() -> pd.DataFrame:
    df = pd.read_csv(PLATEAU)
    df = df[df["policy"] == "plateau_middle_tol_1"].copy()
    return from_common(
        df,
        policy="learned_trigger_plateau_middle_tol_1",
        chosen_col="chosen_ir",
        chosen_gain_col="chosen_true_net_gain",
        correct_col="decision_correct",
        regret_col="decision_regret",
    )


def load_stage_cap(policy_name: str, out_name: str) -> pd.DataFrame:
    df = pd.read_csv(STAGE_CAP)
    df = df[df["policy"] == policy_name].copy()
    return from_common(
        df,
        policy=out_name,
        chosen_col="chosen_ir",
        chosen_gain_col="chosen_true_net_gain",
        correct_col="decision_correct",
        regret_col="decision_regret",
    )


def summarize(all_decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in all_decisions.groupby("policy", sort=False):
        ir_error = group["chosen_ir"] - group["true_best_ir"]
        gain_diff = group["chosen_true_net_gain"] - group["true_best_net_gain"]
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
                "mean_ir_error": float(ir_error.mean()),
                "mean_abs_ir_error": float(ir_error.abs().mean()),
                "max_abs_ir_error": float(ir_error.abs().max()),
                "total_true_best_ir": float(group["true_best_ir"].sum()),
                "total_chosen_ir": float(group["chosen_ir"].sum()),
                "total_ir_diff": float(group["chosen_ir"].sum() - group["true_best_ir"].sum()),
                "mean_net_gain_diff": float(gain_diff.mean()),
                "total_net_gain_diff": float(gain_diff.sum()),
                "avg_chosen_gain": float(group["chosen_true_net_gain"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_regret", "max_regret"]).reset_index(drop=True)


def main() -> None:
    for path in [TREE, LEARNED, PLATEAU, STAGE_CAP]:
        if not path.exists():
            raise FileNotFoundError(path)

    all_decisions = pd.concat(
        [
            load_tree(),
            load_learned(),
            load_plateau(),
            load_stage_cap("stage_cap_dvs_le_1.3_cap_20", "learned_trigger_plateau_stage_cap_dvs_le_1.3_cap_20"),
            load_stage_cap("stage_cap_doy_le_204_cap_20", "learned_trigger_plateau_stage_cap_doy_le_204_cap_20"),
        ],
        ignore_index=True,
    )
    summary = summarize(all_decisions)
    summary.to_csv(SUMMARY, index=False)

    worst = all_decisions.sort_values(["policy", "decision_regret"], ascending=[True, False])
    worst.to_csv(WORST, index=False)

    best = summary.iloc[0]
    lines = [
        "Expanded policy comparison v3",
        "",
        "Includes teacher-facing irrigation metrics:",
        "- optimal irrigation amount error",
        "- total irrigation difference",
        "- net gain difference / regret",
        "",
        summary.to_string(index=False),
        "",
        "Best current policy:",
        best.to_string(),
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

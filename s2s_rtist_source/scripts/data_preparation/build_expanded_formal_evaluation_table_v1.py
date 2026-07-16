#!/usr/bin/env python3
"""Build a formal expanded policy evaluation table.

This table follows the teacher-facing metric categories that are available in
the current single-site expanded dataset:
  - optimal irrigation amount difference
  - total irrigation difference
  - net gain difference / regret
  - DVS/CWDM/CWSO outcome deviation from the SWAP true-best candidate

Scope note: DVS/CWDM/CWSO here are decision-outcome deviations against the
true-best SWAP candidate, not standalone surrogate prediction errors.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")
SAMPLES = OUT_DIR / "shortterm_surrogate_expanded_samples_v1.csv"
TREE = OUT_DIR / "surrogate_tree_nosklearn_expanded_v1_decision_eval.csv"
LEARNED = OUT_DIR / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"
PLATEAU = OUT_DIR / "expanded_plateau_amount_policy_v1_decision_eval.csv"
STAGE_CAP = OUT_DIR / "expanded_stage_cap_policy_v1_decision_eval.csv"

DETAILS_OUT = OUT_DIR / "expanded_formal_policy_decision_details_v1.csv"
SUMMARY_OUT = OUT_DIR / "expanded_formal_policy_evaluation_v1.csv"
REPORT_OUT = OUT_DIR / "expanded_formal_policy_evaluation_v1.md"


OUTCOME_COLS = ["dvs_7d", "cwdm_7d", "cwso_7d", "target_7d", "net_gain_7d"]
POLICY_PRIORITY = {
    "learned_trigger_plateau_stage_cap_dvs_le_1.3_cap_20": 0,
    "learned_trigger_plateau_stage_cap_doy_le_204_cap_20": 1,
    "learned_trigger_plateau_middle_tol_1": 2,
    "learned_trigger_raw_tree": 3,
    "tree_raw": 4,
}


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(format_value(v) for v in row) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_policy_decisions() -> pd.DataFrame:
    tree = pd.read_csv(TREE)
    learned = pd.read_csv(LEARNED)
    plateau = pd.read_csv(PLATEAU)
    stage = pd.read_csv(STAGE_CAP)

    rows = []
    for _, r in tree.iterrows():
        rows.append({"policy": "tree_raw", "date_t": r["date_t"], "chosen_ir": float(r["pred_best_ir"])})
    for _, r in learned[learned["amount_policy"] == "raw_tree_peak"].iterrows():
        rows.append({"policy": "learned_trigger_raw_tree", "date_t": r["date_t"], "chosen_ir": float(r["chosen_ir"])})
    for _, r in plateau[plateau["policy"] == "plateau_middle_tol_1"].iterrows():
        rows.append({"policy": "learned_trigger_plateau_middle_tol_1", "date_t": r["date_t"], "chosen_ir": float(r["chosen_ir"])})
    for policy_name, out_name in [
        ("stage_cap_dvs_le_1.3_cap_20", "learned_trigger_plateau_stage_cap_dvs_le_1.3_cap_20"),
        ("stage_cap_doy_le_204_cap_20", "learned_trigger_plateau_stage_cap_doy_le_204_cap_20"),
    ]:
        sub = stage[stage["policy"] == policy_name]
        for _, r in sub.iterrows():
            rows.append({"policy": out_name, "date_t": r["date_t"], "chosen_ir": float(r["chosen_ir"])})
    return pd.DataFrame(rows)


def build_details(samples: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    required = {"date_t", "candidate_ir", "best_ir_for_date", "best_target_for_date", "is_best_ir", *OUTCOME_COLS}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"Samples missing required columns: {missing}")

    chosen_lookup = samples[
        ["date_t", "decision_doy", "candidate_ir", *OUTCOME_COLS]
    ].rename(
        columns={
            "candidate_ir": "chosen_ir",
            "dvs_7d": "chosen_dvs_7d",
            "cwdm_7d": "chosen_cwdm_7d",
            "cwso_7d": "chosen_cwso_7d",
            "target_7d": "chosen_target_7d",
            "net_gain_7d": "chosen_net_gain_7d",
        }
    )
    best_lookup = samples[samples["is_best_ir"]].copy()
    best_lookup = best_lookup[
        ["date_t", "decision_doy", "candidate_ir", *OUTCOME_COLS]
    ].rename(
        columns={
            "candidate_ir": "true_best_ir",
            "dvs_7d": "true_best_dvs_7d",
            "cwdm_7d": "true_best_cwdm_7d",
            "cwso_7d": "true_best_cwso_7d",
            "target_7d": "true_best_target_7d",
            "net_gain_7d": "true_best_net_gain_7d",
        }
    )

    details = decisions.merge(chosen_lookup, on=["date_t", "chosen_ir"], how="left")
    details = details.merge(best_lookup, on=["date_t", "decision_doy"], how="left")
    if details["chosen_net_gain_7d"].isna().any():
        raise RuntimeError("Some policy choices did not match sample candidate rows.")

    details["ir_error"] = details["chosen_ir"] - details["true_best_ir"]
    details["abs_ir_error"] = details["ir_error"].abs()
    details["target_diff"] = details["chosen_target_7d"] - details["true_best_target_7d"]
    details["net_gain_diff"] = details["chosen_net_gain_7d"] - details["true_best_net_gain_7d"]
    details["decision_regret"] = -details["net_gain_diff"]
    for col in ["dvs_7d", "cwdm_7d", "cwso_7d"]:
        details[f"{col}_diff"] = details[f"chosen_{col}"] - details[f"true_best_{col}"]
        details[f"abs_{col}_diff"] = details[f"{col}_diff"].abs()
    details["decision_correct"] = details["chosen_ir"] == details["true_best_ir"]
    return details.sort_values(["policy", "decision_doy"]).reset_index(drop=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, g in details.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "n_dates": len(g),
                "top1_accuracy": float(g["decision_correct"].mean()),
                "mean_ir_error": float(g["ir_error"].mean()),
                "mean_abs_ir_error": float(g["abs_ir_error"].mean()),
                "max_abs_ir_error": float(g["abs_ir_error"].max()),
                "total_true_best_ir": float(g["true_best_ir"].sum()),
                "total_chosen_ir": float(g["chosen_ir"].sum()),
                "total_ir_diff": float(g["ir_error"].sum()),
                "mean_regret": float(g["decision_regret"].mean()),
                "median_regret": float(g["decision_regret"].median()),
                "max_regret": float(g["decision_regret"].max()),
                "total_regret": float(g["decision_regret"].sum()),
                "mean_target_diff": float(g["target_diff"].mean()),
                "total_target_diff": float(g["target_diff"].sum()),
                "mean_net_gain_diff": float(g["net_gain_diff"].mean()),
                "total_net_gain_diff": float(g["net_gain_diff"].sum()),
                "mean_abs_dvs_diff": float(g["abs_dvs_7d_diff"].mean()),
                "max_abs_dvs_diff": float(g["abs_dvs_7d_diff"].max()),
                "mean_abs_cwdm_diff": float(g["abs_cwdm_7d_diff"].mean()),
                "max_abs_cwdm_diff": float(g["abs_cwdm_7d_diff"].max()),
                "mean_abs_cwso_diff": float(g["abs_cwso_7d_diff"].mean()),
                "max_abs_cwso_diff": float(g["abs_cwso_7d_diff"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["_policy_priority"] = out["policy"].map(POLICY_PRIORITY).fillna(999)
    out = out.sort_values(["mean_regret", "max_regret", "_policy_priority"]).drop(columns=["_policy_priority"])
    return out.reset_index(drop=True)


def write_report(summary: pd.DataFrame, details: pd.DataFrame) -> None:
    best = summary.iloc[0]
    worst = details.sort_values(["policy", "decision_regret"], ascending=[True, False])
    report = [
        "# Expanded Formal Policy Evaluation V1",
        "",
        "## Scope",
        "",
        "This formal table evaluates the current single-site expanded decision policies.",
        "",
        "Available teacher-facing metrics:",
        "",
        "- Optimal irrigation amount difference.",
        "- Total irrigation amount difference.",
        "- Net gain difference / decision regret.",
        "- DVS/CWDM/CWSO outcome deviation from the SWAP true-best candidate.",
        "",
        "Important limitation:",
        "",
        "The DVS/CWDM/CWSO columns here are decision-outcome deviations, not universal surrogate prediction errors. Leave-one-site universal-model evaluation still needs separate multi-site data.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Best Current Policy",
        "",
        best.to_string(),
        "",
        "## Worst Dates By Policy",
        "",
        markdown_table(worst.groupby("policy").head(5)),
        "",
        "## Output Files",
        "",
        f"- `{SUMMARY_OUT}`",
        f"- `{DETAILS_OUT}`",
    ]
    REPORT_OUT.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    for path in [SAMPLES, TREE, LEARNED, PLATEAU, STAGE_CAP]:
        if not path.exists():
            raise FileNotFoundError(path)

    samples = pd.read_csv(SAMPLES)
    decisions = load_policy_decisions()
    details = build_details(samples, decisions)
    summary = summarize(details)
    details.to_csv(DETAILS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    write_report(summary, details)

    print("Expanded formal policy evaluation v1")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"wrote: {SUMMARY_OUT}")
    print(f"wrote: {DETAILS_OUT}")
    print(f"wrote: {REPORT_OUT}")


if __name__ == "__main__":
    main()

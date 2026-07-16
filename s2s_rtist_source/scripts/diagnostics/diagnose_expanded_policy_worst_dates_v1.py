#!/usr/bin/env python3
"""Diagnose largest-regret dates for expanded irrigation policies.

This script is meant to run after:
  1. train_shortterm_surrogate_tree_nosklearn_expanded_v1.py
  2. evaluate_learned_trigger_curve_policy_expanded_v1.py
  3. compare_expanded_policy_results_v1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUT_DIR = Path("Maize_shortterm_surrogate_expanded_v1")


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
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
        return f"{value:.3f}"
    return str(value)


def load_inputs(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples_path = out_dir / "shortterm_surrogate_expanded_samples_v1.csv"
    pred_path = out_dir / "surrogate_tree_nosklearn_expanded_v1_predictions.csv"
    learned_path = out_dir / "learned_trigger_curve_policy_expanded_v1_decision_eval.csv"

    for path in [samples_path, pred_path, learned_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    samples = pd.read_csv(samples_path)
    pred = pd.read_csv(pred_path)
    learned = pd.read_csv(learned_path)
    return samples, pred, learned


def build_candidate_table(samples: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    keep_pred = pred[["sample_id", "pred_net_gain_7d"]].copy()
    df = samples.merge(keep_pred, on="sample_id", how="left")
    if df["pred_net_gain_7d"].isna().any():
        missing = int(df["pred_net_gain_7d"].isna().sum())
        raise RuntimeError(f"{missing} samples did not match tree predictions.")

    df["true_rank"] = df.groupby("date_t")["net_gain_7d"].rank(ascending=False, method="first")
    df["pred_rank"] = df.groupby("date_t")["pred_net_gain_7d"].rank(ascending=False, method="first")
    df["pred_error"] = df["pred_net_gain_7d"] - df["net_gain_7d"]
    df["abs_pred_error"] = df["pred_error"].abs()
    return df


def pick_target_dates(learned: pd.DataFrame, top_n: int) -> list[str]:
    raw = learned[learned["amount_policy"] == "raw_tree_peak"].copy()
    raw = raw.sort_values(["decision_regret", "decision_doy"], ascending=[False, True])
    return raw["date_t"].head(top_n).tolist()


def classify_choice(chosen_ir: float, true_ir: float, regret: float) -> str:
    if regret <= 1e-9:
        return "correct"
    if regret <= 3.0:
        return "near miss"
    if chosen_ir > true_ir:
        return "over-irrigation"
    if chosen_ir < true_ir:
        return "under-irrigation"
    return "same amount but lower gain"


def diagnose_date(date_t: str, candidates: pd.DataFrame, learned_raw: pd.DataFrame) -> dict:
    group = candidates[candidates["date_t"] == date_t].sort_values("candidate_ir").reset_index(drop=True)
    row = learned_raw[learned_raw["date_t"] == date_t].iloc[0]

    true_best = group.loc[group["net_gain_7d"].idxmax()]
    tree_best = group.loc[group["pred_net_gain_7d"].idxmax()]
    chosen_ir = float(row["chosen_ir"])
    chosen = group[group["candidate_ir"].astype(float) == chosen_ir]
    chosen_gain = float(row["chosen_true_net_gain"]) if chosen.empty else float(chosen.iloc[0]["net_gain_7d"])
    zero = group[group["candidate_ir"].astype(float) == 0.0].iloc[0]

    true_ir = float(true_best["candidate_ir"])
    tree_ir = float(tree_best["candidate_ir"])
    true_gain = float(true_best["net_gain_7d"])
    tree_gain = float(tree_best["net_gain_7d"])
    regret = float(row["decision_regret"])

    pred_true = float(true_best["pred_net_gain_7d"])
    pred_tree = float(tree_best["pred_net_gain_7d"])
    pred_gap_tree_minus_true = pred_tree - pred_true
    true_gap_true_minus_tree = true_gain - tree_gain

    near = group.copy()
    near["regret_vs_true_best"] = true_gain - near["net_gain_7d"]
    near = near.sort_values("regret_vs_true_best")
    near_correct_count = int((near["regret_vs_true_best"] <= 5.0).sum())

    if bool(row["triggered"]):
        trigger_note = "triggered; amount follows raw tree peak"
    else:
        trigger_note = "not triggered; forced 0 mm"

    if not bool(row["triggered"]) and tree_ir != 0.0 and chosen_ir == 0.0:
        trigger_effect = "trigger blocked tree irrigation"
    elif bool(row["triggered"]):
        trigger_effect = "trigger allowed tree amount"
    else:
        trigger_effect = "trigger selected zero"

    if regret <= 1e-9:
        main_cause = "no remaining error"
    elif not bool(row["triggered"]) and true_ir > 0.0:
        main_cause = "false negative trigger"
    elif bool(row["triggered"]) and tree_ir != true_ir:
        main_cause = "tree ranked the wrong irrigation amount highest"
    else:
        main_cause = "policy mismatch"

    return {
        "date_t": date_t,
        "decision_doy": int(row["decision_doy"]),
        "true_best_ir": true_ir,
        "learned_chosen_ir": chosen_ir,
        "tree_raw_ir": tree_ir,
        "true_best_gain": round(true_gain, 3),
        "chosen_true_gain": round(chosen_gain, 3),
        "zero_true_gain": round(float(zero["net_gain_7d"]), 3),
        "decision_regret": round(regret, 3),
        "choice_type": classify_choice(chosen_ir, true_ir, regret),
        "trigger_prob": round(float(row["trigger_prob"]), 3),
        "trigger_threshold": round(float(row["trigger_threshold"]), 3),
        "triggered": bool(row["triggered"]),
        "trigger_note": trigger_note,
        "trigger_effect": trigger_effect,
        "main_cause": main_cause,
        "pred_gap_tree_minus_true": round(pred_gap_tree_minus_true, 3),
        "true_gap_true_minus_tree": round(true_gap_true_minus_tree, 3),
        "near_correct_count_regret_le_5": near_correct_count,
    }


def build_report(summary: pd.DataFrame, candidates: pd.DataFrame, target_dates: list[str], out_dir: Path) -> str:
    sections = [
        "# Expanded Policy Worst-Date Diagnostics",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Candidate Curves",
        "",
    ]

    curve_cols = [
        "date_t",
        "candidate_ir",
        "net_gain_7d",
        "pred_net_gain_7d",
        "true_rank",
        "pred_rank",
        "pred_error",
        "is_true_best",
        "is_tree_peak",
        "is_learned_choice",
    ]
    for date_t in target_dates:
        sections.append(f"### {date_t}")
        sections.append("")
        sub = candidates[candidates["date_t"] == date_t].sort_values("candidate_ir")
        sections.append(markdown_table(sub[curve_cols]))
        sections.append("")

    sections.extend(
        [
            "## Outputs",
            "",
            f"- Summary: `{out_dir / 'expanded_policy_worst_date_diagnostics_v1.csv'}`",
            f"- Candidate curves: `{out_dir / 'expanded_policy_worst_date_candidates_v1.csv'}`",
        ]
    )
    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    samples, pred, learned = load_inputs(out_dir)
    candidates = build_candidate_table(samples, pred)
    learned_raw = learned[learned["amount_policy"] == "raw_tree_peak"].copy()
    target_dates = pick_target_dates(learned, args.top_n)

    summary_rows = [diagnose_date(date_t, candidates, learned_raw) for date_t in target_dates]
    summary = pd.DataFrame(summary_rows)

    mark = learned_raw[["date_t", "chosen_ir"]].rename(columns={"chosen_ir": "learned_chosen_ir"})
    candidates = candidates.merge(mark, on="date_t", how="left")
    candidates["is_true_best"] = candidates["true_rank"] == 1.0
    candidates["is_tree_peak"] = candidates["pred_rank"] == 1.0
    candidates["is_learned_choice"] = candidates["candidate_ir"].astype(float) == candidates["learned_chosen_ir"].astype(float)
    candidates = candidates[candidates["date_t"].isin(target_dates)].copy()
    candidates = candidates.sort_values(["decision_doy", "candidate_ir"]).reset_index(drop=True)

    summary_out = out_dir / "expanded_policy_worst_date_diagnostics_v1.csv"
    candidates_out = out_dir / "expanded_policy_worst_date_candidates_v1.csv"
    report_out = out_dir / "expanded_policy_worst_date_diagnostics_v1.md"

    summary.to_csv(summary_out, index=False)
    candidates.to_csv(candidates_out, index=False)
    report_out.write_text(build_report(summary, candidates, target_dates, out_dir), encoding="utf-8")

    print("Expanded policy worst-date diagnostics v1")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(f"wrote: {summary_out}")
    print(f"wrote: {candidates_out}")
    print(f"wrote: {report_out}")


if __name__ == "__main__":
    main()

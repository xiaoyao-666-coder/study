#!/usr/bin/env python3
"""Evaluate conservative irrigation-selection policies on surrogate predictions.

The baseline model predicts candidate net gain. Raw argmax often over-selects
high irrigation. This evaluator keeps the same predictions but chooses the
smallest irrigation candidate whose predicted score is within a threshold of the
predicted best score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "confirmed_5site_true_input_surrogate_baseline_v1_6dates_prestate"
    / "confirmed_5site_true_input_surrogate_baseline_v1_predictions.csv"
)
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "confirmed_5site_true_input_surrogate_policy_v1_6dates_prestate"
DEFAULT_THRESHOLDS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]


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


def select_candidate(group: pd.DataFrame, threshold: float) -> pd.Series:
    ranked = group.sort_values(["pred_net_gain_7d", "candidate_ir"], ascending=[False, True])
    max_pred = float(ranked["pred_net_gain_7d"].iloc[0])
    close = group[group["pred_net_gain_7d"] >= max_pred - threshold].sort_values("candidate_ir")
    return close.iloc[0]


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for site_date_id, group in df.groupby("site_date_id", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        chosen = select_candidate(group, threshold)
        rows.append(
            {
                "threshold": threshold,
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "same_date_duplicate_target_curve": bool(true_best["same_date_duplicate_target_curve"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "chosen_ir": float(chosen["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "chosen_true_net_gain": float(chosen["net_gain_7d"]),
                "chosen_pred_net_gain": float(chosen["pred_net_gain_7d"]),
                "decision_correct": float(true_best["candidate_ir"]) == float(chosen["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
            }
        )
    return pd.DataFrame(rows)


def summarize(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold, group in decision_df.groupby("threshold", sort=True):
        collapse = group[group["target_collapse"]]
        noncollapse = group[~group["target_collapse"]]
        rows.append(
            {
                "threshold": threshold,
                "decision_correct": int(group["decision_correct"].sum()),
                "decision_total": int(len(group)),
                "decision_accuracy": float(group["decision_correct"].mean()),
                "mean_decision_regret": float(group["decision_regret"].mean()),
                "median_decision_regret": float(group["decision_regret"].median()),
                "max_decision_regret": float(group["decision_regret"].max()),
                "collapse_decision_accuracy": float(collapse["decision_correct"].mean()) if not collapse.empty else float("nan"),
                "noncollapse_decision_accuracy": float(noncollapse["decision_correct"].mean()) if not noncollapse.empty else float("nan"),
                "mean_chosen_ir": float(group["chosen_ir"].mean()),
                "zero_choice_rate": float((group["chosen_ir"] == 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_site(decision_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (threshold, site_id), group in decision_df.groupby(["threshold", "site_id"], sort=True):
        collapse = group[group["target_collapse"]]
        noncollapse = group[~group["target_collapse"]]
        rows.append(
            {
                "threshold": threshold,
                "site_id": site_id,
                "decision_correct": int(group["decision_correct"].sum()),
                "decision_total": int(len(group)),
                "decision_accuracy": float(group["decision_correct"].mean()),
                "mean_decision_regret": float(group["decision_regret"].mean()),
                "median_decision_regret": float(group["decision_regret"].median()),
                "max_decision_regret": float(group["decision_regret"].max()),
                "collapse_decision_accuracy": float(collapse["decision_correct"].mean()) if not collapse.empty else float("nan"),
                "noncollapse_decision_accuracy": float(noncollapse["decision_correct"].mean()) if not noncollapse.empty else float("nan"),
                "mean_chosen_ir": float(group["chosen_ir"].mean()),
                "zero_choice_rate": float((group["chosen_ir"] == 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    pred_path = Path(args.input)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {pred_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    required = {
        "site_date_id",
        "site_id",
        "date_t",
        "decision_doy",
        "candidate_ir",
        "net_gain_7d",
        "pred_net_gain_7d",
        "target_collapse",
        "same_date_duplicate_target_curve",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in ["candidate_ir", "net_gain_7d", "pred_net_gain_7d"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target_collapse"] = bool_series(df["target_collapse"])
    df["same_date_duplicate_target_curve"] = bool_series(df["same_date_duplicate_target_curve"])

    decisions = pd.concat([evaluate_threshold(df, t) for t in args.thresholds], ignore_index=True)
    summary = summarize(decisions)
    by_site = summarize_by_site(decisions)
    best = summary.sort_values(["mean_decision_regret", "decision_accuracy"], ascending=[True, False]).head(1)
    best_by_site = by_site.sort_values(["site_id", "mean_decision_regret", "decision_accuracy"], ascending=[True, True, False]).groupby("site_id").head(1)

    decision_path = out_dir / "confirmed_5site_true_input_surrogate_policy_v1_decision_eval.csv"
    summary_path = out_dir / "confirmed_5site_true_input_surrogate_policy_v1_summary.csv"
    by_site_path = out_dir / "confirmed_5site_true_input_surrogate_policy_v1_by_site.csv"
    report_path = out_dir / "confirmed_5site_true_input_surrogate_policy_v1.md"
    decisions.to_csv(decision_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# Confirmed 5-Site True-Input Surrogate Policy V1",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Best By Mean Regret",
        "",
        markdown_table(best),
        "",
        "## Best By Site",
        "",
        markdown_table(best_by_site),
        "",
        "## Outputs",
        "",
        f"- `{decision_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site true-input surrogate policy v1")
    print(f"input: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print(summary.to_string(index=False))
    print("")
    print("best_by_mean_regret:")
    print(best.to_string(index=False))
    print("")
    print("best_by_site:")
    print(best_by_site.to_string(index=False))


if __name__ == "__main__":
    main()

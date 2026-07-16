#!/usr/bin/env python3
"""Diagnose true/predicted irrigation-response curves for tree surrogate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_v1")
PRED = OUT_DIR / "surrogate_tree_nosklearn_v1_predictions.csv"
CURVES = OUT_DIR / "surrogate_tree_nosklearn_v1_curve_diagnostics.csv"
SUMMARY = OUT_DIR / "surrogate_tree_nosklearn_v1_curve_summary.csv"
REPORT = OUT_DIR / "surrogate_tree_nosklearn_v1_curve_diagnostics.md"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    if not PRED.exists():
        raise FileNotFoundError(f"Missing prediction file: {PRED}")

    df = pd.read_csv(PRED)
    df["true_rank"] = df.groupby("date_t")["net_gain_7d"].rank(ascending=False, method="first")
    df["pred_rank"] = df.groupby("date_t")["pred_net_gain_7d"].rank(ascending=False, method="first")
    df["pred_error"] = df["pred_net_gain_7d"] - df["net_gain_7d"]
    df["abs_error"] = df["pred_error"].abs()

    curve_cols = [
        "date_t",
        "decision_doy",
        "candidate_ir",
        "net_gain_7d",
        "pred_net_gain_7d",
        "pred_error",
        "true_rank",
        "pred_rank",
        "is_best_ir",
    ]
    curves = df[curve_cols].sort_values(["decision_doy", "candidate_ir"]).reset_index(drop=True)
    curves.to_csv(CURVES, index=False)

    rows = []
    for date_t, group in df.groupby("date_t", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        pred_best = group.loc[group["pred_net_gain_7d"].idxmax()]
        corr = group[["net_gain_7d", "pred_net_gain_7d"]].corr().iloc[0, 1]
        rows.append(
            {
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "true_best_gain": round(float(true_best["net_gain_7d"]), 3),
                "pred_best_true_gain": round(float(pred_best["net_gain_7d"]), 3),
                "regret": round(float(true_best["net_gain_7d"] - pred_best["net_gain_7d"]), 3),
                "curve_corr": round(float(corr), 3) if pd.notna(corr) else None,
                "mean_abs_error": round(float(group["abs_error"].mean()), 3),
                "max_abs_error": round(float(group["abs_error"].max()), 3),
                "diagnosis": "",
            }
        )

    summary = pd.DataFrame(rows)
    for i, row in summary.iterrows():
        if row["regret"] == 0:
            note = "correct"
        elif row["regret"] <= 3:
            note = "near miss"
        elif row["pred_best_ir"] > row["true_best_ir"]:
            note = "over-irrigation choice"
        elif row["pred_best_ir"] < row["true_best_ir"]:
            note = "under-irrigation choice"
        else:
            note = "wrong ranking"
        summary.loc[i, "diagnosis"] = note
    summary.to_csv(SUMMARY, index=False)

    worst = summary.sort_values("regret", ascending=False).head(5)
    report = [
        "# Tree Surrogate Curve Diagnostics",
        "",
        f"- Prediction file: `{PRED}`",
        f"- Curve diagnostics: `{CURVES}`",
        f"- Summary: `{SUMMARY}`",
        "",
        "## Per-Date Summary",
        "",
        markdown_table(summary),
        "",
        "## Largest Regret Dates",
        "",
        markdown_table(worst),
        "",
        "## How To Read",
        "",
        "- `curve_corr`: correlation between the 8-point true and predicted irrigation-response curves for one date.",
        "- `regret`: true best net gain minus the true net gain of the model-selected irrigation.",
        "- `under-irrigation choice`: model selected less water than the true best.",
        "- `over-irrigation choice`: model selected more water than the true best.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {CURVES}")
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {REPORT}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

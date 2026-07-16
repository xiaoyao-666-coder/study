#!/usr/bin/env python3
"""Diagnose LOSO site-generalization failures for continuous irrigation models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "net_gain_7d"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def static_feature_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col.startswith("static_") or col in {
            "longitude",
            "latitude",
            "site_ir_min",
            "site_ir_max",
        }:
            candidates.append(col)
    numeric = []
    for col in candidates:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            numeric.append(col)
    return numeric


def compute_static_novelty(site_static: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for site_id, row in site_static.iterrows():
        train = site_static.drop(index=site_id)
        z_parts = []
        for col in cols:
            train_values = pd.to_numeric(train[col], errors="coerce")
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(value) or train_values.notna().sum() < 2:
                continue
            mean = float(train_values.mean())
            std = float(train_values.std(ddof=0))
            if std <= 1e-12:
                continue
            z_parts.append(((float(value) - mean) / std) ** 2)
        rows.append(
            {
                "site_id": site_id,
                "static_feature_count": len(z_parts),
                "leave_one_site_static_rms_z": float(np.sqrt(np.mean(z_parts))) if z_parts else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--decision-eval", required=True)
    parser.add_argument("--by-site", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    pred = pd.read_csv(args.predictions)
    decision = pd.read_csv(args.decision_eval)
    by_site = pd.read_csv(args.by_site)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static_cols = static_feature_columns(samples)
    site_static = samples.sort_values(["site_id"]).groupby("site_id", sort=False).head(1)
    site_static = site_static.set_index("site_id")
    novelty = compute_static_novelty(site_static, static_cols)

    by_site_diag = by_site.merge(novelty, on="site_id", how="left")
    by_site_diag = by_site_diag.sort_values("mean_decision_regret", ascending=False)

    worst = decision.sort_values("decision_regret", ascending=False).head(args.top_n)
    pred_cols = [
        col
        for col in [
            "site_date_id",
            "site_id",
            "date_t",
            "candidate_ir",
            TARGET,
            "target_7d",
            "target_collapse",
            "same_date_duplicate_target_curve",
            "pred_net_gain_7d",
        ]
        if col in pred.columns
    ]
    worst_curves = pred.loc[pred["site_date_id"].isin(worst["site_date_id"]), pred_cols].copy()
    worst_curves["_site_date_order"] = worst_curves["site_date_id"].map(
        {site_date: i for i, site_date in enumerate(worst["site_date_id"])}
    )
    worst_curves["candidate_ir"] = pd.to_numeric(worst_curves["candidate_ir"], errors="coerce")
    worst_curves = worst_curves.sort_values(["_site_date_order", "candidate_ir"]).drop(columns=["_site_date_order"])

    by_site_path = out_dir / "continuous_ir_site_generalization_by_site_v1.csv"
    worst_path = out_dir / "continuous_ir_site_generalization_worst_decisions_v1.csv"
    curves_path = out_dir / "continuous_ir_site_generalization_worst_curves_v1.csv"
    report_path = out_dir / "continuous_ir_site_generalization_diagnostic_v1.md"
    by_site_diag.to_csv(by_site_path, index=False)
    worst.to_csv(worst_path, index=False)
    worst_curves.to_csv(curves_path, index=False)

    lines = [
        "# Continuous Irrigation Site-Generalization Diagnostic V1",
        "",
        "## By Site",
        "",
        markdown_table(by_site_diag.head(20)),
        "",
        "## Worst Decisions",
        "",
        markdown_table(worst.head(20)),
        "",
        "## Interpretation Hints",
        "",
        "- High mean regret with high static RMS-Z suggests unseen-site heterogeneity.",
        "- High regret with low static RMS-Z suggests model capacity or feature-window limits.",
        "- Worst curves show whether the model over-irrigates collapse dates or under-irrigates positive-response dates.",
        "",
        "## Outputs",
        "",
        f"- `{by_site_path}`",
        f"- `{worst_path}`",
        f"- `{curves_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation site-generalization diagnostic v1")
    print(f"by_site: {by_site_path}")
    print(f"worst_decisions: {worst_path}")
    print(f"worst_curves: {curves_path}")
    print(f"report: {report_path}")
    print("")
    print(by_site_diag.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare paper-style discrete irrigation search with dense/LSTM optimization.

The source paper evaluates a fixed candidate list. This diagnostic compares:

1. Dense SWAP-sampled oracle: best available sampled label for each site-date.
2. Paper discrete list evaluated by interpolating the sampled SWAP curve.
3. LSTM continuous optimization evaluated by the same interpolation rule.

The goal is to quantify whether the fixed paper list loses value relative to a
denser continuous candidate space and how close the surrogate optimizer gets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


TARGET = "net_gain_7d"
DEFAULT_PAPER_CANDIDATES = "0,10,15,20,25,30,40,60"


def parse_candidates(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Candidate list cannot be empty")
    return sorted(set(values))


def interp_gain(curve: pd.DataFrame, ir: float) -> float:
    tmp = curve[["candidate_ir", TARGET]].copy()
    tmp["candidate_ir"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    tmp[TARGET] = pd.to_numeric(tmp[TARGET], errors="coerce")
    tmp = tmp.dropna().sort_values("candidate_ir")
    return float(np.interp(float(ir), tmp["candidate_ir"].to_numpy(), tmp[TARGET].to_numpy()))


def candidate_set_for_site(max_ir: float, candidates: list[float]) -> list[float]:
    valid = [value for value in candidates if value <= max_ir + 1e-9]
    if 0.0 not in valid:
        valid.insert(0, 0.0)
    # Keep the site-specific cap as a deployable option when the paper's 60mm
    # candidate is infeasible, such as code_C1 with a 27.5mm cap.
    if all(abs(value - max_ir) > 1e-9 for value in valid):
        valid.append(float(max_ir))
    return sorted(set(valid))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--lstm-decisions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    args = parser.parse_args()

    samples = pd.read_csv(args.samples)
    lstm = pd.read_csv(args.lstm_decisions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = parse_candidates(args.paper_candidates)

    rows = []
    for site_date_id, curve in samples.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        oracle = curve.loc[curve[TARGET].idxmax()]
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        paper_values = candidate_set_for_site(site_ir_max, candidates)
        paper_scores = [(ir, interp_gain(curve, ir)) for ir in paper_values]
        paper_best_ir, paper_best_gain = max(paper_scores, key=lambda item: item[1])

        lstm_row = lstm.loc[lstm["site_date_id"] == site_date_id]
        if lstm_row.empty:
            raise ValueError(f"Missing LSTM decision for site_date_id={site_date_id}")
        lstm_row = lstm_row.iloc[0]

        rows.append(
            {
                "site_date_id": site_date_id,
                "site_id": str(oracle["site_id"]),
                "date_t": str(oracle["date_t"]),
                "target_collapse": bool(oracle["target_collapse"]),
                "site_ir_max": site_ir_max,
                "dense_oracle_ir": float(oracle["candidate_ir"]),
                "dense_oracle_gain": float(oracle[TARGET]),
                "paper_best_ir": float(paper_best_ir),
                "paper_best_interp_gain": float(paper_best_gain),
                "paper_regret_vs_dense_oracle": float(oracle[TARGET] - paper_best_gain),
                "lstm_opt_ir": float(lstm_row["surrogate_opt_ir"]),
                "lstm_opt_interp_gain": float(lstm_row["surrogate_opt_interp_true_net_gain"]),
                "lstm_regret_vs_dense_oracle": float(lstm_row["surrogate_opt_interp_regret"]),
                "lstm_minus_paper_regret": float(lstm_row["surrogate_opt_interp_regret"] - (oracle[TARGET] - paper_best_gain)),
            }
        )

    df = pd.DataFrame(rows)
    metrics = pd.DataFrame(
        [
            {
                "site_dates": int(len(df)),
                "paper_candidates": ",".join(str(v).rstrip("0").rstrip(".") for v in candidates),
                "paper_mean_regret": float(df["paper_regret_vs_dense_oracle"].mean()),
                "paper_median_regret": float(df["paper_regret_vs_dense_oracle"].median()),
                "lstm_mean_regret": float(df["lstm_regret_vs_dense_oracle"].mean()),
                "lstm_median_regret": float(df["lstm_regret_vs_dense_oracle"].median()),
                "lstm_better_than_paper_rate": safe_mean(df["lstm_regret_vs_dense_oracle"] < df["paper_regret_vs_dense_oracle"]),
                "same_or_better_than_paper_rate": safe_mean(df["lstm_regret_vs_dense_oracle"] <= df["paper_regret_vs_dense_oracle"]),
                "paper_zero_regret_rate": safe_mean(df["paper_regret_vs_dense_oracle"].abs() <= 1e-9),
                "lstm_zero_regret_rate": safe_mean(df["lstm_regret_vs_dense_oracle"].abs() <= 1e-9),
            }
        ]
    )
    by_site = (
        df.groupby("site_id")
        .agg(
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            lstm_mean_regret=("lstm_regret_vs_dense_oracle", "mean"),
            lstm_better_than_paper_rate=("lstm_minus_paper_regret", lambda s: float((s < 0).mean())),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    by_site["lstm_minus_paper_mean_regret"] = by_site["lstm_mean_regret"] - by_site["paper_mean_regret"]
    by_site = by_site.sort_values("lstm_minus_paper_mean_regret")

    decisions_path = out_dir / "continuous_ir_discrete_vs_lstm_continuous_decisions_v1.csv"
    metrics_path = out_dir / "continuous_ir_discrete_vs_lstm_continuous_metrics_v1.csv"
    by_site_path = out_dir / "continuous_ir_discrete_vs_lstm_continuous_by_site_v1.csv"
    report_path = out_dir / "continuous_ir_discrete_vs_lstm_continuous_v1.md"
    df.to_csv(decisions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    worst_paper = df.sort_values("paper_regret_vs_dense_oracle", ascending=False).head(15)
    worst_lstm = df.sort_values("lstm_regret_vs_dense_oracle", ascending=False).head(15)
    lines = [
        "# Discrete Candidate Search vs LSTM Continuous Optimization V1",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Paper-List Regret",
        "",
        markdown_table(worst_paper),
        "",
        "## Worst LSTM Continuous Regret",
        "",
        markdown_table(worst_lstm),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Discrete candidate search vs LSTM continuous optimization v1")
    print(f"decisions: {decisions_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(metrics.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize the fixed-list anchored local-refinement branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_fixed_list_local_refinement_mainline_summary_v1"


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def add_row(rows: list[dict], **kwargs: object) -> None:
    rows.append(kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headroom = read_required(
        root
        / "continuous_ir_12site_10k_fixed_list_local_refinement_headroom_v1"
        / "fixed_list_local_refinement_headroom_summary_v1.csv"
    )
    lstm_summary = read_required(
        root
        / "continuous_ir_12site_10k_fixed_list_local_surrogate_refinement_v1"
        / "fixed_list_local_surrogate_refinement_summary_v1.csv"
    )
    lstm_sweep = read_required(
        root
        / "continuous_ir_12site_10k_fixed_list_local_surrogate_refinement_v1"
        / "fixed_list_local_surrogate_refinement_threshold_sweep_v1.csv"
    )
    tree_summary = read_required(
        root
        / "continuous_ir_12site_10k_fixed_list_local_refinement_tree_v1"
        / "fixed_list_local_refinement_tree_summary_v1.csv"
    )
    tree_sweep = read_required(
        root
        / "continuous_ir_12site_10k_fixed_list_local_refinement_tree_v1"
        / "fixed_list_local_refinement_tree_threshold_sweep_v1.csv"
    )

    paper_mean = float(headroom["paper_mean_regret"].iloc[0])
    rows: list[dict] = []
    add_row(
        rows,
        policy="paper fixed-list",
        deployable="yes",
        window_mm="",
        mean_regret=paper_mean,
        gain_vs_paper=0.0,
        local_use_rate="",
        note="strong deployable baseline",
    )
    for row in headroom.itertuples(index=False):
        add_row(
            rows,
            policy="oracle local refinement",
            deployable="no",
            window_mm=float(row.window_mm),
            mean_regret=float(row.local_oracle_mean_regret),
            gain_vs_paper=paper_mean - float(row.local_oracle_mean_regret),
            local_use_rate="oracle",
            note="diagnostic upper bound only",
        )
    for row in lstm_summary.itertuples(index=False):
        add_row(
            rows,
            policy="existing LSTM local argmax",
            deployable="yes",
            window_mm=float(row.window_mm),
            mean_regret=float(row.local_pred_mean_regret),
            gain_vs_paper=paper_mean - float(row.local_pred_mean_regret),
            local_use_rate=1.0,
            note="uses saved dense LSTM predictions",
        )
    best_lstm = lstm_sweep.sort_values("mean_regret").iloc[0]
    add_row(
        rows,
        policy="existing LSTM margin guard",
        deployable="yes",
        window_mm=float(best_lstm["window_mm"]),
        mean_regret=float(best_lstm["mean_regret"]),
        gain_vs_paper=paper_mean - float(best_lstm["mean_regret"]),
        local_use_rate=float(best_lstm["local_use_rate"]),
        note=f"best threshold {float(best_lstm['threshold'])}",
    )
    for row in tree_summary.itertuples(index=False):
        add_row(
            rows,
            policy="TinyForest local delta argmax",
            deployable="yes",
            window_mm=float(row.window_mm),
            mean_regret=float(row.pred_local_mean_regret),
            gain_vs_paper=paper_mean - float(row.pred_local_mean_regret),
            local_use_rate=1.0,
            note="dedicated local delta model",
        )
    best_tree = tree_sweep.sort_values("mean_regret").iloc[0]
    add_row(
        rows,
        policy="TinyForest local delta guard",
        deployable="yes",
        window_mm=float(best_tree["window_mm"]),
        mean_regret=float(best_tree["mean_regret"]),
        gain_vs_paper=paper_mean - float(best_tree["mean_regret"]),
        local_use_rate=float(best_tree["local_use_rate"]),
        note=f"best threshold {float(best_tree['threshold'])}",
    )

    summary = pd.DataFrame(rows).sort_values(["deployable", "mean_regret"], ascending=[True, True])
    deployable = summary.loc[summary["deployable"] == "yes"].copy()
    best_deployable = deployable.sort_values("mean_regret").iloc[0]
    conclusion = pd.DataFrame(
        [
            {
                "paper_fixed_list_mean_regret": paper_mean,
                "best_deployable_policy": str(best_deployable["policy"]),
                "best_deployable_mean_regret": float(best_deployable["mean_regret"]),
                "best_deployable_gain_vs_paper": float(best_deployable["gain_vs_paper"]),
                "oracle_headroom_best_mean_regret": float(headroom["local_oracle_mean_regret"].min()),
                "conclusion": (
                    "Large oracle local headroom exists, but tested deployable learned local selectors "
                    "do not produce a meaningful improvement over the paper fixed-list baseline."
                ),
            }
        ]
    )

    summary_path = out_dir / "fixed_list_local_refinement_mainline_summary_v1.csv"
    conclusion_path = out_dir / "fixed_list_local_refinement_mainline_conclusion_v1.csv"
    report_path = out_dir / "fixed_list_local_refinement_mainline_summary_v1.md"
    summary.to_csv(summary_path, index=False)
    conclusion.to_csv(conclusion_path, index=False)

    lines = [
        "# Fixed-List Local Refinement Mainline Summary V1",
        "",
        "## Conclusion",
        "",
        markdown_table(conclusion),
        "",
        "## Policy Table",
        "",
        markdown_table(summary),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{conclusion_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Fixed-list local refinement mainline summary v1")
    print(f"summary: {summary_path}")
    print(f"conclusion: {conclusion_path}")
    print(f"report: {report_path}")
    print("")
    print(conclusion.to_string(index=False))
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

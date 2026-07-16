#!/usr/bin/env python3
"""Summarize continuous-irrigation surrogate model outputs.

Reads the metrics/by-site files emitted by ridge, TinyForest, and two-head
baselines, then writes one compact comparison table plus a short markdown
report. The script is intentionally filename-based so it can run on the server
without extra dependencies or project state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("site_general_surrogate_eval")


MODEL_FILES = {
    "ridge": {
        "metrics": "continuous_irrigation_surrogate_baseline_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_baseline_v1_by_site.csv",
    },
    "tiny_forest": {
        "metrics": "continuous_irrigation_surrogate_tree_nosklearn_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_tree_nosklearn_v1_by_site.csv",
    },
    "twohead_tiny_forest": {
        "metrics": "continuous_irrigation_surrogate_twohead_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_twohead_v1_by_site.csv",
    },
    "mlp_nosklearn": {
        "metrics": "continuous_irrigation_surrogate_mlp_nosklearn_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_mlp_nosklearn_v1_by_site.csv",
    },
    "persite_tiny_forest": {
        "metrics": "continuous_irrigation_surrogate_persite_tree_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_persite_tree_v1_by_site.csv",
    },
    "lstm": {
        "metrics": "continuous_irrigation_surrogate_lstm_v1_metrics.csv",
        "by_site": "continuous_irrigation_surrogate_lstm_v1_by_site.csv",
    },
}


def read_metrics(model: str, directory: Path) -> pd.DataFrame:
    path = directory / MODEL_FILES[model]["metrics"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {model} metrics: {path}")
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model"] = model
    else:
        df.insert(0, "model", model)
    if "model_dir" in df.columns:
        df["model_dir"] = str(directory)
    else:
        insert_at = 1 if "model" in df.columns else 0
        df.insert(insert_at, "model_dir", str(directory))
    return df


def read_by_site(model: str, directory: Path) -> pd.DataFrame:
    path = directory / MODEL_FILES[model]["by_site"]
    if not path.exists():
        raise FileNotFoundError(f"Missing {model} by-site metrics: {path}")
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model"] = model
    else:
        df.insert(0, "model", model)
    if "model_dir" in df.columns:
        df["model_dir"] = str(directory)
    else:
        insert_at = 1 if "model" in df.columns else 0
        df.insert(insert_at, "model_dir", str(directory))
    return df


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-dir", required=True)
    parser.add_argument("--tree-dir", required=True)
    parser.add_argument("--twohead-dir", required=True)
    parser.add_argument("--mlp-dir")
    parser.add_argument("--persite-dir")
    parser.add_argument("--lstm-dir")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ROOT / "continuous_ir_12site_10k_surrogate_model_comparison_v1"),
    )
    args = parser.parse_args()

    model_dirs = {
        "ridge": Path(args.ridge_dir),
        "tiny_forest": Path(args.tree_dir),
        "twohead_tiny_forest": Path(args.twohead_dir),
    }
    if args.mlp_dir:
        model_dirs["mlp_nosklearn"] = Path(args.mlp_dir)
    if args.persite_dir:
        model_dirs["persite_tiny_forest"] = Path(args.persite_dir)
    if args.lstm_dir:
        model_dirs["lstm"] = Path(args.lstm_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat(
        [read_metrics(model, directory) for model, directory in model_dirs.items()],
        ignore_index=True,
    )
    by_site = pd.concat(
        [read_by_site(model, directory) for model, directory in model_dirs.items()],
        ignore_index=True,
    )

    metric_cols = [
        "model",
        "mae",
        "rmse",
        "r2",
        "decision_accuracy",
        "mean_decision_regret",
        "median_decision_regret",
        "collapse_decision_accuracy",
        "noncollapse_decision_accuracy",
    ]
    metric_cols = [col for col in metric_cols if col in metrics.columns]
    compact = metrics[metric_cols].copy()
    compact = compact.sort_values(["mean_decision_regret", "decision_accuracy"], ascending=[True, False])

    by_site_compact = by_site[
        [
            col
            for col in [
                "model",
                "site_id",
                "decision_accuracy",
                "mean_decision_regret",
                "max_decision_regret",
                "n_site_dates",
            ]
            if col in by_site.columns
        ]
    ].sort_values(["site_id", "mean_decision_regret", "model"])

    metrics_path = out_dir / "continuous_ir_surrogate_model_metrics_comparison_v1.csv"
    by_site_path = out_dir / "continuous_ir_surrogate_model_by_site_comparison_v1.csv"
    report_path = out_dir / "continuous_ir_surrogate_model_comparison_v1.md"
    compact.to_csv(metrics_path, index=False)
    by_site_compact.to_csv(by_site_path, index=False)

    best = compact.iloc[0]
    lines = [
        "# Continuous Irrigation Surrogate Model Comparison V1",
        "",
        "## Overall Metrics",
        "",
        markdown_table(compact),
        "",
        "## Current Best",
        "",
        f"- Best by mean decision regret: `{best['model']}`.",
        f"- Mean decision regret: `{best['mean_decision_regret']}`.",
        f"- Decision accuracy: `{best['decision_accuracy']}`.",
        "",
        "## By-Site Metrics",
        "",
        markdown_table(by_site_compact),
        "",
        "## Outputs",
        "",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation surrogate model comparison v1")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()

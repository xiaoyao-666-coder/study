#!/usr/bin/env python3
"""Train a no-sklearn local refiner anchored to the paper fixed list.

The oracle local-refinement diagnostic showed large headroom near the fixed-list
anchor, while the existing global LSTM dense predictions could not exploit it.
This script trains a dedicated local delta model: for each site-date, restrict
candidate rows to a small window around the paper fixed-list amount and predict
the true gain delta relative to that anchor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    TARGET,
    candidate_set_for_site,
    interp_gain,
    parse_candidates,
)
from diagnose_fixed_list_local_refinement_headroom_v1 import (
    float_list,
    interpolated_grid_best,
    parse_bool,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import build_features, markdown_table
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics, usable_columns


DEFAULT_ROOT = Path("site_general_surrogate_eval")
DEFAULT_INPUT = (
    DEFAULT_ROOT
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = DEFAULT_ROOT / "continuous_ir_12site_10k_fixed_list_local_refinement_tree_v1"


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def parse_thresholds(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return sorted(set(values))


def build_local_rows(samples: pd.DataFrame, windows: list[float], paper_candidates: list[float]) -> pd.DataFrame:
    rows = []
    for site_date_id, curve in samples.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        dense_oracle_gain = float(dense_oracle[TARGET])
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        paper_values = candidate_set_for_site(site_ir_max, paper_candidates)
        paper_scores = [(ir, interp_gain(curve, ir)) for ir in paper_values]
        paper_ir, paper_gain = max(paper_scores, key=lambda item: item[1])
        paper_regret = dense_oracle_gain - paper_gain

        for window in windows:
            lower = max(0.0, paper_ir - window)
            upper = min(site_ir_max, paper_ir + window)
            local_oracle_ir, local_oracle_gain = interpolated_grid_best(curve, lower, upper, 0.5)
            local = curve.loc[
                (curve["candidate_ir"] >= lower - 1e-9) & (curve["candidate_ir"] <= upper + 1e-9)
            ].copy()
            if local.empty:
                continue
            for _, row in local.iterrows():
                candidate_ir = float(row["candidate_ir"])
                out = row.to_dict()
                out["window_mm"] = float(window)
                out["paper_best_ir"] = float(paper_ir)
                out["paper_true_gain"] = float(paper_gain)
                out["paper_regret"] = float(paper_regret)
                out["local_oracle_ir"] = float(local_oracle_ir)
                out["local_oracle_true_gain"] = float(local_oracle_gain)
                out["local_oracle_regret"] = float(dense_oracle_gain - local_oracle_gain)
                out["dense_oracle_ir"] = float(dense_oracle["candidate_ir"])
                out["dense_oracle_gain"] = dense_oracle_gain
                out["candidate_offset_from_paper"] = candidate_ir - float(paper_ir)
                out["candidate_abs_offset_from_paper"] = abs(candidate_ir - float(paper_ir))
                out["candidate_offset_fraction_of_window"] = (
                    (candidate_ir - float(paper_ir)) / float(window) if float(window) > 0 else 0.0
                )
                out["candidate_true_delta_over_paper"] = float(row[TARGET]) - float(paper_gain)
                out["candidate_regret_vs_dense_oracle"] = dense_oracle_gain - float(row[TARGET])
                out["target_collapse"] = parse_bool(row["target_collapse"]) if "target_collapse" in row else False
                rows.append(out)
    return pd.DataFrame(rows)


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    x = build_features(df)
    extra_cols = [
        "window_mm",
        "paper_best_ir",
        "candidate_offset_from_paper",
        "candidate_abs_offset_from_paper",
        "candidate_offset_fraction_of_window",
    ]
    for col in extra_cols:
        x[col] = pd.to_numeric(df[col], errors="coerce")
    x["candidate_offset_x_paper_ir"] = x["candidate_offset_from_paper"] * x["paper_best_ir"]
    x["abs_offset_x_window"] = x["candidate_abs_offset_from_paper"] * x["window_mm"]
    return x


def fit_predict_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> np.ndarray:
    cols = usable_columns(x_train)
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train[cols], y_train)
    return model.predict(x_test[cols])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--windows-mm", default="1,2.5,5")
    parser.add_argument("--thresholds", default="-10,-5,-2,-1,0,0.5,1,2,5,10,20")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing sample table: {input_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(input_path)
    paper_candidates = parse_candidates(args.paper_candidates)
    windows = float_list(args.windows_mm)
    thresholds = parse_thresholds(args.thresholds)
    local_df = build_local_rows(samples, windows, paper_candidates)
    if local_df.empty:
        raise ValueError("No local refinement rows were created")

    x_all = build_model_features(local_df)
    y_all = pd.to_numeric(local_df["candidate_true_delta_over_paper"], errors="coerce")
    groups = sorted(local_df["site_id"].astype(str).unique())
    pred_parts = []
    for i, site_id in enumerate(groups):
        print(f"[local-tree] fold {i + 1}/{len(groups)} holdout site_id={site_id}", flush=True)
        test_mask = local_df["site_id"].astype(str) == str(site_id)
        train_mask = ~test_mask
        pred = fit_predict_forest(
            x_all.loc[train_mask],
            y_all.loc[train_mask],
            x_all.loc[test_mask],
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state + i,
        )
        part = local_df.loc[test_mask].copy()
        part["pred_delta_over_paper"] = pred
        pred_parts.append(part)

    pred_df = pd.concat(pred_parts, ignore_index=True)
    decisions = []
    for (window, site_date_id), part in pred_df.groupby(["window_mm", "site_date_id"], sort=False):
        best = part.loc[part["pred_delta_over_paper"].idxmax()]
        decisions.append(
            {
                "window_mm": float(window),
                "site_date_id": str(site_date_id),
                "site_id": str(best["site_id"]),
                "date_t": str(best["date_t"]),
                "target_collapse": parse_bool(best["target_collapse"]),
                "paper_best_ir": float(best["paper_best_ir"]),
                "paper_true_gain": float(best["paper_true_gain"]),
                "paper_regret": float(best["paper_regret"]),
                "local_oracle_ir": float(best["local_oracle_ir"]),
                "local_oracle_regret": float(best["local_oracle_regret"]),
                "pred_local_ir": float(best["candidate_ir"]),
                "pred_local_true_gain": float(best[TARGET]),
                "pred_local_delta_over_paper": float(best["candidate_true_delta_over_paper"]),
                "pred_local_pred_delta_over_paper": float(best["pred_delta_over_paper"]),
                "pred_local_regret": float(best["candidate_regret_vs_dense_oracle"]),
                "pred_local_better_than_paper": float(best["candidate_true_delta_over_paper"]) > 1e-9,
                "pred_local_worse_than_paper": float(best["candidate_true_delta_over_paper"]) < -1e-9,
                "pred_local_changed_ir": abs(float(best["candidate_ir"]) - float(best["paper_best_ir"])) > 1e-9,
            }
        )
    decision_df = pd.DataFrame(decisions)

    summary = (
        decision_df.groupby("window_mm")
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            pred_local_mean_regret=("pred_local_regret", "mean"),
            pred_local_mean_improvement=("pred_local_delta_over_paper", "mean"),
            pred_local_better_rate=("pred_local_better_than_paper", "mean"),
            pred_local_worse_rate=("pred_local_worse_than_paper", "mean"),
            pred_local_changed_ir_rate=("pred_local_changed_ir", "mean"),
            mean_pred_delta=("pred_local_pred_delta_over_paper", "mean"),
            median_pred_delta=("pred_local_pred_delta_over_paper", "median"),
        )
        .reset_index()
        .sort_values("window_mm")
    )

    sweep_rows = []
    for window, part in decision_df.groupby("window_mm", sort=True):
        for threshold in thresholds:
            use_local = part["pred_local_pred_delta_over_paper"] >= threshold
            regret = np.where(use_local, part["pred_local_regret"], part["paper_regret"])
            sweep_rows.append(
                {
                    "window_mm": float(window),
                    "threshold": float(threshold),
                    "mean_regret": float(np.mean(regret)),
                    "median_regret": float(np.median(regret)),
                    "local_use_rate": safe_mean(use_local),
                    "good_override_rate": safe_mean(use_local & part["pred_local_better_than_paper"]),
                    "bad_override_rate": safe_mean(use_local & part["pred_local_worse_than_paper"]),
                    "paper_mean_regret": float(part["paper_regret"].mean()),
                    "pred_local_mean_regret": float(part["pred_local_regret"].mean()),
                    "local_oracle_mean_regret": float(part["local_oracle_regret"].mean()),
                }
            )
    sweep = pd.DataFrame(sweep_rows).sort_values(["mean_regret", "window_mm", "threshold"])
    best = sweep.iloc[0]
    best_window = float(best["window_mm"])
    best_threshold = float(best["threshold"])
    best_decisions = decision_df.loc[decision_df["window_mm"] == best_window].copy()
    best_decisions["guarded_use_local"] = best_decisions["pred_local_pred_delta_over_paper"] >= best_threshold
    best_decisions["guarded_regret"] = np.where(
        best_decisions["guarded_use_local"],
        best_decisions["pred_local_regret"],
        best_decisions["paper_regret"],
    )

    by_site = (
        best_decisions.groupby("site_id")
        .agg(
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            pred_local_mean_regret=("pred_local_regret", "mean"),
            guarded_mean_regret=("guarded_regret", "mean"),
            guarded_use_local_rate=("guarded_use_local", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values("guarded_mean_regret", ascending=False)
    )

    pred_metrics = score_metrics(
        pred_df["candidate_true_delta_over_paper"].to_numpy(dtype=float),
        pred_df["pred_delta_over_paper"].to_numpy(dtype=float),
    )
    metrics = pd.DataFrame([{**pred_metrics, "local_rows": int(len(pred_df)), "site_dates": int(len(decision_df))}])

    pred_path = out_dir / "fixed_list_local_refinement_tree_predictions_v1.csv"
    decision_path = out_dir / "fixed_list_local_refinement_tree_decisions_v1.csv"
    summary_path = out_dir / "fixed_list_local_refinement_tree_summary_v1.csv"
    sweep_path = out_dir / "fixed_list_local_refinement_tree_threshold_sweep_v1.csv"
    by_site_path = out_dir / "fixed_list_local_refinement_tree_by_site_v1.csv"
    metrics_path = out_dir / "fixed_list_local_refinement_tree_metrics_v1.csv"
    report_path = out_dir / "fixed_list_local_refinement_tree_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    summary.to_csv(summary_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    lines = [
        "# Fixed-List Local Refinement Tree V1",
        "",
        "## Metrics",
        "",
        markdown_table(metrics),
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Threshold Sweep",
        "",
        markdown_table(sweep),
        "",
        "## By Site At Best Guard",
        "",
        markdown_table(by_site),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decision_path}`",
        f"- `{summary_path}`",
        f"- `{sweep_path}`",
        f"- `{by_site_path}`",
        f"- `{metrics_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Fixed-list local refinement tree v1")
    print(f"summary: {summary_path}")
    print(f"sweep: {sweep_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(metrics.to_string(index=False))
    print("")
    print(summary.to_string(index=False))
    print("")
    print(sweep.head(20).to_string(index=False))
    print("")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

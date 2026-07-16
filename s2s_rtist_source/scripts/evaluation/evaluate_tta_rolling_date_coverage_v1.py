#!/usr/bin/env python3
"""Evaluate rolling per-site calibration on expanded TTA date-coverage data.

This diagnostic is the first step after generating extra SWAP date/state
coverage. It trains each site's surrogate on earlier expanded dates and tests on
later dates, approximating target-site onboarding / TTA with supervised SWAP
labels before adding unsupervised water-balance losses.
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
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics
from train_persite_tinyforest_profit_surrogate_v1 import (
    build_candidate_rows,
    dense_values,
    predict_forest,
    select_feature_mode,
)


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_date_coverage_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_date_coverage_rolling_eval_v1"
)


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one calibration count is required")
    return sorted(set(values))


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def date_order_table(site_df: pd.DataFrame) -> pd.DataFrame:
    dates = site_df[["site_date_id", "date_t", "decision_doy"]].drop_duplicates().copy()
    dates["sort_key"] = pd.to_numeric(dates["decision_doy"], errors="coerce")
    dates = dates.sort_values(["sort_key", "site_date_id"]).reset_index(drop=True)
    dates["date_order"] = np.arange(len(dates), dtype=int)
    return dates


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if not x[col].isna().all()]


def fit_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[TinyForest, list[str]]:
    cols = usable_columns(x_train)
    if not cols:
        raise ValueError("No usable feature columns")
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train[cols], y_train)
    return model, cols


def add_truth(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["interp_true_net_gain_7d"] = [
        interp_gain(curve, float(ir)) for ir in rows["candidate_ir"].to_numpy(dtype=float)
    ]
    return rows


def evaluate_decisions(
    *,
    site_id: str,
    mode: str,
    calibration_dates: int,
    train_site_dates: int,
    test_df: pd.DataFrame,
    model: TinyForest,
    feature_cols: list[str],
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
    feature_mode: str,
) -> list[dict]:
    rows = []
    for site_date_id, curve in test_df.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        if curve.empty:
            continue
        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        dense_oracle_gain = float(dense_oracle[TARGET])
        dense_oracle_ir = float(dense_oracle["candidate_ir"])
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)

        fixed_rows = add_truth(
            build_candidate_rows(curve, fixed_values, horizon_days=horizon_days, prefix="fixedlist"),
            curve,
        )
        fixed_x = select_feature_mode(build_features(fixed_rows), feature_mode)
        fixed_rows["pred_net_gain_7d"] = predict_forest(model, feature_cols, fixed_x)
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred_best = fixed_rows.loc[fixed_rows["pred_net_gain_7d"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = add_truth(
            build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt"),
            curve,
        )
        dense_x = select_feature_mode(build_features(dense_rows), feature_mode)
        dense_rows["pred_net_gain_7d"] = predict_forest(model, feature_cols, dense_x)
        continuous_pred_best = dense_rows.loc[dense_rows["pred_net_gain_7d"].idxmax()]

        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
        rows.append(
            {
                "site_id": site_id,
                "mode": mode,
                "calibration_dates": int(calibration_dates),
                "train_site_dates": int(train_site_dates),
                "site_date_id": str(site_date_id),
                "date_t": str(dense_oracle["date_t"]),
                "decision_doy": int(dense_oracle["decision_doy"]),
                "dense_oracle_ir": dense_oracle_ir,
                "dense_oracle_gain": dense_oracle_gain,
                "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                "paper_regret_vs_dense_oracle": dense_oracle_gain - fixed_oracle_gain,
                "fixed_list_surrogate_ir": float(fixed_pred_best["candidate_ir"]),
                "fixed_list_surrogate_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                "fixed_list_surrogate_regret_vs_fixed_oracle": fixed_oracle_gain
                - float(fixed_pred_best["interp_true_net_gain_7d"]),
                "continuous_surrogate_ir": float(continuous_pred_best["candidate_ir"]),
                "continuous_surrogate_true_gain": continuous_gain,
                "continuous_surrogate_regret_vs_dense_oracle": dense_oracle_gain - continuous_gain,
                "continuous_surrogate_gain_over_paper": continuous_gain - fixed_oracle_gain,
                "continuous_surrogate_better_than_paper": continuous_gain > fixed_oracle_gain + 1e-9,
                "continuous_surrogate_worse_than_paper": continuous_gain < fixed_oracle_gain - 1e-9,
            }
        )
    return rows


def summarize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["mode", "calibration_dates"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixed_list_surrogate_mean_regret_vs_fixed_oracle=(
                "fixed_list_surrogate_regret_vs_fixed_oracle",
                "mean",
            ),
            continuous_surrogate_mean_regret_vs_dense=(
                "continuous_surrogate_regret_vs_dense_oracle",
                "mean",
            ),
            continuous_surrogate_mean_gain_over_paper=("continuous_surrogate_gain_over_paper", "mean"),
            continuous_surrogate_better_than_paper_rate=("continuous_surrogate_better_than_paper", "mean"),
            continuous_surrogate_worse_than_paper_rate=("continuous_surrogate_worse_than_paper", "mean"),
        )
        .reset_index()
        .sort_values(["mode", "calibration_dates"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-dates", default="2,4,6")
    parser.add_argument("--mode", default="rolling", choices=["rolling", "prefix"])
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing input table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_int_list(args.calibration_dates)
    paper_candidates = parse_candidates(args.paper_candidates)

    df = pd.read_csv(data_path)
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    required = {"site_id", "site_date_id", "date_t", "decision_doy", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    decision_rows = []
    metric_rows = []
    sites = sorted(df["site_id"].astype(str).unique())
    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        order = date_order_table(site_df)
        site_df = site_df.merge(order[["site_date_id", "date_order"]], on="site_date_id", how="left")
        x_all = select_feature_mode(build_features(site_df), args.feature_mode)
        y_all = pd.to_numeric(site_df[TARGET], errors="coerce")
        if y_all.isna().any():
            raise ValueError(f"Target contains NaN for site {site_id}")
        n_dates = int(site_df["date_order"].max()) + 1
        print(f"[rolling-tta] site {site_idx + 1}/{len(sites)} {site_id} dates={n_dates}", flush=True)
        for k in k_values:
            if k >= n_dates:
                continue
            if args.mode == "prefix":
                train_mask = site_df["date_order"] < k
                test_mask = site_df["date_order"] >= k
                eval_slices = [(k, train_mask, test_mask)]
            else:
                eval_slices = []
                for test_order in range(k, n_dates):
                    train_mask = (site_df["date_order"] >= test_order - k) & (site_df["date_order"] < test_order)
                    test_mask = site_df["date_order"] == test_order
                    eval_slices.append((test_order, train_mask, test_mask))
            for offset, train_mask, test_mask in eval_slices:
                if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
                    continue
                model, cols = fit_forest(
                    x_all.loc[train_mask],
                    y_all.loc[train_mask],
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    random_state=args.random_state + site_idx * 1000 + k * 100 + offset,
                )
                preds = predict_forest(model, cols, x_all.loc[test_mask])
                metrics = score_metrics(y_all.loc[test_mask].to_numpy(dtype=float), preds)
                metrics.update(
                    {
                        "site_id": site_id,
                        "mode": args.mode,
                        "calibration_dates": int(k),
                        "train_rows": int(train_mask.sum()),
                        "test_rows": int(test_mask.sum()),
                        "train_site_dates": int(site_df.loc[train_mask, "site_date_id"].nunique()),
                        "test_site_dates": int(site_df.loc[test_mask, "site_date_id"].nunique()),
                    }
                )
                metric_rows.append(metrics)
                decision_rows.extend(
                    evaluate_decisions(
                        site_id=site_id,
                        mode=args.mode,
                        calibration_dates=k,
                        train_site_dates=int(site_df.loc[train_mask, "site_date_id"].nunique()),
                        test_df=site_df.loc[test_mask].copy(),
                        model=model,
                        feature_cols=cols,
                        paper_candidates=paper_candidates,
                        horizon_days=args.horizon_days,
                        grid_step=args.grid_step,
                        feature_mode=args.feature_mode,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    decisions = pd.DataFrame(decision_rows)
    summary = summarize_decisions(decisions)
    prediction_summary = (
        metrics.groupby(["mode", "calibration_dates"])
        .agg(
            sites=("site_id", "nunique"),
            train_rows=("train_rows", "sum"),
            test_rows=("test_rows", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(["mode", "calibration_dates"])
    ) if not metrics.empty else pd.DataFrame()
    by_site = (
        decisions.groupby(["mode", "calibration_dates", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            continuous_surrogate_mean_regret_vs_dense=("continuous_surrogate_regret_vs_dense_oracle", "mean"),
            continuous_surrogate_mean_gain_over_paper=("continuous_surrogate_gain_over_paper", "mean"),
            continuous_surrogate_better_than_paper_rate=("continuous_surrogate_better_than_paper", "mean"),
        )
        .reset_index()
        .sort_values(["mode", "calibration_dates", "continuous_surrogate_mean_regret_vs_dense"], ascending=[True, True, False])
    ) if not decisions.empty else pd.DataFrame()

    decisions_path = out_dir / "tta_rolling_date_coverage_decisions_v1.csv"
    metrics_path = out_dir / "tta_rolling_date_coverage_prediction_metrics_v1.csv"
    summary_path = out_dir / "tta_rolling_date_coverage_summary_v1.csv"
    by_site_path = out_dir / "tta_rolling_date_coverage_by_site_v1.csv"
    report_path = out_dir / "tta_rolling_date_coverage_v1.md"
    decisions.to_csv(decisions_path, index=False)
    prediction_summary.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# TTA Rolling Date Coverage Eval V1",
        "",
        f"- Input: `{data_path}`",
        f"- Mode: `{args.mode}`",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_summary),
        "",
        "## Decision Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("TTA rolling date coverage eval v1")
    print(f"prediction_metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(prediction_summary.to_string(index=False))
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

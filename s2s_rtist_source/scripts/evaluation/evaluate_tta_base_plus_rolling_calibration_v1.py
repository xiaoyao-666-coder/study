#!/usr/bin/env python3
"""Evaluate base expert plus rolling calibration on expanded TTA data.

The rolling-only diagnostic trains from scratch on a few recent expanded dates,
which is not how TTA would normally be used. This script keeps the original
per-site SWAP surrogate data as the base expert training set, then adds recent
expanded target-site dates as calibration/update data before evaluating the
next expanded date.
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


DEFAULT_BASE_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_ADAPT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_date_coverage_fast_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_base_plus_rolling_calibration_v1"
)


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one calibration count is required")
    if any(value < 0 for value in values):
        raise ValueError("Calibration counts must be non-negative")
    return sorted(set(values))


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def prepare_frame(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {source_name} table: {path}")
    df = pd.read_csv(path)
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    required = {"site_id", "site_date_id", "date_t", "decision_doy", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source_name} table missing columns: {missing}")
    df["data_source"] = source_name
    return df


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
    update_mode: str,
    calibration_dates: int,
    train_base_rows: int,
    train_adapt_dates: int,
    train_adapt_rows: int,
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
                "update_mode": update_mode,
                "calibration_dates": int(calibration_dates),
                "train_base_rows": int(train_base_rows),
                "train_adapt_dates": int(train_adapt_dates),
                "train_adapt_rows": int(train_adapt_rows),
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
        decisions.groupby(["update_mode", "calibration_dates"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            mean_train_base_rows=("train_base_rows", "mean"),
            mean_train_adapt_dates=("train_adapt_dates", "mean"),
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
        .sort_values(["update_mode", "calibration_dates"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=str(DEFAULT_BASE_INPUT))
    parser.add_argument("--adapt-input", default=str(DEFAULT_ADAPT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-dates", default="0,2,4,6")
    parser.add_argument("--update-mode", default="expanding", choices=["rolling", "expanding"])
    parser.add_argument("--include-base-same-date", action="store_true")
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = prepare_frame(Path(args.base_input), "base")
    adapt = prepare_frame(Path(args.adapt_input), "adapt")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_int_list(args.calibration_dates)
    paper_candidates = parse_candidates(args.paper_candidates)

    decision_rows = []
    metric_rows = []
    sites = sorted(adapt["site_id"].astype(str).unique())
    for site_idx, site_id in enumerate(sites):
        base_site = base.loc[base["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        adapt_site = adapt.loc[adapt["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        if base_site.empty:
            print(f"[base+rolling] skipping {site_id}: no base rows", flush=True)
            continue
        order = date_order_table(adapt_site)
        adapt_site = adapt_site.merge(order[["site_date_id", "date_order"]], on="site_date_id", how="left")
        n_dates = int(adapt_site["date_order"].max()) + 1
        print(
            f"[base+rolling] site {site_idx + 1}/{len(sites)} {site_id} "
            f"base_rows={len(base_site)} adapt_dates={n_dates}",
            flush=True,
        )
        for k in k_values:
            for test_order in range(n_dates):
                if test_order < k:
                    continue
                test_df = adapt_site.loc[adapt_site["date_order"] == test_order].copy()
                if test_df.empty:
                    continue
                test_dates = set(test_df["date_t"].astype(str))
                if args.include_base_same_date:
                    base_train = base_site
                else:
                    base_train = base_site.loc[~base_site["date_t"].astype(str).isin(test_dates)].copy()
                if args.update_mode == "rolling":
                    adapt_train = adapt_site.loc[
                        (adapt_site["date_order"] >= test_order - k) & (adapt_site["date_order"] < test_order)
                    ].copy()
                else:
                    adapt_train = adapt_site.loc[adapt_site["date_order"] < test_order].copy()
                if k == 0 and args.update_mode == "rolling":
                    adapt_train = adapt_train.iloc[0:0].copy()
                train_df = pd.concat([base_train, adapt_train], ignore_index=True, sort=False)
                if train_df.empty:
                    continue
                x_train = select_feature_mode(build_features(train_df), args.feature_mode)
                y_train = pd.to_numeric(train_df[TARGET], errors="coerce")
                x_test = select_feature_mode(build_features(test_df), args.feature_mode)
                y_test = pd.to_numeric(test_df[TARGET], errors="coerce")
                if y_train.isna().any() or y_test.isna().any():
                    raise ValueError(f"Target contains NaN for site {site_id}")
                model, cols = fit_forest(
                    x_train,
                    y_train,
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    random_state=args.random_state + site_idx * 10000 + k * 100 + test_order,
                )
                preds = predict_forest(model, cols, x_test)
                metrics = score_metrics(y_test.to_numpy(dtype=float), preds)
                metrics.update(
                    {
                        "site_id": site_id,
                        "update_mode": args.update_mode,
                        "calibration_dates": int(k),
                        "train_base_rows": int(len(base_train)),
                        "train_adapt_rows": int(len(adapt_train)),
                        "train_adapt_dates": int(adapt_train["site_date_id"].nunique()),
                        "test_rows": int(len(test_df)),
                        "test_site_dates": int(test_df["site_date_id"].nunique()),
                    }
                )
                metric_rows.append(metrics)
                decision_rows.extend(
                    evaluate_decisions(
                        site_id=site_id,
                        update_mode=args.update_mode,
                        calibration_dates=k,
                        train_base_rows=int(len(base_train)),
                        train_adapt_dates=int(adapt_train["site_date_id"].nunique()),
                        train_adapt_rows=int(len(adapt_train)),
                        test_df=test_df,
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
        metrics.groupby(["update_mode", "calibration_dates"])
        .agg(
            sites=("site_id", "nunique"),
            train_base_rows=("train_base_rows", "sum"),
            train_adapt_rows=("train_adapt_rows", "sum"),
            test_rows=("test_rows", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(["update_mode", "calibration_dates"])
    ) if not metrics.empty else pd.DataFrame()
    by_site = (
        decisions.groupby(["update_mode", "calibration_dates", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            continuous_surrogate_mean_regret_vs_dense=("continuous_surrogate_regret_vs_dense_oracle", "mean"),
            continuous_surrogate_mean_gain_over_paper=("continuous_surrogate_gain_over_paper", "mean"),
            continuous_surrogate_better_than_paper_rate=("continuous_surrogate_better_than_paper", "mean"),
        )
        .reset_index()
        .sort_values(
            ["update_mode", "calibration_dates", "continuous_surrogate_mean_regret_vs_dense"],
            ascending=[True, True, False],
        )
    ) if not decisions.empty else pd.DataFrame()

    decisions_path = out_dir / "tta_base_plus_rolling_decisions_v1.csv"
    metrics_path = out_dir / "tta_base_plus_rolling_prediction_metrics_v1.csv"
    summary_path = out_dir / "tta_base_plus_rolling_summary_v1.csv"
    by_site_path = out_dir / "tta_base_plus_rolling_by_site_v1.csv"
    report_path = out_dir / "tta_base_plus_rolling_v1.md"
    decisions.to_csv(decisions_path, index=False)
    prediction_summary.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# TTA Base Plus Rolling Calibration V1",
        "",
        f"- Base input: `{args.base_input}`",
        f"- Adapt input: `{args.adapt_input}`",
        f"- Update mode: `{args.update_mode}`",
        f"- Include base same date: `{args.include_base_same_date}`",
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

    print("TTA base plus rolling calibration v1")
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

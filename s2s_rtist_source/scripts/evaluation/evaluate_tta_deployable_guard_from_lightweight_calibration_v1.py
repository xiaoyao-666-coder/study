#!/usr/bin/env python3
"""Evaluate deployable guards for lightweight TTA continuous irrigation.

The curve-top diagnostic showed large oracle-guard headroom. This script tests
whether surrogate-visible confidence signals can recover some of that headroom:
predicted gain over fixed-list, top-1/top-2 predicted margin, and distance from
the fixed-list fallback irrigation amount.
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
from evaluate_tta_lightweight_output_calibration_v1 import (
    apply_calibrator,
    date_order_table,
    fit_calibrator,
    fit_forest,
    prepare_frame,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import build_features, markdown_table
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics
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
    / "continuous_ir_12site_tta_deployable_guard_from_lightweight_calibration_v1"
)
DEFAULT_MODES = "none,bias,ir_linear,pred_ir_quadratic"


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one calibration count is required")
    if any(value < 0 for value in values):
        raise ValueError("Calibration counts must be non-negative")
    return sorted(set(values))


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one float value is required")
    return sorted(set(values))


def parse_modes(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {"none", "bias", "ir_linear", "ir_quadratic", "ir_doy_quadratic", "pred_ir_quadratic"}
    bad = sorted(set(modes).difference(allowed))
    if bad:
        raise ValueError(f"Unknown calibration modes: {bad}")
    return modes


def add_interp_truth(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["interp_true_net_gain_7d"] = [
        interp_gain(curve, float(ir)) for ir in rows["candidate_ir"].to_numpy(dtype=float)
    ]
    return rows


def calibrated_predict(
    rows: pd.DataFrame,
    *,
    model,
    feature_cols: list[str],
    calibrator: dict,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = select_feature_mode(build_features(rows), feature_mode)
    base_pred = predict_forest(model, feature_cols, x)
    pred = apply_calibrator(rows, base_pred, calibrator)
    return base_pred, pred


def top_margin(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    order = np.argsort(values)[::-1]
    return float(values[order[0]] - values[order[1]])


def local_margin(dense_rows: pd.DataFrame, best_idx: int) -> float:
    pred = dense_rows["pred_net_gain_7d"].to_numpy(dtype=float)
    if len(pred) <= 1:
        return 0.0
    left = max(0, best_idx - 1)
    right = min(len(pred) - 1, best_idx + 1)
    neighbor_idxs = [idx for idx in range(left, right + 1) if idx != best_idx]
    if not neighbor_idxs:
        return 0.0
    return float(pred[best_idx] - max(pred[neighbor_idxs]))


def evaluate_one_curve(
    *,
    site_id: str,
    calibration_scope: str,
    calibration_dates: int,
    calibration_mode: str,
    train_base_rows: int,
    train_calibration_dates: int,
    train_calibration_rows: int,
    curve: pd.DataFrame,
    model,
    feature_cols: list[str],
    calibrator: dict,
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
    feature_mode: str,
) -> dict:
    curve = curve.copy()
    curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
    curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
    curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
    if curve.empty:
        raise ValueError("Empty curve")

    dense_oracle = curve.loc[curve[TARGET].idxmax()]
    dense_oracle_gain = float(dense_oracle[TARGET])
    dense_oracle_ir = float(dense_oracle["candidate_ir"])
    site_ir_max = float(curve["site_ir_max"].iloc[0])
    fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)

    fixed_rows = add_interp_truth(
        build_candidate_rows(curve, fixed_values, horizon_days=horizon_days, prefix="fixedlist"),
        curve,
    )
    _, fixed_pred = calibrated_predict(
        fixed_rows,
        model=model,
        feature_cols=feature_cols,
        calibrator=calibrator,
        feature_mode=feature_mode,
    )
    fixed_rows["pred_net_gain_7d"] = fixed_pred
    paper_fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
    fixed_pred_best = fixed_rows.loc[fixed_rows["pred_net_gain_7d"].idxmax()]

    dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
    dense_rows = add_interp_truth(
        build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt"),
        curve,
    )
    _, dense_pred = calibrated_predict(
        dense_rows,
        model=model,
        feature_cols=feature_cols,
        calibrator=calibrator,
        feature_mode=feature_mode,
    )
    dense_rows["pred_net_gain_7d"] = dense_pred
    best_idx = int(dense_rows["pred_net_gain_7d"].to_numpy(dtype=float).argmax())
    continuous_pred_best = dense_rows.iloc[best_idx]
    pred_order = np.argsort(dense_pred)[::-1]

    paper_gain = float(paper_fixed_oracle["interp_true_net_gain_7d"])
    continuous_true_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
    continuous_pred_gain = float(continuous_pred_best["pred_net_gain_7d"])
    fixed_pred_best_pred_gain = float(fixed_pred_best["pred_net_gain_7d"])
    paper_ir = float(paper_fixed_oracle["candidate_ir"])
    fixed_pred_best_ir = float(fixed_pred_best["candidate_ir"])
    continuous_ir = float(continuous_pred_best["candidate_ir"])

    paper_pred_at_paper_ir = float(
        fixed_rows.loc[
            (fixed_rows["candidate_ir"].to_numpy(dtype=float) - paper_ir).round(6) == 0,
            "pred_net_gain_7d",
        ].iloc[0]
    )
    continuous_rank_by_true = int(
        dense_rows["interp_true_net_gain_7d"].rank(method="min", ascending=False).iloc[best_idx]
    )

    return {
        "site_id": site_id,
        "calibration_scope": calibration_scope,
        "calibration_dates": int(calibration_dates),
        "calibration_mode": calibration_mode,
        "train_base_rows": int(train_base_rows),
        "train_calibration_dates": int(train_calibration_dates),
        "train_calibration_rows": int(train_calibration_rows),
        "site_date_id": str(dense_oracle["site_date_id"]),
        "date_t": str(dense_oracle["date_t"]),
        "decision_doy": int(dense_oracle["decision_doy"]),
        "site_ir_max": site_ir_max,
        "dense_oracle_ir": dense_oracle_ir,
        "dense_oracle_gain": dense_oracle_gain,
        "paper_fixed_ir": paper_ir,
        "paper_fixed_gain": paper_gain,
        "paper_regret_vs_dense": dense_oracle_gain - paper_gain,
        "fixed_pred_best_ir": fixed_pred_best_ir,
        "fixed_pred_best_pred_gain": fixed_pred_best_pred_gain,
        "fixed_pred_best_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
        "continuous_ir": continuous_ir,
        "continuous_pred_gain": continuous_pred_gain,
        "continuous_true_gain": continuous_true_gain,
        "continuous_regret_vs_dense": dense_oracle_gain - continuous_true_gain,
        "continuous_gain_over_paper": continuous_true_gain - paper_gain,
        "continuous_true_rank": continuous_rank_by_true,
        "pred_gain_over_fixed_pred_best": continuous_pred_gain - fixed_pred_best_pred_gain,
        "pred_gain_over_paper_ir": continuous_pred_gain - paper_pred_at_paper_ir,
        "pred_top1_top2_margin": top_margin(dense_pred),
        "pred_top1_top3_margin": float(dense_pred[pred_order[0]] - dense_pred[pred_order[2]]) if len(pred_order) >= 3 else 0.0,
        "pred_local_margin": local_margin(dense_rows, best_idx),
        "pred_curve_range": float(np.max(dense_pred) - np.min(dense_pred)),
        "pred_curve_std": float(np.std(dense_pred)),
        "ir_distance_to_paper": abs(continuous_ir - paper_ir),
        "ir_distance_to_fixed_pred_best": abs(continuous_ir - fixed_pred_best_ir),
        "ir_distance_to_dense_oracle": abs(continuous_ir - dense_oracle_ir),
        "continuous_is_zero": abs(continuous_ir) <= 1e-9,
        "continuous_is_upper_bound": continuous_ir >= site_ir_max - max(grid_step * 0.5, 1e-9),
    }


def build_policy_masks(rows: pd.DataFrame, *, gain_thresholds: list[float], margin_thresholds: list[float], distance_thresholds: list[float]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {
        "always_paper": np.zeros(len(rows), dtype=bool),
        "always_continuous": np.ones(len(rows), dtype=bool),
    }
    gain_fixed = rows["pred_gain_over_fixed_pred_best"].to_numpy(dtype=float)
    gain_paper = rows["pred_gain_over_paper_ir"].to_numpy(dtype=float)
    margin = rows["pred_top1_top2_margin"].to_numpy(dtype=float)
    local = rows["pred_local_margin"].to_numpy(dtype=float)
    dist_paper = rows["ir_distance_to_paper"].to_numpy(dtype=float)
    dist_fixed = rows["ir_distance_to_fixed_pred_best"].to_numpy(dtype=float)

    for gain_thr in gain_thresholds:
        masks[f"gain_fixed_ge_{gain_thr:g}"] = gain_fixed >= gain_thr
        masks[f"gain_paper_ge_{gain_thr:g}"] = gain_paper >= gain_thr
        for margin_thr in margin_thresholds:
            masks[f"gain_fixed_ge_{gain_thr:g}__margin_ge_{margin_thr:g}"] = (
                (gain_fixed >= gain_thr) & (margin >= margin_thr)
            )
        for dist_thr in distance_thresholds:
            masks[f"gain_fixed_ge_{gain_thr:g}__dist_paper_le_{dist_thr:g}"] = (
                (gain_fixed >= gain_thr) & (dist_paper <= dist_thr)
            )
            masks[f"gain_fixed_ge_{gain_thr:g}__dist_fixed_le_{dist_thr:g}"] = (
                (gain_fixed >= gain_thr) & (dist_fixed <= dist_thr)
            )
    for margin_thr in margin_thresholds:
        masks[f"margin_ge_{margin_thr:g}"] = margin >= margin_thr
        masks[f"local_margin_ge_{margin_thr:g}"] = local >= margin_thr
    return masks


def apply_policies(base_rows: pd.DataFrame, *, gain_thresholds: list[float], margin_thresholds: list[float], distance_thresholds: list[float]) -> pd.DataFrame:
    rows = []
    group_cols = ["calibration_scope", "calibration_dates", "calibration_mode"]
    for keys, part in base_rows.groupby(group_cols, sort=False):
        part = part.reset_index(drop=True)
        masks = build_policy_masks(
            part,
            gain_thresholds=gain_thresholds,
            margin_thresholds=margin_thresholds,
            distance_thresholds=distance_thresholds,
        )
        dense_gain = part["dense_oracle_gain"].to_numpy(dtype=float)
        paper_gain = part["paper_fixed_gain"].to_numpy(dtype=float)
        cont_gain = part["continuous_true_gain"].to_numpy(dtype=float)
        for policy_name, use_cont in masks.items():
            selected_gain = np.where(use_cont, cont_gain, paper_gain)
            regret = dense_gain - selected_gain
            gain_over_paper = selected_gain - paper_gain
            for idx, row in part.iterrows():
                rows.append(
                    {
                        **{col: row[col] for col in group_cols},
                        "guard_policy": policy_name,
                        "site_id": row["site_id"],
                        "site_date_id": row["site_date_id"],
                        "date_t": row["date_t"],
                        "decision_doy": row["decision_doy"],
                        "use_continuous": bool(use_cont[idx]),
                        "guard_selected_ir": float(row["continuous_ir"] if use_cont[idx] else row["paper_fixed_ir"]),
                        "guard_selected_true_gain": float(selected_gain[idx]),
                        "guard_regret_vs_dense": float(regret[idx]),
                        "guard_gain_over_paper": float(gain_over_paper[idx]),
                        "paper_regret_vs_dense": float(row["paper_regret_vs_dense"]),
                        "continuous_regret_vs_dense": float(row["continuous_regret_vs_dense"]),
                        "continuous_gain_over_paper": float(row["continuous_gain_over_paper"]),
                        "continuous_true_rank": int(row["continuous_true_rank"]),
                        "pred_gain_over_fixed_pred_best": float(row["pred_gain_over_fixed_pred_best"]),
                        "pred_gain_over_paper_ir": float(row["pred_gain_over_paper_ir"]),
                        "pred_top1_top2_margin": float(row["pred_top1_top2_margin"]),
                        "pred_local_margin": float(row["pred_local_margin"]),
                        "ir_distance_to_paper": float(row["ir_distance_to_paper"]),
                        "ir_distance_to_fixed_pred_best": float(row["ir_distance_to_fixed_pred_best"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_policy(policy_rows: pd.DataFrame) -> pd.DataFrame:
    if policy_rows.empty:
        return pd.DataFrame()
    return (
        policy_rows.groupby(["calibration_scope", "calibration_dates", "calibration_mode", "guard_policy"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            paper_mean_regret=("paper_regret_vs_dense", "mean"),
            continuous_mean_regret=("continuous_regret_vs_dense", "mean"),
            guard_mean_regret=("guard_regret_vs_dense", "mean"),
            guard_median_regret=("guard_regret_vs_dense", "median"),
            guard_p90_regret=("guard_regret_vs_dense", lambda x: float(np.quantile(x, 0.9))),
            guard_gain_over_paper=("guard_gain_over_paper", "mean"),
            use_continuous_rate=("use_continuous", "mean"),
            guard_better_than_paper_rate=("guard_gain_over_paper", lambda x: float(np.mean(x > 1e-9))),
            guard_worse_than_paper_rate=("guard_gain_over_paper", lambda x: float(np.mean(x < -1e-9))),
            guard_large_regret_gt_2_rate=("guard_regret_vs_dense", lambda x: float(np.mean(x > 2.0))),
            guard_large_regret_gt_5_rate=("guard_regret_vs_dense", lambda x: float(np.mean(x > 5.0))),
            guard_large_regret_gt_10_rate=("guard_regret_vs_dense", lambda x: float(np.mean(x > 10.0))),
        )
        .reset_index()
        .sort_values(
            ["calibration_scope", "calibration_dates", "calibration_mode", "guard_mean_regret", "use_continuous_rate"]
        )
    )


def summarize_by_site(policy_rows: pd.DataFrame) -> pd.DataFrame:
    if policy_rows.empty:
        return pd.DataFrame()
    return (
        policy_rows.groupby(["calibration_scope", "calibration_dates", "calibration_mode", "guard_policy", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_mean_regret=("paper_regret_vs_dense", "mean"),
            continuous_mean_regret=("continuous_regret_vs_dense", "mean"),
            guard_mean_regret=("guard_regret_vs_dense", "mean"),
            guard_gain_over_paper=("guard_gain_over_paper", "mean"),
            use_continuous_rate=("use_continuous", "mean"),
            guard_large_regret_gt_5_rate=("guard_regret_vs_dense", lambda x: float(np.mean(x > 5.0))),
        )
        .reset_index()
        .sort_values(
            ["calibration_scope", "calibration_dates", "calibration_mode", "guard_policy", "guard_mean_regret"],
            ascending=[True, True, True, True, False],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=str(DEFAULT_BASE_INPUT))
    parser.add_argument("--adapt-input", default=str(DEFAULT_ADAPT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-dates", default="0,2,4,6")
    parser.add_argument("--calibration-scope", default="rolling", choices=["rolling", "expanding"])
    parser.add_argument("--calibration-modes", default=DEFAULT_MODES)
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=40)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--correction-clip", type=float, default=20.0)
    parser.add_argument("--gain-thresholds", default="0,1,2,5,10")
    parser.add_argument("--margin-thresholds", default="0,0.5,1,2,5")
    parser.add_argument("--distance-thresholds", default="1,2,3,5,8")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = prepare_frame(Path(args.base_input), "base")
    adapt = prepare_frame(Path(args.adapt_input), "adapt")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_int_list(args.calibration_dates)
    modes = parse_modes(args.calibration_modes)
    paper_candidates = parse_candidates(args.paper_candidates)
    gain_thresholds = parse_float_list(args.gain_thresholds)
    margin_thresholds = parse_float_list(args.margin_thresholds)
    distance_thresholds = parse_float_list(args.distance_thresholds)

    decision_rows = []
    prediction_rows = []
    sites = sorted(adapt["site_id"].astype(str).unique())
    for site_idx, site_id in enumerate(sites):
        base_site = base.loc[base["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        adapt_site = adapt.loc[adapt["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        if base_site.empty:
            print(f"[deployable-guard] skipping {site_id}: no base rows", flush=True)
            continue
        order = date_order_table(adapt_site)
        adapt_site = adapt_site.merge(order[["site_date_id", "date_order"]], on="site_date_id", how="left")
        n_dates = int(adapt_site["date_order"].max()) + 1
        print(
            f"[deployable-guard] site {site_idx + 1}/{len(sites)} {site_id} "
            f"base_rows={len(base_site)} adapt_dates={n_dates}",
            flush=True,
        )

        for test_order in range(n_dates):
            test_df = adapt_site.loc[adapt_site["date_order"] == test_order].copy()
            if test_df.empty:
                continue
            test_dates = set(test_df["date_t"].astype(str))
            base_train = base_site.loc[~base_site["date_t"].astype(str).isin(test_dates)].copy()
            if base_train.empty:
                continue
            x_train = select_feature_mode(build_features(base_train), args.feature_mode)
            y_train = pd.to_numeric(base_train[TARGET], errors="coerce")
            if y_train.isna().any():
                raise ValueError(f"Base target contains NaN for site {site_id}")
            model, cols = fit_forest(
                x_train,
                y_train,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state + site_idx * 10000 + test_order,
            )

            adapt_x = select_feature_mode(build_features(adapt_site), args.feature_mode)
            adapt_base_pred = predict_forest(model, cols, adapt_x)
            base_pred_series = pd.Series(adapt_base_pred, index=adapt_site.index)

            for k in k_values:
                if test_order < k:
                    continue
                if args.calibration_scope == "rolling":
                    if k == 0:
                        calib_df = adapt_site.iloc[0:0].copy()
                    else:
                        calib_df = adapt_site.loc[
                            (adapt_site["date_order"] >= test_order - k)
                            & (adapt_site["date_order"] < test_order)
                        ].copy()
                else:
                    calib_df = adapt_site.loc[adapt_site["date_order"] < test_order].copy()
                calib_base_pred = base_pred_series.loc[calib_df.index].to_numpy(dtype=float)

                for mode in modes:
                    calibrator = fit_calibrator(
                        calib_df,
                        calib_base_pred,
                        mode=mode,
                        ridge_alpha=args.ridge_alpha,
                        correction_clip=args.correction_clip,
                    )
                    test_base_pred = base_pred_series.loc[test_df.index].to_numpy(dtype=float)
                    test_pred = apply_calibrator(test_df, test_base_pred, calibrator)
                    y_test = pd.to_numeric(test_df[TARGET], errors="coerce").to_numpy(dtype=float)
                    metric = score_metrics(y_test, test_pred)
                    metric.update(
                        {
                            "site_id": site_id,
                            "calibration_scope": args.calibration_scope,
                            "calibration_dates": int(k),
                            "calibration_mode": mode,
                            "test_rows": int(len(test_df)),
                        }
                    )
                    prediction_rows.append(metric)
                    for site_date_id, curve in test_df.groupby("site_date_id", sort=False):
                        decision_rows.append(
                            evaluate_one_curve(
                                site_id=site_id,
                                calibration_scope=args.calibration_scope,
                                calibration_dates=k,
                                calibration_mode=mode,
                                train_base_rows=int(len(base_train)),
                                train_calibration_dates=int(calib_df["site_date_id"].nunique()),
                                train_calibration_rows=int(len(calib_df)),
                                curve=curve,
                                model=model,
                                feature_cols=cols,
                                calibrator=calibrator,
                                paper_candidates=paper_candidates,
                                horizon_days=args.horizon_days,
                                grid_step=args.grid_step,
                                feature_mode=args.feature_mode,
                            )
                        )

    decisions = pd.DataFrame(decision_rows)
    predictions = pd.DataFrame(prediction_rows)
    policy_rows = apply_policies(
        decisions,
        gain_thresholds=gain_thresholds,
        margin_thresholds=margin_thresholds,
        distance_thresholds=distance_thresholds,
    )
    policy_summary = summarize_policy(policy_rows)
    by_site = summarize_by_site(policy_rows)
    prediction_summary = (
        predictions.groupby(["calibration_scope", "calibration_dates", "calibration_mode"])
        .agg(
            sites=("site_id", "nunique"),
            test_rows=("test_rows", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
        .sort_values(["calibration_scope", "calibration_dates", "calibration_mode"])
    ) if not predictions.empty else pd.DataFrame()
    best_policies = (
        policy_summary.sort_values(["guard_mean_regret", "use_continuous_rate"], ascending=[True, False]).head(60)
        if not policy_summary.empty
        else pd.DataFrame()
    )
    worst_cases = (
        policy_rows.sort_values("guard_regret_vs_dense", ascending=False).head(100)
        if not policy_rows.empty
        else pd.DataFrame()
    )

    decisions_path = out_dir / "deployable_guard_base_decisions_v1.csv"
    policy_rows_path = out_dir / "deployable_guard_policy_decisions_v1.csv"
    prediction_path = out_dir / "deployable_guard_prediction_metrics_v1.csv"
    summary_path = out_dir / "deployable_guard_policy_summary_v1.csv"
    by_site_path = out_dir / "deployable_guard_policy_by_site_v1.csv"
    best_path = out_dir / "deployable_guard_best_policies_v1.csv"
    worst_path = out_dir / "deployable_guard_worst_cases_v1.csv"
    report_path = out_dir / "deployable_guard_diagnostic_v1.md"
    decisions.to_csv(decisions_path, index=False)
    policy_rows.to_csv(policy_rows_path, index=False)
    prediction_summary.to_csv(prediction_path, index=False)
    policy_summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    best_policies.to_csv(best_path, index=False)
    worst_cases.to_csv(worst_path, index=False)

    lines = [
        "# Deployable Guard Diagnostic V1",
        "",
        f"- Base input: `{args.base_input}`",
        f"- Adapt input: `{args.adapt_input}`",
        f"- Calibration scope: `{args.calibration_scope}`",
        f"- Calibration modes: `{','.join(modes)}`",
        "",
        "## Best Policies",
        "",
        markdown_table(best_policies),
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_summary),
        "",
        "## Policy Summary",
        "",
        markdown_table(policy_summary),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{policy_rows_path}`",
        f"- `{prediction_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{best_path}`",
        f"- `{worst_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Deployable guard diagnostic v1")
    print(f"prediction_metrics: {prediction_path}")
    print(f"policy_summary: {summary_path}")
    print(f"best_policies: {best_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(best_policies.to_string(index=False))


if __name__ == "__main__":
    main()

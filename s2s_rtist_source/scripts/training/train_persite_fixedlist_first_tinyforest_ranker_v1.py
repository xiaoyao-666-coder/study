#!/usr/bin/env python3
"""Train per-site fixed-list-first TinyForest rankers.

The curve-top / zero-margin / dual-score diagnostics all still fail held-out
dates, and their fixed-list regret remains large. This script therefore steps
back to the teacher's first requirement: before maximizing over arbitrary
continuous irrigation, verify that each per-site expert can rank the same
paper fixed-list inputs close to SWAP.

Workflow:

1. For each site-date SWAP curve, reconstruct paper fixed-list candidates by
   interpolation.
2. Train a per-site TinyForest on fixed-list-local topness.
3. Evaluate fixed-list ranking CV first.
4. Optionally evaluate a small local continuous search around the predicted
   fixed-list amount; this is diagnostic only and should be ignored if the
   fixed-list ranker itself fails.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    TARGET,
    candidate_set_for_site,
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics
from train_persite_curve_top_tinyforest_ranker_v1 import fit_top_forest, predict_top_score
from train_persite_tinyforest_profit_surrogate_v1 import (
    add_interp_truth,
    build_candidate_rows,
    dense_values,
    make_group_folds,
    sanitize_name,
    select_feature_mode,
)


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_fixedlist_first_tinyforest_ranker_v1"
)
FIXED_TARGET = "fixedlist_top_score"


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def add_rank_columns(rows: pd.DataFrame, gain_col: str) -> pd.DataFrame:
    out = rows.copy()
    out["true_rank"] = out[gain_col].rank(method="min", ascending=False).astype(int)
    best_gain = float(out[gain_col].max())
    out["fixedlist_regret"] = best_gain - out[gain_col].to_numpy(dtype=float)
    return out


def build_fixedlist_table(
    df: pd.DataFrame,
    *,
    paper_candidates: list[float],
    horizon_days: int,
    top_temperature: float,
    top_regret_eps: float,
    top_rank_k: int,
) -> pd.DataFrame:
    fixed_parts: list[pd.DataFrame] = []
    for site_date_id, curve in df.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        if curve.empty:
            continue
        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)
        fixed_rows = build_candidate_rows(curve, fixed_values, horizon_days=horizon_days, prefix="fixedtrain")
        fixed_rows = add_interp_truth(fixed_rows, curve)
        fixed_rows = add_rank_columns(fixed_rows, "interp_true_net_gain_7d")
        fixed_rows["source_dense_oracle_ir"] = float(dense_oracle["candidate_ir"])
        fixed_rows["source_dense_oracle_gain"] = float(dense_oracle[TARGET])
        fixed_rows["source_site_date_id"] = str(site_date_id)
        fixed_parts.append(fixed_rows)
    if not fixed_parts:
        raise ValueError("No fixed-list rows could be built")
    out = pd.concat(fixed_parts, ignore_index=True)
    temp = max(float(top_temperature), 1e-9)
    out[FIXED_TARGET] = np.exp(-np.clip(out["fixedlist_regret"].to_numpy(dtype=float), 0.0, np.inf) / temp)
    out["curve_regret"] = out["fixedlist_regret"]
    out["curve_top_label"] = (
        (out["fixedlist_regret"] <= float(top_regret_eps) + 1e-9)
        | (out["true_rank"] <= int(top_rank_k))
    )
    return out


def sampled_rank_rows(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    fixed_df: pd.DataFrame,
    scores: np.ndarray,
) -> list[dict]:
    scored = fixed_df[
        [
            "site_date_id",
            "site_id",
            "date_t",
            "candidate_ir",
            "interp_true_net_gain_7d",
            "true_rank",
        ]
    ].copy()
    scored["pred_score"] = scores
    rows: list[dict] = []
    for site_date_id, part in scored.groupby("site_date_id", sort=False):
        true_best = part.loc[part["interp_true_net_gain_7d"].idxmax()]
        pred_best = part.loc[part["pred_score"].idxmax()]
        true_best_pred_rank = int(part["pred_score"].rank(method="min", ascending=False).loc[true_best.name])
        rows.append(
            {
                "eval_mode": eval_mode,
                "site_id": site_id,
                "site_fold": fold_id,
                "site_date_id": str(site_date_id),
                "date_t": str(true_best["date_t"]),
                "true_best_fixed_ir": float(true_best["candidate_ir"]),
                "pred_best_fixed_ir": float(pred_best["candidate_ir"]),
                "true_best_fixed_gain": float(true_best["interp_true_net_gain_7d"]),
                "pred_best_fixed_true_gain": float(pred_best["interp_true_net_gain_7d"]),
                "fixedlist_curve_regret": float(
                    true_best["interp_true_net_gain_7d"] - pred_best["interp_true_net_gain_7d"]
                ),
                "pred_best_true_rank": int(pred_best["true_rank"]),
                "fixed_top1_correct": int(pred_best["true_rank"]) <= 1,
                "fixed_top3_correct": int(pred_best["true_rank"]) <= 3,
                "true_best_pred_rank": true_best_pred_rank,
            }
        )
    return rows


def evaluate_decisions(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    fixed_eval_df: pd.DataFrame,
    raw_curves_df: pd.DataFrame,
    model,
    cols: list[str],
    feature_mode: str,
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
    local_radius: float,
) -> list[dict]:
    raw_curves = {str(k): v.copy() for k, v in raw_curves_df.groupby("site_date_id", sort=False)}
    decision_rows: list[dict] = []
    for site_date_id, fixed_rows in fixed_eval_df.groupby("site_date_id", sort=False):
        fixed_rows = fixed_rows.copy()
        fixed_x = select_feature_mode(build_features(fixed_rows), feature_mode)
        fixed_rows["pred_score"] = predict_top_score(model, cols, fixed_x)
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred = fixed_rows.loc[fixed_rows["pred_score"].idxmax()]

        dense_oracle_gain = float(fixed_rows["source_dense_oracle_gain"].iloc[0])
        dense_oracle_ir = float(fixed_rows["source_dense_oracle_ir"].iloc[0])
        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        fixed_pred_gain = float(fixed_pred["interp_true_net_gain_7d"])
        fixed_pred_ir = float(fixed_pred["candidate_ir"])
        local_ir = fixed_pred_ir
        local_gain = fixed_pred_gain
        local_nonfixed = False

        if float(local_radius) > 0 and str(site_date_id) in raw_curves:
            curve = raw_curves[str(site_date_id)]
            site_ir_max = float(curve["site_ir_max"].iloc[0])
            fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)
            grid = dense_values(site_ir_max, grid_step, fixed_values)
            grid = grid[
                (grid >= fixed_pred_ir - float(local_radius) - 1e-9)
                & (grid <= fixed_pred_ir + float(local_radius) + 1e-9)
            ]
            if len(grid):
                local_rows = build_candidate_rows(curve, grid, horizon_days=horizon_days, prefix="fixedlocal")
                local_rows = add_interp_truth(local_rows, curve)
                local_x = select_feature_mode(build_features(local_rows), feature_mode)
                local_rows["pred_score"] = predict_top_score(model, cols, local_x)
                local_pred = local_rows.loc[local_rows["pred_score"].idxmax()]
                local_ir = float(local_pred["candidate_ir"])
                local_gain = float(local_pred["interp_true_net_gain_7d"])
                nearest_fixed = min(fixed_values, key=lambda value: abs(float(value) - local_ir))
                local_nonfixed = abs(local_ir - float(nearest_fixed)) > 1e-9

        paper_regret = dense_oracle_gain - fixed_oracle_gain
        decision_rows.append(
            {
                "eval_mode": eval_mode,
                "site_id": site_id,
                "site_fold": fold_id,
                "site_date_id": str(site_date_id),
                "date_t": str(fixed_oracle["date_t"]),
                "dense_oracle_ir": dense_oracle_ir,
                "dense_oracle_gain": dense_oracle_gain,
                "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                "paper_regret_vs_dense_oracle": paper_regret,
                "fixedlist_ranker_ir": fixed_pred_ir,
                "fixedlist_ranker_true_gain": fixed_pred_gain,
                "fixedlist_ranker_regret_vs_fixed_oracle": fixed_oracle_gain - fixed_pred_gain,
                "fixedlist_ranker_regret_vs_dense_oracle": dense_oracle_gain - fixed_pred_gain,
                "fixedlist_ranker_gain_over_paper": fixed_pred_gain - fixed_oracle_gain,
                "local_refine_ir": local_ir,
                "local_refine_true_gain": local_gain,
                "local_refine_regret_vs_dense_oracle": dense_oracle_gain - local_gain,
                "local_refine_gain_over_paper": local_gain - fixed_oracle_gain,
                "local_refine_better_than_paper": local_gain > fixed_oracle_gain + 1e-9,
                "local_refine_worse_than_paper": local_gain < fixed_oracle_gain - 1e-9,
                "local_refine_nonfixed_ir": local_nonfixed,
                "local_refine_large_regret_gt_5": dense_oracle_gain - local_gain > 5.0,
            }
        )
    return decision_rows


def sampled_summary(sampled: pd.DataFrame) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame()
    return (
        sampled.groupby("eval_mode")
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            fixed_top1_accuracy=("fixed_top1_correct", "mean"),
            fixed_top3_accuracy=("fixed_top3_correct", "mean"),
            fixed_mean_curve_regret=("fixedlist_curve_regret", "mean"),
            fixed_median_curve_regret=("fixedlist_curve_regret", "median"),
            mean_pred_best_true_rank=("pred_best_true_rank", "mean"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
        )
        .reset_index()
    )


def decision_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby("eval_mode")
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixedlist_ranker_mean_regret_vs_fixed_oracle=(
                "fixedlist_ranker_regret_vs_fixed_oracle",
                "mean",
            ),
            fixedlist_ranker_mean_regret_vs_dense=("fixedlist_ranker_regret_vs_dense_oracle", "mean"),
            fixedlist_ranker_mean_gain_over_paper=("fixedlist_ranker_gain_over_paper", "mean"),
            local_refine_mean_regret_vs_dense=("local_refine_regret_vs_dense_oracle", "mean"),
            local_refine_median_regret_vs_dense=("local_refine_regret_vs_dense_oracle", "median"),
            local_refine_p90_regret_vs_dense=(
                "local_refine_regret_vs_dense_oracle",
                lambda x: float(np.quantile(x, 0.9)),
            ),
            local_refine_mean_gain_over_paper=("local_refine_gain_over_paper", "mean"),
            local_refine_better_than_paper_rate=("local_refine_better_than_paper", "mean"),
            local_refine_worse_than_paper_rate=("local_refine_worse_than_paper", "mean"),
            local_refine_nonfixed_ir_rate=("local_refine_nonfixed_ir", "mean"),
            local_refine_large_regret_gt_5_rate=("local_refine_large_regret_gt_5", "mean"),
        )
        .reset_index()
    )


def by_site_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["eval_mode", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixedlist_ranker_mean_regret_vs_fixed_oracle=(
                "fixedlist_ranker_regret_vs_fixed_oracle",
                "mean",
            ),
            local_refine_mean_regret_vs_dense=("local_refine_regret_vs_dense_oracle", "mean"),
            local_refine_mean_gain_over_paper=("local_refine_gain_over_paper", "mean"),
            local_refine_large_regret_gt_5_rate=("local_refine_large_regret_gt_5", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "local_refine_mean_regret_vs_dense"], ascending=[True, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--folds-per-site", type=int, default=3)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--local-radius", type=float, default=2.0)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--top-temperature", type=float, default=2.0)
    parser.add_argument("--top-regret-eps", type=float, default=1.0)
    parser.add_argument("--top-rank-k", type=int, default=2)
    parser.add_argument("--top-oversample-factor", type=int, default=8)
    parser.add_argument("--shoulder-regret-eps", type=float, default=3.0)
    parser.add_argument("--shoulder-oversample-factor", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--site-limit", type=int, default=0)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-final-experts", action="store_true")
    args = parser.parse_args()

    if args.skip_cv and args.skip_capacity:
        raise ValueError("At least one of CV or capacity check must run")

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sequence-wide sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = out_dir / "final_site_experts_v1"
    if not args.skip_final_experts:
        expert_dir.mkdir(parents=True, exist_ok=True)

    paper_candidates = parse_candidates(args.paper_candidates)
    raw_df = pd.read_csv(data_path)
    required = {"site_id", "site_date_id", "date_t", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in raw_df.columns:
            raw_df[col] = bool_series(raw_df[col])

    fixed_df = build_fixedlist_table(
        raw_df,
        paper_candidates=paper_candidates,
        horizon_days=args.horizon_days,
        top_temperature=args.top_temperature,
        top_regret_eps=args.top_regret_eps,
        top_rank_k=args.top_rank_k,
    )
    sites = sorted(fixed_df["site_id"].astype(str).unique())
    if args.site_limit and args.site_limit > 0:
        sites = sites[: args.site_limit]

    sampled_rows: list[dict] = []
    decision_rows: list[dict] = []
    metric_rows: list[dict] = []
    expert_rows: list[dict] = []

    print(
        f"[persite-fixedlist-first] fixed_rows={len(fixed_df)} sites={len(sites)} "
        f"capacity={not args.skip_capacity} cv={not args.skip_cv}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_fixed = fixed_df.loc[fixed_df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        site_raw = raw_df.loc[raw_df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        x_site = select_feature_mode(build_features(site_fixed), args.feature_mode)
        y_site = pd.to_numeric(site_fixed[FIXED_TARGET], errors="coerce")
        groups = sorted(site_fixed["site_date_id"].astype(str).unique())
        group_values = site_fixed["site_date_id"].astype(str).to_numpy()
        print(f"[persite-fixedlist-first] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            model, cols, train_rows = fit_top_forest(
                x_site,
                y_site,
                site_fixed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                top_oversample_factor=args.top_oversample_factor,
                shoulder_oversample_factor=args.shoulder_oversample_factor,
                shoulder_regret_eps=args.shoulder_regret_eps,
                random_state=args.random_state + site_idx,
            )
            scores = predict_top_score(model, cols, x_site)
            sampled_rows.extend(
                sampled_rank_rows(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    fixed_df=site_fixed,
                    scores=scores,
                )
            )
            metrics = score_metrics(y_site.to_numpy(dtype=float), scores)
            metrics.update(
                {
                    "eval_mode": "capacity",
                    "site_id": site_id,
                    "site_fold": 0,
                    "rows": int(len(site_fixed)),
                    "site_dates": int(len(groups)),
                    "oversampled_train_rows": int(train_rows),
                }
            )
            metric_rows.append(metrics)
            decision_rows.extend(
                evaluate_decisions(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    fixed_eval_df=site_fixed,
                    raw_curves_df=site_raw,
                    model=model,
                    cols=cols,
                    feature_mode=args.feature_mode,
                    paper_candidates=paper_candidates,
                    horizon_days=args.horizon_days,
                    grid_step=args.grid_step,
                    local_radius=args.local_radius,
                )
            )
            if not args.skip_final_experts:
                expert_path = expert_dir / f"persite_fixedlist_first_tinyforest_ranker_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": model,
                            "feature_columns": cols,
                            "site_id": site_id,
                            "target_column": FIXED_TARGET,
                            "paper_candidates": paper_candidates,
                            "horizon_days": int(args.horizon_days),
                            "local_radius": float(args.local_radius),
                            "training_rows": int(len(site_fixed)),
                            "training_site_dates": int(len(groups)),
                        },
                        handle,
                    )
                expert_rows.append(
                    {
                        "site_id": site_id,
                        "expert_path": str(expert_path),
                        "training_rows": int(len(site_fixed)),
                        "training_site_dates": int(len(groups)),
                    }
                )

        if not args.skip_cv:
            folds = make_group_folds(groups, args.folds_per_site, args.random_state + 1000 + site_idx)
            for fold_idx, holdout_groups in enumerate(folds, start=1):
                print(
                    f"[persite-fixedlist-first] site={site_id} cv fold {fold_idx}/{len(folds)}",
                    flush=True,
                )
                test_mask = np.isin(group_values, np.array(holdout_groups, dtype=str))
                train_mask = ~test_mask
                model, cols, train_rows = fit_top_forest(
                    x_site.loc[train_mask],
                    y_site.loc[train_mask],
                    site_fixed.loc[train_mask].reset_index(drop=True),
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    top_oversample_factor=args.top_oversample_factor,
                    shoulder_oversample_factor=args.shoulder_oversample_factor,
                    shoulder_regret_eps=args.shoulder_regret_eps,
                    random_state=args.random_state + site_idx * 100 + fold_idx,
                )
                scores = predict_top_score(model, cols, x_site.loc[test_mask])
                sampled_rows.extend(
                    sampled_rank_rows(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        fixed_df=site_fixed.loc[test_mask],
                        scores=scores,
                    )
                )
                metrics = score_metrics(y_site.loc[test_mask].to_numpy(dtype=float), scores)
                metrics.update(
                    {
                        "eval_mode": "heldout_date_cv",
                        "site_id": site_id,
                        "site_fold": int(fold_idx),
                        "rows": int(test_mask.sum()),
                        "site_dates": int(len(holdout_groups)),
                        "oversampled_train_rows": int(train_rows),
                    }
                )
                metric_rows.append(metrics)
                decision_rows.extend(
                    evaluate_decisions(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        fixed_eval_df=site_fixed.loc[test_mask].copy(),
                        raw_curves_df=site_raw.loc[
                            site_raw["site_date_id"].astype(str).isin(holdout_groups)
                        ].copy(),
                        model=model,
                        cols=cols,
                        feature_mode=args.feature_mode,
                        paper_candidates=paper_candidates,
                        horizon_days=args.horizon_days,
                        grid_step=args.grid_step,
                        local_radius=args.local_radius,
                    )
                )

    sampled = pd.DataFrame(sampled_rows)
    decisions = pd.DataFrame(decision_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    prediction_metrics = (
        fold_metrics.groupby("eval_mode")
        .agg(
            folds=("site_fold", "count"),
            rows=("rows", "sum"),
            mean_oversampled_train_rows=("oversampled_train_rows", "mean"),
            fixed_top_score_mae=("mae", "mean"),
            fixed_top_score_rmse=("rmse", "mean"),
            fixed_top_score_r2=("r2", "mean"),
        )
        .reset_index()
        if not fold_metrics.empty
        else pd.DataFrame()
    )
    sampled_metrics = sampled_summary(sampled)
    summary = decision_summary(decisions)
    by_site = by_site_summary(decisions)
    manifest = pd.DataFrame(expert_rows)

    sampled_path = out_dir / "persite_fixedlist_first_ranker_sampled_rank_eval_v1.csv"
    decisions_path = out_dir / "persite_fixedlist_first_ranker_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_fixedlist_first_ranker_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_fixedlist_first_ranker_prediction_metrics_v1.csv"
    sampled_metrics_path = out_dir / "persite_fixedlist_first_ranker_sampled_rank_metrics_v1.csv"
    summary_path = out_dir / "persite_fixedlist_first_ranker_summary_v1.csv"
    by_site_path = out_dir / "persite_fixedlist_first_ranker_by_site_v1.csv"
    manifest_path = out_dir / "persite_fixedlist_first_ranker_manifest_v1.csv"
    config_path = out_dir / "persite_fixedlist_first_ranker_config_v1.json"
    report_path = out_dir / "persite_fixedlist_first_ranker_v1.md"

    sampled.to_csv(sampled_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    prediction_metrics.to_csv(prediction_metrics_path, index=False)
    sampled_metrics.to_csv(sampled_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "input": str(data_path),
                "target": FIXED_TARGET,
                "feature_mode": args.feature_mode,
                "top_temperature": float(args.top_temperature),
                "top_regret_eps": float(args.top_regret_eps),
                "top_rank_k": int(args.top_rank_k),
                "local_radius": float(args.local_radius),
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "min_samples_leaf": int(args.min_samples_leaf),
                "grid_step": float(args.grid_step),
                "folds_per_site": int(args.folds_per_site),
                "paper_candidates": paper_candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Per-Site Fixed-List-First TinyForest Ranker V1",
        "",
        "## Scope",
        "",
        "- Per-site expert only; no cross-site MoE or gating.",
        "- Trains only on paper fixed-list candidates reconstructed from SWAP curves.",
        "- Local continuous refinement is diagnostic and should be considered only if fixed-list ranking passes.",
        f"- Input: `{data_path}`.",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_metrics),
        "",
        "## Fixed-List Rank Metrics",
        "",
        markdown_table(sampled_metrics),
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
        f"- `{sampled_path}`",
        f"- `{decisions_path}`",
        f"- `{fold_metrics_path}`",
        f"- `{prediction_metrics_path}`",
        f"- `{sampled_metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{manifest_path}`",
        f"- `{config_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site fixed-list-first TinyForest ranker v1")
    print(f"sampled_rank_eval: {sampled_path}")
    print(f"decisions: {decisions_path}")
    print(f"prediction_metrics: {prediction_metrics_path}")
    print(f"sampled_rank_metrics: {sampled_metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    print("")
    print("Prediction metrics")
    print(prediction_metrics.to_string(index=False))
    print("")
    print("Fixed-list rank metrics")
    print(sampled_metrics.to_string(index=False))
    print("")
    print("Decision summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

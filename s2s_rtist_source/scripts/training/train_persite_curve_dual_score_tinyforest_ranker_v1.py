#!/usr/bin/env python3
"""Train per-site dual-score TinyForest rankers for SWAP curve tops.

This diagnostic follows the zero-margin result. Topness training has excellent
capacity but can select 0 mm on held-out dates. Zero-margin training reduces
mean regret, but creates more off-list large failures. This script keeps the
experiment per-site and trains both targets in the same fold:

- topness score: exp(-curve_regret / temperature)
- zero-margin score: gain(candidate) - gain(0 mm), curve-normalized

It then evaluates simple score/rank fusions inside each response curve. This is
an expert-quality diagnostic before MoE, not a cross-site or TTA experiment.
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
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics
from train_persite_curve_top_tinyforest_ranker_v1 import (
    TOP_SCORE,
    add_curve_top_targets,
    add_true_rank_columns,
    fit_top_forest,
    predict_top_score,
)
from train_persite_curve_zero_margin_tinyforest_ranker_v1 import (
    MARGIN_TARGET,
    add_zero_margin_target,
    fit_margin_forest,
)
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
    / "continuous_ir_12site_10k_persite_curve_dual_score_tinyforest_ranker_v1"
)


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float")
    return values


def alpha_token(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def rank_score(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="min", ascending=False)
    n = len(values)
    if n <= 1:
        return pd.Series(np.ones(n, dtype=float), index=values.index)
    return 1.0 - (ranks - 1.0) / float(n - 1)


def add_policy_scores(rows: pd.DataFrame, alphas: list[float]) -> tuple[pd.DataFrame, list[str]]:
    out = rows.copy()
    out["score_top_only"] = out["pred_top_score"]
    out["score_zero_margin_only"] = out["pred_zero_margin"]
    out["top_rank_score"] = rank_score(out["pred_top_score"])
    out["margin_rank_score"] = rank_score(out["pred_zero_margin"])
    policy_scores = ["score_top_only", "score_zero_margin_only"]
    for alpha in alphas:
        token = alpha_token(alpha)
        raw_col = f"score_raw_fusion_top{token}"
        rank_col = f"score_rank_fusion_top{token}"
        out[raw_col] = float(alpha) * out["pred_top_score"] + (1.0 - float(alpha)) * out[
            "pred_zero_margin"
        ]
        out[rank_col] = float(alpha) * out["top_rank_score"] + (1.0 - float(alpha)) * out[
            "margin_rank_score"
        ]
        policy_scores.extend([raw_col, rank_col])
    return out, policy_scores


def policy_name(score_col: str) -> str:
    return score_col.removeprefix("score_")


def fit_dual_models(
    x_train: pd.DataFrame,
    meta_train: pd.DataFrame,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    top_oversample_factor: int,
    shoulder_oversample_factor: int,
    shoulder_regret_eps: float,
    random_state: int,
) -> tuple[TinyForest, list[str], TinyForest, list[str], int, int]:
    top_model, top_cols, top_rows = fit_top_forest(
        x_train,
        pd.to_numeric(meta_train[TOP_SCORE], errors="coerce"),
        meta_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        top_oversample_factor=top_oversample_factor,
        shoulder_oversample_factor=shoulder_oversample_factor,
        shoulder_regret_eps=shoulder_regret_eps,
        random_state=random_state,
    )
    margin_model, margin_cols, margin_rows = fit_margin_forest(
        x_train,
        pd.to_numeric(meta_train[MARGIN_TARGET], errors="coerce"),
        meta_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        top_oversample_factor=top_oversample_factor,
        shoulder_oversample_factor=shoulder_oversample_factor,
        shoulder_regret_eps=shoulder_regret_eps,
        random_state=random_state + 50000,
    )
    return top_model, top_cols, margin_model, margin_cols, top_rows, margin_rows


def predict_dual_scores(
    rows: pd.DataFrame,
    *,
    top_model: TinyForest,
    top_cols: list[str],
    margin_model: TinyForest,
    margin_cols: list[str],
    feature_mode: str,
) -> pd.DataFrame:
    out = rows.copy()
    x_rows = select_feature_mode(build_features(out), feature_mode)
    out["pred_top_score"] = predict_top_score(top_model, top_cols, x_rows)
    out["pred_zero_margin"] = predict_top_score(margin_model, margin_cols, x_rows)
    return out


def sampled_rank_rows(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    eval_df: pd.DataFrame,
    top_scores: np.ndarray,
    margin_scores: np.ndarray,
    alphas: list[float],
) -> list[dict]:
    scored = eval_df[
        [
            "site_date_id",
            "site_id",
            "date_t",
            "candidate_ir",
            TARGET,
            "curve_true_rank",
        ]
    ].copy()
    scored["pred_top_score"] = top_scores
    scored["pred_zero_margin"] = margin_scores
    rows = []
    for site_date_id, part in scored.groupby("site_date_id", sort=False):
        part, score_cols = add_policy_scores(part, alphas)
        true_best = part.loc[part[TARGET].idxmax()]
        for score_col in score_cols:
            pred_best = part.loc[part[score_col].idxmax()]
            true_best_pred_rank = int(part[score_col].rank(method="min", ascending=False).loc[true_best.name])
            rows.append(
                {
                    "eval_mode": eval_mode,
                    "policy": policy_name(score_col),
                    "site_id": site_id,
                    "site_fold": fold_id,
                    "site_date_id": str(site_date_id),
                    "date_t": str(true_best["date_t"]),
                    "true_best_ir": float(true_best["candidate_ir"]),
                    "pred_best_ir": float(pred_best["candidate_ir"]),
                    "true_best_net_gain": float(true_best[TARGET]),
                    "pred_best_true_net_gain": float(pred_best[TARGET]),
                    "sampled_curve_regret": float(true_best[TARGET] - pred_best[TARGET]),
                    "sampled_pred_best_true_rank": int(pred_best["curve_true_rank"]),
                    "sampled_top1_correct": int(pred_best["curve_true_rank"]) <= 1,
                    "sampled_top3_correct": int(pred_best["curve_true_rank"]) <= 3,
                    "sampled_top5_correct": int(pred_best["curve_true_rank"]) <= 5,
                    "true_best_pred_rank": true_best_pred_rank,
                }
            )
    return rows


def evaluate_curves(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    curves_df: pd.DataFrame,
    top_model: TinyForest,
    top_cols: list[str],
    margin_model: TinyForest,
    margin_cols: list[str],
    feature_mode: str,
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
    alphas: list[float],
) -> list[dict]:
    decision_rows: list[dict] = []
    for site_date_id, curve in curves_df.groupby("site_date_id", sort=False):
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

        fixed_rows = build_candidate_rows(curve, fixed_values, horizon_days=horizon_days, prefix="fixedlist")
        fixed_rows = add_interp_truth(fixed_rows, curve)
        fixed_rows = add_true_rank_columns(fixed_rows)
        fixed_rows = predict_dual_scores(
            fixed_rows,
            top_model=top_model,
            top_cols=top_cols,
            margin_model=margin_model,
            margin_cols=margin_cols,
            feature_mode=feature_mode,
        )
        fixed_rows, fixed_score_cols = add_policy_scores(fixed_rows, alphas)
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt")
        dense_rows = add_interp_truth(dense_rows, curve)
        dense_rows = add_true_rank_columns(dense_rows)
        dense_rows = predict_dual_scores(
            dense_rows,
            top_model=top_model,
            top_cols=top_cols,
            margin_model=margin_model,
            margin_cols=margin_cols,
            feature_mode=feature_mode,
        )
        dense_rows, dense_score_cols = add_policy_scores(dense_rows, alphas)
        if fixed_score_cols != dense_score_cols:
            raise RuntimeError("Fixed and dense policy score columns differ")

        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        paper_regret = dense_oracle_gain - fixed_oracle_gain
        for score_col in dense_score_cols:
            fixed_pred_best = fixed_rows.loc[fixed_rows[score_col].idxmax()]
            continuous_pred_best = dense_rows.loc[dense_rows[score_col].idxmax()]
            continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
            continuous_rank = int(continuous_pred_best["dense_true_rank"])
            nearest_fixed = min(
                fixed_values,
                key=lambda value: abs(float(value) - float(continuous_pred_best["candidate_ir"])),
            )
            decision_rows.append(
                {
                    "eval_mode": eval_mode,
                    "policy": policy_name(score_col),
                    "site_id": site_id,
                    "site_fold": fold_id,
                    "site_date_id": str(site_date_id),
                    "date_t": str(dense_oracle["date_t"]),
                    "site_ir_max": site_ir_max,
                    "dense_oracle_ir": dense_oracle_ir,
                    "dense_oracle_gain": dense_oracle_gain,
                    "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                    "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                    "paper_regret_vs_dense_oracle": paper_regret,
                    "fixed_list_dual_ranker_ir": float(fixed_pred_best["candidate_ir"]),
                    "fixed_list_dual_ranker_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                    "fixed_list_dual_ranker_regret_vs_fixed_oracle": fixed_oracle_gain
                    - float(fixed_pred_best["interp_true_net_gain_7d"]),
                    "continuous_dual_ranker_ir": float(continuous_pred_best["candidate_ir"]),
                    "continuous_dual_ranker_true_gain": continuous_gain,
                    "continuous_dual_ranker_regret_vs_dense_oracle": dense_oracle_gain - continuous_gain,
                    "continuous_dual_ranker_gain_over_paper": continuous_gain - fixed_oracle_gain,
                    "continuous_dual_ranker_better_than_paper": continuous_gain > fixed_oracle_gain + 1e-9,
                    "continuous_dual_ranker_worse_than_paper": continuous_gain < fixed_oracle_gain - 1e-9,
                    "continuous_dual_ranker_nearest_fixed_ir": float(nearest_fixed),
                    "continuous_dual_ranker_distance_to_nearest_fixed_ir": abs(
                        float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                    ),
                    "continuous_dual_ranker_nonfixed_ir": abs(
                        float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                    )
                    > 1e-9,
                    "continuous_selected_dense_rank": continuous_rank,
                    "continuous_selected_top1": continuous_rank <= 1,
                    "continuous_selected_top3": continuous_rank <= 3,
                    "continuous_selected_top5": continuous_rank <= 5,
                    "continuous_large_regret_gt_2": dense_oracle_gain - continuous_gain > 2.0,
                    "continuous_large_regret_gt_5": dense_oracle_gain - continuous_gain > 5.0,
                    "continuous_large_regret_gt_10": dense_oracle_gain - continuous_gain > 10.0,
                }
            )
    return decision_rows


def sampled_summary(sampled: pd.DataFrame) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame()
    return (
        sampled.groupby(["eval_mode", "policy"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            sampled_top1_accuracy=("sampled_top1_correct", "mean"),
            sampled_top3_accuracy=("sampled_top3_correct", "mean"),
            sampled_top5_accuracy=("sampled_top5_correct", "mean"),
            sampled_mean_curve_regret=("sampled_curve_regret", "mean"),
            sampled_median_curve_regret=("sampled_curve_regret", "median"),
            mean_pred_best_true_rank=("sampled_pred_best_true_rank", "mean"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "sampled_mean_curve_regret"])
    )


def decision_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["eval_mode", "policy"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixed_list_dual_ranker_mean_regret_vs_fixed_oracle=(
                "fixed_list_dual_ranker_regret_vs_fixed_oracle",
                "mean",
            ),
            continuous_dual_ranker_mean_regret_vs_dense=(
                "continuous_dual_ranker_regret_vs_dense_oracle",
                "mean",
            ),
            continuous_dual_ranker_median_regret_vs_dense=(
                "continuous_dual_ranker_regret_vs_dense_oracle",
                "median",
            ),
            continuous_dual_ranker_p90_regret_vs_dense=(
                "continuous_dual_ranker_regret_vs_dense_oracle",
                lambda x: float(np.quantile(x, 0.9)),
            ),
            continuous_dual_ranker_mean_gain_over_paper=(
                "continuous_dual_ranker_gain_over_paper",
                "mean",
            ),
            continuous_dual_ranker_better_than_paper_rate=(
                "continuous_dual_ranker_better_than_paper",
                "mean",
            ),
            continuous_dual_ranker_worse_than_paper_rate=(
                "continuous_dual_ranker_worse_than_paper",
                "mean",
            ),
            continuous_dual_ranker_nonfixed_ir_rate=("continuous_dual_ranker_nonfixed_ir", "mean"),
            continuous_selected_top1_rate=("continuous_selected_top1", "mean"),
            continuous_selected_top3_rate=("continuous_selected_top3", "mean"),
            continuous_selected_top5_rate=("continuous_selected_top5", "mean"),
            mean_continuous_selected_dense_rank=("continuous_selected_dense_rank", "mean"),
            large_regret_gt_5_rate=("continuous_large_regret_gt_5", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "continuous_dual_ranker_mean_regret_vs_dense"])
    )


def by_site_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["eval_mode", "policy", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            continuous_dual_ranker_mean_regret_vs_dense=(
                "continuous_dual_ranker_regret_vs_dense_oracle",
                "mean",
            ),
            continuous_dual_ranker_mean_gain_over_paper=(
                "continuous_dual_ranker_gain_over_paper",
                "mean",
            ),
            continuous_dual_ranker_better_than_paper_rate=(
                "continuous_dual_ranker_better_than_paper",
                "mean",
            ),
            continuous_dual_ranker_nonfixed_ir_rate=("continuous_dual_ranker_nonfixed_ir", "mean"),
            continuous_selected_top3_rate=("continuous_selected_top3", "mean"),
            mean_continuous_selected_dense_rank=("continuous_selected_dense_rank", "mean"),
            large_regret_gt_5_rate=("continuous_large_regret_gt_5", "mean"),
        )
        .reset_index()
        .sort_values(
            ["eval_mode", "policy", "continuous_dual_ranker_mean_regret_vs_dense"],
            ascending=[True, True, False],
        )
    )


def prediction_metrics_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    return (
        fold_metrics.groupby(["eval_mode", "prediction_target"])
        .agg(
            folds=("site_fold", "count"),
            rows=("rows", "sum"),
            mean_oversampled_train_rows=("oversampled_train_rows", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--folds-per-site", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--top-temperature", type=float, default=2.0)
    parser.add_argument("--top-regret-eps", type=float, default=1.0)
    parser.add_argument("--top-rank-k", type=int, default=3)
    parser.add_argument("--top-oversample-factor", type=int, default=8)
    parser.add_argument("--shoulder-regret-eps", type=float, default=5.0)
    parser.add_argument("--shoulder-oversample-factor", type=int, default=3)
    parser.add_argument("--fusion-alphas", default="0.25,0.5,0.75")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--site-limit", type=int, default=0)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-final-experts", action="store_true")
    args = parser.parse_args()

    if args.skip_cv and args.skip_capacity:
        raise ValueError("At least one of CV or capacity check must run")
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive")
    if args.top_temperature <= 0:
        raise ValueError("--top-temperature must be positive")

    alphas = parse_float_list(args.fusion_alphas)
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("--fusion-alphas must be in [0, 1]")

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sequence-wide sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = out_dir / "final_site_experts_v1"
    if not args.skip_final_experts:
        expert_dir.mkdir(parents=True, exist_ok=True)

    paper_candidates = parse_candidates(args.paper_candidates)
    df = pd.read_csv(data_path)
    required = {"site_id", "site_date_id", "date_t", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    df = add_curve_top_targets(
        df,
        top_temperature=args.top_temperature,
        top_regret_eps=args.top_regret_eps,
        top_rank_k=args.top_rank_k,
    )
    df = add_zero_margin_target(df)

    sites = sorted(df["site_id"].astype(str).unique())
    if args.site_limit and args.site_limit > 0:
        sites = sites[: args.site_limit]

    sampled_rows: list[dict] = []
    decision_rows: list[dict] = []
    metric_rows: list[dict] = []
    expert_rows: list[dict] = []

    print(
        f"[persite-dual-score-tinyforest] rows={len(df)} sites={len(sites)} "
        f"capacity={not args.skip_capacity} cv={not args.skip_cv}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        x_site = select_feature_mode(build_features(site_df), args.feature_mode)
        groups = sorted(site_df["site_date_id"].astype(str).unique())
        print(f"[persite-dual-score-tinyforest] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            print(f"[persite-dual-score-tinyforest] site={site_id} capacity fit", flush=True)
            top_model, top_cols, margin_model, margin_cols, top_rows, margin_rows = fit_dual_models(
                x_site,
                site_df,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                top_oversample_factor=args.top_oversample_factor,
                shoulder_oversample_factor=args.shoulder_oversample_factor,
                shoulder_regret_eps=args.shoulder_regret_eps,
                random_state=args.random_state + site_idx,
            )
            top_scores = predict_top_score(top_model, top_cols, x_site)
            margin_scores = predict_top_score(margin_model, margin_cols, x_site)
            sampled_rows.extend(
                sampled_rank_rows(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    eval_df=site_df,
                    top_scores=top_scores,
                    margin_scores=margin_scores,
                    alphas=alphas,
                )
            )
            for target_name, target_col, scores, oversampled_rows in [
                ("top_score", TOP_SCORE, top_scores, top_rows),
                ("zero_margin", MARGIN_TARGET, margin_scores, margin_rows),
            ]:
                metrics = score_metrics(pd.to_numeric(site_df[target_col]).to_numpy(dtype=float), scores)
                metrics.update(
                    {
                        "eval_mode": "capacity",
                        "prediction_target": target_name,
                        "site_id": site_id,
                        "site_fold": 0,
                        "rows": int(len(site_df)),
                        "site_dates": int(len(groups)),
                        "oversampled_train_rows": int(oversampled_rows),
                    }
                )
                metric_rows.append(metrics)
            decision_rows.extend(
                evaluate_curves(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    curves_df=site_df,
                    top_model=top_model,
                    top_cols=top_cols,
                    margin_model=margin_model,
                    margin_cols=margin_cols,
                    feature_mode=args.feature_mode,
                    paper_candidates=paper_candidates,
                    horizon_days=args.horizon_days,
                    grid_step=args.grid_step,
                    alphas=alphas,
                )
            )
            if not args.skip_final_experts:
                expert_path = expert_dir / f"persite_curve_dual_score_tinyforest_ranker_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "top_model": top_model,
                            "top_feature_columns": top_cols,
                            "margin_model": margin_model,
                            "margin_feature_columns": margin_cols,
                            "site_id": site_id,
                            "top_target_column": TOP_SCORE,
                            "margin_target_column": MARGIN_TARGET,
                            "paper_candidates": paper_candidates,
                            "horizon_days": int(args.horizon_days),
                            "grid_step": float(args.grid_step),
                            "fusion_alphas": alphas,
                            "training_rows": int(len(site_df)),
                            "training_site_dates": int(len(groups)),
                        },
                        handle,
                    )
                expert_rows.append(
                    {
                        "site_id": site_id,
                        "expert_path": str(expert_path),
                        "training_rows": int(len(site_df)),
                        "training_site_dates": int(len(groups)),
                    }
                )

        if not args.skip_cv:
            folds = make_group_folds(groups, args.folds_per_site, args.random_state + 1000 + site_idx)
            group_values = site_df["site_date_id"].astype(str).to_numpy()
            for fold_idx, holdout_groups in enumerate(folds, start=1):
                print(
                    f"[persite-dual-score-tinyforest] site={site_id} cv fold {fold_idx}/{len(folds)} "
                    f"holdout_dates={len(holdout_groups)}",
                    flush=True,
                )
                test_mask = np.isin(group_values, np.array(holdout_groups, dtype=str))
                train_mask = ~test_mask
                top_model, top_cols, margin_model, margin_cols, top_rows, margin_rows = fit_dual_models(
                    x_site.loc[train_mask],
                    site_df.loc[train_mask].reset_index(drop=True),
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    top_oversample_factor=args.top_oversample_factor,
                    shoulder_oversample_factor=args.shoulder_oversample_factor,
                    shoulder_regret_eps=args.shoulder_regret_eps,
                    random_state=args.random_state + site_idx * 100 + fold_idx,
                )
                top_scores = predict_top_score(top_model, top_cols, x_site.loc[test_mask])
                margin_scores = predict_top_score(margin_model, margin_cols, x_site.loc[test_mask])
                sampled_rows.extend(
                    sampled_rank_rows(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        eval_df=site_df.loc[test_mask],
                        top_scores=top_scores,
                        margin_scores=margin_scores,
                        alphas=alphas,
                    )
                )
                for target_name, target_col, scores, oversampled_rows in [
                    ("top_score", TOP_SCORE, top_scores, top_rows),
                    ("zero_margin", MARGIN_TARGET, margin_scores, margin_rows),
                ]:
                    metrics = score_metrics(pd.to_numeric(site_df.loc[test_mask, target_col]).to_numpy(dtype=float), scores)
                    metrics.update(
                        {
                            "eval_mode": "heldout_date_cv",
                            "prediction_target": target_name,
                            "site_id": site_id,
                            "site_fold": int(fold_idx),
                            "rows": int(test_mask.sum()),
                            "site_dates": int(len(holdout_groups)),
                            "oversampled_train_rows": int(oversampled_rows),
                        }
                    )
                    metric_rows.append(metrics)
                decision_rows.extend(
                    evaluate_curves(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        curves_df=site_df.loc[test_mask].copy(),
                        top_model=top_model,
                        top_cols=top_cols,
                        margin_model=margin_model,
                        margin_cols=margin_cols,
                        feature_mode=args.feature_mode,
                        paper_candidates=paper_candidates,
                        horizon_days=args.horizon_days,
                        grid_step=args.grid_step,
                        alphas=alphas,
                    )
                )

    sampled = pd.DataFrame(sampled_rows)
    decisions = pd.DataFrame(decision_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    prediction_metrics = prediction_metrics_table(fold_metrics)
    sampled_metrics = sampled_summary(sampled)
    summary = decision_summary(decisions)
    by_site = by_site_summary(decisions)
    manifest = pd.DataFrame(expert_rows)

    sampled_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_sampled_rank_eval_v1.csv"
    decisions_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_prediction_metrics_v1.csv"
    sampled_metrics_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_sampled_rank_metrics_v1.csv"
    summary_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_summary_v1.csv"
    by_site_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_by_site_v1.csv"
    manifest_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_manifest_v1.csv"
    config_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_config_v1.json"
    report_path = out_dir / "persite_curve_dual_score_tinyforest_ranker_v1.md"

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
                "feature_mode": args.feature_mode,
                "top_target": TOP_SCORE,
                "margin_target": MARGIN_TARGET,
                "top_temperature": float(args.top_temperature),
                "top_regret_eps": float(args.top_regret_eps),
                "top_rank_k": int(args.top_rank_k),
                "top_oversample_factor": int(args.top_oversample_factor),
                "shoulder_regret_eps": float(args.shoulder_regret_eps),
                "shoulder_oversample_factor": int(args.shoulder_oversample_factor),
                "n_estimators": int(args.n_estimators),
                "max_depth": int(args.max_depth),
                "min_samples_leaf": int(args.min_samples_leaf),
                "grid_step": float(args.grid_step),
                "folds_per_site": int(args.folds_per_site),
                "fusion_alphas": alphas,
                "paper_candidates": paper_candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Per-Site Curve Dual-Score TinyForest Ranker V1",
        "",
        "## Scope",
        "",
        "- Per-site expert only; no cross-site MoE or gating.",
        "- Trains topness and zero-margin experts in the same folds.",
        "- Evaluates raw-score and within-curve rank fusions.",
        f"- Input: `{data_path}`.",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_metrics),
        "",
        "## Sampled Curve Rank Metrics",
        "",
        markdown_table(sampled_metrics),
        "",
        "## Dense Decision Summary",
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

    print("Per-site curve dual-score TinyForest ranker v1")
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
    print("Sampled rank metrics")
    print(sampled_metrics.to_string(index=False))
    print("")
    print("Dense decision summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

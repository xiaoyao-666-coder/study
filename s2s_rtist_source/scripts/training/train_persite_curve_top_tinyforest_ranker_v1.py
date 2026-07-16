#!/usr/bin/env python3
"""Train per-site curve-top TinyForest rankers for SWAP response curves.

This diagnostic follows the teacher's latest direction: before MoE or any
cross-site gating, each per-site expert must first select the top of its own
SWAP irrigation response curves. The target is not global profit R2. Instead,
each site-date curve is normalized by its own best SWAP gain and the model is
trained to score candidates near the curve top highest.

The implementation intentionally stays CPU-only and per-site:

- build curve-local regret = curve_best_gain - candidate_gain;
- train top_score = exp(-regret / temperature);
- oversample top candidates so TinyForest capacity is spent near the argmax;
- report sampled top1/top3/top5 and dense continuous decision ranks.
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
    / "continuous_ir_12site_10k_persite_curve_top_tinyforest_ranker_v1"
)
TOP_SCORE = "curve_top_score"


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if not x[col].isna().all()]


def add_curve_top_targets(
    df: pd.DataFrame,
    *,
    top_temperature: float,
    top_regret_eps: float,
    top_rank_k: int,
) -> pd.DataFrame:
    out = df.copy()
    out["candidate_ir"] = pd.to_numeric(out["candidate_ir"], errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = out.dropna(subset=["candidate_ir", TARGET]).copy()
    group = out.groupby("site_date_id", sort=False)[TARGET]
    out["curve_best_gain"] = group.transform("max")
    out["curve_regret"] = out["curve_best_gain"] - out[TARGET]
    out["curve_true_rank"] = (
        out.groupby("site_date_id", sort=False)[TARGET]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    temp = max(float(top_temperature), 1e-9)
    out[TOP_SCORE] = np.exp(-np.clip(out["curve_regret"].to_numpy(dtype=float), 0.0, np.inf) / temp)
    out["curve_top_label"] = (
        (out["curve_regret"] <= float(top_regret_eps) + 1e-9)
        | (out["curve_true_rank"] <= int(top_rank_k))
    )
    return out


def oversampled_training_frame(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    *,
    top_oversample_factor: int,
    shoulder_oversample_factor: int,
    shoulder_regret_eps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    repeats = np.ones(len(meta_train), dtype=int)
    top_mask = meta_train["curve_top_label"].to_numpy(dtype=bool)
    shoulder_mask = meta_train["curve_regret"].to_numpy(dtype=float) <= float(shoulder_regret_eps) + 1e-9
    repeats[shoulder_mask] += max(0, int(shoulder_oversample_factor) - 1)
    repeats[top_mask] += max(0, int(top_oversample_factor) - 1)
    idx = np.repeat(np.arange(len(meta_train)), repeats)
    x_os = x_train.iloc[idx].reset_index(drop=True)
    y_os = y_train.iloc[idx].reset_index(drop=True)
    return x_os, y_os


def fit_top_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    top_oversample_factor: int,
    shoulder_oversample_factor: int,
    shoulder_regret_eps: float,
    random_state: int,
) -> tuple[TinyForest, list[str], int]:
    cols = usable_columns(x_train)
    if not cols:
        raise ValueError("No usable feature columns for curve-top TinyForest")
    x_os, y_os = oversampled_training_frame(
        x_train[cols],
        y_train,
        meta_train,
        top_oversample_factor=top_oversample_factor,
        shoulder_oversample_factor=shoulder_oversample_factor,
        shoulder_regret_eps=shoulder_regret_eps,
    )
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_os, y_os)
    return model, cols, int(len(x_os))


def predict_top_score(model: TinyForest, cols: list[str], x: pd.DataFrame) -> np.ndarray:
    aligned = x.reindex(columns=cols, fill_value=0.0)
    return model.predict(aligned)


def add_true_rank_columns(rows: pd.DataFrame, gain_col: str = "interp_true_net_gain_7d") -> pd.DataFrame:
    out = rows.copy()
    out["dense_true_rank"] = out[gain_col].rank(method="min", ascending=False).astype(int)
    best_gain = float(out[gain_col].max())
    out["dense_curve_regret"] = best_gain - out[gain_col].to_numpy(dtype=float)
    return out


def sampled_rank_rows(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    eval_df: pd.DataFrame,
    scores: np.ndarray,
) -> list[dict]:
    scored = eval_df[
        [
            "site_date_id",
            "site_id",
            "date_t",
            "candidate_ir",
            TARGET,
            "curve_best_gain",
            "curve_regret",
            "curve_true_rank",
            "curve_top_label",
            TOP_SCORE,
        ]
    ].copy()
    scored["pred_top_score"] = scores
    rows = []
    for site_date_id, part in scored.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_top_score"].idxmax()]
        score_rank = part["pred_top_score"].rank(method="min", ascending=False)
        rows.append(
            {
                "eval_mode": eval_mode,
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
                "true_best_pred_rank": int(score_rank.loc[true_best.name]),
                "pred_best_top_score": float(pred_best["pred_top_score"]),
                "true_best_pred_top_score": float(true_best["pred_top_score"]),
            }
        )
    return rows


def evaluate_curves(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    curves_df: pd.DataFrame,
    model: TinyForest,
    feature_cols: list[str],
    feature_mode: str,
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
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
        fixed_x = select_feature_mode(build_features(fixed_rows), feature_mode)
        fixed_rows["pred_top_score"] = predict_top_score(model, feature_cols, fixed_x)
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred_best = fixed_rows.loc[fixed_rows["pred_top_score"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt")
        dense_rows = add_interp_truth(dense_rows, curve)
        dense_rows = add_true_rank_columns(dense_rows)
        dense_x = select_feature_mode(build_features(dense_rows), feature_mode)
        dense_rows["pred_top_score"] = predict_top_score(model, feature_cols, dense_x)
        continuous_pred_best = dense_rows.loc[dense_rows["pred_top_score"].idxmax()]

        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
        continuous_rank = int(continuous_pred_best["dense_true_rank"])
        nearest_fixed = min(fixed_values, key=lambda value: abs(float(value) - float(continuous_pred_best["candidate_ir"])))
        decision_rows.append(
            {
                "eval_mode": eval_mode,
                "site_id": site_id,
                "site_fold": fold_id,
                "site_date_id": str(site_date_id),
                "date_t": str(dense_oracle["date_t"]),
                "site_ir_max": site_ir_max,
                "dense_oracle_ir": dense_oracle_ir,
                "dense_oracle_gain": dense_oracle_gain,
                "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                "paper_regret_vs_dense_oracle": dense_oracle_gain - fixed_oracle_gain,
                "fixed_list_top_ranker_ir": float(fixed_pred_best["candidate_ir"]),
                "fixed_list_top_ranker_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                "fixed_list_top_ranker_pred_score": float(fixed_pred_best["pred_top_score"]),
                "fixed_list_top_ranker_regret_vs_fixed_oracle": fixed_oracle_gain
                - float(fixed_pred_best["interp_true_net_gain_7d"]),
                "continuous_top_ranker_ir": float(continuous_pred_best["candidate_ir"]),
                "continuous_top_ranker_true_gain": continuous_gain,
                "continuous_top_ranker_pred_score": float(continuous_pred_best["pred_top_score"]),
                "continuous_top_ranker_regret_vs_dense_oracle": dense_oracle_gain - continuous_gain,
                "continuous_top_ranker_gain_over_paper": continuous_gain - fixed_oracle_gain,
                "continuous_top_ranker_better_than_paper": continuous_gain > fixed_oracle_gain + 1e-9,
                "continuous_top_ranker_worse_than_paper": continuous_gain < fixed_oracle_gain - 1e-9,
                "continuous_top_ranker_nearest_fixed_ir": float(nearest_fixed),
                "continuous_top_ranker_distance_to_nearest_fixed_ir": abs(
                    float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                ),
                "continuous_top_ranker_nonfixed_ir": abs(
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


def decision_summary(decisions: pd.DataFrame, mode: str) -> pd.DataFrame:
    part = decisions.loc[decisions["eval_mode"] == mode].copy()
    if part.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "eval_mode": mode,
                "sites": int(part["site_id"].nunique()),
                "site_dates": int(part["site_date_id"].nunique()),
                "paper_fixed_list_mean_regret_vs_dense": float(part["paper_regret_vs_dense_oracle"].mean()),
                "fixed_list_top_ranker_mean_regret_vs_fixed_oracle": float(
                    part["fixed_list_top_ranker_regret_vs_fixed_oracle"].mean()
                ),
                "continuous_top_ranker_mean_regret_vs_dense": float(
                    part["continuous_top_ranker_regret_vs_dense_oracle"].mean()
                ),
                "continuous_top_ranker_median_regret_vs_dense": float(
                    part["continuous_top_ranker_regret_vs_dense_oracle"].median()
                ),
                "continuous_top_ranker_p90_regret_vs_dense": float(
                    np.quantile(part["continuous_top_ranker_regret_vs_dense_oracle"], 0.9)
                ),
                "continuous_top_ranker_mean_gain_over_paper": float(
                    part["continuous_top_ranker_gain_over_paper"].mean()
                ),
                "continuous_top_ranker_better_than_paper_rate": safe_mean(
                    part["continuous_top_ranker_better_than_paper"].to_numpy(dtype=float)
                ),
                "continuous_top_ranker_worse_than_paper_rate": safe_mean(
                    part["continuous_top_ranker_worse_than_paper"].to_numpy(dtype=float)
                ),
                "continuous_top_ranker_nonfixed_ir_rate": safe_mean(
                    part["continuous_top_ranker_nonfixed_ir"].to_numpy(dtype=float)
                ),
                "continuous_selected_top1_rate": safe_mean(part["continuous_selected_top1"].to_numpy(dtype=float)),
                "continuous_selected_top3_rate": safe_mean(part["continuous_selected_top3"].to_numpy(dtype=float)),
                "continuous_selected_top5_rate": safe_mean(part["continuous_selected_top5"].to_numpy(dtype=float)),
                "mean_continuous_selected_dense_rank": float(part["continuous_selected_dense_rank"].mean()),
                "large_regret_gt_5_rate": safe_mean(part["continuous_large_regret_gt_5"].to_numpy(dtype=float)),
            }
        ]
    )


def by_site_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["eval_mode", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            continuous_top_ranker_mean_regret_vs_dense=("continuous_top_ranker_regret_vs_dense_oracle", "mean"),
            continuous_top_ranker_mean_gain_over_paper=("continuous_top_ranker_gain_over_paper", "mean"),
            continuous_top_ranker_better_than_paper_rate=("continuous_top_ranker_better_than_paper", "mean"),
            continuous_top_ranker_nonfixed_ir_rate=("continuous_top_ranker_nonfixed_ir", "mean"),
            continuous_selected_top3_rate=("continuous_selected_top3", "mean"),
            mean_continuous_selected_dense_rank=("continuous_selected_dense_rank", "mean"),
            large_regret_gt_5_rate=("continuous_large_regret_gt_5", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "continuous_top_ranker_mean_regret_vs_dense"], ascending=[True, False])
    )


def sampled_summary(sampled: pd.DataFrame) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame()
    return (
        sampled.groupby("eval_mode")
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

    sites = sorted(df["site_id"].astype(str).unique())
    if args.site_limit and args.site_limit > 0:
        sites = sites[: args.site_limit]

    sampled_rows: list[dict] = []
    decision_rows: list[dict] = []
    metric_rows: list[dict] = []
    expert_rows: list[dict] = []

    print(
        f"[persite-curve-top-tinyforest] rows={len(df)} sites={len(sites)} "
        f"capacity={not args.skip_capacity} cv={not args.skip_cv}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        x_site = select_feature_mode(build_features(site_df), args.feature_mode)
        y_site = pd.to_numeric(site_df[TOP_SCORE], errors="coerce")
        if y_site.isna().any():
            raise ValueError(f"Top-score target contains NaN for site {site_id}")
        groups = sorted(site_df["site_date_id"].astype(str).unique())
        print(f"[persite-curve-top-tinyforest] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            print(f"[persite-curve-top-tinyforest] site={site_id} capacity fit", flush=True)
            model, cols, oversampled_rows = fit_top_forest(
                x_site,
                y_site,
                site_df,
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
                    eval_df=site_df,
                    scores=scores,
                )
            )
            metrics = score_metrics(y_site.to_numpy(dtype=float), scores)
            metrics.update(
                {
                    "eval_mode": "capacity",
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
                    model=model,
                    feature_cols=cols,
                    feature_mode=args.feature_mode,
                    paper_candidates=paper_candidates,
                    horizon_days=args.horizon_days,
                    grid_step=args.grid_step,
                )
            )
            if not args.skip_final_experts:
                expert_path = expert_dir / f"persite_curve_top_tinyforest_ranker_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": model,
                            "feature_columns": cols,
                            "site_id": site_id,
                            "target_column": TOP_SCORE,
                            "source_profit_column": TARGET,
                            "paper_candidates": paper_candidates,
                            "horizon_days": int(args.horizon_days),
                            "grid_step": float(args.grid_step),
                            "top_temperature": float(args.top_temperature),
                            "top_regret_eps": float(args.top_regret_eps),
                            "top_rank_k": int(args.top_rank_k),
                            "training_rows": int(len(site_df)),
                            "oversampled_training_rows": int(oversampled_rows),
                            "training_site_dates": int(len(groups)),
                        },
                        handle,
                    )
                expert_rows.append(
                    {
                        "site_id": site_id,
                        "expert_path": str(expert_path),
                        "training_rows": int(len(site_df)),
                        "oversampled_training_rows": int(oversampled_rows),
                        "training_site_dates": int(len(groups)),
                    }
                )

        if not args.skip_cv:
            folds = make_group_folds(groups, args.folds_per_site, args.random_state + 1000 + site_idx)
            group_values = site_df["site_date_id"].astype(str).to_numpy()
            for fold_idx, holdout_groups in enumerate(folds, start=1):
                print(
                    f"[persite-curve-top-tinyforest] site={site_id} cv fold {fold_idx}/{len(folds)} "
                    f"holdout_dates={len(holdout_groups)}",
                    flush=True,
                )
                test_mask = np.isin(group_values, np.array(holdout_groups, dtype=str))
                train_mask = ~test_mask
                model, cols, oversampled_rows = fit_top_forest(
                    x_site.loc[train_mask],
                    y_site.loc[train_mask],
                    site_df.loc[train_mask].reset_index(drop=True),
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
                        eval_df=site_df.loc[test_mask],
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
                        model=model,
                        feature_cols=cols,
                        feature_mode=args.feature_mode,
                        paper_candidates=paper_candidates,
                        horizon_days=args.horizon_days,
                        grid_step=args.grid_step,
                    )
                )

    sampled = pd.DataFrame(sampled_rows)
    decisions = pd.DataFrame(decision_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    sampled_metrics = sampled_summary(sampled)
    summary_parts = []
    for mode in ["capacity", "heldout_date_cv"]:
        part = decision_summary(decisions, mode)
        if not part.empty:
            summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    by_site = by_site_summary(decisions)
    prediction_metrics = (
        fold_metrics.groupby("eval_mode")
        .agg(
            folds=("site_fold", "count"),
            rows=("rows", "sum"),
            mean_oversampled_train_rows=("oversampled_train_rows", "mean"),
            top_score_mae=("mae", "mean"),
            top_score_rmse=("rmse", "mean"),
            top_score_r2=("r2", "mean"),
        )
        .reset_index()
        if not fold_metrics.empty
        else pd.DataFrame()
    )
    manifest = pd.DataFrame(expert_rows)

    sampled_path = out_dir / "persite_curve_top_tinyforest_ranker_sampled_rank_eval_v1.csv"
    decisions_path = out_dir / "persite_curve_top_tinyforest_ranker_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_curve_top_tinyforest_ranker_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_curve_top_tinyforest_ranker_prediction_metrics_v1.csv"
    sampled_metrics_path = out_dir / "persite_curve_top_tinyforest_ranker_sampled_rank_metrics_v1.csv"
    summary_path = out_dir / "persite_curve_top_tinyforest_ranker_summary_v1.csv"
    by_site_path = out_dir / "persite_curve_top_tinyforest_ranker_by_site_v1.csv"
    manifest_path = out_dir / "persite_curve_top_tinyforest_ranker_manifest_v1.csv"
    config_path = out_dir / "persite_curve_top_tinyforest_ranker_config_v1.json"
    report_path = out_dir / "persite_curve_top_tinyforest_ranker_v1.md"

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
                "top_score": TOP_SCORE,
                "source_profit_column": TARGET,
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
                "paper_candidates": paper_candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    worst = (
        decisions.sort_values("continuous_top_ranker_regret_vs_dense_oracle", ascending=False).head(30)
        if not decisions.empty
        else pd.DataFrame()
    )
    best = (
        decisions.sort_values("continuous_top_ranker_gain_over_paper", ascending=False).head(30)
        if not decisions.empty
        else pd.DataFrame()
    )
    lines = [
        "# Per-Site Curve-Top TinyForest Ranker V1",
        "",
        "## Scope",
        "",
        "- Per-site expert only; no cross-site MoE or gating.",
        "- Target is curve-local topness, not absolute SWAP profit R2.",
        "- Top candidates are oversampled to emphasize argmax quality.",
        f"- Input: `{data_path}`.",
        "",
        "## Top-Score Prediction Metrics",
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
        "## Worst Dense Decisions",
        "",
        markdown_table(worst),
        "",
        "## Best Gains Over Paper Fixed List",
        "",
        markdown_table(best),
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

    print("Per-site curve-top TinyForest ranker v1")
    print(f"sampled_rank_eval: {sampled_path}")
    print(f"decisions: {decisions_path}")
    print(f"prediction_metrics: {prediction_metrics_path}")
    print(f"sampled_rank_metrics: {sampled_metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    print("")
    print("Top-score prediction metrics")
    print(prediction_metrics.to_string(index=False))
    print("")
    print("Sampled rank metrics")
    print(sampled_metrics.to_string(index=False))
    print("")
    print("Dense decision summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

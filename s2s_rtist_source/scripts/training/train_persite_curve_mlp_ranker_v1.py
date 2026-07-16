#!/usr/bin/env python3
"""Train per-site curve-aware MLP rankers for irrigation response curves.

This is the next diagnostic after per-site TinyForest capacity passed but
held-out-date and few-shot decision selection remained poor. Instead of fitting
absolute SWAP profit point by point, this model trains a listwise ranking loss
inside each site-date irrigation curve, so the training target is aligned with
the downstream argmax decision.

The script intentionally stays per-site. It must pass before any MoE or
cross-site generalization stage is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
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
    / "continuous_ir_12site_10k_persite_curve_mlp_ranker_v1"
)


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("At least one hidden layer size is required")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Hidden sizes must be positive: {sizes}")
    return sizes


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def usable_columns(x_train: pd.DataFrame) -> list[str]:
    return [col for col in x_train.columns if not x_train[col].isna().all()]


def softmax_np(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values.astype(float) / max(float(temperature), 1e-9)
    scaled = scaled - np.max(scaled)
    exp_values = np.exp(np.clip(scaled, -60.0, 60.0))
    denom = float(exp_values.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(exp_values, 1.0 / len(exp_values), dtype=float)
    return exp_values / denom


def init_params(layer_sizes: list[int], rng: np.random.Generator) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {}
    for i in range(len(layer_sizes) - 1):
        fan_in = max(1, int(layer_sizes[i]))
        scale = math.sqrt(2.0 / fan_in)
        params[f"W{i}"] = rng.normal(0.0, scale, size=(layer_sizes[i], layer_sizes[i + 1]))
        params[f"b{i}"] = np.zeros((1, layer_sizes[i + 1]), dtype=float)
    return params


def forward(
    x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    caches: list[tuple[np.ndarray, np.ndarray]] = []
    a = x
    n_layers = len(params) // 2
    for i in range(n_layers):
        z = a @ params[f"W{i}"] + params[f"b{i}"]
        caches.append((a, z))
        if i == n_layers - 1:
            return z.reshape(-1), caches
        a = np.maximum(z, 0.0)
    raise RuntimeError("Invalid MLP parameter state")


def backward(
    grad_out: np.ndarray,
    caches: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, np.ndarray],
    weight_decay: float,
) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}
    grad = grad_out.reshape(-1, 1)
    n_layers = len(params) // 2
    for i in reversed(range(n_layers)):
        a_prev, _z = caches[i]
        grads[f"W{i}"] = a_prev.T @ grad + float(weight_decay) * params[f"W{i}"]
        grads[f"b{i}"] = grad.sum(axis=0, keepdims=True)
        if i > 0:
            grad = grad @ params[f"W{i}"].T
            prev_z = caches[i - 1][1]
            grad = grad * (prev_z > 0.0)
    return grads


def fit_transformer(
    x_train: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray | list[str]]]:
    cols = usable_columns(x_train)
    if not cols:
        raise ValueError("No usable feature columns")
    arr = x_train[cols].to_numpy(dtype=float)
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(med, inds[1])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    return (arr - mean) / std, {"columns": cols, "median": med, "mean": mean, "std": std}


def transform_features(x: pd.DataFrame, transformer: dict[str, np.ndarray | list[str]]) -> np.ndarray:
    cols = list(transformer["columns"])
    arr = x.reindex(columns=cols).to_numpy(dtype=float)
    med = np.asarray(transformer["median"], dtype=float)
    mean = np.asarray(transformer["mean"], dtype=float)
    std = np.asarray(transformer["std"], dtype=float)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(med, inds[1])
    return (arr - mean) / std


def make_group_indices(site_date_ids: np.ndarray) -> list[np.ndarray]:
    groups = []
    for site_date_id in pd.Series(site_date_ids).drop_duplicates().tolist():
        groups.append(np.where(site_date_ids == site_date_id)[0])
    return groups


def fit_ranker(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    site_date_train: pd.Series,
    *,
    hidden_sizes: list[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    target_temperature: float,
    profit_aux_weight: float,
    random_state: int,
    verbose: bool,
) -> dict:
    x_arr, transformer = fit_transformer(x_train)
    y = y_train.to_numpy(dtype=float)
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std <= 1e-12:
        y_std = 1.0
    y_scaled_all = (y - y_mean) / y_std

    rng = np.random.default_rng(random_state)
    params = init_params([x_arr.shape[1], *hidden_sizes, 1], rng)
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    velocities = {key: np.zeros_like(value) for key, value in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    group_indices = make_group_indices(site_date_train.astype(str).to_numpy())
    for epoch in range(1, int(epochs) + 1):
        losses = []
        for group_pos in rng.permutation(len(group_indices)):
            idx = group_indices[int(group_pos)]
            if len(idx) < 2:
                continue
            scores, caches = forward(x_arr[idx], params)
            target_dist = softmax_np(y[idx], target_temperature)
            pred_dist = softmax_np(scores, 1.0)
            grad_scores = pred_dist - target_dist
            ce_loss = -float(np.sum(target_dist * np.log(np.clip(pred_dist, 1e-12, 1.0))))
            if profit_aux_weight > 0.0:
                aux_err = scores - y_scaled_all[idx]
                grad_scores = grad_scores + float(profit_aux_weight) * (2.0 / len(idx)) * aux_err
                ce_loss += float(profit_aux_weight) * float(np.mean(aux_err * aux_err))
            losses.append(ce_loss)
            grads = backward(grad_scores, caches, params, weight_decay)
            step += 1
            for key in params:
                moments[key] = beta1 * moments[key] + (1.0 - beta1) * grads[key]
                velocities[key] = beta2 * velocities[key] + (1.0 - beta2) * (grads[key] ** 2)
                m_hat = moments[key] / (1.0 - beta1**step)
                v_hat = velocities[key] / (1.0 - beta2**step)
                params[key] -= float(lr) * m_hat / (np.sqrt(v_hat) + eps)
        if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == epochs):
            print(
                f"[curve-mlp] epoch {epoch}/{epochs} listwise_loss={safe_mean(np.asarray(losses)):.6f}",
                flush=True,
            )
    return {
        "params": params,
        "transformer": transformer,
        "hidden_sizes": hidden_sizes,
        "target_temperature": float(target_temperature),
        "profit_aux_weight": float(profit_aux_weight),
        "y_mean": y_mean,
        "y_std": y_std,
    }


def predict_ranker(model: dict, x: pd.DataFrame) -> np.ndarray:
    arr = transform_features(x, model["transformer"])
    scores, _caches = forward(arr, model["params"])
    return scores


def sampled_rank_rows(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    eval_df: pd.DataFrame,
    scores: np.ndarray,
) -> list[dict]:
    scored = eval_df[["site_date_id", "site_id", "date_t", "candidate_ir", TARGET]].copy()
    scored["pred_rank_score"] = scores
    rows = []
    for site_date_id, part in scored.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_rank_score"].idxmax()]
        score_rank = part["pred_rank_score"].rank(method="min", ascending=False)
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
                "sampled_top1_correct": abs(float(true_best["candidate_ir"]) - float(pred_best["candidate_ir"])) <= 1e-9,
                "true_best_pred_rank": int(score_rank.loc[true_best.name]),
            }
        )
    return rows


def evaluate_curves(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    curves_df: pd.DataFrame,
    model: dict,
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
        fixed_x = select_feature_mode(build_features(fixed_rows), feature_mode)
        fixed_rows["pred_rank_score"] = predict_ranker(model, fixed_x)
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred_best = fixed_rows.loc[fixed_rows["pred_rank_score"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt")
        dense_rows = add_interp_truth(dense_rows, curve)
        dense_x = select_feature_mode(build_features(dense_rows), feature_mode)
        dense_rows["pred_rank_score"] = predict_ranker(model, dense_x)
        continuous_pred_best = dense_rows.loc[dense_rows["pred_rank_score"].idxmax()]

        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
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
                "fixed_list_ranker_ir": float(fixed_pred_best["candidate_ir"]),
                "fixed_list_ranker_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                "fixed_list_ranker_regret_vs_fixed_oracle": fixed_oracle_gain
                - float(fixed_pred_best["interp_true_net_gain_7d"]),
                "continuous_ranker_ir": float(continuous_pred_best["candidate_ir"]),
                "continuous_ranker_true_gain": continuous_gain,
                "continuous_ranker_regret_vs_dense_oracle": dense_oracle_gain - continuous_gain,
                "continuous_ranker_gain_over_paper": continuous_gain - fixed_oracle_gain,
                "continuous_ranker_better_than_paper": continuous_gain > fixed_oracle_gain + 1e-9,
                "continuous_ranker_worse_than_paper": continuous_gain < fixed_oracle_gain - 1e-9,
                "continuous_ranker_nearest_fixed_ir": float(nearest_fixed),
                "continuous_ranker_distance_to_nearest_fixed_ir": abs(
                    float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                ),
                "continuous_ranker_nonfixed_ir": abs(
                    float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                )
                > 1e-9,
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
                "fixed_list_ranker_mean_regret_vs_fixed_oracle": float(
                    part["fixed_list_ranker_regret_vs_fixed_oracle"].mean()
                ),
                "continuous_ranker_mean_regret_vs_dense": float(
                    part["continuous_ranker_regret_vs_dense_oracle"].mean()
                ),
                "continuous_ranker_mean_gain_over_paper": float(part["continuous_ranker_gain_over_paper"].mean()),
                "continuous_ranker_better_than_paper_rate": safe_mean(
                    part["continuous_ranker_better_than_paper"].to_numpy(dtype=float)
                ),
                "continuous_ranker_worse_than_paper_rate": safe_mean(
                    part["continuous_ranker_worse_than_paper"].to_numpy(dtype=float)
                ),
                "continuous_ranker_nonfixed_ir_rate": safe_mean(
                    part["continuous_ranker_nonfixed_ir"].to_numpy(dtype=float)
                ),
            }
        ]
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
            sampled_mean_curve_regret=("sampled_curve_regret", "mean"),
            sampled_median_curve_regret=("sampled_curve_regret", "median"),
            mean_true_best_pred_rank=("true_best_pred_rank", "mean"),
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
            fixed_list_ranker_mean_regret_vs_fixed_oracle=(
                "fixed_list_ranker_regret_vs_fixed_oracle",
                "mean",
            ),
            continuous_ranker_mean_regret_vs_dense=("continuous_ranker_regret_vs_dense_oracle", "mean"),
            continuous_ranker_mean_gain_over_paper=("continuous_ranker_gain_over_paper", "mean"),
            continuous_ranker_better_than_paper_rate=("continuous_ranker_better_than_paper", "mean"),
            continuous_ranker_nonfixed_ir_rate=("continuous_ranker_nonfixed_ir", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "continuous_ranker_mean_regret_vs_dense"], ascending=[True, False])
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
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=10.0)
    parser.add_argument("--profit-aux-weight", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--site-limit", type=int, default=0)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--skip-final-experts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.skip_cv and args.skip_capacity:
        raise ValueError("At least one of CV or capacity check must run")
    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sequence-wide sample table: {data_path}")
    if args.grid_step <= 0:
        raise ValueError("--grid-step must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = out_dir / "final_site_experts_v1"
    if not args.skip_final_experts:
        expert_dir.mkdir(parents=True, exist_ok=True)

    paper_candidates = parse_candidates(args.paper_candidates)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    df = pd.read_csv(data_path)
    required = {"site_id", "site_date_id", "date_t", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])

    sites = sorted(df["site_id"].astype(str).unique())
    if args.site_limit and args.site_limit > 0:
        sites = sites[: args.site_limit]

    sampled_rows: list[dict] = []
    decision_rows: list[dict] = []
    expert_rows: list[dict] = []
    print(
        f"[persite-curve-mlp] rows={len(df)} sites={len(sites)} "
        f"capacity={not args.skip_capacity} cv={not args.skip_cv}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        x_site = select_feature_mode(build_features(site_df), args.feature_mode)
        y_site = pd.to_numeric(site_df[TARGET], errors="coerce")
        if y_site.isna().any():
            raise ValueError(f"Target contains NaN for site {site_id}")
        groups = sorted(site_df["site_date_id"].astype(str).unique())
        print(f"[persite-curve-mlp] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            print(f"[persite-curve-mlp] site={site_id} capacity fit", flush=True)
            model = fit_ranker(
                x_site,
                y_site,
                site_df["site_date_id"].astype(str),
                hidden_sizes=hidden_sizes,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                target_temperature=args.target_temperature,
                profit_aux_weight=args.profit_aux_weight,
                random_state=args.random_state + site_idx,
                verbose=args.verbose,
            )
            scores = predict_ranker(model, x_site)
            sampled_rows.extend(
                sampled_rank_rows(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    eval_df=site_df,
                    scores=scores,
                )
            )
            decision_rows.extend(
                evaluate_curves(
                    eval_mode="capacity",
                    site_id=site_id,
                    fold_id=0,
                    curves_df=site_df,
                    model=model,
                    feature_mode=args.feature_mode,
                    paper_candidates=paper_candidates,
                    horizon_days=args.horizon_days,
                    grid_step=args.grid_step,
                )
            )
            if not args.skip_final_experts:
                expert_path = expert_dir / f"persite_curve_mlp_ranker_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as f:
                    pickle.dump(model, f)
                expert_rows.append(
                    {
                        "site_id": site_id,
                        "expert_path": str(expert_path),
                        "rows": int(len(site_df)),
                        "site_dates": int(len(groups)),
                    }
                )

        if not args.skip_cv:
            folds = make_group_folds(groups, args.folds_per_site, args.random_state + 1000 + site_idx)
            for fold_idx, test_groups in enumerate(folds, start=1):
                print(
                    f"[persite-curve-mlp] site={site_id} cv fold {fold_idx}/{len(folds)}",
                    flush=True,
                )
                test_mask = site_df["site_date_id"].astype(str).isin(test_groups)
                train_mask = ~test_mask
                model = fit_ranker(
                    x_site.loc[train_mask],
                    y_site.loc[train_mask],
                    site_df.loc[train_mask, "site_date_id"].astype(str),
                    hidden_sizes=hidden_sizes,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    target_temperature=args.target_temperature,
                    profit_aux_weight=args.profit_aux_weight,
                    random_state=args.random_state + site_idx * 100 + fold_idx,
                    verbose=args.verbose,
                )
                scores = predict_ranker(model, x_site.loc[test_mask])
                sampled_rows.extend(
                    sampled_rank_rows(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        eval_df=site_df.loc[test_mask],
                        scores=scores,
                    )
                )
                decision_rows.extend(
                    evaluate_curves(
                        eval_mode="heldout_date_cv",
                        site_id=site_id,
                        fold_id=fold_idx,
                        curves_df=site_df.loc[test_mask],
                        model=model,
                        feature_mode=args.feature_mode,
                        paper_candidates=paper_candidates,
                        horizon_days=args.horizon_days,
                        grid_step=args.grid_step,
                    )
                )

    sampled = pd.DataFrame(sampled_rows)
    decisions = pd.DataFrame(decision_rows)
    sampled_metrics = sampled_summary(sampled)
    summary_parts = []
    for mode in ["capacity", "heldout_date_cv"]:
        part = decision_summary(decisions, mode)
        if not part.empty:
            summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    by_site = by_site_summary(decisions)
    manifest = pd.DataFrame(expert_rows)

    sampled_path = out_dir / "persite_curve_mlp_ranker_sampled_rank_eval_v1.csv"
    decisions_path = out_dir / "persite_curve_mlp_ranker_decisions_v1.csv"
    sampled_metrics_path = out_dir / "persite_curve_mlp_ranker_sampled_rank_metrics_v1.csv"
    summary_path = out_dir / "persite_curve_mlp_ranker_summary_v1.csv"
    by_site_path = out_dir / "persite_curve_mlp_ranker_by_site_v1.csv"
    manifest_path = out_dir / "persite_curve_mlp_ranker_manifest_v1.csv"
    config_path = out_dir / "persite_curve_mlp_ranker_config_v1.json"
    report_path = out_dir / "persite_curve_mlp_ranker_v1.md"

    sampled.to_csv(sampled_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    sampled_metrics.to_csv(sampled_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "input": str(data_path),
                "feature_mode": args.feature_mode,
                "hidden_sizes": hidden_sizes,
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "target_temperature": float(args.target_temperature),
                "profit_aux_weight": float(args.profit_aux_weight),
                "grid_step": float(args.grid_step),
                "folds_per_site": int(args.folds_per_site),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Per-Site Curve-Aware MLP Ranker V1",
        "",
        "## Scope",
        "",
        "- Per-site listwise MLP ranker over irrigation response curves.",
        "- Trains within each site-date curve, then evaluates fixed-list and dense continuous decisions.",
        f"- Input: `{data_path}`.",
        "",
        "## Sampled Curve Rank Metrics",
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
        f"- `{sampled_metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{manifest_path}`",
        f"- `{config_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site curve-aware MLP ranker v1")
    print(f"sampled_rank_eval: {sampled_path}")
    print(f"decisions: {decisions_path}")
    print(f"sampled_rank_metrics: {sampled_metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    print("")
    print(sampled_metrics.to_string(index=False))
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

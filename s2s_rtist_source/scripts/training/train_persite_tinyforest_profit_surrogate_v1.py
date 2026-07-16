#!/usr/bin/env python3
"""Per-site TinyForest profit surrogate capacity and validation diagnostics.

This script is the no-GPU follow-up after the failed per-site LSTM run. It asks
two teacher-aligned questions before MoE:

1. Capacity check: if each site expert is trained on all dates from that site,
   can it fit the SWAP profit curve on sampled/fixed-list irrigation inputs?
2. Held-out-date CV: when dates are held out within the same site, does the
   per-site expert still select reasonable fixed-list and continuous irrigation
   amounts?

The script evaluates decisions by interpolating the SWAP-sampled response curve.
It intentionally avoids cross-site generalization.
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
    interp_gain,
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics


DEFAULT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_persite_tinyforest_profit_surrogate_v1"
)


def safe_mean(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def sample_id(site: str, date_text: str, ir: float, prefix: str) -> str:
    date_token = str(date_text).replace("-", "").replace("/", "").replace(" ", "").lower()
    ir_token = f"{float(ir):.6g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{site}_{date_token}_ir{ir_token}"


def candidate_sequence(ir: float, horizon_days: int) -> str:
    return json.dumps([float(ir)] + [0.0] * max(horizon_days - 1, 0), separators=(",", ":"))


def update_candidate_columns(row: pd.Series, ir: float, horizon_days: int, prefix: str) -> pd.Series:
    row = row.copy()
    site = str(row["site_id"])
    row["candidate_ir"] = float(ir)
    row["ir"] = float(ir)
    row["candidate_ir_sq"] = float(ir) ** 2
    row["is_zero_ir"] = 1 if abs(float(ir)) <= 1e-12 else 0
    site_ir_max = float(row["site_ir_max"])
    row["candidate_ir_fraction"] = float(ir) / site_ir_max if site_ir_max > 0 else 0.0
    row["candidate_ir_fraction_sq"] = row["candidate_ir_fraction"] ** 2
    row["candidate_ir_sequence"] = candidate_sequence(float(ir), horizon_days)
    row["sample_id"] = sample_id(site, str(row["date_t"]), float(ir), prefix)
    for day in range(1, horizon_days + 1):
        col = f"future_ir_day{day:02d}"
        if col in row.index:
            row[col] = float(ir) if day == 1 else 0.0
    return row


def dense_values(max_ir: float, step: float, extra_values: list[float]) -> np.ndarray:
    grid = np.arange(0.0, float(max_ir) + step * 0.5, step, dtype=float)
    values = np.concatenate([grid, np.array(extra_values + [float(max_ir)], dtype=float)])
    values = np.unique(np.round(values, 6))
    return values[(values >= -1e-9) & (values <= float(max_ir) + 1e-9)]


def build_candidate_rows(
    curve: pd.DataFrame,
    irrigation_values: list[float] | np.ndarray,
    *,
    horizon_days: int,
    prefix: str,
) -> pd.DataFrame:
    base = curve.sort_values("candidate_ir").iloc[0]
    rows = [update_candidate_columns(base, float(ir), horizon_days, prefix) for ir in irrigation_values]
    return pd.DataFrame(rows).reset_index(drop=True)


def add_interp_truth(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["interp_true_net_gain_7d"] = [
        interp_gain(curve, float(ir)) for ir in rows["candidate_ir"].to_numpy(dtype=float)
    ]
    return rows


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if not x[col].isna().all()]


def select_feature_mode(x: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return x
    if mode != "compact":
        raise ValueError(f"Unknown feature mode: {mode}")

    compact_cols = []
    keep_exact = {
        "candidate_ir",
        "candidate_ir_sq",
        "is_zero_ir",
        "decision_doy_sin",
        "decision_doy_cos",
        "candidate_ir_x_doy_sin",
        "candidate_ir_x_doy_cos",
        "longitude",
        "latitude",
        "candidate_ir_x_latitude",
        "candidate_ir_x_longitude",
        "site_ir_min",
        "site_ir_max",
        "candidate_ir_fraction",
        "candidate_ir_fraction_sq",
    }
    keep_prefixes = (
        "state_",
        "soil_",
        "static_",
        "hist_days_available",
        "hist_solar_mean",
        "hist_tmax_mean",
        "hist_tmin_mean",
        "hist_relhum_mean",
        "hist_precip_sum",
        "hist_windspeed_mean",
        "future_days_available",
        "future_solar_mean",
        "future_tmax_mean",
        "future_tmin_mean",
        "future_relhum_mean",
        "future_precip_sum",
        "future_windspeed_mean",
        "candidate_ir_x_state_",
        "candidate_ir_x_soil_",
        "candidate_ir_x_static_",
        "candidate_ir_x_hist_",
        "candidate_ir_x_future_",
    )
    drop_prefixes = (
        "hist_lag",
        "future_day",
        "future_ir_day",
        "candidate_ir_x_hist_lag",
        "candidate_ir_x_future_day",
        "candidate_ir_x_future_ir_day",
        "site_",
        "candidate_ir_x_site_",
        "candidate_ir_sq_x_site_",
    )
    for col in x.columns:
        if col.startswith(drop_prefixes):
            continue
        if col in keep_exact or col.startswith(keep_prefixes):
            compact_cols.append(col)
    if not compact_cols:
        raise ValueError("Compact feature mode selected no columns")
    return x[compact_cols].copy()


def make_group_folds(groups: list[str], folds: int, random_state: int) -> list[list[str]]:
    if folds <= 1:
        raise ValueError("--folds-per-site must be at least 2")
    folds = min(int(folds), len(groups))
    rng = np.random.default_rng(random_state)
    shuffled = np.array(groups, dtype=object)
    rng.shuffle(shuffled)
    parts = [list(part.astype(str)) for part in np.array_split(shuffled, folds)]
    return [part for part in parts if part]


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
        raise ValueError("No usable feature columns for TinyForest")
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train[cols], y_train)
    return model, cols


def predict_forest(model: TinyForest, cols: list[str], x: pd.DataFrame) -> np.ndarray:
    aligned = x.reindex(columns=cols, fill_value=0.0)
    return model.predict(aligned)


def mode_summary(decisions: pd.DataFrame, mode: str) -> pd.DataFrame:
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
                "fixed_list_surrogate_mean_regret_vs_fixed_oracle": float(
                    part["fixed_list_surrogate_regret_vs_fixed_oracle"].mean()
                ),
                "continuous_surrogate_mean_regret_vs_dense": float(
                    part["continuous_surrogate_regret_vs_dense_oracle"].mean()
                ),
                "continuous_surrogate_mean_gain_over_paper": float(
                    part["continuous_surrogate_gain_over_paper"].mean()
                ),
                "continuous_surrogate_better_than_paper_rate": safe_mean(
                    part["continuous_surrogate_gain_over_paper"] > 1e-9
                ),
                "continuous_surrogate_worse_than_paper_rate": safe_mean(
                    part["continuous_surrogate_gain_over_paper"] < -1e-9
                ),
                "continuous_surrogate_nonfixed_ir_rate": safe_mean(part["continuous_surrogate_nonfixed_ir"]),
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
            continuous_surrogate_nonfixed_ir_rate=("continuous_surrogate_nonfixed_ir", "mean"),
        )
        .reset_index()
        .sort_values(["eval_mode", "continuous_surrogate_mean_regret_vs_dense"], ascending=[True, False])
    )


def evaluate_curves(
    *,
    eval_mode: str,
    site_id: str,
    fold_id: int,
    curves_df: pd.DataFrame,
    model: TinyForest,
    feature_cols: list[str],
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    decision_rows: list[dict] = []
    fixed_parts: list[pd.DataFrame] = []
    dense_parts: list[pd.DataFrame] = []
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
        fixed_x = build_features(fixed_rows)
        fixed_rows["pred_net_gain_7d"] = predict_forest(model, feature_cols, fixed_x)
        fixed_rows["eval_mode"] = eval_mode
        fixed_rows["site_fold"] = fold_id
        fixed_parts.append(
            fixed_rows[
                [
                    "sample_id",
                    "site_date_id",
                    "site_id",
                    "date_t",
                    "candidate_ir",
                    "interp_true_net_gain_7d",
                    "pred_net_gain_7d",
                    "eval_mode",
                    "site_fold",
                ]
            ].copy()
        )
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred_best = fixed_rows.loc[fixed_rows["pred_net_gain_7d"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt")
        dense_rows = add_interp_truth(dense_rows, curve)
        dense_x = build_features(dense_rows)
        dense_rows["pred_net_gain_7d"] = predict_forest(model, feature_cols, dense_x)
        dense_rows["eval_mode"] = eval_mode
        dense_rows["site_fold"] = fold_id
        dense_parts.append(
            dense_rows[
                [
                    "sample_id",
                    "site_date_id",
                    "site_id",
                    "date_t",
                    "candidate_ir",
                    "interp_true_net_gain_7d",
                    "pred_net_gain_7d",
                    "eval_mode",
                    "site_fold",
                ]
            ].copy()
        )
        continuous_pred_best = dense_rows.loc[dense_rows["pred_net_gain_7d"].idxmax()]
        nearest_fixed = min(fixed_values, key=lambda value: abs(float(value) - float(continuous_pred_best["candidate_ir"])))
        continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        paper_regret = dense_oracle_gain - fixed_oracle_gain
        continuous_regret = dense_oracle_gain - continuous_gain
        continuous_gain_over_paper = continuous_gain - fixed_oracle_gain
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
                "paper_regret_vs_dense_oracle": paper_regret,
                "fixed_list_surrogate_ir": float(fixed_pred_best["candidate_ir"]),
                "fixed_list_surrogate_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                "fixed_list_surrogate_pred_gain": float(fixed_pred_best["pred_net_gain_7d"]),
                "fixed_list_surrogate_regret_vs_fixed_oracle": fixed_oracle_gain
                - float(fixed_pred_best["interp_true_net_gain_7d"]),
                "continuous_surrogate_ir": float(continuous_pred_best["candidate_ir"]),
                "continuous_surrogate_true_gain": continuous_gain,
                "continuous_surrogate_pred_gain": float(continuous_pred_best["pred_net_gain_7d"]),
                "continuous_surrogate_regret_vs_dense_oracle": continuous_regret,
                "continuous_surrogate_gain_over_paper": continuous_gain_over_paper,
                "continuous_surrogate_better_than_paper": continuous_gain_over_paper > 1e-9,
                "continuous_surrogate_worse_than_paper": continuous_gain_over_paper < -1e-9,
                "continuous_surrogate_nearest_fixed_ir": float(nearest_fixed),
                "continuous_surrogate_distance_to_nearest_fixed_ir": abs(
                    float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                ),
                "continuous_surrogate_nonfixed_ir": abs(
                    float(continuous_pred_best["candidate_ir"]) - float(nearest_fixed)
                )
                > 1e-9,
            }
        )
    return decision_rows, fixed_parts, dense_parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--folds-per-site", type=int, default=5)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--site-limit", type=int, default=0)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-capacity", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--skip-final-experts", action="store_true")
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

    sampled_parts: list[pd.DataFrame] = []
    fixed_parts_all: list[pd.DataFrame] = []
    dense_parts_all: list[pd.DataFrame] = []
    decision_rows: list[dict] = []
    metric_rows: list[dict] = []
    expert_rows: list[dict] = []

    print(
        f"[persite-tinyforest] rows={len(df)} sites={len(sites)} "
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
        print(f"[persite-tinyforest] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            print(f"[persite-tinyforest] site={site_id} capacity fit", flush=True)
            model, cols = fit_forest(
                x_site,
                y_site,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state + site_idx,
            )
            pred = predict_forest(model, cols, x_site)
            sampled = site_df[
                ["sample_id", "site_date_id", "site_id", "date_t", "candidate_ir", TARGET]
            ].copy()
            sampled["pred_net_gain_7d"] = pred
            sampled["eval_mode"] = "capacity"
            sampled["site_fold"] = 0
            sampled_parts.append(sampled)
            metrics = score_metrics(y_site.to_numpy(dtype=float), pred)
            metrics.update(
                {
                    "eval_mode": "capacity",
                    "site_id": site_id,
                    "site_fold": 0,
                    "rows": int(len(site_df)),
                    "site_dates": int(len(groups)),
                }
            )
            metric_rows.append(metrics)
            rows, fixed_parts, dense_parts = evaluate_curves(
                eval_mode="capacity",
                site_id=site_id,
                fold_id=0,
                curves_df=site_df,
                model=model,
                feature_cols=cols,
                paper_candidates=paper_candidates,
                horizon_days=args.horizon_days,
                grid_step=args.grid_step,
            )
            decision_rows.extend(rows)
            fixed_parts_all.extend(fixed_parts)
            dense_parts_all.extend(dense_parts)

            if not args.skip_final_experts:
                expert_path = expert_dir / f"persite_tinyforest_expert_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": model,
                            "feature_columns": cols,
                            "site_id": site_id,
                            "target_column": TARGET,
                            "paper_candidates": paper_candidates,
                            "horizon_days": int(args.horizon_days),
                            "grid_step": float(args.grid_step),
                            "training_rows": int(len(site_df)),
                            "training_site_dates": int(len(groups)),
                        },
                        handle,
                    )
                expert_rows.append(
                    {
                        "site_id": site_id,
                        "expert_checkpoint": str(expert_path),
                        "training_rows": int(len(site_df)),
                        "training_site_dates": int(len(groups)),
                        "n_estimators": int(args.n_estimators),
                        "max_depth": int(args.max_depth),
                        "min_samples_leaf": int(args.min_samples_leaf),
                    }
                )

        if not args.skip_cv:
            folds = make_group_folds(groups, args.folds_per_site, args.random_state + site_idx * 10)
            group_values = site_df["site_date_id"].astype(str).to_numpy()
            for fold_idx, holdout_groups in enumerate(folds):
                print(
                    f"[persite-tinyforest] site={site_id} cv fold {fold_idx + 1}/{len(folds)} "
                    f"holdout_dates={len(holdout_groups)}",
                    flush=True,
                )
                test_mask = np.isin(group_values, np.array(holdout_groups, dtype=str))
                train_mask = ~test_mask
                model, cols = fit_forest(
                    x_site.loc[train_mask],
                    y_site.loc[train_mask],
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    random_state=args.random_state + site_idx * 100 + fold_idx,
                )
                pred = predict_forest(model, cols, x_site.loc[test_mask])
                sampled = site_df.loc[
                    test_mask,
                    ["sample_id", "site_date_id", "site_id", "date_t", "candidate_ir", TARGET],
                ].copy()
                sampled["pred_net_gain_7d"] = pred
                sampled["eval_mode"] = "heldout_date_cv"
                sampled["site_fold"] = fold_idx
                sampled_parts.append(sampled)
                metrics = score_metrics(sampled[TARGET].to_numpy(dtype=float), pred)
                metrics.update(
                    {
                        "eval_mode": "heldout_date_cv",
                        "site_id": site_id,
                        "site_fold": int(fold_idx),
                        "rows": int(test_mask.sum()),
                        "site_dates": int(len(holdout_groups)),
                    }
                )
                metric_rows.append(metrics)
                rows, fixed_parts, dense_parts = evaluate_curves(
                    eval_mode="heldout_date_cv",
                    site_id=site_id,
                    fold_id=fold_idx,
                    curves_df=site_df.loc[test_mask].copy(),
                    model=model,
                    feature_cols=cols,
                    paper_candidates=paper_candidates,
                    horizon_days=args.horizon_days,
                    grid_step=args.grid_step,
                )
                decision_rows.extend(rows)
                fixed_parts_all.extend(fixed_parts)
                dense_parts_all.extend(dense_parts)

    sampled_predictions = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    fixed_predictions = pd.concat(fixed_parts_all, ignore_index=True) if fixed_parts_all else pd.DataFrame()
    dense_predictions = pd.concat(dense_parts_all, ignore_index=True) if dense_parts_all else pd.DataFrame()
    decisions = pd.DataFrame(decision_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    summary = pd.concat(
        [
            mode_summary(decisions, "capacity"),
            mode_summary(decisions, "heldout_date_cv"),
        ],
        ignore_index=True,
    )
    by_site = by_site_summary(decisions)
    prediction_metrics = (
        fold_metrics.groupby("eval_mode")
        .agg(
            folds=("site_fold", "count"),
            rows=("rows", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .reset_index()
    )

    decisions_path = out_dir / "persite_tinyforest_profit_surrogate_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_tinyforest_profit_surrogate_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_tinyforest_profit_surrogate_prediction_metrics_v1.csv"
    summary_path = out_dir / "persite_tinyforest_profit_surrogate_summary_v1.csv"
    by_site_path = out_dir / "persite_tinyforest_profit_surrogate_by_site_v1.csv"
    expert_manifest_path = out_dir / "persite_tinyforest_profit_surrogate_expert_manifest_v1.csv"
    report_path = out_dir / "persite_tinyforest_profit_surrogate_v1.md"

    decisions.to_csv(decisions_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    prediction_metrics.to_csv(prediction_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    pd.DataFrame(expert_rows).to_csv(expert_manifest_path, index=False)
    if args.save_predictions:
        sampled_predictions.to_csv(out_dir / "persite_tinyforest_profit_surrogate_sampled_predictions_v1.csv", index=False)
        fixed_predictions.to_csv(out_dir / "persite_tinyforest_profit_surrogate_fixed_list_predictions_v1.csv", index=False)
        dense_predictions.to_csv(out_dir / "persite_tinyforest_profit_surrogate_dense_predictions_v1.csv", index=False)

    worst = decisions.sort_values("continuous_surrogate_regret_vs_dense_oracle", ascending=False).head(30)
    best = decisions.sort_values("continuous_surrogate_gain_over_paper", ascending=False).head(30)
    lines = [
        "# Per-Site TinyForest Profit Surrogate V1",
        "",
        "## Purpose",
        "",
        "- Capacity check: train and evaluate within the same site/date set.",
        "- Held-out-date CV: validate within site while holding out dates.",
        "- Evaluate paper fixed-list and continuous-grid decisions by SWAP-curve interpolation.",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_metrics),
        "",
        "## Decision Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Largest Continuous Regrets",
        "",
        markdown_table(worst),
        "",
        "## Largest Gains Over Paper Fixed List",
        "",
        markdown_table(best),
        "",
        "## Outputs",
        "",
        f"- `{decisions_path}`",
        f"- `{fold_metrics_path}`",
        f"- `{prediction_metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{expert_manifest_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Per-site TinyForest profit surrogate v1")
    print(f"decisions: {decisions_path}")
    print(f"prediction_metrics: {prediction_metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"expert_manifest: {expert_manifest_path}")
    print(f"report: {report_path}")
    print("")
    print("Prediction metrics")
    print(prediction_metrics.to_string(index=False))
    print("")
    print("Decision summary")
    print(summary.to_string(index=False))
    print("")
    print("By site")
    print(by_site.to_string(index=False))


if __name__ == "__main__":
    main()

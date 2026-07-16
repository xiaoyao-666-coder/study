#!/usr/bin/env python3
"""Train per-site zero-anchor margin TinyForest rankers for SWAP curves.

The curve-top ranker passed in-sample capacity but failed held-out-date CV by
often selecting 0 mm or near-zero irrigation on high-irrigation dates. This
script tests a targeted expert-quality fix before MoE: keep the model per-site,
but train it to predict each candidate's curve-local gain over the 0 mm anchor.

The target preserves the within-curve argmax while making 0 mm a stable anchor:

    zero_margin_score = (gain(candidate) - gain(0 mm)) / max_abs_margin(curve)

If all nonzero candidates are worse than 0 mm, the best score should remain at
0. If irrigation is beneficial, nonzero candidates must beat the zero anchor.
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
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics
from train_persite_curve_top_tinyforest_ranker_v1 import (
    add_curve_top_targets,
    by_site_summary,
    decision_summary,
    evaluate_curves,
    oversampled_training_frame,
    predict_top_score,
    sampled_rank_rows,
    sampled_summary,
    usable_columns,
)
from train_persite_tinyforest_profit_surrogate_v1 import (
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
    / "continuous_ir_12site_10k_persite_curve_zero_margin_tinyforest_ranker_v1"
)
MARGIN_TARGET = "zero_margin_score"


def zero_gain_by_curve(df: pd.DataFrame) -> pd.Series:
    zero_gain: dict[str, float] = {}
    for site_date_id, curve in df.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET])
        if curve.empty:
            continue
        idx = int(np.argmin(np.abs(curve["candidate_ir"].to_numpy(dtype=float))))
        zero_gain[str(site_date_id)] = float(curve.iloc[idx][TARGET])
    return df["site_date_id"].astype(str).map(zero_gain)


def add_zero_margin_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zero_anchor_gain"] = zero_gain_by_curve(out)
    if out["zero_anchor_gain"].isna().any():
        missing = out.loc[out["zero_anchor_gain"].isna(), "site_date_id"].astype(str).unique()
        raise ValueError(f"Missing zero-anchor gain for site-date ids: {missing[:5]}")
    out["gain_over_zero"] = pd.to_numeric(out[TARGET], errors="coerce") - out["zero_anchor_gain"]
    max_abs = out.groupby("site_date_id", sort=False)["gain_over_zero"].transform(
        lambda s: max(float(np.max(np.abs(s.to_numpy(dtype=float)))), 1.0)
    )
    out[MARGIN_TARGET] = out["gain_over_zero"] / max_abs
    out[MARGIN_TARGET] = out[MARGIN_TARGET].clip(-1.0, 1.0)
    out["zero_anchor_is_best"] = out["curve_best_gain"] <= out["zero_anchor_gain"] + 1e-9
    return out


def fit_margin_forest(
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
        raise ValueError("No usable feature columns for zero-margin TinyForest")
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


def prediction_metrics_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    return (
        fold_metrics.groupby("eval_mode")
        .agg(
            folds=("site_fold", "count"),
            rows=("rows", "sum"),
            mean_oversampled_train_rows=("oversampled_train_rows", "mean"),
            zero_margin_mae=("mae", "mean"),
            zero_margin_rmse=("rmse", "mean"),
            zero_margin_r2=("r2", "mean"),
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
        top_temperature=2.0,
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
        f"[persite-zero-margin-tinyforest] rows={len(df)} sites={len(sites)} "
        f"capacity={not args.skip_capacity} cv={not args.skip_cv}",
        flush=True,
    )

    for site_idx, site_id in enumerate(sites):
        site_df = df.loc[df["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        x_site = select_feature_mode(build_features(site_df), args.feature_mode)
        y_site = pd.to_numeric(site_df[MARGIN_TARGET], errors="coerce")
        if y_site.isna().any():
            raise ValueError(f"Zero-margin target contains NaN for site {site_id}")
        groups = sorted(site_df["site_date_id"].astype(str).unique())
        print(f"[persite-zero-margin-tinyforest] site {site_idx + 1}/{len(sites)} {site_id}", flush=True)

        if not args.skip_capacity:
            print(f"[persite-zero-margin-tinyforest] site={site_id} capacity fit", flush=True)
            model, cols, oversampled_rows = fit_margin_forest(
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
                expert_path = expert_dir / f"persite_curve_zero_margin_tinyforest_ranker_{sanitize_name(site_id)}_v1.pkl"
                with expert_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "model": model,
                            "feature_columns": cols,
                            "site_id": site_id,
                            "target_column": MARGIN_TARGET,
                            "source_profit_column": TARGET,
                            "paper_candidates": paper_candidates,
                            "horizon_days": int(args.horizon_days),
                            "grid_step": float(args.grid_step),
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
                    f"[persite-zero-margin-tinyforest] site={site_id} cv fold {fold_idx}/{len(folds)} "
                    f"holdout_dates={len(holdout_groups)}",
                    flush=True,
                )
                test_mask = np.isin(group_values, np.array(holdout_groups, dtype=str))
                train_mask = ~test_mask
                model, cols, oversampled_rows = fit_margin_forest(
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
    prediction_metrics = prediction_metrics_table(fold_metrics)
    sampled_metrics = sampled_summary(sampled)
    summary_parts = []
    for mode in ["capacity", "heldout_date_cv"]:
        part = decision_summary(decisions, mode)
        if not part.empty:
            summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    by_site = by_site_summary(decisions)
    manifest = pd.DataFrame(expert_rows)

    sampled_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_sampled_rank_eval_v1.csv"
    decisions_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_decisions_v1.csv"
    fold_metrics_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_fold_metrics_v1.csv"
    prediction_metrics_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_prediction_metrics_v1.csv"
    sampled_metrics_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_sampled_rank_metrics_v1.csv"
    summary_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_summary_v1.csv"
    by_site_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_by_site_v1.csv"
    manifest_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_manifest_v1.csv"
    config_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_config_v1.json"
    report_path = out_dir / "persite_curve_zero_margin_tinyforest_ranker_v1.md"

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
                "target": MARGIN_TARGET,
                "target_definition": "(gain(candidate)-gain(0mm))/max_abs_margin(curve)",
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
    lines = [
        "# Per-Site Curve Zero-Margin TinyForest Ranker V1",
        "",
        "## Scope",
        "",
        "- Per-site expert only; no cross-site MoE or gating.",
        "- Target is curve-local gain over the 0 mm anchor.",
        "- Motivation: previous curve-top CV failures often selected 0 mm or near-zero irrigation.",
        f"- Input: `{data_path}`.",
        "",
        "## Zero-Margin Prediction Metrics",
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

    print("Per-site curve zero-margin TinyForest ranker v1")
    print(f"sampled_rank_eval: {sampled_path}")
    print(f"decisions: {decisions_path}")
    print(f"prediction_metrics: {prediction_metrics_path}")
    print(f"sampled_rank_metrics: {sampled_metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"manifest: {manifest_path}")
    print(f"report: {report_path}")
    print("")
    print("Zero-margin prediction metrics")
    print(prediction_metrics.to_string(index=False))
    print("")
    print("Sampled rank metrics")
    print(sampled_metrics.to_string(index=False))
    print("")
    print("Dense decision summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a collapse-aware two-head continuous-irrigation surrogate baseline.

The first head predicts whether a site-date is a collapse curve where zero
irrigation is optimal. The second head predicts non-collapse candidate net gain.
When the collapse probability is high, prediction falls back to the physical
water-cost curve, which keeps the continuous surrogate aligned with SWAP smoke
outputs without hand-tuning station-specific guards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import (
    TinyForest,
    score_metrics,
    usable_columns,
)


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_features_v1"
    / "confirmed_5site_true_input_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_surrogate_twohead_v1"
)


def fit_forest(
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
    if not cols:
        raise ValueError("No usable feature columns in training fold")
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train[cols], y_train)
    return model.predict(x_test[cols])


def site_date_representatives(df: pd.DataFrame) -> pd.Index:
    tmp = df.copy()
    tmp["_candidate_ir_num"] = pd.to_numeric(tmp["candidate_ir"], errors="coerce")
    return tmp.sort_values(["site_date_id", "_candidate_ir_num"]).groupby("site_date_id", sort=False).head(1).index


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def evaluate(
    df: pd.DataFrame,
    *,
    cv_group_col: str,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    collapse_threshold: float,
    water_cost_per_mm: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if cv_group_col not in df.columns:
        raise ValueError(f"Missing CV group column: {cv_group_col}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])

    x_all = build_features(df)
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    rep_idx = site_date_representatives(df)
    rep_df = df.loc[rep_idx].copy()
    x_rep = x_all.loc[rep_idx]
    y_collapse = rep_df["target_collapse"].astype(float)

    groups = sorted(df[cv_group_col].astype(str).unique())
    pred_parts = []
    for i, group_id in enumerate(groups):
        print(
            f"[twohead] fold {i + 1}/{len(groups)} holdout {cv_group_col}={group_id}",
            flush=True,
        )
        test_mask = df[cv_group_col].astype(str) == str(group_id)
        train_mask = ~test_mask
        rep_test_mask = rep_df[cv_group_col].astype(str) == str(group_id)
        rep_train_mask = ~rep_test_mask

        collapse_pred = fit_forest(
            x_rep.loc[rep_train_mask],
            y_collapse.loc[rep_train_mask],
            x_rep.loc[rep_test_mask],
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + i,
        )
        collapse_prob_by_site_date = {
            site_date: float(np.clip(prob, 0.0, 1.0))
            for site_date, prob in zip(rep_df.loc[rep_test_mask, "site_date_id"], collapse_pred)
        }

        noncollapse_train_mask = train_mask & (~df["target_collapse"])
        if int(noncollapse_train_mask.sum()) < max(10, min_samples_leaf * 4):
            noncollapse_train_mask = train_mask
        response_pred = fit_forest(
            x_all.loc[noncollapse_train_mask],
            y_all.loc[noncollapse_train_mask],
            x_all.loc[test_mask],
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1000 + i,
        )

        part = df.loc[
            test_mask,
            [
                "sample_id",
                "site_date_id",
                "site_id",
                "date_t",
                "decision_doy",
                "candidate_ir",
                TARGET,
                "target_7d",
                "best_ir_for_date",
                "best_target_for_date",
                "is_best_ir",
                "target_collapse",
                "same_date_duplicate_target_curve",
            ],
        ].copy()
        part["cv_group_col"] = cv_group_col
        part["cv_group_value"] = str(group_id)
        part["model"] = "twohead_tiny_forest"
        part["collapse_prob"] = part["site_date_id"].map(collapse_prob_by_site_date).astype(float)
        part["response_pred_net_gain_7d"] = response_pred
        part["collapse_cost_pred_net_gain_7d"] = -water_cost_per_mm * pd.to_numeric(part["candidate_ir"], errors="coerce")
        part["pred_net_gain_7d"] = np.where(
            part["collapse_prob"] >= collapse_threshold,
            part["collapse_cost_pred_net_gain_7d"],
            part["response_pred_net_gain_7d"],
        )
        pred_parts.append(part)
        print(
            f"[twohead] fold {i + 1}/{len(groups)} completed rows={int(test_mask.sum())}",
            flush=True,
        )

    pred_df = pd.concat(pred_parts, ignore_index=True)

    decisions = []
    for site_date_id, part in pred_df.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
                "site_date_id": site_date_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "same_date_duplicate_target_curve": bool(true_best["same_date_duplicate_target_curve"]),
                "collapse_prob": float(true_best["collapse_prob"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "true_best_net_gain": float(true_best[TARGET]),
                "pred_best_true_net_gain": float(pred_best[TARGET]),
                "pred_best_pred_net_gain": float(pred_best["pred_net_gain_7d"]),
                "decision_correct": float(true_best["candidate_ir"]) == float(pred_best["candidate_ir"]),
                "decision_regret": float(true_best[TARGET] - pred_best[TARGET]),
            }
        )

    decision_df = pd.DataFrame(decisions)
    metrics = score_metrics(
        pred_df[TARGET].to_numpy(dtype=float),
        pred_df["pred_net_gain_7d"].to_numpy(dtype=float),
    )
    metrics.update(
        {
            "cv_group_col": cv_group_col,
            "cv_folds": int(len(groups)),
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            "collapse_threshold": float(collapse_threshold),
            "water_cost_per_mm": float(water_cost_per_mm),
            "decision_correct": int(decision_df["decision_correct"].sum()),
            "decision_total": int(len(decision_df)),
            "decision_accuracy": safe_mean(decision_df["decision_correct"]),
            "mean_decision_regret": float(decision_df["decision_regret"].mean()),
            "median_decision_regret": float(decision_df["decision_regret"].median()),
            "collapse_decision_accuracy": safe_mean(decision_df.loc[decision_df["target_collapse"], "decision_correct"]),
            "noncollapse_decision_accuracy": safe_mean(decision_df.loc[~decision_df["target_collapse"], "decision_correct"]),
        }
    )
    metrics_df = pd.DataFrame([metrics])
    by_site = (
        decision_df.groupby("site_id")
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_decision_regret=("decision_regret", "mean"),
            max_decision_regret=("decision_regret", "max"),
            mean_collapse_prob=("collapse_prob", "mean"),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    return pred_df, decision_df, metrics_df, by_site, usable_columns(x_all)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--collapse-threshold", type=float, default=0.5)
    parser.add_argument("--water-cost-per-mm", type=float, default=1.4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing continuous-irrigation sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(
        f"[twohead] loaded rows={len(df)} cols={len(df.columns)} cv_group_col={args.cv_group_col}",
        flush=True,
    )
    pred_df, decision_df, metrics_df, by_site, feature_cols = evaluate(
        df,
        cv_group_col=args.cv_group_col,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        collapse_threshold=args.collapse_threshold,
        water_cost_per_mm=args.water_cost_per_mm,
        random_state=args.random_state,
    )

    pred_path = out_dir / "continuous_irrigation_surrogate_twohead_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_twohead_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_twohead_v1_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_twohead_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_twohead_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_twohead_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    worst = decision_df.sort_values("decision_regret", ascending=False).head(15)
    lines = [
        "# Continuous Irrigation Surrogate Two-Head V1",
        "",
        "## Scope",
        "",
        "- Collapse-aware continuous-irrigation surrogate smoke baseline.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        "- Head 1 predicts collapse probability for each site-date.",
        "- Head 2 predicts non-collapse candidate net gain.",
        "- Collapse predictions use the physical water-cost curve for candidate ranking.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Decision Rows",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decision_path}`",
        f"- `{metrics_path}`",
        f"- `{by_site_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Continuous irrigation surrogate two-head v1")
    print(f"input: {data_path}")
    print(f"predictions: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print(metrics_df.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print("")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a tiny baseline surrogate sanity check on confirmed 5-site samples.

This is a workflow sanity check, not the final universal surrogate. It uses only
features already present in the candidate-level table and evaluates by leaving
one site-date group out at a time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "confirmed_5site_true_input_surrogate_table_v1_6dates"
    / "confirmed_5site_true_input_surrogate_samples_v1.csv"
)
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "confirmed_5site_true_input_surrogate_baseline_v1"
TARGET = "net_gain_7d"


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "candidate_ir",
        "candidate_ir_sq",
        "is_zero_ir",
        "decision_doy_sin",
        "decision_doy_cos",
        "longitude",
        "latitude",
        "site_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feature_parts = {}
    for col in ["candidate_ir", "candidate_ir_sq", "is_zero_ir", "decision_doy_sin", "decision_doy_cos", "longitude", "latitude"]:
        feature_parts[col] = pd.to_numeric(df[col], errors="coerce")

    candidate_ir = feature_parts["candidate_ir"]
    candidate_ir_sq = feature_parts["candidate_ir_sq"]
    feature_parts["candidate_ir_x_doy_sin"] = candidate_ir * feature_parts["decision_doy_sin"]
    feature_parts["candidate_ir_x_doy_cos"] = candidate_ir * feature_parts["decision_doy_cos"]
    feature_parts["candidate_ir_x_latitude"] = candidate_ir * feature_parts["latitude"]
    feature_parts["candidate_ir_x_longitude"] = candidate_ir * feature_parts["longitude"]

    site = pd.get_dummies(df["site_id"].astype(str), prefix="site", dtype=float)
    for col in site.columns:
        feature_parts[col] = site[col]
        feature_parts[f"candidate_ir_x_{col}"] = candidate_ir * site[col]
        feature_parts[f"candidate_ir_sq_x_{col}"] = candidate_ir_sq * site[col]

    optional_numeric = [
        "state_dvs",
        "state_lai",
        "state_rootd",
        "state_cwdm",
        "state_cwso",
        "soil_layer_count",
        "soil_depth_min_cm",
        "soil_depth_max_cm",
        "soil_h_mean_0_30_cm",
        "soil_h_mean_30_60_cm",
        "soil_h_mean_60_100_cm",
        "soil_h_mean_0_100_cm",
        "soil_h_min_0_100_cm",
        "soil_h_max_0_100_cm",
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
    ]
    optional_numeric.extend(
        [
            "site_ir_min",
            "site_ir_max",
            "candidate_ir_fraction",
            "candidate_ir_fraction_sq",
            "static_dem_m",
            "static_tile_drain",
            "static_theta_s_0_60_mean",
            "static_theta_s_0_60_min",
            "static_theta_s_0_60_max",
            "static_theta_r_0_60_mean",
            "static_theta_r_0_60_min",
            "static_theta_r_0_60_max",
            "static_ksat_0_60_mean",
            "static_ksat_0_60_min",
            "static_ksat_0_60_max",
            "static_alpha_0_60_mean",
            "static_alpha_0_60_min",
            "static_alpha_0_60_max",
            "static_n_0_60_mean",
            "static_n_0_60_min",
            "static_n_0_60_max",
            "static_lambda_0_60_mean",
            "static_lambda_0_60_min",
            "static_lambda_0_60_max",
            "static_t2m_mean",
            "static_t2m_std",
            "static_tmin_mean",
            "static_tmax_mean",
            "static_precip_mean",
            "static_precip_sum",
            "static_pet_mean",
            "static_rad_mean",
        ]
    )
    sequence_prefixes = ("hist_lag", "future_day", "future_ir_day")
    for col in df.columns:
        if col.startswith(sequence_prefixes) and col not in optional_numeric:
            optional_numeric.append(col)
    for col in optional_numeric:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            feature_parts[col] = values
            feature_parts[f"candidate_ir_x_{col}"] = candidate_ir * values
            if col in {"candidate_ir_fraction", "candidate_ir_fraction_sq"}:
                feature_parts[f"{col}_x_doy_sin"] = values * feature_parts["decision_doy_sin"]
                feature_parts[f"{col}_x_doy_cos"] = values * feature_parts["decision_doy_cos"]
    return pd.DataFrame(feature_parts, index=df.index)


def fit_ridge(x: pd.DataFrame, y: pd.Series, alpha: float) -> dict:
    arr = x.to_numpy(dtype=float)
    valid_cols = ~np.all(np.isnan(arr), axis=0)
    if not np.any(valid_cols):
        raise ValueError("All feature columns are NaN in the training fold")
    arr = arr[:, valid_cols]
    columns = [col for col, keep in zip(x.columns, valid_cols) if keep]
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(med, inds[1])

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    z = (arr - mean) / std
    design = np.column_stack([np.ones(len(z)), z])
    target = y.to_numpy(dtype=float)

    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return {"coef": coef, "median": med, "mean": mean, "std": std, "columns": columns}


def predict(model: dict, x: pd.DataFrame) -> np.ndarray:
    arr = x[model["columns"]].to_numpy(dtype=float)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(model["median"], inds[1])
    z = (arr - model["mean"]) / model["std"]
    design = np.column_stack([np.ones(len(z)), z])
    return design @ model["coef"]


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate(df: pd.DataFrame, alpha: float, cv_group_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "site_date_id" not in df.columns:
        raise ValueError("Missing site_date_id column")
    if cv_group_col not in df.columns:
        raise ValueError(f"Missing CV group column: {cv_group_col}")
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])

    features = build_features(df)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    groups = sorted(df[cv_group_col].astype(str).unique())

    pred_parts = []
    for group_id in groups:
        test_mask = df[cv_group_col].astype(str) == str(group_id)
        train_mask = ~test_mask
        model = fit_ridge(features.loc[train_mask], y.loc[train_mask], alpha=alpha)
        pred = predict(model, features.loc[test_mask])

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
        part["pred_net_gain_7d"] = pred
        pred_parts.append(part)

    pred_df = pd.concat(pred_parts, ignore_index=True)

    decisions = []
    for group_id, part in pred_df.groupby("site_date_id", sort=False):
        true_best = part.loc[part[TARGET].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
                "site_date_id": group_id,
                "site_id": str(true_best["site_id"]),
                "date_t": str(true_best["date_t"]),
                "decision_doy": int(true_best["decision_doy"]),
                "target_collapse": bool(true_best["target_collapse"]),
                "same_date_duplicate_target_curve": bool(true_best["same_date_duplicate_target_curve"]),
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
    metrics = score_metrics(pred_df[TARGET].to_numpy(dtype=float), pred_df["pred_net_gain_7d"].to_numpy(dtype=float))
    metrics["cv_group_col"] = cv_group_col
    metrics["cv_folds"] = int(len(groups))
    metrics["decision_correct"] = int(decision_df["decision_correct"].sum())
    metrics["decision_total"] = int(len(decision_df))
    metrics["decision_accuracy"] = float(decision_df["decision_correct"].mean())
    metrics["mean_decision_regret"] = float(decision_df["decision_regret"].mean())
    metrics["median_decision_regret"] = float(decision_df["decision_regret"].median())
    metrics["collapse_decision_accuracy"] = float(decision_df.loc[decision_df["target_collapse"], "decision_correct"].mean())
    metrics["noncollapse_decision_accuracy"] = float(decision_df.loc[~decision_df["target_collapse"], "decision_correct"].mean())
    metrics_df = pd.DataFrame([metrics])
    return pred_df, decision_df, metrics_df, list(features.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cv-group-col", default="site_date_id", help="Column to leave out for each CV fold, e.g. site_date_id, site_id, date_t.")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    pred_df, decision_df, metrics_df, feature_cols = evaluate(df, alpha=args.alpha, cv_group_col=args.cv_group_col)

    pred_path = out_dir / "confirmed_5site_true_input_surrogate_baseline_v1_predictions.csv"
    decision_path = out_dir / "confirmed_5site_true_input_surrogate_baseline_v1_decision_eval.csv"
    metrics_path = out_dir / "confirmed_5site_true_input_surrogate_baseline_v1_metrics.csv"
    feature_path = out_dir / "confirmed_5site_true_input_surrogate_baseline_v1_features.json"
    report_path = out_dir / "confirmed_5site_true_input_surrogate_baseline_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    lines = [
        "# Confirmed 5-Site True-Input Surrogate Baseline V1",
        "",
        "## Scope",
        "",
        "- Ridge baseline with configurable leave-one-group-out evaluation.",
        f"- CV group column: `{args.cv_group_col}`.",
        "- Uses only current candidate table fields: site, coordinates, decision date, and candidate irrigation.",
        "- This is a sanity check, not a final universal surrogate.",
        "",
        "## Metrics",
        "",
        markdown_table(metrics_df),
        "",
        "## Decision Evaluation",
        "",
        markdown_table(decision_df),
        "",
        "## Outputs",
        "",
        f"- `{pred_path}`",
        f"- `{decision_path}`",
        f"- `{metrics_path}`",
        f"- `{feature_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Confirmed 5-site true-input surrogate baseline v1")
    print(f"input: {data_path}")
    print(f"predictions: {pred_path}")
    print(f"decision_eval: {decision_path}")
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")
    print(metrics_df.to_string(index=False))
    print("")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()

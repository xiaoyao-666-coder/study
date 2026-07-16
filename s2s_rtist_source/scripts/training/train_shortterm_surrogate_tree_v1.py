#!/usr/bin/env python3
"""Train/evaluate tree-based nonlinear baselines for shortterm surrogate v1."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
PRED_OUT = OUT_DIR / "surrogate_tree_v1_predictions.csv"
DECISION_OUT = OUT_DIR / "surrogate_tree_v1_decision_eval.csv"
METRICS_OUT = OUT_DIR / "surrogate_tree_v1_metrics.txt"

TARGET = "net_gain_7d"


BASE_FEATURES = [
    "candidate_ir",
    "state_dvs",
    "state_lai",
    "state_rootd",
    "state_cwdm",
    "state_cwso",
    "soil_layer_count",
    "soil_h_mean_0_30_cm",
    "soil_h_mean_30_60_cm",
    "soil_h_mean_60_100_cm",
    "soil_h_mean_0_100_cm",
    "soil_h_min_0_100_cm",
    "soil_h_max_0_100_cm",
    "hist_precip_sum",
    "hist_solar_mean",
    "hist_tmax_mean",
    "hist_tmin_mean",
    "hist_relhum_mean",
    "hist_windspeed_mean",
    "future_precip_sum",
    "future_solar_mean",
    "future_tmax_mean",
    "future_tmin_mean",
    "future_relhum_mean",
    "future_windspeed_mean",
]


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in BASE_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        x[col] = pd.to_numeric(df[col], errors="coerce")

    x["candidate_ir_sq"] = x["candidate_ir"] ** 2
    x["candidate_ir_x_state_dvs"] = x["candidate_ir"] * x["state_dvs"]
    x["candidate_ir_x_state_lai"] = x["candidate_ir"] * x["state_lai"]
    x["candidate_ir_x_soil_h_0_30"] = x["candidate_ir"] * x["soil_h_mean_0_30_cm"]
    x["candidate_ir_x_future_precip"] = x["candidate_ir"] * x["future_precip_sum"]
    x["candidate_ir_x_future_tmax"] = x["candidate_ir"] * x["future_tmax_mean"]
    return x


def make_models():
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for tree baseline. Install/use sklearn or keep ridge baseline only."
        ) from exc

    return {
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42,
            ),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=2,
                min_samples_leaf=2,
                random_state=42,
            ),
        ),
    }


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_model(df: pd.DataFrame, x_all: pd.DataFrame, y_all: pd.Series, model_name: str, model) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = []
    dates = sorted(df["date_t"].unique(), key=lambda x: int(df.loc[df["date_t"] == x, "decision_doy"].iloc[0]))
    for date_t in dates:
        test_mask = df["date_t"] == date_t
        train_mask = ~test_mask
        model.fit(x_all.loc[train_mask], y_all.loc[train_mask])
        pred = model.predict(x_all.loc[test_mask])

        part = df.loc[test_mask, [
            "sample_id",
            "site_id",
            "date_t",
            "decision_doy",
            "candidate_ir",
            "net_gain_7d",
            "target_7d",
            "best_ir_for_date",
            "is_best_ir",
        ]].copy()
        part["pred_net_gain_7d"] = pred
        part["model"] = model_name
        parts.append(part)

    pred_df = pd.concat(parts, ignore_index=True)
    pred_df["pred_rank"] = pred_df.groupby(["model", "date_t"])["pred_net_gain_7d"].rank(ascending=False, method="first")

    decisions = []
    for date_t, group in pred_df.groupby("date_t", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        pred_best = group.loc[group["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
                "model": model_name,
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "pred_best_ir": float(pred_best["candidate_ir"]),
                "pred_best_pred_net_gain": float(pred_best["pred_net_gain_7d"]),
                "pred_best_true_net_gain": float(pred_best["net_gain_7d"]),
                "decision_correct": float(pred_best["candidate_ir"]) == float(true_best["candidate_ir"]),
                "decision_regret": float(true_best["net_gain_7d"] - pred_best["net_gain_7d"]),
            }
        )
    decision_df = pd.DataFrame(decisions)
    m = score_metrics(pred_df["net_gain_7d"].to_numpy(dtype=float), pred_df["pred_net_gain_7d"].to_numpy(dtype=float))
    m["decision_correct"] = int(decision_df["decision_correct"].sum())
    m["decision_total"] = int(len(decision_df))
    m["decision_accuracy"] = float(m["decision_correct"] / m["decision_total"])
    m["mean_decision_regret"] = float(decision_df["decision_regret"].mean())
    return pred_df, decision_df, m


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")

    df = pd.read_csv(DATA)
    x_all = add_interactions(df)
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    pred_all = []
    decision_all = []
    metric_rows = []
    for model_name, model in make_models().items():
        pred_df, decision_df, m = evaluate_model(df, x_all, y_all, model_name, model)
        pred_all.append(pred_df)
        decision_all.append(decision_df)
        metric_rows.append({"model": model_name, **m})

    pred_out = pd.concat(pred_all, ignore_index=True)
    decision_out = pd.concat(decision_all, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    pred_out.to_csv(PRED_OUT, index=False)
    decision_out.to_csv(DECISION_OUT, index=False)

    lines = [
        "Shortterm surrogate tree baseline v1",
        "",
        f"dataset: {DATA}",
        f"rows: {len(df)}",
        f"decision_dates: {df['date_t'].nunique()}",
        f"features_with_interactions: {x_all.shape[1]}",
        "evaluation: leave-one-decision-date-out",
        "",
        metrics_df.to_string(index=False),
        "",
        f"wrote: {PRED_OUT}",
        f"wrote: {DECISION_OUT}",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nDecision eval:")
    print(decision_out.to_string(index=False))


if __name__ == "__main__":
    main()

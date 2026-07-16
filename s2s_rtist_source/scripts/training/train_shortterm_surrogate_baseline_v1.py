#!/usr/bin/env python3
"""Train/evaluate a small baseline surrogate on shortterm v1 samples.

The evaluation is leave-one-decision-date-out: train on 9 dates, predict the
held-out date, and choose the candidate with the largest predicted net_gain_7d.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
PRED_OUT = OUT_DIR / "surrogate_baseline_v1_predictions.csv"
DECISION_OUT = OUT_DIR / "surrogate_baseline_v1_decision_eval.csv"
METRICS_OUT = OUT_DIR / "surrogate_baseline_v1_metrics.txt"

TARGET = "net_gain_7d"

FEATURES = [
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


def numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def fit_predict_sklearn(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray | None:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
    except Exception:
        return None

    # Use a conservative linear model for tiny data. Random forest is tempting,
    # but date-level extrapolation with 80 rows is less jumpy with ridge.
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(x_train, y_train)
    return model.predict(x_test)


def fit_predict_numpy(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    train = x_train.copy()
    test = x_test.copy()
    med = train.median(numeric_only=True)
    train = train.fillna(med).fillna(0.0)
    test = test.fillna(med).fillna(0.0)

    mean = train.mean()
    std = train.std().replace(0, 1.0)
    xtr = ((train - mean) / std).to_numpy(dtype=float)
    xte = ((test - mean) / std).to_numpy(dtype=float)
    y = y_train.to_numpy(dtype=float)

    xtr = np.column_stack([np.ones(len(xtr)), xtr])
    xte = np.column_stack([np.ones(len(xte)), xte])
    alpha = 1.0
    eye = np.eye(xtr.shape[1])
    eye[0, 0] = 0.0
    beta = np.linalg.solve(xtr.T @ xtr + alpha * eye, xtr.T @ y)
    return xte @ beta


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")

    df = pd.read_csv(DATA)
    x_all = numeric_frame(df, FEATURES)
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    preds = []
    for date_t in sorted(df["date_t"].unique(), key=lambda x: int(df.loc[df["date_t"] == x, "decision_doy"].iloc[0])):
        test_mask = df["date_t"] == date_t
        train_mask = ~test_mask
        x_train = x_all.loc[train_mask]
        y_train = y_all.loc[train_mask]
        x_test = x_all.loc[test_mask]

        pred = fit_predict_sklearn(x_train, y_train, x_test)
        model_name = "ridge_sklearn"
        if pred is None:
            pred = fit_predict_numpy(x_train, y_train, x_test)
            model_name = "ridge_numpy"

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
        preds.append(part)

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["pred_rank"] = pred_df.groupby("date_t")["pred_net_gain_7d"].rank(ascending=False, method="first")
    pred_df.to_csv(PRED_OUT, index=False)

    decisions = []
    for date_t, group in pred_df.groupby("date_t", sort=False):
        true_best = group.loc[group["net_gain_7d"].idxmax()]
        pred_best = group.loc[group["pred_net_gain_7d"].idxmax()]
        decisions.append(
            {
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
    decision_df.to_csv(DECISION_OUT, index=False)

    m = metrics(pred_df["net_gain_7d"].to_numpy(dtype=float), pred_df["pred_net_gain_7d"].to_numpy(dtype=float))
    correct = int(decision_df["decision_correct"].sum())
    total = int(len(decision_df))
    avg_regret = float(decision_df["decision_regret"].mean())
    lines = [
        "Shortterm surrogate baseline v1",
        "",
        f"dataset: {DATA}",
        f"rows: {len(df)}",
        f"decision_dates: {total}",
        f"features: {len(FEATURES)}",
        "evaluation: leave-one-decision-date-out",
        "",
        f"MAE: {m['mae']:.6f}",
        f"RMSE: {m['rmse']:.6f}",
        f"R2: {m['r2']:.6f}",
        f"decision_correct: {correct}/{total}",
        f"decision_accuracy: {correct / total:.3f}",
        f"mean_decision_regret: {avg_regret:.6f}",
        "",
        f"wrote: {PRED_OUT}",
        f"wrote: {DECISION_OUT}",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nDecision eval:")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()

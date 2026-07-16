#!/usr/bin/env python3
"""Decision-oriented ranker for shortterm surrogate v1.

The model directly optimizes a listwise softmax loss within each decision date,
so it learns to pick the best irrigation candidate rather than just fit the
numeric net_gain_7d value.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
PRED_OUT = OUT_DIR / "surrogate_ranker_v1_predictions.csv"
DECISION_OUT = OUT_DIR / "surrogate_ranker_v1_decision_eval.csv"
METRICS_OUT = OUT_DIR / "surrogate_ranker_v1_metrics.txt"


BASE_FEATURES = [
    "candidate_ir",
    "decision_doy_sin",
    "decision_doy_cos",
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    doy = pd.to_numeric(df["decision_doy"], errors="coerce")
    x["candidate_ir"] = pd.to_numeric(df["candidate_ir"], errors="coerce")
    x["decision_doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    x["decision_doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    for col in BASE_FEATURES[3:]:
        if col not in df.columns:
            raise ValueError(f"Missing feature column: {col}")
        x[col] = pd.to_numeric(df[col], errors="coerce")

    x["candidate_ir_sq"] = x["candidate_ir"] ** 2
    x["candidate_ir_x_state_dvs"] = x["candidate_ir"] * x["state_dvs"]
    x["candidate_ir_x_state_lai"] = x["candidate_ir"] * x["state_lai"]
    x["candidate_ir_x_soil_h_0_30"] = x["candidate_ir"] * x["soil_h_mean_0_30_cm"]
    x["candidate_ir_x_future_precip"] = x["candidate_ir"] * x["future_precip_sum"]
    x["candidate_ir_x_future_tmax"] = x["candidate_ir"] * x["future_tmax_mean"]
    x["candidate_ir_x_future_tmin"] = x["candidate_ir"] * x["future_tmin_mean"]
    return x


def group_indices(df: pd.DataFrame):
    for date_t, group in df.groupby("date_t", sort=False):
        idx = group.index.to_numpy()
        yield date_t, idx


def standardize(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = train_x.copy()
    test = test_x.copy()
    med = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(med)
    test = test.fillna(med)
    mean = train.mean()
    std = train.std().replace(0, 1.0).fillna(1.0)
    return (
        ((train - mean) / std).to_numpy(dtype=float),
        ((test - mean) / std).to_numpy(dtype=float),
        mean.to_numpy(dtype=float),
        std.to_numpy(dtype=float),
    )


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(z) / len(z)
    return e / s


def train_listwise_ranker(x_train: np.ndarray, groups: list[np.ndarray], y_train: np.ndarray, *, lr: float = 0.05, reg: float = 1e-3, steps: int = 2000) -> tuple[np.ndarray, float]:
    n, p = x_train.shape
    w = np.zeros(p + 1, dtype=float)  # bias + weights
    y = y_train.astype(float)
    group_y = []
    for g in groups:
        yy = y[g]
        best = np.argmax(yy)
        group_y.append(best)

    for step in range(steps):
        grad = np.zeros_like(w)
        loss = 0.0
        for g, best_idx in zip(groups, group_y):
            Xg = x_train[g]
            Xg1 = np.column_stack([np.ones(len(g)), Xg])
            scores = Xg1 @ w
            probs = softmax(scores)
            loss -= math.log(max(probs[best_idx], 1e-12))
            y_onehot = np.zeros(len(g), dtype=float)
            y_onehot[best_idx] = 1.0
            diff = probs - y_onehot
            grad += Xg1.T @ diff
        loss = loss / len(groups) + 0.5 * reg * float(np.sum(w[1:] ** 2))
        grad = grad / len(groups)
        grad[1:] += reg * w[1:]
        w -= lr * grad
        if step % 200 == 0:
            # small decay for stability
            lr *= 0.95
    return w, loss


def predict_scores(w: np.ndarray, x: np.ndarray, group_list: list[np.ndarray]) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    return x1 @ w


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
    x_all = add_features(df)
    y_all = pd.to_numeric(df["net_gain_7d"], errors="coerce")
    if y_all.isna().any():
        raise ValueError("Target contains NaN")

    pred_parts = []
    decision_rows = []
    date_order = sorted(df["date_t"].unique(), key=lambda x: int(df.loc[df["date_t"] == x, "decision_doy"].iloc[0]))

    for date_t in date_order:
        test_mask = df["date_t"] == date_t
        train_mask = ~test_mask
        train_x, test_x, _, _ = standardize(x_all.loc[train_mask], x_all.loc[test_mask])
        train_y = y_all.loc[train_mask].to_numpy(dtype=float)
        test_y = y_all.loc[test_mask].to_numpy(dtype=float)

        train_groups = []
        for _, idx in group_indices(df.loc[train_mask]):
            train_groups.append(np.arange(len(idx)) if False else idx)  # placeholder to satisfy linter-like clarity

        # Re-map training indices to 0..n_train-1 while preserving group membership.
        train_idx = df.index[train_mask].to_numpy()
        pos = {idx: i for i, idx in enumerate(train_idx)}
        groups = []
        for _, idx in group_indices(df.loc[train_mask]):
            groups.append(np.array([pos[i] for i in idx], dtype=int))

        w, train_loss = train_listwise_ranker(train_x, groups, train_y)
        test_scores = predict_scores(w, test_x, [np.arange(len(test_y))])

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
        part["pred_net_gain_7d"] = test_scores
        part["model"] = "listwise_ranker_numpy"
        pred_parts.append(part)

        true_best = part.loc[part["net_gain_7d"].idxmax()]
        pred_best = part.loc[part["pred_net_gain_7d"].idxmax()]
        probs = softmax(test_scores)
        decision_rows.append(
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
                "train_loss": float(train_loss),
                "pred_entropy": float(-np.sum(probs * np.log(probs + 1e-12))),
            }
        )

    pred_df = pd.concat(pred_parts, ignore_index=True)
    decision_df = pd.DataFrame(decision_rows)
    pred_df["pred_rank"] = pred_df.groupby("date_t")["pred_net_gain_7d"].rank(ascending=False, method="first")

    pred_df.to_csv(PRED_OUT, index=False)
    decision_df.to_csv(DECISION_OUT, index=False)

    y_pred = pred_df["pred_net_gain_7d"].to_numpy(dtype=float)
    m = metrics(pred_df["net_gain_7d"].to_numpy(dtype=float), y_pred)
    correct = int(decision_df["decision_correct"].sum())
    total = int(len(decision_df))
    avg_regret = float(decision_df["decision_regret"].mean())
    lines = [
        "Shortterm surrogate ranker v1",
        "",
        f"dataset: {DATA}",
        f"rows: {len(df)}",
        f"decision_dates: {df['date_t'].nunique()}",
        f"features: {x_all.shape[1]}",
        "model: listwise softmax ranker in numpy",
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

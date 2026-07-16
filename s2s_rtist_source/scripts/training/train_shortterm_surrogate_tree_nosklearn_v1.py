#!/usr/bin/env python3
"""Tree baseline without scikit-learn for shortterm surrogate v1.

This is a small CART-style regression tree plus bagging forest implemented with
only numpy/pandas, for servers where scikit-learn is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
PRED_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_predictions.csv"
DECISION_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_decision_eval.csv"
METRICS_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_metrics.txt"

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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
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


@dataclass
class Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class TinyRegressionTree:
    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 3, max_features: int | None = None, rng: np.random.Generator | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root: Node | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyRegressionTree":
        self.root = self._build(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")
        return np.array([self._predict_one(row, self.root) for row in x], dtype=float)

    def _predict_one(self, row: np.ndarray, node: Node) -> float:
        while node.feature is not None and node.left is not None and node.right is not None:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(value=float(np.mean(y)))
        if depth >= self.max_depth or len(y) < self.min_samples_leaf * 2 or float(np.var(y)) <= 1e-12:
            return node

        split = self._best_split(x, y)
        if split is None:
            return node
        feature, threshold = split
        mask = x[:, feature] <= threshold
        if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
            return node

        node.feature = int(feature)
        node.threshold = float(threshold)
        node.left = self._build(x[mask], y[mask], depth + 1)
        node.right = self._build(x[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        n_features = x.shape[1]
        if self.max_features is None or self.max_features >= n_features:
            features = np.arange(n_features)
        else:
            features = self.rng.choice(n_features, size=self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_loss = np.inf
        for feature in features:
            values = np.unique(x[:, feature])
            if len(values) <= 1:
                continue
            if len(values) > 20:
                thresholds = np.quantile(values, np.linspace(0.1, 0.9, 9))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in np.unique(thresholds):
                mask = x[:, feature] <= threshold
                left_n = int(mask.sum())
                right_n = int((~mask).sum())
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                left = y[mask]
                right = y[~mask]
                loss = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best_feature = int(feature)
                    best_threshold = float(threshold)
        if best_feature is None:
            return None
        return best_feature, best_threshold


class TinyForest:
    def __init__(self, n_estimators: int = 200, max_depth: int = 3, min_samples_leaf: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: list[TinyRegressionTree] = []
        self.medians: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TinyForest":
        arr = x.to_numpy(dtype=float)
        self.medians = np.nanmedian(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(self.medians, inds[1])
        target = y.to_numpy(dtype=float)

        rng = np.random.default_rng(self.random_state)
        n, p = arr.shape
        max_features = max(1, int(math.sqrt(p)))
        self.trees = []
        for _ in range(self.n_estimators):
            sample_idx = rng.integers(0, n, size=n)
            tree = TinyRegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features,
                rng=rng,
            )
            tree.fit(arr[sample_idx], target[sample_idx])
            self.trees.append(tree)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("Forest has not been fitted")
        arr = x.to_numpy(dtype=float)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(self.medians, inds[1])
        preds = np.vstack([tree.predict(arr) for tree in self.trees])
        return preds.mean(axis=0)


def score_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate(df: pd.DataFrame, x_all: pd.DataFrame, y_all: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    parts = []
    dates = sorted(df["date_t"].unique(), key=lambda x: int(df.loc[df["date_t"] == x, "decision_doy"].iloc[0]))
    for date_t in dates:
        test_mask = df["date_t"] == date_t
        train_mask = ~test_mask
        model = TinyForest(n_estimators=300, max_depth=3, min_samples_leaf=2, random_state=42)
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
        part["model"] = "tiny_forest_nosklearn"
        parts.append(part)

    pred_df = pd.concat(parts, ignore_index=True)
    pred_df["pred_rank"] = pred_df.groupby("date_t")["pred_net_gain_7d"].rank(ascending=False, method="first")

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
    metric = score_metrics(pred_df["net_gain_7d"].to_numpy(dtype=float), pred_df["pred_net_gain_7d"].to_numpy(dtype=float))
    metric["decision_correct"] = int(decision_df["decision_correct"].sum())
    metric["decision_total"] = int(len(decision_df))
    metric["decision_accuracy"] = float(metric["decision_correct"] / metric["decision_total"])
    metric["mean_decision_regret"] = float(decision_df["decision_regret"].mean())
    return pred_df, decision_df, metric


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")

    df = pd.read_csv(DATA)
    x_all = build_features(df)
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    pred_df, decision_df, metric = evaluate(df, x_all, y_all)
    pred_df.to_csv(PRED_OUT, index=False)
    decision_df.to_csv(DECISION_OUT, index=False)

    lines = [
        "Shortterm surrogate tree baseline without sklearn v1",
        "",
        f"dataset: {DATA}",
        f"rows: {len(df)}",
        f"decision_dates: {df['date_t'].nunique()}",
        f"features_with_interactions: {x_all.shape[1]}",
        "model: tiny bagged regression forest implemented with numpy/pandas",
        "evaluation: leave-one-decision-date-out",
        "",
        f"MAE: {metric['mae']:.6f}",
        f"RMSE: {metric['rmse']:.6f}",
        f"R2: {metric['r2']:.6f}",
        f"decision_correct: {metric['decision_correct']}/{metric['decision_total']}",
        f"decision_accuracy: {metric['decision_accuracy']:.3f}",
        f"mean_decision_regret: {metric['mean_decision_regret']:.6f}",
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

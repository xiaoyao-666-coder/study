#!/usr/bin/env python3
"""Train a nonlinear continuous-irrigation surrogate baseline without sklearn.

This is the second tabular baseline after the ridge sanity check. It keeps the
same continuous-irrigation feature table and leave-one-site-out workflow, but
uses the local TinyForest implementation so the baseline can run on the server
without installing scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    build_features,
    markdown_table,
)


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_features_v1"
    / "confirmed_5site_true_input_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_sampling_smoke_surrogate_tree_nosklearn_v1"
)


@dataclass
class Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class TinyRegressionTree:
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 3,
        max_features: int | None = None,
        rng: np.random.Generator | None = None,
    ):
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
            node = node.left if row[node.feature] <= node.threshold else node.right
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
            thresholds = np.quantile(values, np.linspace(0.1, 0.9, 9)) if len(values) > 20 else (values[:-1] + values[1:]) / 2.0
            for threshold in np.unique(thresholds):
                mask = x[:, feature] <= threshold
                if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
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
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: list[TinyRegressionTree] = []
        self.medians: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TinyForest":
        arr = x.to_numpy(dtype=float)
        self.medians = np.nanmedian(arr, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
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


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if not x[col].isna().all()]


def fit_predict_forest(
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


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def evaluate(
    df: pd.DataFrame,
    *,
    cv_group_col: str,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if "site_date_id" not in df.columns:
        raise ValueError("Missing site_date_id column")
    if cv_group_col not in df.columns:
        raise ValueError(f"Missing CV group column: {cv_group_col}")

    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])

    x_all = build_features(df)
    y_all = pd.to_numeric(df[TARGET], errors="coerce")
    if y_all.isna().any():
        raise ValueError(f"Target column {TARGET} contains NaN")

    groups = sorted(df[cv_group_col].astype(str).unique())
    pred_parts = []
    for i, group_id in enumerate(groups):
        print(
            f"[tree] fold {i + 1}/{len(groups)} holdout {cv_group_col}={group_id}",
            flush=True,
        )
        test_mask = df[cv_group_col].astype(str) == str(group_id)
        train_mask = ~test_mask
        pred = fit_predict_forest(
            x_all.loc[train_mask],
            y_all.loc[train_mask],
            x_all.loc[test_mask],
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + i,
        )
        print(
            f"[tree] fold {i + 1}/{len(groups)} completed rows={int(test_mask.sum())}",
            flush=True,
        )

        keep_cols = [
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
        ]
        part = df.loc[test_mask, keep_cols].copy()
        part["cv_group_col"] = cv_group_col
        part["cv_group_value"] = str(group_id)
        part["model"] = "tiny_forest_nosklearn"
        part["pred_net_gain_7d"] = pred
        pred_parts.append(part)

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
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
    )
    feature_cols = usable_columns(x_all)
    return pred_df, decision_df, metrics_df, by_site, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing continuous-irrigation sample table: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(
        f"[tree] loaded rows={len(df)} cols={len(df.columns)} cv_group_col={args.cv_group_col}",
        flush=True,
    )
    pred_df, decision_df, metrics_df, by_site, feature_cols = evaluate(
        df,
        cv_group_col=args.cv_group_col,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )

    pred_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_tree_nosklearn_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    worst = decision_df.sort_values("decision_regret", ascending=False).head(15)
    lines = [
        "# Continuous Irrigation Surrogate Tree No-Sklearn V1",
        "",
        "## Scope",
        "",
        "- Nonlinear tabular baseline for continuous-irrigation surrogate smoke.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        "- Default validation is leave-one-site-out.",
        "- Uses the local TinyForest implementation; no scikit-learn dependency.",
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

    print("Continuous irrigation surrogate tree nosklearn v1")
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

#!/usr/bin/env python3
"""Train a no-sklearn MLP baseline for site-general continuous irrigation.

This is the first neural-style baseline after the tabular TinyForest checks. It
keeps the same candidate-level feature table and leave-one-site-out protocol,
but uses a small fully connected network implemented with NumPy only so it can
run on the server without installing PyTorch or scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    TARGET,
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import score_metrics


DEFAULT_DATA = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_features_v1"
    / "continuous_ir_12site_surrogate_features_samples_v1.csv"
)
DEFAULT_OUT_DIR = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_mlp_loso_v1"
)


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not sizes:
        raise ValueError("At least one hidden layer size is required")
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Hidden sizes must be positive: {sizes}")
    return sizes


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [
        col
        for col in x.columns
        if not x[col].isna().all()
        and not col.startswith("site_")
        and not col.startswith("candidate_ir_x_site_")
        and not col.startswith("candidate_ir_sq_x_site_")
    ]


def prepare_arrays(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    cols = usable_columns(x_train)
    if not cols:
        raise ValueError("No usable feature columns in training fold")
    train = x_train[cols].to_numpy(dtype=float)
    test = x_test[cols].to_numpy(dtype=float)
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    train_inds = np.where(np.isnan(train))
    test_inds = np.where(np.isnan(test))
    train[train_inds] = np.take(med, train_inds[1])
    test[test_inds] = np.take(med, test_inds[1])
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std <= 1e-12] = 1.0
    return (train - mean) / std, (test - mean) / std, cols, {"median": med, "mean": mean, "std": std}


def init_params(layer_sizes: list[int], rng: np.random.Generator) -> dict:
    params = {}
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        scale = math.sqrt(2.0 / max(1, fan_in))
        params[f"W{i}"] = rng.normal(0.0, scale, size=(fan_in, fan_out))
        params[f"b{i}"] = np.zeros((1, fan_out), dtype=float)
    return params


def forward(x: np.ndarray, params: dict) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    caches = []
    a = x
    n_layers = len(params) // 2
    for i in range(n_layers):
        z = a @ params[f"W{i}"] + params[f"b{i}"]
        if i == n_layers - 1:
            caches.append((a, z))
            return z, caches
        caches.append((a, z))
        a = np.maximum(z, 0.0)
    raise RuntimeError("Invalid MLP parameter state")


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hidden_sizes: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_state: int,
    verbose: bool,
) -> dict:
    rng = np.random.default_rng(random_state)
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std <= 1e-12:
        y_std = 1.0
    y_scaled = ((y - y_mean) / y_std).reshape(-1, 1)

    params = init_params([x.shape[1], *hidden_sizes, 1], rng)
    moments = {f"m_{k}": np.zeros_like(v) for k, v in params.items()}
    velocities = {f"v_{k}": np.zeros_like(v) for k, v in params.items()}
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    n = x.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = y_scaled[idx]
            pred, caches = forward(xb, params)
            grad = (2.0 / len(xb)) * (pred - yb)

            grads = {}
            n_layers = len(params) // 2
            for i in reversed(range(n_layers)):
                a_prev, z = caches[i]
                grads[f"W{i}"] = a_prev.T @ grad + weight_decay * params[f"W{i}"]
                grads[f"b{i}"] = grad.sum(axis=0, keepdims=True)
                if i > 0:
                    grad = grad @ params[f"W{i}"].T
                    grad = grad * (caches[i - 1][1] > 0.0)

            step += 1
            for key in params:
                moments[f"m_{key}"] = beta1 * moments[f"m_{key}"] + (1 - beta1) * grads[key]
                velocities[f"v_{key}"] = beta2 * velocities[f"v_{key}"] + (1 - beta2) * (grads[key] ** 2)
                m_hat = moments[f"m_{key}"] / (1 - beta1**step)
                v_hat = velocities[f"v_{key}"] / (1 - beta2**step)
                params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            train_pred, _ = forward(x, params)
            loss = float(np.mean((train_pred - y_scaled) ** 2))
            print(f"[mlp] epoch {epoch}/{epochs} scaled_mse={loss:.6f}", flush=True)

    return {"params": params, "y_mean": y_mean, "y_std": y_std}


def predict_mlp(model: dict, x: np.ndarray) -> np.ndarray:
    pred_scaled, _ = forward(x, model["params"])
    return pred_scaled.reshape(-1) * model["y_std"] + model["y_mean"]


def fit_predict_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    hidden_sizes: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_state: int,
    verbose: bool,
) -> tuple[np.ndarray, list[str]]:
    train_arr, test_arr, cols, _stats = prepare_arrays(x_train, x_test)
    model = train_mlp(
        train_arr,
        y_train.to_numpy(dtype=float),
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        random_state=random_state,
        verbose=verbose,
    )
    return predict_mlp(model, test_arr), cols


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def evaluate(
    df: pd.DataFrame,
    *,
    cv_group_col: str,
    hidden_sizes: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    random_state: int,
    verbose: bool,
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
    used_cols = set()
    for i, group_id in enumerate(groups):
        print(f"[mlp] fold {i + 1}/{len(groups)} holdout {cv_group_col}={group_id}", flush=True)
        test_mask = df[cv_group_col].astype(str) == str(group_id)
        train_mask = ~test_mask
        pred, cols = fit_predict_mlp(
            x_all.loc[train_mask],
            y_all.loc[train_mask],
            x_all.loc[test_mask],
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            random_state=random_state + i,
            verbose=verbose,
        )
        used_cols.update(cols)
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
        part["model"] = "mlp_nosklearn"
        part["pred_net_gain_7d"] = pred
        pred_parts.append(part)
        print(f"[mlp] fold {i + 1}/{len(groups)} completed rows={int(test_mask.sum())}", flush=True)

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
            "hidden_sizes": ",".join(str(size) for size in hidden_sizes),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
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
    return pred_df, decision_df, metrics_df, by_site, sorted(used_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cv-group-col", default="site_id")
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--verbose-epochs", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing continuous-irrigation sample table: {data_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(
        f"[mlp] loaded rows={len(df)} cols={len(df.columns)} cv_group_col={args.cv_group_col}",
        flush=True,
    )
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    pred_df, decision_df, metrics_df, by_site, feature_cols = evaluate(
        df,
        cv_group_col=args.cv_group_col,
        hidden_sizes=hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        random_state=args.random_state,
        verbose=args.verbose_epochs,
    )

    pred_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1_predictions.csv"
    decision_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1_decision_eval.csv"
    metrics_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1_metrics.csv"
    by_site_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1_by_site.csv"
    feature_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1_features.json"
    report_path = out_dir / "continuous_irrigation_surrogate_mlp_nosklearn_v1.md"

    pred_df.to_csv(pred_path, index=False)
    decision_df.to_csv(decision_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    worst = decision_df.sort_values("decision_regret", ascending=False).head(15)
    lines = [
        "# Continuous Irrigation Surrogate MLP No-Sklearn V1",
        "",
        "## Scope",
        "",
        "- Site-general continuous-input neural tabular baseline.",
        f"- Input table: `{data_path}`.",
        f"- CV group column: `{args.cv_group_col}`.",
        f"- Hidden sizes: `{args.hidden_sizes}`.",
        "- NumPy implementation; no PyTorch/scikit-learn dependency.",
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

    print("Continuous irrigation surrogate MLP no-sklearn v1")
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

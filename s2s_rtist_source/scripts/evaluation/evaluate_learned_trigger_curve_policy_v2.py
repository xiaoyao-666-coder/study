#!/usr/bin/env python3
"""Evaluate learned trigger + amount decision policies.

Trigger:
  A small numpy logistic classifier predicts whether irrigation should be
  considered for each decision date.

Amount choice if triggered:
  1. raw_tree_peak: candidate with highest tree-predicted net gain
  2. curve_peak: candidate at the peak of a constrained concave response curve

The probability threshold is tuned on training dates in leave-one-date-out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SAMPLES = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
PRED = Path("Maize_shortterm_surrogate_v1/surrogate_tree_nosklearn_v1_predictions.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
DECISION_OUT = OUT_DIR / "learned_trigger_curve_policy_v2_decision_eval.csv"
METRICS_OUT = OUT_DIR / "learned_trigger_curve_policy_v2_metrics.txt"

TRIGGER_FEATURES = [
    "decision_doy_sin",
    "decision_doy_cos",
    "state_dvs",
    "state_lai",
    "state_cwdm",
    "state_cwso",
    "soil_h_mean_0_30_cm",
    "soil_h_mean_30_60_cm",
    "future_precip_sum",
    "future_tmax_mean",
    "future_tmin_mean",
    "hist_precip_sum",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def merge_inputs() -> pd.DataFrame:
    samples = pd.read_csv(SAMPLES)
    pred = pd.read_csv(PRED)
    merged = samples.merge(pred[["sample_id", "pred_net_gain_7d"]], on="sample_id", how="left")
    if merged["pred_net_gain_7d"].isna().any():
        raise RuntimeError("Some samples did not match tree predictions.")
    return merged


def fit_concave_curve(ir: np.ndarray, pred_gain: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    best_sse = np.inf
    best_fit = None
    best_mu = None
    best_beta = None
    best_c = None
    for mu in np.linspace(float(ir.min()), float(ir.max()), 31):
        z = (ir - mu) ** 2
        var_z = float(np.var(z))
        beta = 0.0 if var_z <= 1e-12 else float(np.cov(z, pred_gain, bias=True)[0, 1] / var_z)
        if beta > 0:
            beta = 0.0
        c = float(np.mean(pred_gain - beta * z))
        fit = c + beta * z
        sse = float(np.sum((pred_gain - fit) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_fit = fit
            best_mu = mu
            best_beta = beta
            best_c = c
    return best_fit, float(best_mu), float(best_beta), float(best_c)


def build_date_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date_t, group in df.groupby("date_t", sort=False):
        g = group.sort_values("candidate_ir").reset_index(drop=True)
        ir = g["candidate_ir"].to_numpy(dtype=float)
        pred = g["pred_net_gain_7d"].to_numpy(dtype=float)
        fit, mu, beta, c = fit_concave_curve(ir, pred)

        true_best = g.loc[g["net_gain_7d"].idxmax()]
        raw_best = g.loc[g["pred_net_gain_7d"].idxmax()]
        curve_best = g.iloc[int(np.argmax(fit))]
        zero = g.loc[g["candidate_ir"] == 0].iloc[0]
        first = g.iloc[0]
        row = {
            "date_t": date_t,
            "decision_doy": int(true_best["decision_doy"]),
            "true_best_ir": float(true_best["candidate_ir"]),
            "true_best_net_gain": float(true_best["net_gain_7d"]),
            "true_trigger": float(true_best["candidate_ir"]) > 0.0,
            "zero_true_gain": float(zero["net_gain_7d"]),
            "raw_tree_ir": float(raw_best["candidate_ir"]),
            "raw_tree_true_gain": float(raw_best["net_gain_7d"]),
            "curve_ir": float(curve_best["candidate_ir"]),
            "curve_true_gain": float(curve_best["net_gain_7d"]),
            "curve_mu": mu,
            "curve_beta": beta,
            "curve_c": c,
        }
        doy = float(first["decision_doy"])
        row["decision_doy_sin"] = float(np.sin(2 * np.pi * doy / 366.0))
        row["decision_doy_cos"] = float(np.cos(2 * np.pi * doy / 366.0))
        for col in TRIGGER_FEATURES[2:]:
            row[col] = float(first[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("decision_doy").reset_index(drop=True)


def standardize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    med = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(med)
    test = test.fillna(med)
    mean = train.mean()
    std = train.std().replace(0, 1.0).fillna(1.0)
    return ((train - mean) / std).to_numpy(dtype=float), ((test - mean) / std).to_numpy(dtype=float)


def train_logistic(x: np.ndarray, y: np.ndarray, *, lr: float = 0.05, reg: float = 0.1, steps: int = 2500) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(x1.shape[1], dtype=float)
    for step in range(steps):
        p = sigmoid(x1 @ w)
        grad = x1.T @ (p - y) / len(y)
        grad[1:] += reg * w[1:]
        w -= lr * grad
        if step % 500 == 0 and step > 0:
            lr *= 0.8
    return w


def predict_prob(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    return sigmoid(x1 @ w)


def eval_policy(date_table: pd.DataFrame, dates: list[str], probs: dict[str, float], threshold: float, amount_policy: str) -> tuple[float, float]:
    regrets = []
    correct = []
    for date_t in dates:
        row = date_table[date_table["date_t"] == date_t].iloc[0]
        triggered = probs[date_t] >= threshold
        if not triggered:
            chosen_ir = 0.0
            chosen_gain = row["zero_true_gain"]
        elif amount_policy == "raw_tree_peak":
            chosen_ir = row["raw_tree_ir"]
            chosen_gain = row["raw_tree_true_gain"]
        elif amount_policy == "curve_peak":
            chosen_ir = row["curve_ir"]
            chosen_gain = row["curve_true_gain"]
        else:
            raise ValueError(amount_policy)
        regrets.append(float(row["true_best_net_gain"] - chosen_gain))
        correct.append(float(chosen_ir) == float(row["true_best_ir"]))
    return float(np.mean(regrets)), float(np.mean(correct))


def tune_threshold(date_table: pd.DataFrame, train_dates: list[str], probs: dict[str, float], amount_policy: str) -> tuple[float, float, float]:
    best = None
    for threshold in np.linspace(0.30, 0.85, 12):
        regret, acc = eval_policy(date_table, train_dates, probs, float(threshold), amount_policy)
        key = (regret, -acc, threshold)
        if best is None or key < best[0]:
            best = (key, float(threshold), regret, acc)
    return best[1], best[2], best[3]


def main() -> None:
    df = merge_inputs()
    date_table = build_date_table(df)
    dates = date_table["date_t"].tolist()
    out_rows = []

    for amount_policy in ["raw_tree_peak", "curve_peak"]:
        for held_date in dates:
            train_dates = [d for d in dates if d != held_date]
            train = date_table[date_table["date_t"].isin(train_dates)]
            test = date_table[date_table["date_t"] == held_date]
            x_train, x_test = standardize(train[TRIGGER_FEATURES], test[TRIGGER_FEATURES])
            y_train = train["true_trigger"].astype(float).to_numpy()
            w = train_logistic(x_train, y_train)

            # Need probabilities for train + held date under the same fold model.
            all_fold = date_table[date_table["date_t"].isin(train_dates + [held_date])]
            x_train_again, x_all = standardize(train[TRIGGER_FEATURES], all_fold[TRIGGER_FEATURES])
            probs_arr = predict_prob(w, x_all)
            probs = {d: float(p) for d, p in zip(all_fold["date_t"], probs_arr)}
            threshold, train_regret, train_acc = tune_threshold(date_table, train_dates, probs, amount_policy)

            row = test.iloc[0]
            triggered = probs[held_date] >= threshold
            if not triggered:
                chosen_ir = 0.0
                chosen_gain = row["zero_true_gain"]
            elif amount_policy == "raw_tree_peak":
                chosen_ir = row["raw_tree_ir"]
                chosen_gain = row["raw_tree_true_gain"]
            else:
                chosen_ir = row["curve_ir"]
                chosen_gain = row["curve_true_gain"]
            out_rows.append(
                {
                    "amount_policy": amount_policy,
                    "date_t": held_date,
                    "decision_doy": int(row["decision_doy"]),
                    "trigger_prob": probs[held_date],
                    "trigger_threshold": threshold,
                    "triggered": bool(triggered),
                    "true_trigger": bool(row["true_trigger"]),
                    "true_best_ir": float(row["true_best_ir"]),
                    "true_best_net_gain": float(row["true_best_net_gain"]),
                    "chosen_ir": float(chosen_ir),
                    "chosen_true_net_gain": float(chosen_gain),
                    "decision_correct": float(chosen_ir) == float(row["true_best_ir"]),
                    "decision_regret": float(row["true_best_net_gain"] - chosen_gain),
                    "train_regret": train_regret,
                    "train_accuracy": train_acc,
                    "raw_tree_ir": float(row["raw_tree_ir"]),
                    "curve_ir": float(row["curve_ir"]),
                }
            )

    result = pd.DataFrame(out_rows)
    result.to_csv(DECISION_OUT, index=False)

    summary = (
        result.groupby("amount_policy", as_index=False)
        .agg(
            decision_accuracy=("decision_correct", "mean"),
            mean_regret=("decision_regret", "mean"),
            trigger_accuracy=("triggered", lambda s: float((s.to_numpy() == result.loc[s.index, "true_trigger"].to_numpy()).mean())),
            trigger_rate=("triggered", "mean"),
        )
    )

    lines = [
        "Learned trigger + amount policy v2",
        "",
        f"samples: {SAMPLES}",
        f"tree predictions: {PRED}",
        "trigger model: numpy logistic classifier, leave-one-date-out",
        "threshold: tuned on training dates to minimize regret; lower bound raised to 0.30",
        "",
        summary.to_string(index=False),
        "",
        f"wrote: {DECISION_OUT}",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nDecision eval:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

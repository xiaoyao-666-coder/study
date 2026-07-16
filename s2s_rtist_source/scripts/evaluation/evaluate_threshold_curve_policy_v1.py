#!/usr/bin/env python3
"""Evaluate threshold-trigger + constrained response curve policy.

Policy:
1. Trigger irrigation only when soil/crop/weather thresholds indicate it is
   worth considering irrigation.
2. If triggered, fit a constrained concave response curve to the tree-predicted
   candidate net gains, then choose the candidate near the fitted peak.

Thresholds are selected in leave-one-date-out fashion using only training dates.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLES = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
PRED = Path("Maize_shortterm_surrogate_v1/surrogate_tree_nosklearn_v1_predictions.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
DECISION_OUT = OUT_DIR / "threshold_curve_policy_v1_decision_eval.csv"
DETAIL_OUT = OUT_DIR / "threshold_curve_policy_v1_candidate_detail.csv"
METRICS_OUT = OUT_DIR / "threshold_curve_policy_v1_metrics.txt"


H_THRESHOLDS = [-20000, -12000, -10000, -8000, -6000, -4000, -2000, -1000, -500, -100]
RAIN_THRESHOLDS = [0, 2, 5, 10, 20, 35, 60]
DVS_MAX_THRESHOLDS = [1.3, 1.4, 1.5, 1.55, 1.6, 1.7, 1.85, 2.1]
MIN_GAIN_THRESHOLDS = [0, 1, 2, 5, 10, 15]


def merge_inputs() -> pd.DataFrame:
    if not SAMPLES.exists():
        raise FileNotFoundError(f"Missing samples: {SAMPLES}")
    if not PRED.exists():
        raise FileNotFoundError(f"Missing tree predictions: {PRED}")
    samples = pd.read_csv(SAMPLES)
    pred = pd.read_csv(PRED)
    keep = [
        "sample_id",
        "pred_net_gain_7d",
    ]
    merged = samples.merge(pred[keep], on="sample_id", how="left")
    if merged["pred_net_gain_7d"].isna().any():
        raise RuntimeError("Some samples did not match tree predictions.")
    return merged


def fit_concave_curve(ir: np.ndarray, pred_gain: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Fit y = c + beta * (ir - mu)^2 with beta <= 0 by grid over mu."""
    best_sse = np.inf
    best_fit = None
    best_mu = None
    best_beta = None
    best_c = None
    mus = np.linspace(float(ir.min()), float(ir.max()), 61)
    for mu in mus:
        z = (ir - mu) ** 2
        var_z = float(np.var(z))
        if var_z <= 1e-12:
            beta = 0.0
        else:
            beta = float(np.cov(z, pred_gain, bias=True)[0, 1] / var_z)
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


def apply_policy_to_group(group: pd.DataFrame, params: dict) -> dict:
    g = group.sort_values("candidate_ir").reset_index(drop=True).copy()
    row0 = g.iloc[0]
    h = float(row0["soil_h_mean_0_30_cm"])
    rain = float(row0["future_precip_sum"])
    dvs = float(row0["state_dvs"])

    trigger = (
        h <= params["h_threshold"]
        and rain <= params["rain_threshold"]
        and dvs <= params["dvs_max"]
    )

    ir = g["candidate_ir"].to_numpy(dtype=float)
    pred = g["pred_net_gain_7d"].to_numpy(dtype=float)
    fit, mu, beta, c = fit_concave_curve(ir, pred)
    g["curve_fit_pred_gain"] = fit

    zero_fit = float(g.loc[g["candidate_ir"] == 0, "curve_fit_pred_gain"].iloc[0])
    best_idx = int(np.argmax(fit))
    best_fit_gain = float(fit[best_idx])
    curve_gain_over_zero = best_fit_gain - zero_fit

    if (not trigger) or curve_gain_over_zero < params["min_gain"]:
        chosen = g.loc[g["candidate_ir"] == 0].iloc[0]
        reason = "no_trigger_or_low_gain"
    else:
        chosen = g.iloc[best_idx]
        reason = "trigger_curve_peak"

    true_best = g.loc[g["net_gain_7d"].idxmax()]
    return {
        "date_t": str(true_best["date_t"]),
        "decision_doy": int(true_best["decision_doy"]),
        "true_best_ir": float(true_best["candidate_ir"]),
        "true_best_net_gain": float(true_best["net_gain_7d"]),
        "chosen_ir": float(chosen["candidate_ir"]),
        "chosen_true_net_gain": float(chosen["net_gain_7d"]),
        "decision_correct": float(chosen["candidate_ir"]) == float(true_best["candidate_ir"]),
        "decision_regret": float(true_best["net_gain_7d"] - chosen["net_gain_7d"]),
        "trigger": bool(trigger),
        "reason": reason,
        "h_0_30": h,
        "future_precip_sum": rain,
        "state_dvs": dvs,
        "curve_mu": mu,
        "curve_beta": beta,
        "curve_c": c,
        "curve_gain_over_zero": curve_gain_over_zero,
        **params,
    }, g


def evaluate_params(df: pd.DataFrame, dates: list[str], params: dict) -> tuple[float, float]:
    regrets = []
    correct = []
    for date_t in dates:
        group = df[df["date_t"] == date_t]
        row, _ = apply_policy_to_group(group, params)
        regrets.append(row["decision_regret"])
        correct.append(row["decision_correct"])
    return float(np.mean(regrets)), float(np.mean(correct))


def choose_params(df: pd.DataFrame, train_dates: list[str]) -> dict:
    best = None
    for h_threshold, rain_threshold, dvs_max, min_gain in product(
        H_THRESHOLDS, RAIN_THRESHOLDS, DVS_MAX_THRESHOLDS, MIN_GAIN_THRESHOLDS
    ):
        params = {
            "h_threshold": h_threshold,
            "rain_threshold": rain_threshold,
            "dvs_max": dvs_max,
            "min_gain": min_gain,
        }
        regret, acc = evaluate_params(df, train_dates, params)
        key = (regret, -acc, min_gain, rain_threshold)
        if best is None or key < best[0]:
            best = (key, params, regret, acc)
    return {**best[1], "train_mean_regret": best[2], "train_accuracy": best[3]}


def main() -> None:
    df = merge_inputs()
    dates = sorted(df["date_t"].unique(), key=lambda d: int(df.loc[df["date_t"] == d, "decision_doy"].iloc[0]))

    decisions = []
    detail_parts = []
    for held_date in dates:
        train_dates = [d for d in dates if d != held_date]
        params = choose_params(df, train_dates)
        decision, detail = apply_policy_to_group(df[df["date_t"] == held_date], params)
        decisions.append(decision)
        detail["heldout_date"] = held_date
        detail["curve_selected"] = detail["candidate_ir"] == decision["chosen_ir"]
        detail_parts.append(detail)

    decision_df = pd.DataFrame(decisions)
    detail_df = pd.concat(detail_parts, ignore_index=True)
    decision_df.to_csv(DECISION_OUT, index=False)
    detail_df.to_csv(DETAIL_OUT, index=False)

    correct = int(decision_df["decision_correct"].sum())
    total = len(decision_df)
    mean_regret = float(decision_df["decision_regret"].mean())
    trigger_rate = float(decision_df["trigger"].mean())
    lines = [
        "Threshold-trigger + constrained response curve policy v1",
        "",
        f"samples: {SAMPLES}",
        f"tree predictions: {PRED}",
        "evaluation: leave-one-decision-date-out threshold tuning",
        "",
        f"decision_correct: {correct}/{total}",
        f"decision_accuracy: {correct / total:.3f}",
        f"mean_decision_regret: {mean_regret:.6f}",
        f"trigger_rate: {trigger_rate:.3f}",
        "",
        f"wrote: {DECISION_OUT}",
        f"wrote: {DETAIL_OUT}",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nDecision eval:")
    print(decision_df.to_string(index=False))


if __name__ == "__main__":
    main()

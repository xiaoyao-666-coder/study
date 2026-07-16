#!/usr/bin/env python3
"""Fast threshold-trigger + constrained response curve policy.

This version precomputes the fitted response curve for each date, then runs a
small threshold search with progress logging.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLES = Path("Maize_shortterm_surrogate_v1/shortterm_surrogate_samples_v1_with_weather_v2.csv")
PRED = Path("Maize_shortterm_surrogate_v1/surrogate_tree_nosklearn_v1_predictions.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
DECISION_OUT = OUT_DIR / "threshold_curve_policy_fast_v1_decision_eval.csv"
DETAIL_OUT = OUT_DIR / "threshold_curve_policy_fast_v1_candidate_detail.csv"
METRICS_OUT = OUT_DIR / "threshold_curve_policy_fast_v1_metrics.txt"

H_THRESHOLDS = [-12000, -10000, -8000, -6000, -4000, -1000]
RAIN_THRESHOLDS = [5, 10, 20, 35]
DVS_MAX_THRESHOLDS = [1.45, 1.55, 1.65, 1.85]
MIN_GAIN_THRESHOLDS = [0, 2, 5, 10]


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


def precompute_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    details = []
    for date_t, group in df.groupby("date_t", sort=False):
        g = group.sort_values("candidate_ir").reset_index(drop=True).copy()
        ir = g["candidate_ir"].to_numpy(dtype=float)
        pred = g["pred_net_gain_7d"].to_numpy(dtype=float)
        fit, mu, beta, c = fit_concave_curve(ir, pred)
        g["curve_fit_pred_gain"] = fit
        g["curve_mu"] = mu
        g["curve_beta"] = beta
        g["curve_c"] = c

        zero = g.loc[g["candidate_ir"] == 0].iloc[0]
        curve_best = g.loc[g["curve_fit_pred_gain"].idxmax()]
        true_best = g.loc[g["net_gain_7d"].idxmax()]

        rows.append(
            {
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "zero_true_gain": float(zero["net_gain_7d"]),
                "curve_best_ir": float(curve_best["candidate_ir"]),
                "curve_best_true_gain": float(curve_best["net_gain_7d"]),
                "curve_gain_over_zero": float(curve_best["curve_fit_pred_gain"] - zero["curve_fit_pred_gain"]),
                "h_0_30": float(g.iloc[0]["soil_h_mean_0_30_cm"]),
                "future_precip_sum": float(g.iloc[0]["future_precip_sum"]),
                "state_dvs": float(g.iloc[0]["state_dvs"]),
                "curve_mu": mu,
                "curve_beta": beta,
                "curve_c": c,
            }
        )
        details.append(g)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def choose_for_date(row: pd.Series, params: dict) -> dict:
    trigger = (
        row["h_0_30"] <= params["h_threshold"]
        and row["future_precip_sum"] <= params["rain_threshold"]
        and row["state_dvs"] <= params["dvs_max"]
        and row["curve_gain_over_zero"] >= params["min_gain"]
    )
    if trigger:
        chosen_ir = row["curve_best_ir"]
        chosen_true_gain = row["curve_best_true_gain"]
        reason = "trigger_curve_peak"
    else:
        chosen_ir = 0.0
        chosen_true_gain = row["zero_true_gain"]
        reason = "no_trigger_or_low_gain"
    return {
        "chosen_ir": float(chosen_ir),
        "chosen_true_net_gain": float(chosen_true_gain),
        "decision_correct": float(chosen_ir) == float(row["true_best_ir"]),
        "decision_regret": float(row["true_best_net_gain"] - chosen_true_gain),
        "trigger": bool(trigger),
        "reason": reason,
    }


def evaluate_params(date_summary: pd.DataFrame, dates: list[str], params: dict) -> tuple[float, float]:
    regrets = []
    correct = []
    subset = date_summary[date_summary["date_t"].isin(dates)]
    for row in subset.itertuples(index=False):
        decision = choose_for_date(pd.Series(row._asdict()), params)
        regrets.append(decision["decision_regret"])
        correct.append(decision["decision_correct"])
    return float(np.mean(regrets)), float(np.mean(correct))


def choose_params(date_summary: pd.DataFrame, train_dates: list[str]) -> dict:
    best_key = None
    best_params = None
    best_regret = None
    best_acc = None
    for h_threshold, rain_threshold, dvs_max, min_gain in product(
        H_THRESHOLDS, RAIN_THRESHOLDS, DVS_MAX_THRESHOLDS, MIN_GAIN_THRESHOLDS
    ):
        params = {
            "h_threshold": h_threshold,
            "rain_threshold": rain_threshold,
            "dvs_max": dvs_max,
            "min_gain": min_gain,
        }
        regret, acc = evaluate_params(date_summary, train_dates, params)
        key = (regret, -acc, min_gain, rain_threshold)
        if best_key is None or key < best_key:
            best_key = key
            best_params = params
            best_regret = regret
            best_acc = acc
    return {**best_params, "train_mean_regret": best_regret, "train_accuracy": best_acc}


def main() -> None:
    print("loading data...", flush=True)
    df = merge_inputs()
    print("precomputing response curves...", flush=True)
    date_summary, detail = precompute_by_date(df)
    dates = sorted(date_summary["date_t"].tolist(), key=lambda d: int(date_summary.loc[date_summary["date_t"] == d, "decision_doy"].iloc[0]))

    decisions = []
    for i, held_date in enumerate(dates, start=1):
        print(f"[{i}/{len(dates)}] tuning thresholds for held-out {held_date}", flush=True)
        train_dates = [d for d in dates if d != held_date]
        params = choose_params(date_summary, train_dates)
        row = date_summary.loc[date_summary["date_t"] == held_date].iloc[0]
        decision = choose_for_date(row, params)
        decisions.append(
            {
                "date_t": held_date,
                "decision_doy": int(row["decision_doy"]),
                "true_best_ir": float(row["true_best_ir"]),
                "true_best_net_gain": float(row["true_best_net_gain"]),
                **decision,
                **params,
                "h_0_30": float(row["h_0_30"]),
                "future_precip_sum": float(row["future_precip_sum"]),
                "state_dvs": float(row["state_dvs"]),
                "curve_best_ir": float(row["curve_best_ir"]),
                "curve_gain_over_zero": float(row["curve_gain_over_zero"]),
            }
        )

    decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(DECISION_OUT, index=False)
    detail.to_csv(DETAIL_OUT, index=False)

    correct = int(decision_df["decision_correct"].sum())
    total = len(decision_df)
    mean_regret = float(decision_df["decision_regret"].mean())
    lines = [
        "Threshold-trigger + constrained response curve policy fast v1",
        "",
        f"decision_correct: {correct}/{total}",
        f"decision_accuracy: {correct / total:.3f}",
        f"mean_decision_regret: {mean_regret:.6f}",
        f"trigger_rate: {decision_df['trigger'].mean():.3f}",
        "",
        f"wrote: {DECISION_OUT}",
        f"wrote: {DETAIL_OUT}",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("\nDecision eval:", flush=True)
    print(decision_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

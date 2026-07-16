#!/usr/bin/env python3
"""Smooth tree surrogate candidate curves and re-evaluate decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRED = Path("Maize_shortterm_surrogate_v1/surrogate_tree_nosklearn_v1_predictions.csv")
OUT_DIR = Path("Maize_shortterm_surrogate_v1")
SMOOTH_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_predictions_smoothed.csv"
DECISION_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_decision_eval_smoothed.csv"
METRICS_OUT = OUT_DIR / "surrogate_tree_nosklearn_v1_metrics_smoothed.txt"


def smooth_one_curve(values: np.ndarray, mode: str = "neighbor3") -> np.ndarray:
    """Smooth an ordered 8-point curve over irrigation amounts."""
    if len(values) != 8:
        raise ValueError(f"Expected 8 candidate values, got {len(values)}")

    v = np.asarray(values, dtype=float)
    out = v.copy()

    if mode == "neighbor3":
        for i in range(len(v)):
            lo = max(0, i - 1)
            hi = min(len(v), i + 2)
            out[i] = float(np.mean(v[lo:hi]))
    elif mode == "triangular3":
        w = np.array([1.0, 2.0, 1.0], dtype=float)
        for i in range(len(v)):
            idx = [j for j in (i - 1, i, i + 1) if 0 <= j < len(v)]
            ww = w[[j - (i - 1) for j in idx]]
            out[i] = float(np.dot(v[idx], ww) / ww.sum())
    elif mode == "edge_preserve":
        out[0] = 0.7 * v[0] + 0.3 * v[1]
        out[-1] = 0.7 * v[-1] + 0.3 * v[-2]
        for i in range(1, len(v) - 1):
            out[i] = 0.25 * v[i - 1] + 0.5 * v[i] + 0.25 * v[i + 1]
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")

    return out


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    if not PRED.exists():
        raise FileNotFoundError(f"Missing prediction file: {PRED}")

    df = pd.read_csv(PRED)
    if "candidate_ir" not in df.columns or "pred_net_gain_7d" not in df.columns:
        raise ValueError("Prediction file missing required columns")

    smooth_parts = []
    decision_rows = []

    for date_t, group in df.groupby("date_t", sort=False):
        group = group.sort_values("candidate_ir").reset_index(drop=True)
        pred_raw = group["pred_net_gain_7d"].to_numpy(dtype=float)
        pred_smooth = smooth_one_curve(pred_raw, mode="neighbor3")

        sub = group.copy()
        sub["pred_net_gain_raw"] = pred_raw
        sub["pred_net_gain_smoothed"] = pred_smooth
        sub["smooth_mode"] = "neighbor3"
        smooth_parts.append(sub)

        true_best = sub.loc[sub["net_gain_7d"].idxmax()]
        raw_best = sub.loc[sub["pred_net_gain_raw"].idxmax()]
        sm_best = sub.loc[sub["pred_net_gain_smoothed"].idxmax()]

        decision_rows.append(
            {
                "date_t": date_t,
                "decision_doy": int(true_best["decision_doy"]),
                "true_best_ir": float(true_best["candidate_ir"]),
                "true_best_net_gain": float(true_best["net_gain_7d"]),
                "raw_pred_best_ir": float(raw_best["candidate_ir"]),
                "raw_pred_best_true_gain": float(raw_best["net_gain_7d"]),
                "raw_decision_correct": float(raw_best["candidate_ir"]) == float(true_best["candidate_ir"]),
                "raw_regret": float(true_best["net_gain_7d"] - raw_best["net_gain_7d"]),
                "smoothed_pred_best_ir": float(sm_best["candidate_ir"]),
                "smoothed_pred_best_true_gain": float(sm_best["net_gain_7d"]),
                "smoothed_decision_correct": float(sm_best["candidate_ir"]) == float(true_best["candidate_ir"]),
                "smoothed_regret": float(true_best["net_gain_7d"] - sm_best["net_gain_7d"]),
                "raw_curve_corr": float(np.corrcoef(sub["net_gain_7d"], sub["pred_net_gain_raw"])[0, 1]),
                "smoothed_curve_corr": float(np.corrcoef(sub["net_gain_7d"], sub["pred_net_gain_smoothed"])[0, 1]),
            }
        )

    smooth_df = pd.concat(smooth_parts, ignore_index=True)
    decision_df = pd.DataFrame(decision_rows)
    smooth_df.to_csv(SMOOTH_OUT, index=False)
    decision_df.to_csv(DECISION_OUT, index=False)

    raw_metrics = metrics(smooth_df["net_gain_7d"].to_numpy(dtype=float), smooth_df["pred_net_gain_raw"].to_numpy(dtype=float))
    sm_metrics = metrics(smooth_df["net_gain_7d"].to_numpy(dtype=float), smooth_df["pred_net_gain_smoothed"].to_numpy(dtype=float))
    raw_correct = int(decision_df["raw_decision_correct"].sum())
    sm_correct = int(decision_df["smoothed_decision_correct"].sum())
    lines = [
        "Tree surrogate smoothing v1",
        "",
        f"prediction file: {PRED}",
        f"output smoothed file: {SMOOTH_OUT}",
        f"decision output: {DECISION_OUT}",
        "smoothing mode: neighbor3",
        "",
        "raw prediction metrics:",
        f"  MAE: {raw_metrics['mae']:.6f}",
        f"  RMSE: {raw_metrics['rmse']:.6f}",
        f"  R2: {raw_metrics['r2']:.6f}",
        f"  decision_accuracy: {raw_correct}/{len(decision_df)} = {raw_correct / len(decision_df):.3f}",
        f"  mean_regret: {decision_df['raw_regret'].mean():.6f}",
        "",
        "smoothed prediction metrics:",
        f"  MAE: {sm_metrics['mae']:.6f}",
        f"  RMSE: {sm_metrics['rmse']:.6f}",
        f"  R2: {sm_metrics['r2']:.6f}",
        f"  decision_accuracy: {sm_correct}/{len(decision_df)} = {sm_correct / len(decision_df):.3f}",
        f"  mean_regret: {decision_df['smoothed_regret'].mean():.6f}",
        "",
    ]
    METRICS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nDecision eval:")
    cols = [
        "date_t",
        "decision_doy",
        "true_best_ir",
        "raw_pred_best_ir",
        "raw_pred_best_true_gain",
        "raw_decision_correct",
        "raw_regret",
        "smoothed_pred_best_ir",
        "smoothed_pred_best_true_gain",
        "smoothed_decision_correct",
        "smoothed_regret",
        "raw_curve_corr",
        "smoothed_curve_corr",
    ]
    print(decision_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()

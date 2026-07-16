#!/usr/bin/env python3
"""Evaluate lightweight TTA-style output calibration for per-site experts.

This diagnostic keeps the per-site TinyForest expert fixed for each test date
and adapts only a tiny output correction layer using the previous K expanded
target-site dates. It is meant to test the teacher's TTA idea more directly
than retraining the whole expert on a few recent dates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from compare_discrete_vs_continuous_ir_optimization_v1 import (
    DEFAULT_PAPER_CANDIDATES,
    TARGET,
    candidate_set_for_site,
    interp_gain,
    parse_candidates,
)
from train_confirmed_5site_true_input_surrogate_baseline_v1 import (
    bool_series,
    build_features,
    markdown_table,
)
from train_continuous_irrigation_surrogate_tree_nosklearn_v1 import TinyForest, score_metrics
from train_persite_tinyforest_profit_surrogate_v1 import (
    build_candidate_rows,
    dense_values,
    predict_forest,
    select_feature_mode,
)


DEFAULT_BASE_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_ADAPT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_date_coverage_fast_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_lightweight_output_calibration_v1"
)
DEFAULT_MODES = "none,bias,ir_linear,ir_quadratic,pred_ir_quadratic"


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one calibration count is required")
    if any(value < 0 for value in values):
        raise ValueError("Calibration counts must be non-negative")
    return sorted(set(values))


def parse_modes(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    allowed = {
        "none",
        "bias",
        "ir_linear",
        "ir_quadratic",
        "ir_doy_quadratic",
        "pred_ir_quadratic",
    }
    bad = sorted(set(modes).difference(allowed))
    if bad:
        raise ValueError(f"Unknown calibration modes: {bad}")
    return modes


def prepare_frame(path: Path, source_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {source_name} table: {path}")
    df = pd.read_csv(path)
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    required = {"site_id", "site_date_id", "date_t", "decision_doy", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{source_name} table missing columns: {missing}")
    return df


def date_order_table(site_df: pd.DataFrame) -> pd.DataFrame:
    dates = site_df[["site_date_id", "date_t", "decision_doy"]].drop_duplicates().copy()
    dates["sort_key"] = pd.to_numeric(dates["decision_doy"], errors="coerce")
    dates = dates.sort_values(["sort_key", "site_date_id"]).reset_index(drop=True)
    dates["date_order"] = np.arange(len(dates), dtype=int)
    return dates


def usable_columns(x: pd.DataFrame) -> list[str]:
    return [col for col in x.columns if not x[col].isna().all()]


def fit_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> tuple[TinyForest, list[str]]:
    cols = usable_columns(x_train)
    if not cols:
        raise ValueError("No usable feature columns")
    model = TinyForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(x_train[cols], y_train)
    return model, cols


def add_truth(rows: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["interp_true_net_gain_7d"] = [
        interp_gain(curve, float(ir)) for ir in rows["candidate_ir"].to_numpy(dtype=float)
    ]
    return rows


def ir_fraction(rows: pd.DataFrame) -> np.ndarray:
    ir = pd.to_numeric(rows["candidate_ir"], errors="coerce").to_numpy(dtype=float)
    max_ir = pd.to_numeric(rows["site_ir_max"], errors="coerce").replace(0.0, np.nan).to_numpy(dtype=float)
    frac = ir / max_ir
    return np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)


def doy_terms(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.to_numeric(rows["decision_doy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    angle = 2.0 * math.pi * doy / 366.0
    return np.sin(angle), np.cos(angle)


def calibration_matrix(rows: pd.DataFrame, base_pred: np.ndarray, mode: str) -> np.ndarray:
    n = len(rows)
    ones = np.ones(n, dtype=float)
    if mode in {"none", "bias"}:
        return ones.reshape(-1, 1)

    frac = ir_fraction(rows)
    frac2 = frac**2
    if mode == "ir_linear":
        cols = [ones, frac]
    elif mode == "ir_quadratic":
        cols = [ones, frac, frac2]
    elif mode == "ir_doy_quadratic":
        sin_doy, cos_doy = doy_terms(rows)
        cols = [ones, frac, frac2, sin_doy, cos_doy, frac * sin_doy, frac * cos_doy]
    elif mode == "pred_ir_quadratic":
        pred = np.asarray(base_pred, dtype=float)
        pred_scaled = (pred - np.nanmean(pred)) / (np.nanstd(pred) + 1e-9)
        cols = [ones, pred_scaled, frac, frac2, pred_scaled * frac]
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")
    return np.vstack(cols).T.astype(float)


def calibration_width(mode: str) -> int:
    if mode in {"none", "bias"}:
        return 1
    if mode == "ir_linear":
        return 2
    if mode == "ir_quadratic":
        return 3
    if mode == "ir_doy_quadratic":
        return 7
    if mode == "pred_ir_quadratic":
        return 5
    raise ValueError(f"Unknown calibration mode: {mode}")


def fit_calibrator(
    train_rows: pd.DataFrame,
    train_base_pred: np.ndarray,
    *,
    mode: str,
    ridge_alpha: float,
    correction_clip: float,
) -> dict:
    if mode == "none" or len(train_rows) == 0:
        return {
            "mode": mode,
            "coef": np.zeros(calibration_width(mode), dtype=float),
            "ridge_alpha": float(ridge_alpha),
            "correction_clip": float(correction_clip),
        }
    y = pd.to_numeric(train_rows[TARGET], errors="coerce").to_numpy(dtype=float)
    base_pred = np.asarray(train_base_pred, dtype=float)
    residual = y - base_pred
    x = calibration_matrix(train_rows, base_pred, mode)
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0], dtype=float) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    rhs = x.T @ residual
    try:
        coef = np.linalg.solve(xtx + penalty, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(xtx + penalty, rhs, rcond=None)[0]
    return {
        "mode": mode,
        "coef": coef.astype(float),
        "ridge_alpha": float(ridge_alpha),
        "correction_clip": float(correction_clip),
    }


def apply_calibrator(rows: pd.DataFrame, base_pred: np.ndarray, calibrator: dict) -> np.ndarray:
    mode = str(calibrator["mode"])
    base_pred = np.asarray(base_pred, dtype=float)
    if mode == "none":
        return base_pred
    x = calibration_matrix(rows, base_pred, mode)
    coef = np.asarray(calibrator["coef"], dtype=float)
    correction = x @ coef
    clip = float(calibrator.get("correction_clip", 0.0))
    if clip > 0:
        correction = np.clip(correction, -clip, clip)
    return base_pred + correction


def predict_rows(
    rows: pd.DataFrame,
    *,
    model: TinyForest,
    feature_cols: list[str],
    calibrator: dict,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = select_feature_mode(build_features(rows), feature_mode)
    base_pred = predict_forest(model, feature_cols, x)
    calibrated = apply_calibrator(rows, base_pred, calibrator)
    return base_pred, calibrated


def evaluate_decisions(
    *,
    site_id: str,
    calibration_scope: str,
    calibration_dates: int,
    calibration_mode: str,
    train_base_rows: int,
    train_calibration_dates: int,
    train_calibration_rows: int,
    test_df: pd.DataFrame,
    model: TinyForest,
    feature_cols: list[str],
    calibrator: dict,
    paper_candidates: list[float],
    horizon_days: int,
    grid_step: float,
    feature_mode: str,
) -> list[dict]:
    rows = []
    for site_date_id, curve in test_df.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        if curve.empty:
            continue
        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        dense_oracle_gain = float(dense_oracle[TARGET])
        dense_oracle_ir = float(dense_oracle["candidate_ir"])
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)

        fixed_rows = add_truth(
            build_candidate_rows(curve, fixed_values, horizon_days=horizon_days, prefix="fixedlist"),
            curve,
        )
        _, fixed_pred = predict_rows(
            fixed_rows,
            model=model,
            feature_cols=feature_cols,
            calibrator=calibrator,
            feature_mode=feature_mode,
        )
        fixed_rows["pred_net_gain_7d"] = fixed_pred
        fixed_oracle = fixed_rows.loc[fixed_rows["interp_true_net_gain_7d"].idxmax()]
        fixed_pred_best = fixed_rows.loc[fixed_rows["pred_net_gain_7d"].idxmax()]

        dense_grid = dense_values(site_ir_max, grid_step, fixed_values)
        dense_rows = add_truth(
            build_candidate_rows(curve, dense_grid, horizon_days=horizon_days, prefix="denseopt"),
            curve,
        )
        _, dense_pred = predict_rows(
            dense_rows,
            model=model,
            feature_cols=feature_cols,
            calibrator=calibrator,
            feature_mode=feature_mode,
        )
        dense_rows["pred_net_gain_7d"] = dense_pred
        continuous_pred_best = dense_rows.loc[dense_rows["pred_net_gain_7d"].idxmax()]

        fixed_oracle_gain = float(fixed_oracle["interp_true_net_gain_7d"])
        continuous_gain = float(continuous_pred_best["interp_true_net_gain_7d"])
        rows.append(
            {
                "site_id": site_id,
                "calibration_scope": calibration_scope,
                "calibration_dates": int(calibration_dates),
                "calibration_mode": calibration_mode,
                "train_base_rows": int(train_base_rows),
                "train_calibration_dates": int(train_calibration_dates),
                "train_calibration_rows": int(train_calibration_rows),
                "site_date_id": str(site_date_id),
                "date_t": str(dense_oracle["date_t"]),
                "decision_doy": int(dense_oracle["decision_doy"]),
                "dense_oracle_ir": dense_oracle_ir,
                "dense_oracle_gain": dense_oracle_gain,
                "paper_fixed_list_oracle_ir": float(fixed_oracle["candidate_ir"]),
                "paper_fixed_list_oracle_gain": fixed_oracle_gain,
                "paper_regret_vs_dense_oracle": dense_oracle_gain - fixed_oracle_gain,
                "fixed_list_calibrated_ir": float(fixed_pred_best["candidate_ir"]),
                "fixed_list_calibrated_true_gain": float(fixed_pred_best["interp_true_net_gain_7d"]),
                "fixed_list_calibrated_regret_vs_fixed_oracle": fixed_oracle_gain
                - float(fixed_pred_best["interp_true_net_gain_7d"]),
                "continuous_calibrated_ir": float(continuous_pred_best["candidate_ir"]),
                "continuous_calibrated_true_gain": continuous_gain,
                "continuous_calibrated_regret_vs_dense_oracle": dense_oracle_gain - continuous_gain,
                "continuous_calibrated_gain_over_paper": continuous_gain - fixed_oracle_gain,
                "continuous_calibrated_better_than_paper": continuous_gain > fixed_oracle_gain + 1e-9,
                "continuous_calibrated_worse_than_paper": continuous_gain < fixed_oracle_gain - 1e-9,
            }
        )
    return rows


def summarize_prediction(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["calibration_scope", "calibration_dates", "calibration_mode"], sort=False
    ):
        metrics = score_metrics(
            group["true_net_gain_7d"].to_numpy(dtype=float),
            group["pred_net_gain_7d"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "calibration_scope": keys[0],
                "calibration_dates": int(keys[1]),
                "calibration_mode": keys[2],
                "sites": int(group["site_id"].nunique()),
                "test_rows": int(len(group)),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["calibration_scope", "calibration_dates", "calibration_mode"])


def summarize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    return (
        decisions.groupby(["calibration_scope", "calibration_dates", "calibration_mode"])
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            mean_train_base_rows=("train_base_rows", "mean"),
            mean_train_calibration_dates=("train_calibration_dates", "mean"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            fixed_list_calibrated_mean_regret_vs_fixed_oracle=(
                "fixed_list_calibrated_regret_vs_fixed_oracle",
                "mean",
            ),
            continuous_calibrated_mean_regret_vs_dense=(
                "continuous_calibrated_regret_vs_dense_oracle",
                "mean",
            ),
            continuous_calibrated_mean_gain_over_paper=("continuous_calibrated_gain_over_paper", "mean"),
            continuous_calibrated_better_than_paper_rate=("continuous_calibrated_better_than_paper", "mean"),
            continuous_calibrated_worse_than_paper_rate=("continuous_calibrated_worse_than_paper", "mean"),
        )
        .reset_index()
        .sort_values(["calibration_scope", "calibration_dates", "calibration_mode"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", default=str(DEFAULT_BASE_INPUT))
    parser.add_argument("--adapt-input", default=str(DEFAULT_ADAPT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--calibration-dates", default="0,2,4,6")
    parser.add_argument("--calibration-scope", default="rolling", choices=["rolling", "expanding"])
    parser.add_argument("--calibration-modes", default=DEFAULT_MODES)
    parser.add_argument("--include-base-same-date", action="store_true")
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--feature-mode", default="all", choices=["all", "compact"])
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--correction-clip", type=float, default=20.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = prepare_frame(Path(args.base_input), "base")
    adapt = prepare_frame(Path(args.adapt_input), "adapt")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_int_list(args.calibration_dates)
    modes = parse_modes(args.calibration_modes)
    paper_candidates = parse_candidates(args.paper_candidates)

    prediction_rows = []
    decision_rows = []
    sites = sorted(adapt["site_id"].astype(str).unique())
    for site_idx, site_id in enumerate(sites):
        base_site = base.loc[base["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        adapt_site = adapt.loc[adapt["site_id"].astype(str) == site_id].copy().reset_index(drop=True)
        if base_site.empty:
            print(f"[lightweight-tta] skipping {site_id}: no base rows", flush=True)
            continue
        order = date_order_table(adapt_site)
        adapt_site = adapt_site.merge(order[["site_date_id", "date_order"]], on="site_date_id", how="left")
        n_dates = int(adapt_site["date_order"].max()) + 1
        print(
            f"[lightweight-tta] site {site_idx + 1}/{len(sites)} {site_id} "
            f"base_rows={len(base_site)} adapt_dates={n_dates}",
            flush=True,
        )

        for test_order in range(n_dates):
            test_df = adapt_site.loc[adapt_site["date_order"] == test_order].copy()
            if test_df.empty:
                continue
            test_dates = set(test_df["date_t"].astype(str))
            if args.include_base_same_date:
                base_train = base_site
            else:
                base_train = base_site.loc[~base_site["date_t"].astype(str).isin(test_dates)].copy()
            if base_train.empty:
                continue
            x_train = select_feature_mode(build_features(base_train), args.feature_mode)
            y_train = pd.to_numeric(base_train[TARGET], errors="coerce")
            if y_train.isna().any():
                raise ValueError(f"Base target contains NaN for site {site_id}")
            model, cols = fit_forest(
                x_train,
                y_train,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                random_state=args.random_state + site_idx * 10000 + test_order,
            )

            adapt_base_pred = predict_forest(
                model,
                cols,
                select_feature_mode(build_features(adapt_site), args.feature_mode),
            )
            base_pred_series = pd.Series(adapt_base_pred, index=adapt_site.index)
            test_base_pred = base_pred_series.loc[test_df.index].to_numpy(dtype=float)

            for k in k_values:
                if test_order < k:
                    continue
                if args.calibration_scope == "rolling":
                    if k == 0:
                        calib_df = adapt_site.iloc[0:0].copy()
                    else:
                        calib_df = adapt_site.loc[
                            (adapt_site["date_order"] >= test_order - k)
                            & (adapt_site["date_order"] < test_order)
                        ].copy()
                else:
                    calib_df = adapt_site.loc[adapt_site["date_order"] < test_order].copy()
                calib_base_pred = base_pred_series.loc[calib_df.index].to_numpy(dtype=float)

                for mode in modes:
                    calibrator = fit_calibrator(
                        calib_df,
                        calib_base_pred,
                        mode=mode,
                        ridge_alpha=args.ridge_alpha,
                        correction_clip=args.correction_clip,
                    )
                    test_pred = apply_calibrator(test_df, test_base_pred, calibrator)
                    y_test = pd.to_numeric(test_df[TARGET], errors="coerce").to_numpy(dtype=float)
                    if np.isnan(y_test).any():
                        raise ValueError(f"Adapt target contains NaN for site {site_id}")
                    for idx, pred, true_value in zip(test_df.index, test_pred, y_test):
                        prediction_rows.append(
                            {
                                "site_id": site_id,
                                "calibration_scope": args.calibration_scope,
                                "calibration_dates": int(k),
                                "calibration_mode": mode,
                                "site_date_id": str(test_df.loc[idx, "site_date_id"]),
                                "date_t": str(test_df.loc[idx, "date_t"]),
                                "candidate_ir": float(test_df.loc[idx, "candidate_ir"]),
                                "true_net_gain_7d": float(true_value),
                                "pred_net_gain_7d": float(pred),
                                "train_base_rows": int(len(base_train)),
                                "train_calibration_dates": int(calib_df["site_date_id"].nunique()),
                                "train_calibration_rows": int(len(calib_df)),
                            }
                        )
                    decision_rows.extend(
                        evaluate_decisions(
                            site_id=site_id,
                            calibration_scope=args.calibration_scope,
                            calibration_dates=k,
                            calibration_mode=mode,
                            train_base_rows=int(len(base_train)),
                            train_calibration_dates=int(calib_df["site_date_id"].nunique()),
                            train_calibration_rows=int(len(calib_df)),
                            test_df=test_df,
                            model=model,
                            feature_cols=cols,
                            calibrator=calibrator,
                            paper_candidates=paper_candidates,
                            horizon_days=args.horizon_days,
                            grid_step=args.grid_step,
                            feature_mode=args.feature_mode,
                        )
                    )

    predictions = pd.DataFrame(prediction_rows)
    decisions = pd.DataFrame(decision_rows)
    prediction_summary = summarize_prediction(predictions) if not predictions.empty else pd.DataFrame()
    summary = summarize_decisions(decisions)
    by_site = (
        decisions.groupby(["calibration_scope", "calibration_dates", "calibration_mode", "site_id"])
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_fixed_list_mean_regret_vs_dense=("paper_regret_vs_dense_oracle", "mean"),
            continuous_calibrated_mean_regret_vs_dense=("continuous_calibrated_regret_vs_dense_oracle", "mean"),
            continuous_calibrated_mean_gain_over_paper=("continuous_calibrated_gain_over_paper", "mean"),
            continuous_calibrated_better_than_paper_rate=("continuous_calibrated_better_than_paper", "mean"),
        )
        .reset_index()
        .sort_values(
            [
                "calibration_scope",
                "calibration_dates",
                "calibration_mode",
                "continuous_calibrated_mean_regret_vs_dense",
            ],
            ascending=[True, True, True, False],
        )
    ) if not decisions.empty else pd.DataFrame()

    predictions_path = out_dir / "tta_lightweight_output_calibration_predictions_v1.csv"
    decisions_path = out_dir / "tta_lightweight_output_calibration_decisions_v1.csv"
    metrics_path = out_dir / "tta_lightweight_output_calibration_prediction_metrics_v1.csv"
    summary_path = out_dir / "tta_lightweight_output_calibration_summary_v1.csv"
    by_site_path = out_dir / "tta_lightweight_output_calibration_by_site_v1.csv"
    report_path = out_dir / "tta_lightweight_output_calibration_v1.md"
    predictions.to_csv(predictions_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    prediction_summary.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)

    lines = [
        "# TTA Lightweight Output Calibration V1",
        "",
        f"- Base input: `{args.base_input}`",
        f"- Adapt input: `{args.adapt_input}`",
        f"- Calibration scope: `{args.calibration_scope}`",
        f"- Calibration modes: `{','.join(modes)}`",
        f"- Include base same date: `{args.include_base_same_date}`",
        f"- Ridge alpha: `{args.ridge_alpha}`",
        f"- Correction clip: `{args.correction_clip}`",
        "",
        "This diagnostic keeps the per-site expert fixed for a test date and adapts only a small output correction layer.",
        "",
        "## Prediction Metrics",
        "",
        markdown_table(prediction_summary),
        "",
        "## Decision Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Outputs",
        "",
        f"- `{predictions_path}`",
        f"- `{decisions_path}`",
        f"- `{metrics_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("TTA lightweight output calibration v1")
    print(f"prediction_metrics: {metrics_path}")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"report: {report_path}")
    print("")
    print(prediction_summary.to_string(index=False))
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

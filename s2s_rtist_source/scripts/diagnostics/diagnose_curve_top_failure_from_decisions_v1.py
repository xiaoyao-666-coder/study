#!/usr/bin/env python3
"""Diagnose curve-top ranking failures from decision CSV files.

This is a fast post-processing diagnostic for surrogate irrigation decisions.
It reconstructs each date's dense true SWAP response curve, ranks the selected
irrigation amount on that curve, and estimates oracle guard headroom against
the paper fixed-list baseline.
"""

from __future__ import annotations

import argparse
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
from train_confirmed_5site_true_input_surrogate_baseline_v1 import bool_series, markdown_table
from train_persite_tinyforest_profit_surrogate_v1 import dense_values


DEFAULT_ADAPT_INPUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_date_coverage_fast_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_DECISIONS = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_lightweight_output_calibration_smoke_v1"
    / "tta_lightweight_output_calibration_decisions_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_tta_lightweight_curve_top_failure_diag_v1"
)


def prepare_adapt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing adapt table: {path}")
    df = pd.read_csv(path)
    for col in ["is_best_ir", "target_collapse", "same_date_duplicate_target_curve"]:
        if col in df.columns:
            df[col] = bool_series(df[col])
    required = {"site_id", "site_date_id", "date_t", "decision_doy", "candidate_ir", "site_ir_max", TARGET}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Adapt table missing columns: {missing}")
    df["candidate_ir"] = pd.to_numeric(df["candidate_ir"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df.dropna(subset=["candidate_ir", TARGET]).copy()


def prepare_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing decisions table: {path}")
    df = pd.read_csv(path)
    required = {
        "site_id",
        "site_date_id",
        "dense_oracle_ir",
        "dense_oracle_gain",
        "paper_fixed_list_oracle_gain",
        "paper_regret_vs_dense_oracle",
        "continuous_calibrated_ir",
        "continuous_calibrated_true_gain",
        "continuous_calibrated_regret_vs_dense_oracle",
        "continuous_calibrated_gain_over_paper",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Decision table missing columns: {missing}")
    for col in [
        "dense_oracle_ir",
        "dense_oracle_gain",
        "paper_fixed_list_oracle_gain",
        "paper_regret_vs_dense_oracle",
        "continuous_calibrated_ir",
        "continuous_calibrated_true_gain",
        "continuous_calibrated_regret_vs_dense_oracle",
        "continuous_calibrated_gain_over_paper",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["continuous_calibrated_ir"]).copy()


def dense_curve_for_date(
    curve: pd.DataFrame,
    *,
    paper_candidates: list[float],
    grid_step: float,
) -> pd.DataFrame:
    curve = curve.sort_values("candidate_ir").copy()
    site_ir_max = float(curve["site_ir_max"].iloc[0])
    fixed_values = candidate_set_for_site(site_ir_max, paper_candidates)
    grid = dense_values(site_ir_max, grid_step, fixed_values)
    gains = np.array([interp_gain(curve, float(ir)) for ir in grid], dtype=float)
    out = pd.DataFrame({"candidate_ir": grid, "true_gain": gains})
    out["rank"] = out["true_gain"].rank(method="min", ascending=False).astype(int)
    return out


def nearest_dense_row(dense_curve: pd.DataFrame, selected_ir: float) -> pd.Series:
    idx = (dense_curve["candidate_ir"].to_numpy(dtype=float) - float(selected_ir))
    idx = int(np.argmin(np.abs(idx)))
    return dense_curve.iloc[idx]


def add_rank_diagnostics(
    decisions: pd.DataFrame,
    adapt: pd.DataFrame,
    *,
    paper_candidates: list[float],
    grid_step: float,
) -> pd.DataFrame:
    curves = {str(k): v.copy() for k, v in adapt.groupby("site_date_id", sort=False)}
    dense_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, decision in decisions.iterrows():
        site_date_id = str(decision["site_date_id"])
        if site_date_id not in curves:
            continue
        if site_date_id not in dense_cache:
            dense_cache[site_date_id] = dense_curve_for_date(
                curves[site_date_id],
                paper_candidates=paper_candidates,
                grid_step=grid_step,
            )
        dense_curve = dense_cache[site_date_id]
        selected_ir = float(decision["continuous_calibrated_ir"])
        selected = nearest_dense_row(dense_curve, selected_ir)
        oracle = dense_curve.loc[dense_curve["true_gain"].idxmax()]
        site_ir_max = float(curves[site_date_id]["site_ir_max"].iloc[0])
        row = decision.to_dict()
        row.update(
            {
                "dense_grid_points": int(len(dense_curve)),
                "selected_dense_ir_nearest": float(selected["candidate_ir"]),
                "selected_dense_true_gain_nearest": float(selected["true_gain"]),
                "selected_dense_rank": int(selected["rank"]),
                "selected_top1": int(selected["rank"]) <= 1,
                "selected_top3": int(selected["rank"]) <= 3,
                "selected_top5": int(selected["rank"]) <= 5,
                "selected_ir_abs_error_vs_dense_oracle": abs(selected_ir - float(oracle["candidate_ir"])),
                "selected_is_zero": abs(selected_ir) <= 1e-9,
                "selected_is_upper_bound": selected_ir >= site_ir_max - max(grid_step * 0.5, 1e-9),
                "continuous_beats_paper": float(decision["continuous_calibrated_gain_over_paper"]) > 1e-9,
                "oracle_guard_true_gain": max(
                    float(decision["continuous_calibrated_true_gain"]),
                    float(decision["paper_fixed_list_oracle_gain"]),
                ),
            }
        )
        row["oracle_guard_regret_vs_dense"] = float(decision["dense_oracle_gain"]) - float(
            row["oracle_guard_true_gain"]
        )
        row["oracle_guard_gain_over_paper"] = float(row["oracle_guard_true_gain"]) - float(
            decision["paper_fixed_list_oracle_gain"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    group_cols = [col for col in ["calibration_scope", "calibration_dates", "calibration_mode"] if col in diag.columns]
    return (
        diag.groupby(group_cols)
        .agg(
            sites=("site_id", "nunique"),
            site_dates=("site_date_id", "nunique"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            continuous_mean_regret=("continuous_calibrated_regret_vs_dense_oracle", "mean"),
            continuous_median_regret=("continuous_calibrated_regret_vs_dense_oracle", "median"),
            continuous_p90_regret=("continuous_calibrated_regret_vs_dense_oracle", lambda x: float(np.quantile(x, 0.9))),
            large_regret_gt_2_rate=("continuous_calibrated_regret_vs_dense_oracle", lambda x: float(np.mean(x > 2.0))),
            large_regret_gt_5_rate=("continuous_calibrated_regret_vs_dense_oracle", lambda x: float(np.mean(x > 5.0))),
            large_regret_gt_10_rate=("continuous_calibrated_regret_vs_dense_oracle", lambda x: float(np.mean(x > 10.0))),
            selected_top1_rate=("selected_top1", "mean"),
            selected_top3_rate=("selected_top3", "mean"),
            selected_top5_rate=("selected_top5", "mean"),
            mean_selected_rank=("selected_dense_rank", "mean"),
            mean_ir_abs_error=("selected_ir_abs_error_vs_dense_oracle", "mean"),
            selected_zero_rate=("selected_is_zero", "mean"),
            selected_upper_bound_rate=("selected_is_upper_bound", "mean"),
            continuous_better_than_paper_rate=("continuous_beats_paper", "mean"),
            oracle_guard_mean_regret=("oracle_guard_regret_vs_dense", "mean"),
            oracle_guard_gain_over_paper=("oracle_guard_gain_over_paper", "mean"),
            oracle_guard_use_continuous_rate=("continuous_beats_paper", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def by_site_table(diag: pd.DataFrame) -> pd.DataFrame:
    if diag.empty:
        return pd.DataFrame()
    group_cols = [
        col for col in ["calibration_scope", "calibration_dates", "calibration_mode", "site_id"] if col in diag.columns
    ]
    return (
        diag.groupby(group_cols)
        .agg(
            site_dates=("site_date_id", "nunique"),
            paper_mean_regret=("paper_regret_vs_dense_oracle", "mean"),
            continuous_mean_regret=("continuous_calibrated_regret_vs_dense_oracle", "mean"),
            large_regret_gt_5_rate=("continuous_calibrated_regret_vs_dense_oracle", lambda x: float(np.mean(x > 5.0))),
            selected_top3_rate=("selected_top3", "mean"),
            mean_selected_rank=("selected_dense_rank", "mean"),
            mean_ir_abs_error=("selected_ir_abs_error_vs_dense_oracle", "mean"),
            selected_zero_rate=("selected_is_zero", "mean"),
            oracle_guard_mean_regret=("oracle_guard_regret_vs_dense", "mean"),
        )
        .reset_index()
        .sort_values(group_cols[:-1] + ["continuous_mean_regret"], ascending=[True] * (len(group_cols) - 1) + [False])
    )


def worst_cases_table(diag: pd.DataFrame, limit: int) -> pd.DataFrame:
    keep = [
        col
        for col in [
            "calibration_scope",
            "calibration_dates",
            "calibration_mode",
            "site_id",
            "site_date_id",
            "date_t",
            "decision_doy",
            "dense_oracle_ir",
            "continuous_calibrated_ir",
            "selected_dense_rank",
            "selected_ir_abs_error_vs_dense_oracle",
            "paper_regret_vs_dense_oracle",
            "continuous_calibrated_regret_vs_dense_oracle",
            "continuous_calibrated_gain_over_paper",
            "selected_is_zero",
            "selected_is_upper_bound",
        ]
        if col in diag.columns
    ]
    return diag.sort_values("continuous_calibrated_regret_vs_dense_oracle", ascending=False)[keep].head(limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapt-input", default=str(DEFAULT_ADAPT_INPUT))
    parser.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--grid-step", type=float, default=0.5)
    parser.add_argument("--worst-limit", type=int, default=80)
    args = parser.parse_args()

    adapt = prepare_adapt(Path(args.adapt_input))
    decisions = prepare_decisions(Path(args.decisions))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_candidates = parse_candidates(args.paper_candidates)

    diag = add_rank_diagnostics(
        decisions,
        adapt,
        paper_candidates=paper_candidates,
        grid_step=args.grid_step,
    )
    summary = summary_table(diag)
    by_site = by_site_table(diag)
    worst = worst_cases_table(diag, args.worst_limit)

    diag_path = out_dir / "curve_top_failure_diagnostics_v1.csv"
    summary_path = out_dir / "curve_top_failure_summary_v1.csv"
    by_site_path = out_dir / "curve_top_failure_by_site_v1.csv"
    worst_path = out_dir / "curve_top_failure_worst_cases_v1.csv"
    report_path = out_dir / "curve_top_failure_diagnostics_v1.md"
    diag.to_csv(diag_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_site.to_csv(by_site_path, index=False)
    worst.to_csv(worst_path, index=False)

    lines = [
        "# Curve-Top Failure Diagnostics V1",
        "",
        f"- Adapt input: `{args.adapt_input}`",
        f"- Decisions: `{args.decisions}`",
        f"- Grid step: `{args.grid_step}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## Worst Cases",
        "",
        markdown_table(worst),
        "",
        "## Outputs",
        "",
        f"- `{diag_path}`",
        f"- `{summary_path}`",
        f"- `{by_site_path}`",
        f"- `{worst_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Curve-top failure diagnostics v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"worst_cases: {worst_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

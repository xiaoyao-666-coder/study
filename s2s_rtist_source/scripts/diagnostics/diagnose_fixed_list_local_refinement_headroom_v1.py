#!/usr/bin/env python3
"""Diagnose oracle headroom for fixed-list anchored local refinement.

The paper fixed list is a very strong deployable baseline. Instead of asking a
surrogate to globally choose irrigation over the full feasible range, this
diagnostic asks a safer question: if the paper fixed-list decision is used as an
anchor, how much regret could be removed by searching only a local neighborhood
around that anchor?

This is an oracle headroom diagnostic using the sampled SWAP true curve. It does
not use surrogate predictions and does not claim deployability. If the oracle
headroom is small, a learned local refiner is unlikely to be worth pursuing.
"""

from __future__ import annotations

import argparse
import csv
import errno
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
from train_confirmed_5site_true_input_surrogate_baseline_v1 import markdown_table


DEFAULT_SAMPLES = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_surrogate_sequence_wide_features_v1"
    / "continuous_ir_12site_surrogate_sequence_wide_samples_v1.csv"
)
DEFAULT_OUT = (
    Path("site_general_surrogate_eval")
    / "continuous_ir_12site_10k_fixed_list_local_refinement_headroom_v1"
)
REQUIRED_SAMPLE_COLUMNS = {
    "site_date_id",
    "site_id",
    "date_t",
    "candidate_ir",
    "site_ir_max",
    TARGET,
}


def float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one value is required")
    return sorted(set(values))


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def sniff_csv_header(path: Path) -> set[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return set(next(reader, []))
    except (OSError, UnicodeDecodeError, StopIteration):
        return set()


def find_sample_candidates(root: Path) -> list[Path]:
    patterns = [
        "*sequence_wide_samples*.csv",
        "*surrogate_samples*.csv",
        "*continuous_ir* samples*.csv",
        "*continuous_ir*.csv",
    ]
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    unique = sorted(set(found), key=lambda path: (len(path.parts), str(path)))
    return [path for path in unique if REQUIRED_SAMPLE_COLUMNS.issubset(sniff_csv_header(path))]


def resolve_samples_path(samples_arg: str) -> Path:
    samples_path = Path(samples_arg)
    if samples_path.exists():
        return samples_path

    if samples_path != DEFAULT_SAMPLES:
        raise FileNotFoundError(f"Missing samples file: {samples_path}")

    candidates = find_sample_candidates(Path("site_general_surrogate_eval"))
    if len(candidates) == 1:
        print(f"[info] Using discovered samples file: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates[:20])
        raise FileNotFoundError(
            "Default samples file is missing and multiple possible sample files were found. "
            f"Pass --samples explicitly.\n{formatted}"
        )

    raise FileNotFoundError(
        "Missing 12-site dense samples file. Expected:\n"
        f"  {DEFAULT_SAMPLES}\n"
        "No local CSV with the required sample columns was found under site_general_surrogate_eval. "
        "Sync or generate the SWAP dense sample table, then rerun this diagnostic."
    )


def interpolated_grid_best(curve: pd.DataFrame, lower: float, upper: float, step: float) -> tuple[float, float]:
    lower = max(float(lower), float(curve["candidate_ir"].min()))
    upper = min(float(upper), float(curve["candidate_ir"].max()))
    if upper < lower:
        raise ValueError(f"Invalid local window: lower={lower}, upper={upper}")
    grid = np.arange(lower, upper + step * 0.5, step)
    if len(grid) == 0:
        grid = np.array([lower], dtype=float)
    scores = [(float(ir), interp_gain(curve, float(ir))) for ir in grid]
    return max(scores, key=lambda item: item[1])


def write_csv(path: Path, df: pd.DataFrame) -> bool:
    try:
        df.to_csv(path, index=False)
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def write_text(path: Path, text: str) -> bool:
    try:
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(f"[warn] No space left on device; skipped writing {path}")
            return False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--paper-candidates", default=DEFAULT_PAPER_CANDIDATES)
    parser.add_argument("--windows-mm", default="1,2.5,5,10")
    parser.add_argument("--grid-step-mm", type=float, default=0.5)
    args = parser.parse_args()

    samples_path = resolve_samples_path(args.samples)
    out_dir = Path(args.output_dir)
    can_write = True
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print("[warn] No space left on device; will print summary only and skip writing")
            can_write = False
        else:
            raise

    samples = pd.read_csv(samples_path)
    missing_columns = sorted(REQUIRED_SAMPLE_COLUMNS.difference(samples.columns))
    if missing_columns:
        raise ValueError(f"Missing required sample columns: {missing_columns}")

    paper_candidates = parse_candidates(args.paper_candidates)
    windows = float_list(args.windows_mm)
    rows = []
    for site_date_id, curve in samples.groupby("site_date_id", sort=False):
        curve = curve.copy()
        curve["candidate_ir"] = pd.to_numeric(curve["candidate_ir"], errors="coerce")
        curve[TARGET] = pd.to_numeric(curve[TARGET], errors="coerce")
        curve = curve.dropna(subset=["candidate_ir", TARGET]).sort_values("candidate_ir")
        dense_oracle = curve.loc[curve[TARGET].idxmax()]
        dense_oracle_ir = float(dense_oracle["candidate_ir"])
        dense_oracle_gain = float(dense_oracle[TARGET])
        site_ir_max = float(curve["site_ir_max"].iloc[0])
        paper_values = candidate_set_for_site(site_ir_max, paper_candidates)
        paper_scores = [(ir, interp_gain(curve, ir)) for ir in paper_values]
        paper_ir, paper_gain = max(paper_scores, key=lambda item: item[1])
        paper_regret = dense_oracle_gain - paper_gain

        for window in windows:
            lower = max(0.0, paper_ir - window)
            upper = min(site_ir_max, paper_ir + window)
            local_ir, local_gain = interpolated_grid_best(curve, lower, upper, args.grid_step_mm)
            local_regret = dense_oracle_gain - local_gain
            paper_matches_dense = abs(float(paper_regret)) <= 1e-9
            dense_oracle_within_window = abs(dense_oracle_ir - float(paper_ir)) <= float(window) + 1e-9
            if paper_matches_dense:
                anchor_error_type = "fixed_list_already_optimal"
            elif dense_oracle_within_window:
                anchor_error_type = "best_amount_inside_local_window"
            else:
                anchor_error_type = "best_amount_outside_local_window"
            rows.append(
                {
                    "site_date_id": site_date_id,
                    "site_id": str(dense_oracle["site_id"]),
                    "date_t": str(dense_oracle["date_t"]),
                    "target_collapse": parse_bool(dense_oracle["target_collapse"]) if "target_collapse" in dense_oracle else False,
                    "site_ir_max": site_ir_max,
                    "window_mm": float(window),
                    "paper_best_ir": float(paper_ir),
                    "paper_best_gain": float(paper_gain),
                    "paper_regret": float(paper_regret),
                    "local_oracle_ir": float(local_ir),
                    "local_oracle_gain": float(local_gain),
                    "local_oracle_regret": float(local_regret),
                    "dense_oracle_ir": dense_oracle_ir,
                    "dense_oracle_gain": dense_oracle_gain,
                    "local_improvement_over_paper": float(paper_regret - local_regret),
                    "paper_matches_dense_oracle": paper_matches_dense,
                    "dense_oracle_within_window": dense_oracle_within_window,
                    "anchor_error_type": anchor_error_type,
                    "abs_dense_minus_paper_ir": abs(dense_oracle_ir - float(paper_ir)),
                    "local_changed_ir": abs(float(local_ir) - float(paper_ir)) > 1e-9,
                    "local_reaches_dense_oracle": abs(float(local_regret)) <= 1e-9,
                }
            )

    decisions = pd.DataFrame(rows)
    summary = (
        decisions.groupby("window_mm")
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            mean_improvement_over_paper=("local_improvement_over_paper", "mean"),
            median_improvement_over_paper=("local_improvement_over_paper", "median"),
            improvement_rate=("local_improvement_over_paper", lambda s: float((s > 1e-9).mean())),
            changed_ir_rate=("local_changed_ir", "mean"),
            reaches_dense_oracle_rate=("local_reaches_dense_oracle", "mean"),
        )
        .reset_index()
        .sort_values("window_mm")
    )
    by_site = (
        decisions.groupby(["window_mm", "site_id"])
        .agg(
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            mean_improvement_over_paper=("local_improvement_over_paper", "mean"),
            improvement_rate=("local_improvement_over_paper", lambda s: float((s > 1e-9).mean())),
            n_site_dates=("site_date_id", "count"),
        )
        .reset_index()
        .sort_values(["window_mm", "mean_improvement_over_paper"], ascending=[True, False])
    )
    by_error_type = (
        decisions.groupby(["window_mm", "anchor_error_type"])
        .agg(
            site_dates=("site_date_id", "count"),
            paper_mean_regret=("paper_regret", "mean"),
            local_oracle_mean_regret=("local_oracle_regret", "mean"),
            mean_improvement_over_paper=("local_improvement_over_paper", "mean"),
            total_improvement_over_paper=("local_improvement_over_paper", "sum"),
            changed_ir_rate=("local_changed_ir", "mean"),
            reaches_dense_oracle_rate=("local_reaches_dense_oracle", "mean"),
            mean_abs_dense_minus_paper_ir=("abs_dense_minus_paper_ir", "mean"),
        )
        .reset_index()
        .sort_values(["window_mm", "anchor_error_type"])
    )
    best_window = summary.sort_values("local_oracle_mean_regret").iloc[0]
    best_decisions = decisions.loc[decisions["window_mm"] == float(best_window["window_mm"])].copy()
    worst_paper = best_decisions.sort_values("paper_regret", ascending=False).head(40)
    largest_improvements = best_decisions.sort_values("local_improvement_over_paper", ascending=False).head(40)

    summary_path = out_dir / "fixed_list_local_refinement_headroom_summary_v1.csv"
    decisions_path = out_dir / "fixed_list_local_refinement_headroom_decisions_v1.csv"
    by_site_path = out_dir / "fixed_list_local_refinement_headroom_by_site_v1.csv"
    by_error_type_path = out_dir / "fixed_list_local_refinement_headroom_by_error_type_v1.csv"
    report_path = out_dir / "fixed_list_local_refinement_headroom_v1.md"

    lines = [
        "# Fixed-List Anchored Local Refinement Headroom V1",
        "",
        "## Inputs",
        "",
        f"- Samples: `{samples_path}`",
        f"- Paper candidates: `{args.paper_candidates}`",
        f"- Local windows mm: `{args.windows_mm}`",
        f"- Grid step mm: `{args.grid_step_mm}`",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## By Site",
        "",
        markdown_table(by_site),
        "",
        "## By Anchor Error Type",
        "",
        markdown_table(by_error_type),
        "",
        "## Worst Paper Regret At Best Window",
        "",
        markdown_table(worst_paper),
        "",
        "## Largest Local Improvements At Best Window",
        "",
        markdown_table(largest_improvements),
        "",
        "## Outputs",
        "",
        f"- `{summary_path}`",
        f"- `{decisions_path}`",
        f"- `{by_site_path}`",
        f"- `{by_error_type_path}`",
    ]
    report_text = "\n".join(lines) + "\n"

    if can_write:
        write_csv(summary_path, summary)
        write_csv(decisions_path, decisions)
        write_csv(by_site_path, by_site)
        write_csv(by_error_type_path, by_error_type)
        write_text(report_path, report_text)

    print("Fixed-list local refinement headroom v1")
    print(f"summary: {summary_path}")
    print(f"by_site: {by_site_path}")
    print(f"by_error_type: {by_error_type_path}")
    print(f"report: {report_path}")
    print("")
    print(summary.to_string(index=False))
    print("")
    print(by_site.to_string(index=False))
    print("")
    print(by_error_type.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize old-year paper-fixed-list ERA5 single-scenario SWAP relabeling runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("site_general_surrogate_eval") / "continuous_ir_12site_restart_generation_older_year_v1"
DEFAULT_OUT = Path("site_general_surrogate_eval") / "older_year_swap_label_generation_v1"
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019]
DEFAULT_SITES = ["code_C2", "code_N1", "code_N2", "code_N4"]
PAPER_IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    rows = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--run-name-template", default="continuous_ir_{year}_failure_sites_high_response_paper_fixed_list_v1")
    parser.add_argument("--output-prefix", default="older_year_2015_2019_paper_fixed_list_era5_single_scenario")
    return parser.parse_args()


def read_existing_metadata(run: Path) -> dict[str, object]:
    path = run / "continuous_ir_12site_restart_generation_metadata_v1.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def infer_irrigation_values(best_or_merged: pd.DataFrame) -> list[float]:
    if "ir" in best_or_merged.columns:
        return sorted(pd.to_numeric(best_or_merged["ir"], errors="coerce").dropna().unique().tolist())
    return []


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_frames = []
    merged_frames = []
    summary_frames = []
    error_frames = []
    run_metadata = []

    for year in args.years:
        run = run_root / args.run_name_template.format(year=year)
        if not run.exists():
            raise FileNotFoundError(f"Missing run directory for {year}: {run}")

        merged_path = run / "continuous_ir_12site_restart_generation_merged_v1.csv"
        best_path = run / "continuous_ir_12site_restart_generation_best_by_date_v1.csv"
        summary_path = run / "continuous_ir_12site_restart_generation_summary_v1.csv"
        if not merged_path.exists():
            raise FileNotFoundError(f"Missing merged candidate response table: {merged_path}")
        if not best_path.exists():
            raise FileNotFoundError(f"Missing best-by-date table: {best_path}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary table: {summary_path}")

        merged = pd.read_csv(merged_path)
        best = pd.read_csv(best_path)
        summary = pd.read_csv(summary_path)
        merged["year"] = year
        best["year"] = year
        summary["year"] = year

        merged_frames.append(merged)
        best_frames.append(best)
        summary_frames.append(summary[["year", "site", "status", "rows", "error_rows", "plan_rows", "plan_dates"]])

        for site in args.sites:
            err = run / site / "site_restart_generation_errors.csv"
            if err.exists():
                df = pd.read_csv(err)
                df["year"] = year
                df["site"] = site
                error_frames.append(df)

        metadata = read_existing_metadata(run)
        irrigation_values = infer_irrigation_values(merged)
        run_metadata.append(
            {
                "year": year,
                "run_dir": str(run),
                "weather_scenario_count": metadata.get("weather_scenario_count", 1),
                "weather_scenario_label": metadata.get("weather_scenario_label", "era5_single_scenario"),
                "uses_paper_fixed_irrigation_list": irrigation_values == PAPER_IRRIGATION_OPTIONS_MM,
                "irrigation_option_count": len(irrigation_values),
                "irrigation_option_values_mm": irrigation_values,
                "paper_irrigation_option_values_mm": PAPER_IRRIGATION_OPTIONS_MM,
            }
        )

    best_all = pd.concat(best_frames, ignore_index=True)
    merged_all = pd.concat(merged_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    agg = best_all.groupby(["year", "site"]).agg(
        successful_dates=("date_t", "count"),
        nonzero_dates=("best_ir_for_date", lambda s: int((s > 0).sum())),
        nonzero_rate=("best_ir_for_date", lambda s: round(float((s > 0).mean()), 3)),
        max_best_ir=("best_ir_for_date", "max"),
        mean_best_target=("best_target_for_date", "mean"),
    ).reset_index()
    agg["mean_best_target"] = agg["mean_best_target"].round(3)
    agg = agg.merge(summary_all, on=["year", "site"], how="left")
    agg["planned_dates"] = agg["plan_dates"]
    agg["failed_dates"] = agg["error_rows"].astype(int)
    agg["missing_candidate_rows"] = agg["plan_rows"] - agg["rows"]
    agg["coverage_rate"] = (agg["successful_dates"] / agg["planned_dates"]).round(3)

    site_total = agg.groupby("site").agg(
        successful_dates=("successful_dates", "sum"),
        planned_dates=("planned_dates", "sum"),
        failed_dates=("failed_dates", "sum"),
        nonzero_dates=("nonzero_dates", "sum"),
        mean_nonzero_rate=("nonzero_rate", "mean"),
        max_best_ir=("max_best_ir", "max"),
        mean_best_target=("mean_best_target", "mean"),
    ).reset_index()
    site_total["coverage_rate"] = (site_total["successful_dates"] / site_total["planned_dates"]).round(3)
    site_total["mean_nonzero_rate"] = site_total["mean_nonzero_rate"].round(3)
    site_total["mean_best_target"] = site_total["mean_best_target"].round(3)

    prefix = args.output_prefix
    agg.to_csv(out_dir / f"{prefix}_summary_by_year_site_v1.csv", index=False)
    site_total.to_csv(out_dir / f"{prefix}_summary_by_site_v1.csv", index=False)
    merged_all.to_csv(out_dir / f"{prefix}_candidate_response_surface_v1.csv", index=False)
    best_all.to_csv(out_dir / f"{prefix}_best_by_date_v1.csv", index=False)
    summary_all.to_csv(out_dir / f"{prefix}_run_rows_v1.csv", index=False)
    pd.DataFrame(run_metadata).to_csv(out_dir / f"{prefix}_run_metadata_v1.csv", index=False)
    if not errors_all.empty:
        errors_all.to_csv(out_dir / f"{prefix}_failed_dates_v1.csv", index=False)

    report_lines = [
        "# Paper-Fixed-List ERA5 Single-Scenario SWAP Relabeling Summary",
        "",
        "This artifact is paper-aligned on the irrigation candidate list, but it is not a full paper label reproduction.",
        "",
        "- Weather scenario count: 1",
        "- Weather scenario label: ERA5 single scenario",
        f"- Paper irrigation options: {PAPER_IRRIGATION_OPTIONS_MM}",
        "- Missing paper element: 9-member S2S forecast ensemble response surface",
        "",
        "## Year-Site Summary",
        markdown_table(
            agg[
                [
                    "year",
                    "site",
                    "successful_dates",
                    "planned_dates",
                    "failed_dates",
                    "coverage_rate",
                    "nonzero_dates",
                    "nonzero_rate",
                    "max_best_ir",
                    "mean_best_target",
                ]
            ]
        ),
        "",
        "## Site Totals",
        markdown_table(site_total),
    ]
    if not errors_all.empty:
        report_lines.extend(
            [
                "",
                "## Failed Dates",
                markdown_table(
                    errors_all[["year", "site", "date_t", "decision_doy", "error_type", "error"]].sort_values(
                        ["year", "site", "decision_doy"]
                    )
                ),
            ]
        )
    report = out_dir / f"{prefix}_summary_v1.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("wrote:")
    for name in [
        f"{prefix}_summary_by_year_site_v1.csv",
        f"{prefix}_summary_by_site_v1.csv",
        f"{prefix}_candidate_response_surface_v1.csv",
        f"{prefix}_best_by_date_v1.csv",
        f"{prefix}_run_metadata_v1.csv",
        f"{prefix}_summary_v1.md",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()

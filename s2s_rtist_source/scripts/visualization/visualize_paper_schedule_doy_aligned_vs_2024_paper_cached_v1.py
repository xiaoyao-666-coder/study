#!/usr/bin/env python3
"""Visualize paper-schedule-DOY aligned old-year labels vs 2024 paper cache.

Both sides are reduced to the same decision-label level:

- old years: best irrigation / best target from each successful site-date
- 2024 paper cache: date-level mean schedule and per-ensemble winning rows

This is not a response-surface comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path("site_general_surrogate_eval") / "continuous_ir_12site_restart_generation_older_year_v1"
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "paper_schedule_doy_aligned_vs_2024_paper_cached_visualization_v1"
DEFAULT_PAPER_DIR = Path("model3_opt_sto_upload") / "Maize"
DEFAULT_YEARS = [2015, 2016, 2017, 2018, 2019]
DEFAULT_SITES = ["code_C2", "code_N1", "code_N2", "code_N4"]
RUN_TEMPLATE = "continuous_ir_{year}_failure_sites_paper_schedule_doy_aligned_v3"
FIGURE_PREFIX = "paper_schedule_doy_aligned_era5_single_vs_2024_paper_cached"
PAPER_IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--paper-dir", default=str(DEFAULT_PAPER_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--sites", nargs="+", default=DEFAULT_SITES)
    parser.add_argument("--run-template", default=RUN_TEMPLATE)
    return parser.parse_args()


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


def date_to_doy(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%d-%b-%Y", errors="coerce").dt.dayofyear


def read_old_year_runs(run_root: Path, years: list[int], run_template: str, sites: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_frames = []
    merged_frames = []
    summary_frames = []
    error_frames = []

    for year in years:
        run = run_root / run_template.format(year=year)
        best_path = run / "continuous_ir_12site_restart_generation_best_by_date_v1.csv"
        merged_path = run / "continuous_ir_12site_restart_generation_merged_v1.csv"
        summary_path = run / "continuous_ir_12site_restart_generation_summary_v1.csv"
        missing = [p for p in [best_path, merged_path, summary_path] if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing old-year run output(s): " + ", ".join(str(p) for p in missing))

        best = pd.read_csv(best_path)
        merged = pd.read_csv(merged_path)
        summary = pd.read_csv(summary_path)
        best["year"] = year
        merged["year"] = year
        summary["year"] = year
        best_frames.append(best)
        merged_frames.append(merged)
        summary_frames.append(summary)

        for site in sites:
            err_path = run / site / "site_restart_generation_errors.csv"
            if err_path.exists():
                err = pd.read_csv(err_path)
                err["year"] = year
                err["site"] = site
                error_frames.append(err)

    best_all = pd.concat(best_frames, ignore_index=True)
    merged_all = pd.concat(merged_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)
    errors_all = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    best_all["decision_doy"] = pd.to_numeric(best_all["decision_doy"], errors="coerce")
    best_all["best_ir_for_date"] = pd.to_numeric(best_all["best_ir_for_date"], errors="coerce")
    best_all["best_target_for_date"] = pd.to_numeric(best_all["best_target_for_date"], errors="coerce")
    merged_all["ir"] = pd.to_numeric(merged_all["ir"], errors="coerce")
    return best_all, merged_all, summary_all, errors_all


def read_paper_cache(paper_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_day_path = paper_dir / "all_day_ir_var_results.csv"
    day_path = paper_dir / "day_scheduled.csv"
    missing = [p for p in [all_day_path, day_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing paper cache file(s): " + ", ".join(str(p) for p in missing))
    all_day = pd.read_csv(all_day_path)
    day = pd.read_csv(day_path)
    all_day["decision_doy"] = date_to_doy(all_day["date_t"])
    day["decision_doy"] = date_to_doy(day["date_t"])
    all_day["ir"] = pd.to_numeric(all_day["ir"], errors="coerce")
    all_day["target_value"] = pd.to_numeric(all_day["target_value"], errors="coerce")
    day["mean_ir"] = pd.to_numeric(day["mean_ir"], errors="coerce")
    day["mean_target_value"] = pd.to_numeric(day["mean_target_value"], errors="coerce")
    return all_day, day


def build_old_year_summary(best: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_doy = (
        best.groupby("decision_doy")
        .agg(
            old_successful_site_dates=("best_ir_for_date", "count"),
            old_mean_best_ir=("best_ir_for_date", "mean"),
            old_nonzero_rate=("best_ir_for_date", lambda s: float((s > 0).mean())),
            old_mean_best_target=("best_target_for_date", "mean"),
        )
        .reset_index()
    )
    by_year_site = (
        best.groupby(["year", "site"])
        .agg(
            successful_dates=("best_ir_for_date", "count"),
            nonzero_dates=("best_ir_for_date", lambda s: int((s > 0).sum())),
            nonzero_rate=("best_ir_for_date", lambda s: float((s > 0).mean())),
            mean_best_ir=("best_ir_for_date", "mean"),
            mean_best_target=("best_target_for_date", "mean"),
        )
        .reset_index()
    )
    keep = ["year", "site", "rows", "error_rows", "plan_rows", "plan_dates"]
    by_year_site = by_year_site.merge(summary[keep], on=["year", "site"], how="left")
    by_year_site["planned_dates"] = by_year_site["plan_dates"]
    by_year_site["failed_dates"] = by_year_site["error_rows"].astype(int)
    by_year_site["coverage_rate"] = by_year_site["successful_dates"] / by_year_site["planned_dates"]

    by_site = (
        by_year_site.groupby("site")
        .agg(
            successful_dates=("successful_dates", "sum"),
            planned_dates=("planned_dates", "sum"),
            failed_dates=("failed_dates", "sum"),
            nonzero_dates=("nonzero_dates", "sum"),
            mean_best_ir=("mean_best_ir", "mean"),
            mean_best_target=("mean_best_target", "mean"),
        )
        .reset_index()
    )
    by_site["coverage_rate"] = by_site["successful_dates"] / by_site["planned_dates"]
    by_site["nonzero_rate"] = by_site["nonzero_dates"] / by_site["successful_dates"]
    return by_doy, by_year_site, by_site


def build_paper_summary(paper_all: pd.DataFrame, paper_day: pd.DataFrame) -> pd.DataFrame:
    by_doy_member = (
        paper_all.groupby("decision_doy")
        .agg(
            paper_ensemble_rows=("ir", "count"),
            paper_mean_winner_ir=("ir", "mean"),
            paper_winner_nonzero_rate=("ir", lambda s: float((s > 0).mean())),
            paper_mean_winner_target=("target_value", "mean"),
        )
        .reset_index()
    )
    by_doy_date = paper_day.rename(
        columns={
            "mean_ir": "paper_date_mean_ir",
            "mean_target_value": "paper_date_mean_target",
        }
    )[["decision_doy", "date_t", "paper_date_mean_ir", "paper_date_mean_target"]]
    return by_doy_member.merge(by_doy_date, on="decision_doy", how="outer").sort_values("decision_doy")


def build_comparison(old_by_doy: pd.DataFrame, paper_by_doy: pd.DataFrame) -> pd.DataFrame:
    out = old_by_doy.merge(paper_by_doy, on="decision_doy", how="outer").sort_values("decision_doy")
    out["has_old"] = out["old_successful_site_dates"].notna()
    out["has_paper"] = out["paper_ensemble_rows"].notna()
    out["has_both"] = out["has_old"] & out["has_paper"]
    return out


def distribution_table(old_best: pd.DataFrame, paper_all: pd.DataFrame, paper_day: pd.DataFrame) -> pd.DataFrame:
    old_counts = old_best["best_ir_for_date"].value_counts().reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0).sort_index()
    paper_counts = paper_all["ir"].value_counts().reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0).sort_index()
    # Date-level paper mean_ir is not on the fixed list after averaging, so bin it
    # only for a separate mean schedule distribution diagnostic.
    return pd.DataFrame(
        {
            "irrigation_mm": PAPER_IRRIGATION_OPTIONS_MM,
            "old_doy_aligned_best_count": old_counts.to_numpy(),
            "old_doy_aligned_best_rate": old_counts.to_numpy() / max(1, int(old_counts.sum())),
            "paper_cached_ensemble_winner_count": paper_counts.to_numpy(),
            "paper_cached_ensemble_winner_rate": paper_counts.to_numpy() / max(1, int(paper_counts.sum())),
        }
    )


def save_distribution_plot(dist: pd.DataFrame, out_dir: Path) -> Path:
    x = np.arange(len(dist))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, dist["old_doy_aligned_best_rate"], width, label="Old-year DOY-aligned best labels", color="#4C78A8")
    ax.bar(x + width / 2, dist["paper_cached_ensemble_winner_rate"], width, label="2024 paper cached ensemble winners", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in dist["irrigation_mm"]])
    ax.set_xlabel("chosen / winning irrigation (mm)")
    ax.set_ylabel("rate")
    ax.set_title("Same-level decision-label irrigation distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_ir_distribution_rate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_doy_trend_plot(comp: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    x = comp["decision_doy"]

    axes[0].plot(x, comp["old_mean_best_ir"], marker="o", label="Old-year DOY-aligned mean best_ir", color="#4C78A8")
    axes[0].plot(x, comp["paper_date_mean_ir"], marker="s", label="2024 paper cached date mean_ir", color="#F58518")
    axes[0].set_ylabel("mean irrigation (mm)")
    axes[0].set_title("Paper-schedule-DOY aligned decision-label comparison")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, comp["old_nonzero_rate"], marker="o", label="Old-year best_ir nonzero rate", color="#4C78A8")
    axes[1].plot(x, comp["paper_winner_nonzero_rate"], marker="s", label="2024 paper ensemble winner nonzero rate", color="#F58518")
    axes[1].set_ylabel("nonzero rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    axes[2].plot(x, comp["old_mean_best_target"], marker="o", label="Old-year mean best target", color="#4C78A8")
    axes[2].plot(x, comp["paper_date_mean_target"], marker="s", label="2024 paper cached date mean target", color="#F58518")
    axes[2].set_ylabel("mean target")
    axes[2].set_xlabel("paper schedule decision DOY")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_doy_mean_ir_nonzero_target.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_common_doy_scatter(comp: pd.DataFrame, out_dir: Path) -> Path:
    common = comp[comp["has_both"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = [
        ("old_mean_best_ir", "paper_date_mean_ir", "mean irrigation (mm)"),
        ("old_nonzero_rate", "paper_winner_nonzero_rate", "nonzero rate"),
        ("old_mean_best_target", "paper_date_mean_target", "mean target"),
    ]
    for ax, (old_col, paper_col, label) in zip(axes, specs):
        ax.scatter(common[old_col], common[paper_col], s=60, color="#4C78A8")
        for row in common.itertuples(index=False):
            ax.text(getattr(row, old_col), getattr(row, paper_col), f"{int(row.decision_doy)}", fontsize=8)
        vals = pd.concat([common[old_col], common[paper_col]]).dropna()
        if not vals.empty:
            lo, hi = float(vals.min()), float(vals.max())
            pad = (hi - lo) * 0.08 if hi > lo else 1.0
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888888", linestyle="--", linewidth=1)
        ax.set_xlabel("old-year DOY-aligned aggregated")
        ax.set_ylabel("2024 paper cached")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Common-DOY same-level comparison")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_common_doy_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_coverage_plot(by_year_site: pd.DataFrame, out_dir: Path) -> Path:
    sites = list(by_year_site["site"].drop_duplicates())
    years = sorted(int(v) for v in by_year_site["year"].dropna().unique())
    x = np.arange(len(years))
    width = 0.18
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    for i, site in enumerate(sites):
        part = by_year_site[by_year_site["site"] == site].set_index("year").reindex(years)
        offset = (i - (len(sites) - 1) / 2) * width
        axes[0].bar(x + offset, part["coverage_rate"], width=width, color=colors[i], label=site)
        axes[1].bar(x + offset, part["nonzero_rate"], width=width, color=colors[i], label=site)
    axes[0].set_ylabel("coverage rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Old-year DOY-aligned label coverage")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("nonzero best_ir rate")
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(y) for y in years])
    axes[1].legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_old_year_coverage_nonzero_by_year_site.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(
    out_dir: Path,
    old_best: pd.DataFrame,
    old_merged: pd.DataFrame,
    paper_all: pd.DataFrame,
    paper_day: pd.DataFrame,
    by_site: pd.DataFrame,
    dist: pd.DataFrame,
    comp: pd.DataFrame,
    paths: list[Path],
) -> Path:
    bad = (
        old_merged.groupby(["year", "site", "date_t"])
        .agg(n_ir=("ir", "nunique"), rows=("ir", "count"))
        .reset_index()
        .query("n_ir != 8 or rows != 8")
    )
    summary = pd.DataFrame(
        [
            {"metric": "old-year successful site-dates", "value": len(old_best)},
            {"metric": "old-year candidate rows", "value": len(old_merged)},
            {"metric": "old-year complete successful response surfaces", "value": "yes" if bad.empty else "no"},
            {"metric": "2024 paper cached ensemble-winner rows", "value": len(paper_all)},
            {"metric": "2024 paper cached date-mean rows", "value": len(paper_day)},
            {"metric": "old-year nonzero best-label rate", "value": round(float((old_best["best_ir_for_date"] > 0).mean()), 3)},
            {"metric": "2024 paper ensemble-winner nonzero rate", "value": round(float((paper_all["ir"] > 0).mean()), 3)},
            {"metric": "2024 paper date-mean nonzero rate", "value": round(float((paper_day["mean_ir"] > 0).mean()), 3)},
        ]
    )
    report = out_dir / f"{FIGURE_PREFIX}_visualization_index_v1.md"
    lines = [
        "# Paper-Schedule-DOY Aligned Old-Year Labels vs 2024 Paper Cache",
        "",
        "Comparison level: decision labels and paper schedule DOYs.",
        "This is not a full response-surface reproduction of the paper ensemble process.",
        "",
        "Old-year data: 2015-2019 ERA5 single-scenario SWAP relabeling, aligned to paper schedule DOYs.",
        "Paper data: 2024 public cached schedule outputs.",
        "",
        "## Summary",
        markdown_table(summary),
        "",
        "## Old-Year Site Summary",
        markdown_table(by_site.round(6)),
        "",
        "## Irrigation Distribution",
        markdown_table(dist.round(6)),
        "",
        "## DOY Summary",
        markdown_table(comp.round(6)),
        "",
        "## Figures",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    if not bad.empty:
        lines.extend(["", "## Incomplete Successful Candidate Surfaces", markdown_table(bad)])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    paper_dir = Path(args.paper_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_best, old_merged, old_summary, _old_errors = read_old_year_runs(run_root, args.years, args.run_template, args.sites)
    paper_all, paper_day = read_paper_cache(paper_dir)

    old_by_doy, old_by_year_site, old_by_site = build_old_year_summary(old_best, old_summary)
    paper_by_doy = build_paper_summary(paper_all, paper_day)
    comp = build_comparison(old_by_doy, paper_by_doy)
    dist = distribution_table(old_best, paper_all, paper_day)

    old_best.to_csv(out_dir / f"{FIGURE_PREFIX}_old_year_best_by_date.csv", index=False)
    old_merged.to_csv(out_dir / f"{FIGURE_PREFIX}_old_year_candidate_response_surface.csv", index=False)
    old_by_year_site.to_csv(out_dir / f"{FIGURE_PREFIX}_old_year_summary_by_year_site.csv", index=False)
    old_by_site.to_csv(out_dir / f"{FIGURE_PREFIX}_old_year_summary_by_site.csv", index=False)
    comp.to_csv(out_dir / f"{FIGURE_PREFIX}_doy_summary.csv", index=False)
    dist.to_csv(out_dir / f"{FIGURE_PREFIX}_ir_distribution_rate.csv", index=False)

    paths = [
        save_distribution_plot(dist, out_dir),
        save_doy_trend_plot(comp, out_dir),
        save_common_doy_scatter(comp, out_dir),
        save_coverage_plot(old_by_year_site, out_dir),
    ]
    report = write_report(out_dir, old_best, old_merged, paper_all, paper_day, old_by_site, dist, comp, paths)

    print("wrote visualization report:")
    print(report)
    print("figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

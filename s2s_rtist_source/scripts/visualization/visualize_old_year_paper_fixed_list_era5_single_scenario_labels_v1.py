#!/usr/bin/env python3
"""Visualize old-year paper-fixed-list ERA5 single-scenario SWAP labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BASE = Path("site_general_surrogate_eval") / "older_year_swap_label_generation_v1"
DEFAULT_PREFIX = "older_year_2015_2019_paper_fixed_list_era5_single_scenario"
FIGURE_PREFIX = "old_year_paper_fixed_list_era5_single_scenario"
PAPER_IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_BASE / "visualizations_paper_fixed_list_era5_single_scenario_v1"),
    )
    parser.add_argument("--top-curves-per-site", type=int, default=4)
    return parser.parse_args()


def read_inputs(base_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_path = base_dir / f"{prefix}_candidate_response_surface_v1.csv"
    best_path = base_dir / f"{prefix}_best_by_date_v1.csv"
    summary_path = base_dir / f"{prefix}_summary_by_year_site_v1.csv"
    missing = [p for p in [candidate_path, best_path, summary_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(p) for p in missing))
    candidates = pd.read_csv(candidate_path)
    best = pd.read_csv(best_path)
    summary = pd.read_csv(summary_path)
    return candidates, best, summary


def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "decision_doy" in out.columns:
        out["decision_doy"] = pd.to_numeric(out["decision_doy"], errors="coerce")
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def save_summary_bars(summary: pd.DataFrame, out_dir: Path) -> Path:
    sites = list(summary["site"].drop_duplicates())
    years = sorted(summary["year"].drop_duplicates())
    x = np.arange(len(years))
    width = 0.18

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))

    for i, site in enumerate(sites):
        part = summary[summary["site"] == site].set_index("year").reindex(years)
        offset = (i - (len(sites) - 1) / 2) * width
        axes[0].bar(x + offset, part["coverage_rate"], width=width, color=colors[i], label=site)
        axes[1].bar(x + offset, part["nonzero_rate"], width=width, color=colors[i], label=site)

    axes[0].set_ylabel("coverage rate")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Successful label coverage by year and site")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_ylabel("nonzero best-ir rate")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Nonzero oracle irrigation rate by year and site")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(y) for y in years])
    axes[1].legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    fig.suptitle("Old-year SWAP relabeling summary: paper fixed irrigation list, ERA5 single scenario", y=0.99)
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_summary_coverage_nonzero_rate_by_year_site.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def heatmap_matrix(best: pd.DataFrame, site: str, value_col: str, years: list[int], doys: list[int]) -> np.ndarray:
    matrix = np.full((len(years), len(doys)), np.nan)
    part = best[best["site"] == site]
    for i, year in enumerate(years):
        for j, doy in enumerate(doys):
            rows = part[(part["year"] == year) & (part["decision_doy"] == doy)]
            if not rows.empty:
                matrix[i, j] = float(rows.iloc[0][value_col])
    return matrix


def save_best_heatmap(best: pd.DataFrame, out_dir: Path, value_col: str, title: str, filename: str, cmap: str) -> Path:
    sites = list(best["site"].drop_duplicates())
    years = sorted(int(v) for v in best["year"].dropna().unique())
    doys = sorted(int(v) for v in best["decision_doy"].dropna().unique())

    fig, axes = plt.subplots(len(sites), 1, figsize=(14, 2.7 * len(sites)), sharex=True)
    if len(sites) == 1:
        axes = [axes]

    finite_values = pd.to_numeric(best[value_col], errors="coerce").dropna()
    vmin = 0
    vmax = float(finite_values.max()) if not finite_values.empty else 1
    if value_col == "best_ir_for_date":
        vmax = max(vmax, max(PAPER_IRRIGATION_OPTIONS_MM))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d9d9d9")
    im = None
    for ax, site in zip(axes, sites):
        matrix = heatmap_matrix(best, site, value_col, years, doys)
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels([str(y) for y in years])
        ax.set_ylabel(site)
        ax.grid(False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    label = f"{matrix[i, j]:.0f}" if value_col == "best_ir_for_date" else f"{matrix[i, j]:.0f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")

    axes[-1].set_xticks(range(len(doys)))
    axes[-1].set_xticklabels([str(d) for d in doys], rotation=45, ha="right")
    axes[-1].set_xlabel("decision DOY")
    fig.suptitle(title, y=0.995)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.01)
        cbar.set_label(value_col)
    fig.tight_layout(rect=[0, 0, 0.98, 0.97])
    path = out_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_best_ir_distribution(best: pd.DataFrame, out_dir: Path) -> Path:
    sites = list(best["site"].drop_duplicates())
    fig, axes = plt.subplots(1, len(sites), figsize=(4.3 * len(sites), 4), sharey=True)
    if len(sites) == 1:
        axes = [axes]
    for ax, site in zip(axes, sites):
        part = best[best["site"] == site]
        counts = (
            part["best_ir_for_date"]
            .value_counts()
            .reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0)
            .sort_index()
        )
        ax.bar([str(int(v)) for v in counts.index], counts.values, color="#4C78A8")
        ax.set_title(site)
        ax.set_xlabel("best irrigation (mm)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("successful dates")
    fig.suptitle("Distribution of best irrigation labels")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_best_ir_distribution_by_site.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_top_response_curves(candidates: pd.DataFrame, best: pd.DataFrame, out_dir: Path, top_n: int) -> Path:
    sites = list(best["site"].drop_duplicates())
    fig, axes = plt.subplots(len(sites), 1, figsize=(12, 3.2 * len(sites)), sharex=True)
    if len(sites) == 1:
        axes = [axes]

    for ax, site in zip(axes, sites):
        top_dates = (
            best[best["site"] == site]
            .sort_values("best_target_for_date", ascending=False)
            .head(top_n)
        )
        for row in top_dates.itertuples(index=False):
            part = candidates[
                (candidates["site"] == site)
                & (candidates["year"] == row.year)
                & (candidates["date_t"] == row.date_t)
            ].sort_values("ir")
            if part.empty:
                continue
            label = f"{int(row.year)} {row.date_t} best={float(row.best_ir_for_date):g}"
            ax.plot(part["ir"], part["target_value"], marker="o", linewidth=1.8, label=label)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{site}\ntarget")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
    axes[-1].set_xlabel("candidate irrigation (mm)")
    fig.suptitle("Top target response curves by site")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_top_response_curves_by_site.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, paths: list[Path], summary: pd.DataFrame) -> Path:
    total_success = int(summary["successful_dates"].sum())
    total_planned = int(summary["planned_dates"].sum())
    total_failed = int(summary["failed_dates"].sum())
    total_nonzero = int(summary["nonzero_dates"].sum())
    report = out_dir / f"{FIGURE_PREFIX}_visualization_index_v1.md"
    lines = [
        "# Old-Year Label Expansion Visualizations",
        "",
        "Scope: paper fixed irrigation list, ERA5 single-scenario SWAP relabeling.",
        "This is not a full paper 9-member S2S ensemble label reproduction.",
        "",
        f"- planned year-site-dates: {total_planned}",
        f"- successful year-site-dates: {total_success}",
        f"- failed year-site-dates: {total_failed}",
        f"- nonzero best-ir successful dates: {total_nonzero}",
        "",
        "## Figures",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates, best, summary = read_inputs(base_dir, args.prefix)
    candidates = add_date_columns(candidates)
    best = add_date_columns(best)
    summary = add_date_columns(summary)

    paths = [
        save_summary_bars(summary, out_dir),
        save_best_heatmap(
            best,
            out_dir,
            value_col="best_ir_for_date",
            title="Best irrigation label by year, site, and decision DOY",
            filename=f"{FIGURE_PREFIX}_best_ir_heatmap_by_site_year_doy.png",
            cmap="YlGnBu",
        ),
        save_best_heatmap(
            best,
            out_dir,
            value_col="best_target_for_date",
            title="Best target value by year, site, and decision DOY",
            filename=f"{FIGURE_PREFIX}_best_target_heatmap_by_site_year_doy.png",
            cmap="YlOrRd",
        ),
        save_best_ir_distribution(best, out_dir),
        save_top_response_curves(candidates, best, out_dir, args.top_curves_per_site),
    ]
    report = write_report(out_dir, paths, summary)

    print("wrote visualization report:")
    print(report)
    print("figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()



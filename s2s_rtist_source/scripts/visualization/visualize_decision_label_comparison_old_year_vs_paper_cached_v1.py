#!/usr/bin/env python3
"""Compare old-year relabeling and cached paper outputs at decision-label level.

This script deliberately avoids response-surface comparisons. The old-year data
has a full single-scenario 8-candidate response surface, while the public paper
cache only has date-level means and per-ensemble winning rows. To make the
visual comparison fair, both sides are reduced to chosen/best irrigation labels
and DOY-level summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OLD_BASE = Path("site_general_surrogate_eval") / "older_year_swap_label_generation_v1"
DEFAULT_OLD_PREFIX = "older_year_2015_2019_paper_fixed_list_era5_single_scenario"
DEFAULT_PAPER_DIR = Path("model3_opt_sto_upload") / "Maize"
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "decision_label_comparison_old_year_vs_paper_cached_v1"
FIGURE_PREFIX = "decision_label_comparison_old_year_era5_single_vs_paper_cached"
PAPER_IRRIGATION_OPTIONS_MM = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--old-base-dir", default=str(DEFAULT_OLD_BASE))
    parser.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX)
    parser.add_argument("--paper-dir", default=str(DEFAULT_PAPER_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def date_to_doy(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series, format="%d-%b-%Y", errors="coerce").dt.dayofyear


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


def read_inputs(project_root: Path, old_base: Path, old_prefix: str, paper_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_best_path = project_root / old_base / f"{old_prefix}_best_by_date_v1.csv"
    paper_all_path = project_root / paper_dir / "all_day_ir_var_results.csv"
    paper_day_path = project_root / paper_dir / "day_scheduled.csv"
    missing = [p for p in [old_best_path, paper_all_path, paper_day_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(str(p) for p in missing))

    old_best = pd.read_csv(old_best_path)
    paper_all = pd.read_csv(paper_all_path)
    paper_day = pd.read_csv(paper_day_path)

    old_best["decision_doy"] = pd.to_numeric(old_best["decision_doy"], errors="coerce")
    old_best["best_ir_for_date"] = pd.to_numeric(old_best["best_ir_for_date"], errors="coerce")
    old_best["best_target_for_date"] = pd.to_numeric(old_best["best_target_for_date"], errors="coerce")

    paper_all["decision_doy"] = date_to_doy(paper_all["date_t"])
    paper_all["ir"] = pd.to_numeric(paper_all["ir"], errors="coerce")
    paper_all["target_value"] = pd.to_numeric(paper_all["target_value"], errors="coerce")

    paper_day["decision_doy"] = date_to_doy(paper_day["date_t"])
    paper_day["mean_ir"] = pd.to_numeric(paper_day["mean_ir"], errors="coerce")
    paper_day["mean_target_value"] = pd.to_numeric(paper_day["mean_target_value"], errors="coerce")
    return old_best, paper_all, paper_day


def make_distribution_table(old_best: pd.DataFrame, paper_all: pd.DataFrame) -> pd.DataFrame:
    old_counts = old_best["best_ir_for_date"].value_counts().reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0).sort_index()
    paper_counts = paper_all["ir"].value_counts().reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0).sort_index()
    out = pd.DataFrame(
        {
            "irrigation_mm": PAPER_IRRIGATION_OPTIONS_MM,
            "old_year_best_count": old_counts.to_numpy(),
            "old_year_best_rate": old_counts.to_numpy() / max(1, int(old_counts.sum())),
            "paper_cached_ensemble_winner_count": paper_counts.to_numpy(),
            "paper_cached_ensemble_winner_rate": paper_counts.to_numpy() / max(1, int(paper_counts.sum())),
        }
    )
    return out


def make_doy_summary(old_best: pd.DataFrame, paper_all: pd.DataFrame, paper_day: pd.DataFrame) -> pd.DataFrame:
    old_doy = (
        old_best.groupby("decision_doy")
        .agg(
            old_year_label_count=("best_ir_for_date", "count"),
            old_year_mean_best_ir=("best_ir_for_date", "mean"),
            old_year_nonzero_rate=("best_ir_for_date", lambda s: float((s > 0).mean())),
            old_year_mean_best_target=("best_target_for_date", "mean"),
        )
        .reset_index()
    )
    paper_member_doy = (
        paper_all.groupby("decision_doy")
        .agg(
            paper_cached_ensemble_rows=("ir", "count"),
            paper_cached_mean_winner_ir_from_rows=("ir", "mean"),
            paper_cached_winner_nonzero_rate=("ir", lambda s: float((s > 0).mean())),
            paper_cached_mean_winner_target_from_rows=("target_value", "mean"),
        )
        .reset_index()
    )
    paper_date_doy = paper_day.rename(
        columns={
            "mean_ir": "paper_cached_date_mean_ir",
            "mean_target_value": "paper_cached_date_mean_target",
        }
    )[["decision_doy", "paper_cached_date_mean_ir", "paper_cached_date_mean_target"]]

    out = old_doy.merge(paper_member_doy, on="decision_doy", how="outer")
    out = out.merge(paper_date_doy, on="decision_doy", how="outer")
    out["has_both_sources"] = out["old_year_label_count"].notna() & out["paper_cached_ensemble_rows"].notna()
    return out.sort_values("decision_doy").reset_index(drop=True)


def save_distribution_plot(dist: pd.DataFrame, out_dir: Path) -> Path:
    x = np.arange(len(dist))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, dist["old_year_best_rate"], width, label="Old-year ERA5 single-scenario best labels", color="#4C78A8")
    ax.bar(x + width / 2, dist["paper_cached_ensemble_winner_rate"], width, label="Paper cached ensemble winners", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in dist["irrigation_mm"]])
    ax.set_ylim(0, max(0.05, float(dist[["old_year_best_rate", "paper_cached_ensemble_winner_rate"]].max().max()) * 1.18))
    ax.set_xlabel("chosen / winning irrigation (mm)")
    ax.set_ylabel("rate")
    ax.set_title("Decision-label irrigation distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_ir_distribution_rate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_doy_trend_plot(doy_summary: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    x = doy_summary["decision_doy"]

    axes[0].plot(x, doy_summary["old_year_mean_best_ir"], marker="o", label="Old-year mean best_ir", color="#4C78A8")
    axes[0].plot(x, doy_summary["paper_cached_date_mean_ir"], marker="s", label="Paper cached date mean_ir", color="#F58518")
    axes[0].set_ylabel("mean irrigation (mm)")
    axes[0].set_title("DOY-level decision-label summary")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, doy_summary["old_year_nonzero_rate"], marker="o", label="Old-year nonzero best_ir rate", color="#4C78A8")
    axes[1].plot(x, doy_summary["paper_cached_winner_nonzero_rate"], marker="s", label="Paper cached ensemble winner nonzero rate", color="#F58518")
    axes[1].set_ylabel("nonzero rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    axes[2].plot(x, doy_summary["old_year_mean_best_target"], marker="o", label="Old-year mean best target", color="#4C78A8")
    axes[2].plot(x, doy_summary["paper_cached_date_mean_target"], marker="s", label="Paper cached date mean target", color="#F58518")
    axes[2].set_ylabel("mean target")
    axes[2].set_xlabel("decision DOY")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    for ax in axes:
        for doy in doy_summary.loc[~doy_summary["has_both_sources"], "decision_doy"]:
            ax.axvline(doy, color="#999999", alpha=0.08, linewidth=4)

    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_doy_mean_ir_nonzero_target.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_common_doy_scatter(doy_summary: pd.DataFrame, out_dir: Path) -> Path:
    common = doy_summary[doy_summary["has_both_sources"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    specs = [
        ("old_year_mean_best_ir", "paper_cached_date_mean_ir", "mean irrigation (mm)"),
        ("old_year_nonzero_rate", "paper_cached_winner_nonzero_rate", "nonzero rate"),
        ("old_year_mean_best_target", "paper_cached_date_mean_target", "mean target"),
    ]
    for ax, (old_col, paper_col, label) in zip(axes, specs):
        ax.scatter(common[old_col], common[paper_col], s=60, color="#4C78A8")
        for row in common.itertuples(index=False):
            ax.text(getattr(row, old_col), getattr(row, paper_col), f"{int(row.decision_doy)}", fontsize=8)
        vals = pd.concat([common[old_col], common[paper_col]]).dropna()
        if not vals.empty:
            lo, hi = float(vals.min()), float(vals.max())
            pad = (hi - lo) * 0.08 if hi > lo else 1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888888", linestyle="--", linewidth=1)
        ax.set_xlabel("old-year aggregated")
        ax.set_ylabel("paper cached")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Common-DOY decision-label comparison")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_common_doy_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(out_dir: Path, old_best: pd.DataFrame, paper_all: pd.DataFrame, paper_day: pd.DataFrame, dist: pd.DataFrame, doy_summary: pd.DataFrame, paths: list[Path]) -> Path:
    report = out_dir / f"{FIGURE_PREFIX}_visualization_index_v1.md"
    common_doys = doy_summary.loc[doy_summary["has_both_sources"], "decision_doy"].dropna().astype(int).tolist()
    summary = pd.DataFrame(
        [
            {"metric": "old-year best-label rows", "value": len(old_best)},
            {"metric": "paper cached ensemble-winner rows", "value": len(paper_all)},
            {"metric": "paper cached date-mean rows", "value": len(paper_day)},
            {"metric": "old-year nonzero best-label rate", "value": round(float((old_best["best_ir_for_date"] > 0).mean()), 3)},
            {"metric": "paper cached ensemble-winner nonzero rate", "value": round(float((paper_all["ir"] > 0).mean()), 3)},
            {"metric": "paper cached date-mean nonzero rate", "value": round(float((paper_day["mean_ir"] > 0).mean()), 3)},
            {"metric": "common decision DOYs", "value": common_doys},
        ]
    )
    lines = [
        "# Decision-Label Comparison: Old-Year Relabeling vs Paper Cached Outputs",
        "",
        "Comparison level: chosen/best irrigation labels and DOY-level summaries.",
        "This is not a response-surface comparison.",
        "",
        "Old-year data: paper-fixed-list ERA5 single-scenario SWAP relabeling.",
        "Paper data: cached public schedule outputs with date means and per-ensemble winning rows.",
        "",
        "## Summary",
        markdown_table(summary),
        "",
        "## Irrigation Distribution",
        markdown_table(dist.round(6)),
        "",
        "## Common-DOY Summary",
        markdown_table(doy_summary[doy_summary["has_both_sources"]].round(6)),
        "",
        "## Figures",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old_best, paper_all, paper_day = read_inputs(
        project_root=project_root,
        old_base=Path(args.old_base_dir),
        old_prefix=args.old_prefix,
        paper_dir=Path(args.paper_dir),
    )
    dist = make_distribution_table(old_best, paper_all)
    doy_summary = make_doy_summary(old_best, paper_all, paper_day)

    dist.to_csv(out_dir / f"{FIGURE_PREFIX}_ir_distribution_rate.csv", index=False)
    doy_summary.to_csv(out_dir / f"{FIGURE_PREFIX}_doy_summary.csv", index=False)

    paths = [
        save_distribution_plot(dist, out_dir),
        save_doy_trend_plot(doy_summary, out_dir),
        save_common_doy_scatter(doy_summary, out_dir),
    ]
    report = write_report(out_dir, old_best, paper_all, paper_day, dist, doy_summary, paths)

    print("wrote decision-label comparison report:")
    print(report)
    print("figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualize cached paper-style irrigation scheduling outputs.

The public cache contains date-level scheduled means plus per-ensemble winning
irrigation rows. It does not contain the full member-by-candidate response
surface, so these figures intentionally visualize only the available cached
paper outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("model3_opt_sto_upload") / "Maize"
DEFAULT_OUT_DIR = Path("site_general_surrogate_eval") / "paper_cached_schedule_outputs_visualization_v1"
FIGURE_PREFIX = "paper_cached_schedule_outputs"
PAPER_IRRIGATION_OPTIONS_MM = [0, 10, 15, 20, 25, 30, 40, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def parse_date_order(df: pd.DataFrame, date_col: str = "date_t") -> pd.DataFrame:
    out = df.copy()
    out["_date"] = pd.to_datetime(out[date_col], format="%d-%b-%Y", errors="coerce")
    return out.sort_values("_date").drop(columns=["_date"])


def ensemble_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["file_num_ens"].astype(str)
        + "_sf"
        + df["sf_time"].astype(str)
        + "_exp"
        + df["exp_n"].astype(str)
    )


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


def save_schedule_over_time(day: pd.DataFrame, out_dir: Path) -> Path:
    day = parse_date_order(day)
    labels = day["date_t"].astype(str).tolist()
    x = np.arange(len(day))

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(x, day["mean_ir"], color="#4C78A8", width=0.65, label="mean irrigation")
    line = ax2.plot(x, day["mean_target_value"], color="#E45756", marker="o", linewidth=2, label="mean target")

    ax1.set_ylabel("mean irrigation (mm)")
    ax2.set_ylabel("mean target value")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_title("Cached paper schedule: date-level ensemble mean")

    handles = [bars, line[0]]
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper left")

    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_date_mean_ir_and_target.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_winner_ir_heatmap(all_day: pd.DataFrame, out_dir: Path) -> Path:
    data = all_day.copy()
    data["ensemble_key"] = ensemble_key(data)
    data = parse_date_order(data)
    pivot = data.pivot_table(index="ensemble_key", columns="date_t", values="ir", aggfunc="first")
    pivot = pivot.reindex(columns=data["date_t"].drop_duplicates().tolist())

    fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(pivot))))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=60)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Cached paper per-ensemble winning irrigation")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{float(value):.0f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("winning irrigation (mm)")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_ensemble_winner_ir_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_winner_target_heatmap(all_day: pd.DataFrame, out_dir: Path) -> Path:
    data = all_day.copy()
    data["ensemble_key"] = ensemble_key(data)
    data = parse_date_order(data)
    pivot = data.pivot_table(index="ensemble_key", columns="date_t", values="target_value", aggfunc="first")
    pivot = pivot.reindex(columns=data["date_t"].drop_duplicates().tolist())

    fig, ax = plt.subplots(figsize=(13, max(5, 0.35 * len(pivot))))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Cached paper per-ensemble winning target value")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("target value")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_ensemble_winner_target_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_ir_distribution(all_day: pd.DataFrame, out_dir: Path) -> Path:
    counts = all_day["ir"].value_counts().reindex(PAPER_IRRIGATION_OPTIONS_MM, fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(v) for v in counts.index], counts.values, color="#59A14F")
    ax.set_title("Cached paper winning irrigation distribution across ensemble rows")
    ax.set_xlabel("winning irrigation (mm)")
    ax.set_ylabel("row count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_winner_ir_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_mean_audit(day: pd.DataFrame, all_day: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    calc = (
        all_day.groupby("date_t")
        .agg(
            cached_ensemble_rows=("ir", "count"),
            mean_ir_from_rows=("ir", "mean"),
            mean_target_from_rows=("target_value", "mean"),
        )
        .reset_index()
    )
    audit = day.merge(calc, on="date_t", how="left")
    audit["mean_ir_delta"] = audit["mean_ir_from_rows"] - audit["mean_ir"]
    audit["mean_target_delta"] = audit["mean_target_from_rows"] - audit["mean_target_value"]
    audit = parse_date_order(audit)
    audit.to_csv(out_dir / f"{FIGURE_PREFIX}_mean_reconstruction_audit.csv", index=False)

    x = np.arange(len(audit))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, audit["mean_ir"], marker="o", label="day_scheduled mean_ir")
    axes[0].plot(x, audit["mean_ir_from_rows"], marker="s", label="mean from all_day rows")
    axes[0].set_ylabel("mean irrigation (mm)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, audit["mean_target_value"], marker="o", label="day_scheduled mean target")
    axes[1].plot(x, audit["mean_target_from_rows"], marker="s", label="mean from all_day rows")
    axes[1].set_ylabel("mean target")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(audit["date_t"], rotation=45, ha="right")
    fig.suptitle("Cached paper mean reconstruction audit")
    fig.tight_layout()
    path = out_dir / f"{FIGURE_PREFIX}_mean_reconstruction_audit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, audit


def write_report(out_dir: Path, all_day: pd.DataFrame, day: pd.DataFrame, audit: pd.DataFrame, paths: list[Path]) -> Path:
    report = out_dir / f"{FIGURE_PREFIX}_visualization_index_v1.md"
    summary = pd.DataFrame(
        [
            {"item": "day_scheduled rows", "value": len(day)},
            {"item": "all_day winning rows", "value": len(all_day)},
            {"item": "unique decision dates", "value": all_day["date_t"].nunique()},
            {"item": "unique ensemble keys", "value": ensemble_key(all_day).nunique()},
            {"item": "available irrigation values", "value": sorted(all_day["ir"].dropna().unique().tolist())},
        ]
    )
    lines = [
        "# Cached Paper Schedule Visualization",
        "",
        "These figures visualize cached public paper-style outputs only.",
        "The cache stores date-level scheduled means and per-ensemble winning irrigation rows; it does not expose the full 9-member x 8-candidate response surface.",
        "",
        "## Data Summary",
        markdown_table(summary),
        "",
        "## Mean Reconstruction Audit",
        markdown_table(
            audit[
                [
                    "date_t",
                    "cached_ensemble_rows",
                    "mean_ir",
                    "mean_ir_from_rows",
                    "mean_ir_delta",
                    "mean_target_value",
                    "mean_target_from_rows",
                    "mean_target_delta",
                ]
            ].round(6)
        ),
        "",
        "## Figures",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_day_path = input_dir / "all_day_ir_var_results.csv"
    day_path = input_dir / "day_scheduled.csv"
    if not all_day_path.exists():
        raise FileNotFoundError(f"Missing {all_day_path}")
    if not day_path.exists():
        raise FileNotFoundError(f"Missing {day_path}")

    all_day = pd.read_csv(all_day_path)
    day = pd.read_csv(day_path)

    paths = [
        save_schedule_over_time(day, out_dir),
        save_winner_ir_heatmap(all_day, out_dir),
        save_winner_target_heatmap(all_day, out_dir),
        save_ir_distribution(all_day, out_dir),
    ]
    audit_path, audit = save_mean_audit(day, all_day, out_dir)
    paths.append(audit_path)
    report = write_report(out_dir, all_day, day, audit, paths)

    print("wrote visualization report:")
    print(report)
    print("figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()



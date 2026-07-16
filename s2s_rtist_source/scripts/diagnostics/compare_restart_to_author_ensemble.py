"""Compare current restart decisions with cached author ensemble decisions.

Run inside Maize_restart_dataset. The copied Maize directory already contains
the author's cached all_day_ir_var_results.csv and day_scheduled.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESTART_BEST = "restart_decision_best_by_date.csv"
AUTHOR_ALL = "all_day_ir_var_results.csv"
AUTHOR_MEAN = "day_scheduled.csv"

OUT_CSV = "restart_vs_author_ensemble.csv"
AUTHOR_SUMMARY_CSV = "author_ensemble_summary.csv"
OUT_MD = "restart_vs_author_ensemble.md"


def classify_gap(abs_gap: float) -> str:
    if abs_gap <= 5:
        return "close"
    if abs_gap <= 10:
        return "moderate"
    return "large"


def main() -> None:
    for file_name in [RESTART_BEST, AUTHOR_ALL, AUTHOR_MEAN]:
        if not Path(file_name).exists():
            raise FileNotFoundError(f"Missing {file_name}; run inside Maize_restart_dataset copied from Maize.")

    restart = pd.read_csv(RESTART_BEST)
    author_all = pd.read_csv(AUTHOR_ALL)
    author_mean = pd.read_csv(AUTHOR_MEAN)

    ens_summary = (
        author_all.groupby("date_t", sort=False)
        .agg(
            ensemble_n=("ir", "size"),
            ensemble_mean_ir_from_members=("ir", "mean"),
            ensemble_median_ir=("ir", "median"),
            ensemble_min_ir=("ir", "min"),
            ensemble_max_ir=("ir", "max"),
            ensemble_zero_count=("ir", lambda s: int((s == 0).sum())),
            ensemble_mean_target_from_members=("target_value", "mean"),
        )
        .reset_index()
    )
    ir_values = (
        author_all.groupby("date_t", sort=False)["ir"]
        .apply(lambda s: "/".join(str(int(v)) for v in s.tolist()))
        .reset_index(name="ensemble_member_best_irs")
    )
    ens_summary = ens_summary.merge(ir_values, on="date_t", how="left")
    ens_summary = ens_summary.merge(author_mean, on="date_t", how="left")
    ens_summary.to_csv(AUTHOR_SUMMARY_CSV, index=False)

    merged = restart.merge(ens_summary, on="date_t", how="left")
    merged = merged.rename(
        columns={
            "best_ir_for_date": "restart_best_ir",
            "best_target_for_date": "restart_best_target",
            "mean_ir": "author_mean_ir",
            "mean_target_value": "author_mean_target",
        }
    )
    merged["restart_minus_author_mean_ir"] = merged["restart_best_ir"] - merged["author_mean_ir"]
    merged["abs_restart_author_gap"] = merged["restart_minus_author_mean_ir"].abs()
    merged["gap_class"] = merged["abs_restart_author_gap"].map(classify_gap)

    cols = [
        "date_t",
        "restart_best_ir",
        "restart_best_target",
        "author_mean_ir",
        "author_mean_target",
        "restart_minus_author_mean_ir",
        "abs_restart_author_gap",
        "gap_class",
        "ensemble_n",
        "ensemble_member_best_irs",
        "ensemble_median_ir",
        "ensemble_min_ir",
        "ensemble_max_ir",
        "ensemble_zero_count",
    ]
    merged[cols].to_csv(OUT_CSV, index=False)

    lines = [
        "# Restart vs Author Ensemble Decisions",
        "",
        "This compares the current single-scenario restart smoke dataset with the author's cached 9-member ensemble decision summary.",
        "",
        "| date | restart best | author mean | gap | class | ensemble member best irrigation amounts |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in merged[cols].itertuples(index=False):
        lines.append(
            f"| {row.date_t} | {row.restart_best_ir:.1f} | {row.author_mean_ir:.1f} | "
            f"{row.restart_minus_author_mean_ir:.1f} | {row.gap_class} | {row.ensemble_member_best_irs} |"
        )

    large = merged[merged["gap_class"] == "large"]
    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            f"- Compared dates: {len(merged)}",
            f"- Close gaps (<=5 mm): {(merged['gap_class'] == 'close').sum()}",
            f"- Moderate gaps (<=10 mm): {(merged['gap_class'] == 'moderate').sum()}",
            f"- Large gaps (>10 mm): {(merged['gap_class'] == 'large').sum()}",
            "",
        ]
    )
    if not large.empty:
        lines.append("Large-gap dates:")
        for row in large.itertuples(index=False):
            lines.append(
                f"- {row.date_t}: restart={row.restart_best_ir:.1f} mm, "
                f"author_mean={row.author_mean_ir:.1f} mm, members={row.ensemble_member_best_irs}"
            )
        lines.append("")
    lines.extend(
        [
            "Interpretation:",
            "",
            "- The restart dataset is currently a single-scenario smoke dataset.",
            "- The author's published schedule averages 9 ensemble-member decisions.",
            "- Large gaps are expected where ensemble uncertainty is high or where the single scenario differs from the ensemble mean.",
            "- The next formal dataset should therefore include ensemble member identity and forecast/weather inputs.",
            "",
            f"Wrote `{OUT_CSV}` and `{AUTHOR_SUMMARY_CSV}`.",
        ]
    )
    Path(OUT_MD).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

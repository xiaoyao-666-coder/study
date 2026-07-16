"""Prepare cached author ensemble decision labels.

This does not create candidate-level SWAP response samples. It only reshapes the
author-provided all_day_ir_var_results.csv into member-level and date-level
decision labels, useful as a reference while raw S2S ensemble weather inputs are
not yet available.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


AUTHOR_ALL = "all_day_ir_var_results.csv"
AUTHOR_MEAN = "day_scheduled.csv"
MEMBER_OUT = "author_ensemble_member_labels.csv"
DATE_OUT = "author_ensemble_date_labels.csv"
REPORT_OUT = "author_ensemble_label_report.md"


def main() -> None:
    for file_name in [AUTHOR_ALL, AUTHOR_MEAN]:
        if not Path(file_name).exists():
            raise FileNotFoundError(f"Missing {file_name}; run inside a copied Maize directory.")

    all_day = pd.read_csv(AUTHOR_ALL)
    day_mean = pd.read_csv(AUTHOR_MEAN)

    all_day = all_day.rename(
        columns={
            "ir": "member_best_ir",
            "target_value": "member_best_target",
        }
    )
    all_day["ensemble_member_id"] = (
        all_day["file_num_ens"].astype(str)
        + "_"
        + all_day["sf_time"].astype(str).str.zfill(2)
        + "_exp"
        + all_day["exp_n"].astype(str).str.zfill(2)
    )
    member_cols = [
        "date_t",
        "ensemble_member_id",
        "file_num_ens",
        "sf_time",
        "exp_n",
        "member_best_ir",
        "member_best_target",
        "cwdm_value",
        "cwso_value",
    ]
    all_day[member_cols].to_csv(MEMBER_OUT, index=False)

    summary = (
        all_day.groupby("date_t", sort=False)
        .agg(
            ensemble_n=("ensemble_member_id", "size"),
            member_mean_ir=("member_best_ir", "mean"),
            member_median_ir=("member_best_ir", "median"),
            member_min_ir=("member_best_ir", "min"),
            member_max_ir=("member_best_ir", "max"),
            zero_member_count=("member_best_ir", lambda s: int((s == 0).sum())),
            member_mean_target=("member_best_target", "mean"),
        )
        .reset_index()
    )
    members = (
        all_day.groupby("date_t", sort=False)["member_best_ir"]
        .apply(lambda s: "/".join(str(int(v)) for v in s.tolist()))
        .reset_index(name="member_best_irs")
    )
    date_labels = summary.merge(members, on="date_t", how="left").merge(day_mean, on="date_t", how="left")
    date_labels = date_labels.rename(
        columns={
            "mean_ir": "author_reported_mean_ir",
            "mean_target_value": "author_reported_mean_target",
        }
    )
    date_labels.to_csv(DATE_OUT, index=False)

    lines = [
        "# Author Ensemble Label Table",
        "",
        "These files reshape cached author outputs only.",
        "",
        "They do not contain all irrigation candidates per ensemble member.",
        "They do not contain raw S2S weather inputs.",
        "",
        f"- Member rows: {len(all_day)}",
        f"- Decision dates: {all_day['date_t'].nunique()}",
        f"- Output: `{MEMBER_OUT}`",
        f"- Output: `{DATE_OUT}`",
        "",
        "## Date-Level Labels",
        "",
        "| date | author mean ir | member best irs | zero members |",
        "|---|---:|---|---:|",
    ]
    for row in date_labels.itertuples(index=False):
        lines.append(
            f"| {row.date_t} | {row.author_reported_mean_ir:.2f} | "
            f"{row.member_best_irs} | {int(row.zero_member_count)} |"
        )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "For surrogate training we still need candidate-level rows:",
            "",
            "```text",
            "date_t x ensemble_member x irrigation_candidate -> CWDM/CWSO/DVS/target_value",
            "```",
            "",
            "The current cached author file only gives the winning irrigation amount for each ensemble member.",
        ]
    )
    Path(REPORT_OUT).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

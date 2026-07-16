"""Create a compact human-readable summary for restart surrogate samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = "surrogate_samples_restart.csv"
SUMMARY_CSV = "restart_dataset_brief.csv"
SUMMARY_MD = "restart_dataset_brief.md"


def main() -> None:
    path = Path(INPUT_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}; run prepare_restart_surrogate_table.py first.")

    df = pd.read_csv(path)
    rows = []
    for date_t, group in df.groupby("date_t", sort=False):
        ranked = group.sort_values("target_value", ascending=False).reset_index(drop=True)
        best = ranked.iloc[0]
        second = ranked.iloc[1]
        rows.append(
            {
                "date_t": date_t,
                "decision_doy": int(best["decision_doy"]),
                "best_ir": float(best["ir"]),
                "best_target": float(best["target_value"]),
                "second_ir": float(second["ir"]),
                "second_target": float(second["target_value"]),
                "best_margin": float(best["target_value"] - second["target_value"]),
                "min_target": float(group["target_value"].min()),
                "max_target": float(group["target_value"].max()),
                "n_candidates": int(len(group)),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    lines = [
        "# Restart Decision Dataset Brief",
        "",
        f"- Rows: {len(df)}",
        f"- Decision dates: {summary['date_t'].nunique()}",
        f"- Candidates per date: {sorted(df['ir'].unique().tolist())}",
        "",
        "## Best Irrigation By Date",
        "",
        "| date | best_ir | best_target | second_ir | second_target | margin | note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        if row.best_ir == 0:
            note = "no irrigation"
        elif row.best_margin < 2:
            note = "close call"
        else:
            note = "clear"
        lines.append(
            f"| {row.date_t} | {row.best_ir:.0f} | {row.best_target:.1f} | "
            f"{row.second_ir:.0f} | {row.second_target:.1f} | {row.best_margin:.1f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `best_ir`: the irrigation candidate with the largest target value.",
            "- `second_ir`: the runner-up candidate.",
            "- `margin`: `best_target - second_target`; small margins mean the decision is sensitive.",
            "- `no irrigation`: best candidate is 0 mm.",
            "",
            f"Wrote `{SUMMARY_CSV}` and `{SUMMARY_MD}`.",
        ]
    )
    Path(SUMMARY_MD).write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()

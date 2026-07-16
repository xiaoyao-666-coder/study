#!/usr/bin/env python3
"""Create compact deliverables for short-term surrogate v1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("Maize_shortterm_surrogate_v1")
FULL = OUT_DIR / "shortterm_surrogate_samples_v1_with_true_state_soil.csv"
COMPACT = OUT_DIR / "shortterm_surrogate_samples_v1_compact.csv"
LABELS = OUT_DIR / "shortterm_surrogate_labels_v1.csv"
REPORT = OUT_DIR / "shortterm_surrogate_v1_final_report.md"


COMPACT_COLS = [
    "sample_id",
    "site_id",
    "date_t",
    "decision_doy",
    "horizon_days",
    "candidate_ir",
    "candidate_ir_sequence",
    "state_dvs",
    "state_lai",
    "state_rootd",
    "state_cwdm",
    "state_cwso",
    "soil_layer_count",
    "soil_h_mean_0_30_cm",
    "soil_h_mean_30_60_cm",
    "soil_h_mean_60_100_cm",
    "soil_h_mean_0_100_cm",
    "soil_h_min_0_100_cm",
    "soil_h_max_0_100_cm",
    "end_daynr",
    "dvs_7d",
    "lai_7d",
    "rootd_7d",
    "cwdm_7d",
    "cwso_7d",
    "target_7d",
    "no_irrigation_target_7d",
    "net_gain_7d",
    "best_ir_for_date",
    "best_target_for_date",
    "is_best_ir",
    "target_regret",
    "current_state_status",
    "forecast_weather_status",
]


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def classify_margin(x: float) -> str:
    if x <= 1.0:
        return "very close"
    if x <= 3.0:
        return "close"
    return "clear"


def main() -> None:
    if not FULL.exists():
        raise FileNotFoundError(f"Missing full table: {FULL}")

    df = pd.read_csv(FULL)
    missing = sorted(set(COMPACT_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Full table is missing expected compact columns: {missing}")

    compact = df[COMPACT_COLS].copy()
    compact.to_csv(COMPACT, index=False)

    best = (
        compact.loc[compact["is_best_ir"], [
            "date_t",
            "decision_doy",
            "best_ir_for_date",
            "best_target_for_date",
            "soil_h_mean_0_30_cm",
            "soil_h_mean_30_60_cm",
            "soil_h_mean_60_100_cm",
        ]]
        .sort_values("decision_doy")
        .reset_index(drop=True)
    )

    margins = []
    for date_t, group in compact.groupby("date_t", sort=False):
        sorted_group = group.sort_values("target_7d", ascending=False).reset_index(drop=True)
        best_row = sorted_group.iloc[0]
        second_row = sorted_group.iloc[1] if len(sorted_group) > 1 else sorted_group.iloc[0]
        margin = float(best_row["target_7d"] - second_row["target_7d"])
        margins.append(
            {
                "date_t": date_t,
                "best_ir": float(best_row["candidate_ir"]),
                "best_target": round(float(best_row["target_7d"]), 3),
                "second_ir": float(second_row["candidate_ir"]),
                "second_target": round(float(second_row["target_7d"]), 3),
                "margin": round(margin, 3),
                "margin_class": classify_margin(margin),
            }
        )
    margin_df = pd.DataFrame(margins)
    margin_csv = OUT_DIR / "shortterm_surrogate_v1_decision_margin.csv"
    margin_df.to_csv(margin_csv, index=False)

    report = [
        "# Short-Term Surrogate V1 Final Report",
        "",
        "## What This Dataset Contains",
        "",
        "This is the first 7-day rolling optimization surrogate dataset built from the validated SWAP restart workflow.",
        "",
        "- Sample unit: one site x one decision date x one candidate irrigation amount.",
        "- Inputs available: candidate irrigation, true pre-decision crop state, and SWAP pressure-head soil state summaries.",
        "- Label available: 7-day target and 7-day net gain relative to no irrigation.",
        "- Weather sequence block: still pending; current file keeps a status flag for later GEFS replacement.",
        "",
        "## Files",
        "",
        f"- Compact table: `{COMPACT}`",
        f"- Full table with soil profile JSON: `{FULL}`",
        f"- Label table: `{LABELS}`",
        f"- Decision margin table: `{margin_csv}`",
        "",
        "## Dataset Size",
        "",
        f"- Rows: {len(compact)}",
        f"- Sites: {compact['site_id'].nunique()}",
        f"- Decision dates: {compact[['site_id', 'date_t']].drop_duplicates().shape[0]}",
        f"- Candidate irrigation amounts: {sorted(compact['candidate_ir'].unique().tolist())}",
        "",
        "## Best Decisions",
        "",
        markdown_table(margin_df),
        "",
        "## Notes",
        "",
        "- `soil_h_*` fields are pressure head summaries from SWAP `.end` files, not volumetric water content.",
        "- More negative pressure head generally indicates drier soil water status.",
        "- The full table preserves `soil_h_profile_json`; the compact table omits it for easier viewing and sharing.",
        "- Next step: add historical weather/irrigation sequences and future 7-day weather forecasts.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote {COMPACT}")
    print(f"Wrote {margin_csv}")
    print(f"Wrote {REPORT}")
    print("\nCompact preview:")
    print(compact.head(12).to_string(index=False))


if __name__ == "__main__":
    main()

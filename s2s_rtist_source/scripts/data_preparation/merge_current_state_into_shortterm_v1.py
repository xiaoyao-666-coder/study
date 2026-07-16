#!/usr/bin/env python3
"""Merge current-state features into short-term surrogate v1 samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("Maize_shortterm_surrogate_v1")
    samples_path = output_dir / "shortterm_surrogate_samples_v1.csv"
    state_path = output_dir / "current_state_by_date.csv"
    merged_path = output_dir / "shortterm_surrogate_samples_v1_with_state.csv"
    report_path = output_dir / "shortterm_surrogate_with_state_report.md"

    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing current-state file: {state_path}")

    samples = pd.read_csv(samples_path)
    state = pd.read_csv(state_path)

    state_cols = [
        "date_t",
        "decision_doy",
        "state_source",
        "state_daynr",
        "state_dvs",
        "state_lai",
        "state_rootd",
        "state_cwdm",
        "state_cwso",
        "state_soil_water_layers_status",
    ]
    missing = sorted(set(state_cols) - set(state.columns))
    if missing:
        raise ValueError(f"Current-state table is missing columns: {missing}")

    merged = samples.merge(state[state_cols], on=["date_t", "decision_doy"], how="left")
    if merged["state_source"].isna().any():
        missing_rows = int(merged["state_source"].isna().sum())
        raise RuntimeError(f"{missing_rows} sample rows did not match a current-state row.")

    merged["current_state_status"] = merged["state_source"].apply(
        lambda x: "filled_from_pre_decision_state"
        if str(x).endswith(".crp")
        else "filled_placeholder_from_0mm_7day_result"
    )
    merged.to_csv(merged_path, index=False)

    source_counts = merged["current_state_status"].value_counts().to_dict()
    lines = [
        "# Short-Term Surrogate V1 With Current State",
        "",
        f"- Input samples: `{samples_path}`",
        f"- Current-state table: `{state_path}`",
        f"- Output: `{merged_path}`",
        f"- Rows: {len(merged)}",
        f"- Source status counts: {source_counts}",
        "",
        "If the status is `filled_placeholder_from_0mm_7day_result`, the next formal run should save true pre-decision `.crp` files.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {merged_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

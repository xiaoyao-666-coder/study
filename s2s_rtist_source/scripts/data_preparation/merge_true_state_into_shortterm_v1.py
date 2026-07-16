#!/usr/bin/env python3
"""Merge true pre-decision state into short-term surrogate v1 samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("Maize_shortterm_surrogate_v1")
    samples_path = output_dir / "shortterm_surrogate_samples_v1.csv"
    state_path = output_dir / "current_state_by_date_true.csv"
    merged_path = output_dir / "shortterm_surrogate_samples_v1_with_true_state.csv"
    report_path = output_dir / "shortterm_surrogate_true_state_report.md"

    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing true state file: {state_path}")

    samples = pd.read_csv(samples_path)
    state = pd.read_csv(state_path)
    state_cols = [
        "date_t",
        "decision_doy",
        "pre_end_doy",
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
        raise ValueError(f"True state table missing columns: {missing}")

    merged = samples.merge(state[state_cols], on=["date_t", "decision_doy"], how="left")
    if merged["state_source"].isna().any():
        missing_rows = int(merged["state_source"].isna().sum())
        raise RuntimeError(f"{missing_rows} sample rows did not match true state rows.")

    merged["current_state_status"] = "filled_from_true_pre_decision_state"
    merged.to_csv(merged_path, index=False)

    report = [
        "# Short-Term Surrogate V1 With True Current State",
        "",
        f"- Input samples: `{samples_path}`",
        f"- True state table: `{state_path}`",
        f"- Output: `{merged_path}`",
        f"- Rows: {len(merged)}",
        f"- Decision dates: {merged[['site_id', 'date_t']].drop_duplicates().shape[0]}",
        "- Current state status: filled from true pre-decision SWAP crop output.",
        "- Soil water layer state is still pending.",
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {merged_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
